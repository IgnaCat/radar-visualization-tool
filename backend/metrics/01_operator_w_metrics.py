#!/usr/bin/env python3
"""
01_operator_w_metrics.py
========================
Mide las características del operador W disperso y lo compara con PyART.

Genera números para:
  - tab:operador_w  (Sección 5.1 — cold start, propiedades de W)
  - tab:cold_warm   (Sección 5.1 — speedup frente a PyART)

Uso (desde backend/):
    python metrics/01_operator_w_metrics.py

Requisitos:
    conda activate radar-env   (o el entorno que tenga GDAL)
    pip install -r requirements.txt

Política de caché:
    Este script NO escribe ni lee el caché del servidor.
    Construye W en memoria y lo descarta al terminar.
    El tamaño en disco se mide guardando en un archivo temporal.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pyart
from scipy.sparse import save_npz

# ── Path setup ────────────────────────────────────────────────────────────────
# Ejecutar desde backend/  →  sys.path ya incluye backend/
# Ejecutar desde otro lugar →  agregamos backend/ manualmente
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# ── Imports del módulo app (sin tocar caché) ──────────────────────────────────
#
#   grid_compute.py  →  build_W_operator   (pura: temp files + scipy, sin caché)
#   grid_interpolate →  apply_operator     (pura: NumPy/scipy)
#   grid_geometry    →  utilidades de geometría (pura: NumPy/math)
#   radar_common     →  resolve_field, safe_range_max_m (puras)
#   constants        →  TOA, ROI_PARAMS_BY_VOLUME
#
#   get_gate_xyz_coords / get_grid_xyz_coords se definen inline abajo
#   para evitar importar grid_builder, que carga cache.py al módulo.
#
from app.services.radar_processing.grid_compute import build_W_operator
from app.services.radar_processing.grid_interpolate import apply_operator
from app.services.radar_processing.grid_geometry import (
    calculate_grid_resolution,
    calculate_grid_points,
    calculate_z_limits,
    infer_blind_range_m,
    infer_last_gate_range_m,
)
from app.services.radar_common import resolve_field, safe_range_max_m
from app.core.constants import TOA, ROI_PARAMS_BY_VOLUME, ROI_PARAMS_VOL01


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN  —  ajustar si el archivo de radar es distinto
# ══════════════════════════════════════════════════════════════════════════════

NC_FILE = Path(_BACKEND).parent / "biribiri" / "RMA1_0315_01_20250819T001715Z.nc"
VOLUME  = "01"
FIELD   = "DBZH"

# Número de repeticiones del warm start (se descarta la 1ª y se promedian las demás)
N_REPS_WARM = 3


# ══════════════════════════════════════════════════════════════════════════════
# Funciones auxiliares puras (sin dependencia de grid_builder ni caché)
# ══════════════════════════════════════════════════════════════════════════════

def get_gate_xyz_coords(radar: pyart.core.Radar) -> np.ndarray:
    """Coordenadas cartesianas (x, y, z) de todos los gates del radar.

    Returns
    -------
    np.ndarray, shape (nrays * ngates, 3), dtype float32
    """
    x = radar.gate_x["data"].ravel()
    y = radar.gate_y["data"].ravel()
    z = radar.gate_z["data"].ravel()
    xyz = np.column_stack((x, y, z))
    expected = radar.nrays * radar.ngates
    if xyz.shape[0] != expected:
        raise ValueError(
            f"Gate XYZ mismatch: got {xyz.shape[0]}, expected {expected}. "
            "Puede haber transiciones filtradas en el archivo."
        )
    return xyz.astype(np.float32, copy=False)


def get_grid_xyz_coords(
    grid_shape: tuple,
    grid_limits: tuple,
    dtype=np.float32,
) -> np.ndarray:
    """Coordenadas (x, y, z) de todos los voxels de una grilla 3D regular.

    Returns
    -------
    np.ndarray, shape (N_voxels, 3)
    """
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = grid_limits
    nz, ny, nx = grid_shape

    def axis(vmin, vmax, n):
        if n == 1:
            return np.array([(vmin + vmax) / 2.0], dtype=dtype)
        return np.linspace(vmin, vmax, n, dtype=dtype)

    z = axis(zmin, zmax, nz)
    y = axis(ymin, ymax, ny)
    x = axis(xmin, xmax, nx)

    X = np.tile(x, nz * ny)
    Y = np.tile(np.repeat(y, nx), nz)
    Z = np.repeat(z, ny * nx)
    xyz = np.column_stack((X, Y, Z)).astype(dtype, copy=False)

    assert xyz.shape == (nz * ny * nx, 3), f"Shape inesperado: {xyz.shape}"
    return xyz


def grid_with_pyart(
    radar: pyart.core.Radar,
    field_name: str,
    grid_shape: tuple,
    grid_limits: tuple,
    h_factor: float,
    nb: float,
    bsp: float,
    min_radius: float,
    weighting_function: str = "Barnes2",
) -> tuple:
    """Grilla usando PyART (implementación de referencia).

    Returns
    -------
    (grid3d, t_pyart_s)
        grid3d: np.ma.MaskedArray (nz, ny, nx)
        t_pyart_s: tiempo de ejecución en segundos
    """
    grid_origin = (
        float(radar.latitude["data"][0]),
        float(radar.longitude["data"][0]),
    )
    t0 = time.time()
    grid = pyart.map.grid_from_radars(
        radar,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        gridding_algo="map_gates_to_grid",
        grid_origin=grid_origin,
        fields=[field_name],
        weighting_function=weighting_function,
        gatefilters=False,
        roi_func="dist_beam",
        nb=nb,
        bsp=bsp,
        h_factor=np.full(1, h_factor, dtype=np.float32),
        min_radius=min_radius,
    )
    t_pyart = time.time() - t0
    fd = grid.fields[field_name]["data"]
    grid3d = fd[0] if fd.ndim == 4 else fd
    return grid3d, t_pyart


def w_ram_mb(W) -> float:
    """Tamaño del operador W en RAM (MB)."""
    return (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1024**2


def w_disk_mb(W) -> float:
    """Tamaño del operador W serializado en disco como .npz comprimido (MB)."""
    fd, tmp = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        save_npz(tmp, W, compressed=True)
        size = os.path.getsize(tmp) / 1024**2
    finally:
        os.unlink(tmp)
    return size


def sep(title: str = "") -> None:
    print(f"\n{'─' * 64}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 64}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("  01_operator_w_metrics.py")
    print("=" * 64)

    # ── Verificar archivo ──────────────────────────────────────
    if not NC_FILE.exists():
        sys.exit(f"\n[ERROR] Archivo no encontrado: {NC_FILE}\n"
                 "Editar NC_FILE al inicio del script.")

    # ── Leer radar ─────────────────────────────────────────────
    sep("Leyendo radar")
    t0 = time.time()
    radar = pyart.io.read(str(NC_FILE))
    t_read = time.time() - t0

    print(f"  Archivo  : {NC_FILE.name}")
    print(f"  Lectura  : {t_read:.2f} s")
    print(f"  Sweeps   : {radar.nsweeps}")
    print(f"  Nrays    : {radar.nrays}")
    print(f"  Ngates   : {radar.ngates}")
    print(f"  N_gates total : {radar.nrays * radar.ngates:,}")

    field_name, _  = resolve_field(radar, FIELD)
    range_max_m    = safe_range_max_m(radar)
    blind_range_m  = infer_blind_range_m(radar)
    last_gate_m    = infer_last_gate_range_m(radar)
    lowest_elev    = float(radar.fixed_angle["data"].min())

    print(f"  Campo    : {field_name}")
    print(f"  range_max: {range_max_m/1e3:.0f} km")
    print(f"  blind    : {blind_range_m:.0f} m")
    print(f"  last_gate: {last_gate_m:.0f} m")
    print(f"  elev_min : {lowest_elev:.2f}°")

    # ── Geometría de grilla (exactamente igual que producción) ─
    sep("Configuración de grilla")
    grid_res_xy, grid_res_z = calculate_grid_resolution(VOLUME)

    z_lim_full = calculate_z_limits(
        range_max_m=range_max_m,
        elevation=0,
        radar_fixed_angles=radar.fixed_angle["data"],
    )
    z_lim = (z_lim_full[0], z_lim_full[1])   # (z_min, z_max) — ignora elev_deg
    y_lim = (-range_max_m,  range_max_m)
    x_lim = (-range_max_m,  range_max_m)
    grid_limits = (z_lim, y_lim, x_lim)

    nz, ny, nx = calculate_grid_points(
        z_lim, y_lim, x_lim, grid_res_xy, grid_res_z
    )
    grid_shape = (nz, ny, nx)

    roi_params          = ROI_PARAMS_BY_VOLUME.get(VOLUME, ROI_PARAMS_VOL01)
    h_factor, nb, bsp, min_radius = roi_params

    print(f"  Resolución XY / Z  : {grid_res_xy} / {grid_res_z} m")
    print(f"  Grid shape (Z,Y,X) : {nz} × {ny} × {nx}")
    print(f"  N_voxels           : {nz * ny * nx:,}")
    print(f"  Z limits           : {z_lim[0]/1e3:.0f} – {z_lim[1]/1e3:.0f} km")
    print(f"  ROI params         : h_factor={h_factor}, nb={nb}°, "
          f"bsp={bsp}, min_radius={min_radius} m")

    # ── Coordenadas ─────────────────────────────────────────────
    sep("Coordenadas")
    gates_xyz  = get_gate_xyz_coords(radar)
    voxels_xyz = get_grid_xyz_coords(grid_shape, grid_limits)
    print(f"  gates_xyz  : {gates_xyz.shape}")
    print(f"  voxels_xyz : {voxels_xyz.shape}")

    # ══════════════════════════════════════════════════════════════
    # COLD START — construcción de W
    # ══════════════════════════════════════════════════════════════
    sep("Cold start — construcción de W")
    print("  (puede tardar varios minutos en la primera ejecución)")

    t0 = time.time()
    W = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz,
        toa=TOA,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        weight_func="nearest",
        max_neighbors=1,
        blind_range_m=blind_range_m,
        last_gate_range_m=last_gate_m,
        lowest_elev_deg=lowest_elev,
        n_workers=None,   # usa min(4, cpu_count()-1) automáticamente
    )
    t_build = time.time() - t0

    # Propiedades del operador
    nnz            = W.nnz
    nvoxels        = W.shape[0]
    ngates_total   = W.shape[1]
    # Voxels con al menos un gate contribuyendo (filas no vacías del CSR)
    voxels_filled  = int((np.diff(W.indptr) > 0).sum())
    avg_neighbors  = nnz / voxels_filled if voxels_filled > 0 else 0.0
    density        = nnz / (nvoxels * ngates_total)
    ram_mb         = w_ram_mb(W)

    print(f"\n  N_voxels (W.shape[0])    : {nvoxels:,}")
    print(f"  N_gates  (W.shape[1])    : {ngates_total:,}")
    print(f"  Dims grilla (Z × Y × X)  : {nz} × {ny} × {nx}")
    print(f"  nnz                      : {nnz:,}")
    print(f"  Voxels con datos         : {voxels_filled:,} "
          f"({100*voxels_filled/nvoxels:.1f}%)")
    print(f"  Vecinos promedio / voxel : {avg_neighbors:.1f}")
    print(f"  Densidad de W            : {density:.3e}")
    print(f"  RAM                      : {ram_mb:.2f} MB")
    print(f"  Tiempo construcción      : {t_build:.2f} s")

    sep("Tamaño en disco (.npz comprimido)")
    print("  Midiendo (guarda en temp, mide, borra)...")
    disk_mb = w_disk_mb(W)
    print(f"  Disco : {disk_mb:.2f} MB")

    # ══════════════════════════════════════════════════════════════
    # WARM START — aplicación de W
    # ══════════════════════════════════════════════════════════════
    sep(f"Warm start — aplicación de W (× {N_REPS_WARM} reps)")
    field_data = radar.fields[field_name]["data"]

    times_apply = []
    for i in range(N_REPS_WARM):
        t0 = time.time()
        grid3d = apply_operator(W, field_data, grid_shape)
        times_apply.append(time.time() - t0)
        print(f"  Rep {i+1}: {times_apply[-1]:.4f} s"
              + ("  (descartada)" if i == 0 else ""))

    t_apply      = float(np.mean(times_apply[1:]))
    vox_valid    = int((~grid3d.mask).sum())
    mean_dbz     = float(grid3d.data[~grid3d.mask].mean()) if vox_valid > 0 else float("nan")

    print(f"\n  t_apply (prom. {N_REPS_WARM-1} reps, excl. 1ª) : {t_apply:.4f} s")
    print(f"  Voxels válidos : {vox_valid:,} / {grid3d.size:,} "
          f"({100*vox_valid/grid3d.size:.1f}%)")
    print(f"  Media {FIELD}    : {mean_dbz:.2f} dBZ")

    # ══════════════════════════════════════════════════════════════
    # PyART — implementación de referencia
    # ══════════════════════════════════════════════════════════════
    sep("PyART — referencia (puede tardar ~10 min)")
    print("  Ejecutando pyart.map.grid_from_radars ...")
    grid3d_pyart, t_pyart = grid_with_pyart(
        radar, field_name, grid_shape, grid_limits,
        h_factor, nb, bsp, min_radius, "nearest",
    )
    vox_valid_pyart = int((~grid3d_pyart.mask).sum())
    mean_dbz_pyart  = float(
        grid3d_pyart.data[~grid3d_pyart.mask].mean()
    ) if vox_valid_pyart > 0 else float("nan")

    print(f"  t_pyart        : {t_pyart:.2f} s")
    print(f"  Voxels válidos : {vox_valid_pyart:,} / {grid3d_pyart.size:,}")
    print(f"  Media {FIELD}    : {mean_dbz_pyart:.2f} dBZ")

    # ══════════════════════════════════════════════════════════════
    # RESUMEN PARA TESIS
    # ══════════════════════════════════════════════════════════════
    speedup_cold  = t_pyart / t_build  if t_build  > 0 else float("nan")
    speedup_warm  = t_pyart / t_apply  if t_apply  > 0 else float("nan")

    sep("RESUMEN — copiar en resultados.tex")
    print(f"""
┌─ tab:operador_w ───────────────────────────────────────────────┐
  Voxels de la grilla (N_voxels)   : {nvoxels:>14,}
  Dimensiones (Z × Y × X)          : {nz} × {ny} × {nx}
  Elementos no nulos (nnz)          : {nnz:>14,}
  Vecinos promedio por voxel        : {avg_neighbors:>14.1f}
  Densidad de W                     : {density:>14.3e}
  Memoria en RAM                    : {ram_mb:>13.2f} MB
  Tamaño en disco (.npz)            : {disk_mb:>13.2f} MB
  Tiempo de construcción            : {t_build:>13.2f} s
└────────────────────────────────────────────────────────────────┘

┌─ tab:cold_warm ────────────────────────────────────────────────┐
  PyART (referencia)                : {t_pyart:>8.2f} s   (1×)
  Cold start (construye W)          : {t_build:>8.2f} s   (~{speedup_cold:.1f}×)
  Warm start (aplica W cacheado)    : {t_apply:>8.4f} s   (~{speedup_warm:.0f}×)
└────────────────────────────────────────────────────────────────┘
""")

    print("─" * 64)
    print("  Rows LaTeX listos para pegar en resultados.tex:")
    print("─" * 64)
    print()
    print("% ── tab:operador_w ──")
    print(f"Voxels de la grilla ($N_{{\\text{{voxels}}}}$) & {nvoxels:,} \\\\")
    print(f"Dimensiones de la grilla          & ${nz} \\\\times {ny} \\\\times {nx}$ \\\\")
    print(f"Elementos no nulos ($\\text{{nnz}}$)  & {nnz:,} \\\\")
    print(f"Vecinos promedio por voxel         & {avg_neighbors:.1f} \\\\")
    print(f"Densidad de $\\mathbf{{W}}$           & ${density:.3e}$ \\\\")
    print(f"Memoria en RAM                     & {ram_mb:.2f}~MB \\\\")
    print(f"Tamaño en disco (\\texttt{{.npz}})    & {disk_mb:.2f}~MB \\\\")
    print(f"Tiempo de construcción             & {t_build:.2f}~s \\\\")
    print()
    print("% ── tab:cold_warm ──")
    print(f"Py-ART (referencia)                & {t_pyart:.2f}~s & $1\\\\times$ \\\\")
    print(f"Cold start (construye $\\mathbf{{W}}$) & {t_build:.2f}~s & $\\\\sim {speedup_cold:.1f}\\\\times$ \\\\")
    print(f"Warm start (aplica $\\mathbf{{W}}$ cacheado) & {t_apply:.2f}~s & $\\\\sim {speedup_warm:.0f}\\\\times$ \\\\")


if __name__ == "__main__":
    main()
