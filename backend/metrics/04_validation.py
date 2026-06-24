#!/usr/bin/env python3
"""
04_validation.py
================
Compara la grilla DBZH producida por el operador W contra PyART.
Genera números para tab:validacion y la figura fig:validacion_ppi.

Uso (desde backend/):
    python metrics/04_validation.py

Política de caché local:
    Las grillas intermedias (grid_W y grid_pyart) se guardan en
    metrics/cache/ la primera vez. Las ejecuciones siguientes las
    cargan desde disco, evitando los ~6 minutos de cómputo.
    Para forzar recálculo: borrar metrics/cache/val_*.npy

Salida:
    - Métricas en consola (para tab:validacion)
    - Rows LaTeX listos para pegar
    - metrics/figures/validation_comparison.png  (para fig:validacion_ppi)
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pyart

# ── Path setup ────────────────────────────────────────────────────────────────
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

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
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

NC_FILE   = Path(_BACKEND).parent / "biribiri" / "RMA1_0315_01_20250819T001715Z.nc"
VOLUME    = "01"
FIELD     = "DBZH"

CACHE_DIR   = Path(__file__).parent / "cache"
FIGURES_DIR = Path(__file__).parent / "figures"


# ══════════════════════════════════════════════════════════════════════════════
# Funciones auxiliares (idénticas a 01_operator_w_metrics.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_gate_xyz_coords(radar: pyart.core.Radar) -> np.ndarray:
    x = radar.gate_x["data"].ravel()
    y = radar.gate_y["data"].ravel()
    z = radar.gate_z["data"].ravel()
    xyz = np.column_stack((x, y, z))
    expected = radar.nrays * radar.ngates
    if xyz.shape[0] != expected:
        raise ValueError(f"Gate XYZ mismatch: got {xyz.shape[0]}, expected {expected}.")
    return xyz.astype(np.float32, copy=False)


def get_grid_xyz_coords(grid_shape, grid_limits, dtype=np.float32) -> np.ndarray:
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = grid_limits
    nz, ny, nx = grid_shape

    def axis(vmin, vmax, n):
        return np.array([(vmin + vmax) / 2.0], dtype=dtype) if n == 1 \
               else np.linspace(vmin, vmax, n, dtype=dtype)

    z = axis(zmin, zmax, nz)
    y = axis(ymin, ymax, ny)
    x = axis(xmin, xmax, nx)

    X = np.tile(x, nz * ny)
    Y = np.tile(np.repeat(y, nx), nz)
    Z = np.repeat(z, ny * nx)
    return np.column_stack((X, Y, Z)).astype(dtype, copy=False)


def grid_with_pyart(radar, field_name, grid_shape, grid_limits,
                    h_factor, nb, bsp, min_radius, weighting_function="nearest"):
    grid_origin = (
        float(radar.latitude["data"][0]),
        float(radar.longitude["data"][0]),
    )
    t0 = time.time()
    grid = pyart.map.grid_from_radars(
        radar,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        gridding_algo="map_to_grid",
        grid_origin=grid_origin,
        fields=[field_name],
        weighting_function=weighting_function,
        gatefilters=None,
        roi_func="dist_beam",
        nb=nb, bsp=bsp, h_factor=h_factor, min_radius=min_radius,
    )
    t_pyart = time.time() - t0
    fd = grid.fields[field_name]["data"]
    grid3d = fd[0] if fd.ndim == 4 else fd
    return grid3d, t_pyart


def sep(title=""):
    print(f"\n{'─' * 64}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 64}")


# ══════════════════════════════════════════════════════════════════════════════
# Caché local de grillas
# ══════════════════════════════════════════════════════════════════════════════

def _cache_paths():
    return {
        "meta":        CACHE_DIR / "val_meta.json",
        "grid_W":      CACHE_DIR / "val_grid_W.npy",
        "mask_W":      CACHE_DIR / "val_mask_W.npy",
        "grid_pyart":  CACHE_DIR / "val_grid_pyart.npy",
        "mask_pyart":  CACHE_DIR / "val_mask_pyart.npy",
    }


def load_grids_from_cache():
    """Carga grillas desde disco. Devuelve (grid_W, grid_pyart, meta) o None."""
    paths = _cache_paths()
    if not all(p.exists() for p in paths.values()):
        return None
    try:
        meta = json.loads(paths["meta"].read_text())
        shape = tuple(meta["grid_shape"])
        data_W     = np.load(paths["grid_W"])
        mask_W     = np.load(paths["mask_W"])
        data_py    = np.load(paths["grid_pyart"])
        mask_py    = np.load(paths["mask_pyart"])
        grid_W     = np.ma.MaskedArray(data_W.reshape(shape),    mask=mask_W.reshape(shape))
        grid_pyart = np.ma.MaskedArray(data_py.reshape(shape),   mask=mask_py.reshape(shape))
        print("  [caché] Grillas cargadas desde metrics/cache/")
        return grid_W, grid_pyart, meta
    except Exception as e:
        print(f"  [caché] Error al cargar: {e} — recalculando.")
        return None


def save_grids_to_cache(grid_W, grid_pyart, meta):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths()
    np.save(paths["grid_W"],     grid_W.data.ravel())
    np.save(paths["mask_W"],     np.ma.getmaskarray(grid_W).ravel())
    np.save(paths["grid_pyart"], grid_pyart.data.ravel())
    np.save(paths["mask_pyart"], np.ma.getmaskarray(grid_pyart).ravel())
    # Convertir listas anidadas a formato serializable
    meta_serial = {k: (list(v) if isinstance(v, tuple) else v) for k, v in meta.items()}
    if "grid_limits" in meta_serial:
        meta_serial["grid_limits"] = [list(lim) for lim in meta["grid_limits"]]
    paths["meta"].write_text(json.dumps(meta_serial, indent=2))
    print("  [caché] Grillas guardadas en metrics/cache/")


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de validación
# ══════════════════════════════════════════════════════════════════════════════

def compute_validation_metrics(grid_W, grid_pyart):
    """
    Compara grid_W (operador) contra grid_pyart (referencia).
    Solo evalúa sobre voxels válidos en AMBAS grillas.
    """
    mask_W    = np.ma.getmaskarray(grid_W)
    mask_py   = np.ma.getmaskarray(grid_pyart)
    valid     = (~mask_W) & (~mask_py)

    v_W   = grid_W.data[valid].astype(np.float64)
    v_py  = grid_pyart.data[valid].astype(np.float64)
    diff  = v_W - v_py

    rmse     = float(np.sqrt(np.mean(diff**2)))
    mae      = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff)))
    mean_W   = float(np.mean(v_W))
    mean_py  = float(np.mean(v_py))
    rel_err  = abs(mean_W - mean_py) / abs(mean_py) * 100 if mean_py != 0 else float("nan")
    n_valid  = int(valid.sum())
    n_total  = grid_W.size

    return dict(
        rmse=rmse, mae=mae, max_diff=max_diff,
        mean_W=mean_W, mean_py=mean_py, rel_err=rel_err,
        n_valid=n_valid, n_total=n_total,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Figura de comparación (3 paneles)
# ══════════════════════════════════════════════════════════════════════════════

def make_comparison_figure(grid_W, grid_pyart, grid_shape, grid_limits,
                           radar_lat: float, radar_lon: float, output_path: Path):
    """
    Genera figura de 3 paneles con proyección Mercator y colormap grc_th
    (igual que png.py):
        izquierda : DBZH operador W
        centro    : DBZH Py-ART
        derecha   : diferencia (W − Py-ART)

    Usa el z-level con mayor cantidad de voxels válidos en ambas grillas.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as e:
        print(f"  [figura] Dependencia no disponible ({e}) — omitiendo figura.")
        return

    # ── Colormaps (igual que png.py) ──────────────────────────────
    from app.utils.colores import get_cmap_grc_th
    cmap_dbz = get_cmap_grc_th()
    cmap_dbz.set_bad("white")         # celdas sin datos → blanco

    cmap_diff = plt.cm.RdBu_r.copy()
    cmap_diff.set_bad("white")

    # ── Extent geográfico (metros → lat/lon) ──────────────────────
    from math import cos, pi
    (_, _), (ymin, ymax), (xmin, xmax) = grid_limits
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * cos(radar_lat * pi / 180.0)
    lat_min = radar_lat + ymin / m_per_deg_lat
    lat_max = radar_lat + ymax / m_per_deg_lat
    lon_min = radar_lon + xmin / m_per_deg_lon
    lon_max = radar_lon + xmax / m_per_deg_lon
    extent = [lon_min, lon_max, lat_min, lat_max]  # para imshow en PlateCarree

    # ── Z-level con más voxels válidos en ambas grillas ───────────
    nz, ny, nx = grid_shape
    valid_both  = (~np.ma.getmaskarray(grid_W)) & (~np.ma.getmaskarray(grid_pyart))
    valid_per_z = valid_both.sum(axis=(1, 2))
    z_idx       = int(np.argmax(valid_per_z))
    z_km        = (grid_limits[0][1] / nz) * z_idx / 1000.0   # altura aproximada

    sl_W    = grid_W[z_idx, :, :]
    sl_py   = grid_pyart[z_idx, :, :]
    sl_diff = np.ma.MaskedArray(
        sl_W.data - sl_py.data,
        mask=(np.ma.getmaskarray(sl_W) | np.ma.getmaskarray(sl_py)),
    )

    vmin, vmax = -30.0, 70.0
    diff_abs = max(abs(float(np.ma.min(sl_diff))), abs(float(np.ma.max(sl_diff)))) \
               if sl_diff.count() > 0 else 5.0
    diff_max = min(diff_abs, 10.0)

    # ── Figura ────────────────────────────────────────────────────
    proj     = ccrs.Mercator()
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6),
        subplot_kw={"projection": proj},
        facecolor="white",
    )
    fig.suptitle(
        f"Validación DBZH — nivel {z_idx}  |  "
        f"{valid_per_z[z_idx]:,} voxels válidos en ambas grillas",
        fontsize=14, fontweight="bold",
    )

    titles = ["Operador W (herramienta)", "Py-ART (referencia)", "Diferencia W − Py-ART"]
    datas  = [sl_W, sl_py, sl_diff]
    cmaps  = [cmap_dbz, cmap_dbz, cmap_diff]
    vmins  = [vmin, vmin, -diff_max]
    vmaxs  = [vmax, vmax,  diff_max]
    labels = ["dBZ", "dBZ", "dBZ"]

    ims = []
    for ax, data, cmap, vm_n, vm_x, title, label in zip(
            axes, datas, cmaps, vmins, vmaxs, titles, labels):

        ax.set_extent(extent, crs=data_crs)
        ax.set_facecolor("white")

        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            transform=data_crs,
            vmin=vm_n, vmax=vm_x,
            cmap=cmap,
            interpolation="nearest",
        )
        ims.append(im)

        # Features geográficas
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="gray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="gray")
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="lightgray")

        ax.set_title(title, fontsize=13, pad=8)

        # Gridlines con etiquetas
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}

        # Colorbar bajo cada panel
        cb = plt.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.04, fraction=0.046, label=label)
        cb.ax.tick_params(labelsize=10)
        cb.set_label(label, fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                pad_inches=0.1, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"  [figura] Guardada en: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  04_validation.py")
    print("=" * 64)

    if not NC_FILE.exists():
        sys.exit(f"\n[ERROR] Archivo no encontrado: {NC_FILE}\n"
                 "Editar NC_FILE al inicio del script.")

    # ── Intentar cargar grillas desde caché ───────────────────────
    sep("Cargando grillas (o calculando si no hay caché)")
    cached = load_grids_from_cache()

    if cached is not None:
        grid_W, grid_pyart, meta = cached
        grid_shape  = tuple(meta["grid_shape"])
        grid_limits = tuple(tuple(lim) for lim in meta["grid_limits"])
    else:
        # Calcular todo desde cero
        t0 = time.time()
        radar = pyart.io.read(str(NC_FILE))
        print(f"  Lectura radar: {time.time()-t0:.2f} s")

        field_name, _  = resolve_field(radar, FIELD)
        range_max_m    = safe_range_max_m(radar)
        blind_range_m  = infer_blind_range_m(radar)
        last_gate_m    = infer_last_gate_range_m(radar)
        lowest_elev    = float(radar.fixed_angle["data"].min())

        grid_res_xy, grid_res_z = calculate_grid_resolution(VOLUME)
        z_lim_full = calculate_z_limits(
            range_max_m=range_max_m, elevation=0,
            radar_fixed_angles=radar.fixed_angle["data"],
        )
        z_lim      = (z_lim_full[0], z_lim_full[1])
        y_lim      = (-range_max_m, range_max_m)
        x_lim      = (-range_max_m, range_max_m)
        grid_limits = (z_lim, y_lim, x_lim)

        nz, ny, nx  = calculate_grid_points(z_lim, y_lim, x_lim, grid_res_xy, grid_res_z)
        grid_shape  = (nz, ny, nx)

        roi_params          = ROI_PARAMS_BY_VOLUME.get(VOLUME, ROI_PARAMS_VOL01)
        h_factor, nb, bsp, min_radius = roi_params

        print(f"  Grid shape (Z,Y,X): {nz} × {ny} × {nx}")

        # Build W
        sep("Cold start — construcción de W")
        print("  (puede tardar varios minutos...)")
        gates_xyz  = get_gate_xyz_coords(radar)
        voxels_xyz = get_grid_xyz_coords(grid_shape, grid_limits)

        t0 = time.time()
        W = build_W_operator(
            gates_xyz=gates_xyz, voxels_xyz=voxels_xyz, toa=TOA,
            h_factor=h_factor, nb=nb, bsp=bsp, min_radius=min_radius,
            weight_func="nearest", max_neighbors=1,
            blind_range_m=blind_range_m, last_gate_range_m=last_gate_m,
            lowest_elev_deg=lowest_elev, n_workers=None,
        )
        t_build = time.time() - t0
        print(f"  W construido en {t_build:.2f} s  (nnz={W.nnz:,})")

        # Apply W
        field_data = radar.fields[field_name]["data"]
        grid_W = apply_operator(W, field_data, grid_shape)

        # PyART reference
        sep("PyART — referencia (puede tardar ~10 min)")
        grid_pyart, t_pyart = grid_with_pyart(
            radar, field_name, grid_shape, grid_limits,
            h_factor, nb, bsp, min_radius, "nearest",
        )
        print(f"  t_pyart: {t_pyart:.2f} s")

        meta = {
            "grid_shape": list(grid_shape),
            "grid_limits": [list(lim) for lim in grid_limits],
            "radar_lat": float(radar.latitude["data"][0]),
            "radar_lon": float(radar.longitude["data"][0]),
            "field": field_name,
            "volume": VOLUME,
            "weight_func": "nearest",
        }
        save_grids_to_cache(grid_W, grid_pyart, meta)

    # ── Métricas de validación ─────────────────────────────────────
    sep("Métricas de validación")
    m = compute_validation_metrics(grid_W, grid_pyart)

    print(f"  Voxels evaluados : {m['n_valid']:,} / {m['n_total']:,} "
          f"({100*m['n_valid']/m['n_total']:.1f}%)")
    print(f"  RMSE             : {m['rmse']:.4f} dBZ")
    print(f"  MAE              : {m['mae']:.4f} dBZ")
    print(f"  Diferencia máx.  : {m['max_diff']:.4f} dBZ")
    print(f"  Media W          : {m['mean_W']:.4f} dBZ")
    print(f"  Media Py-ART     : {m['mean_py']:.4f} dBZ")
    print(f"  Error relativo   : {m['rel_err']:.4f} %")

    # ── Figura ────────────────────────────────────────────────────
    sep("Generando figura de comparación")
    fig_path   = FIGURES_DIR / "validation_comparison.png"
    radar_lat  = float(meta["radar_lat"])
    radar_lon  = float(meta["radar_lon"])
    make_comparison_figure(grid_W, grid_pyart, grid_shape, grid_limits,
                           radar_lat, radar_lon, fig_path)

    # ── Resumen para tesis ─────────────────────────────────────────
    sep("RESUMEN — copiar en resultados.tex")
    print(f"""
┌─ tab:validacion ───────────────────────────────────────────────┐
  RMSE                       : {m['rmse']:.3f} dBZ
  Error absoluto medio (MAE) : {m['mae']:.3f} dBZ
  Diferencia máxima          : {m['max_diff']:.2f} dBZ
  Media (operador W)         : {m['mean_W']:.2f} dBZ
  Media (Py-ART)             : {m['mean_py']:.2f} dBZ
  Error relativo             : {m['rel_err']:.2f} %
└────────────────────────────────────────────────────────────────┘
""")

    print("─" * 64)
    print("  Rows LaTeX listos para pegar en resultados.tex:")
    print("─" * 64)
    print()
    print("% ── tab:validacion ──")
    print(f"RMSE                          & {m['rmse']:.3f}~dBZ \\\\")
    print(f"Error absoluto medio (MAE)    & {m['mae']:.3f}~dBZ \\\\")
    print(f"Diferencia máxima             & {m['max_diff']:.2f}~dBZ \\\\")
    print(f"Media (operador $\\mathbf{{W}}$) & {m['mean_W']:.2f}~dBZ \\\\")
    print(f"Media (Py-ART)                & {m['mean_py']:.2f}~dBZ \\\\")
    print(f"Error relativo                & {m['rel_err']:.2f}~\\% \\\\")
    print()
    print("% ── fig:validacion_ppi ──")
    print(f"% Figura guardada en: {fig_path}")
    print(r"% \includegraphics[width=\textwidth]{figures/validation_comparison.png}")


if __name__ == "__main__":
    main()
