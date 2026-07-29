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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# NC_FILE   = Path(_BACKEND).parent / "biribiri" / "curso-radar-meteo" / "Datos_Radar" / "rma1_rma17" / "rma1_rma17" / "RMA17_0315_01_20250819T001713Z.nc"
# NC_FILE   = Path(_BACKEND).parent / "biribiri" / "curso-radar-meteo" / "Datos_Radar" / "volumenes" / "volumenes" / "RMA1_0315_01_20250830T203606Z.nc"
#NC_FILE   = Path(_BACKEND).parent / "biribiri" / "curso-radar-meteo" / "Datos_Radar" / "RMA6_Mar_del_Plata-20251126T164922Z-1-001" / "RMA6_Mar_del_Plata" / "RMA6_0315_01_20251112T150538Z.nc"
VOLUME    = "01"
FIELD     = "DBZH"
WEIGHT_FUNC = "nearest"  # Cambiar a "Barnes2" para medir diferencia real con interpolación ponderada

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
        gridding_algo="map_gates_to_grid",
        grid_origin=grid_origin,
        fields=[field_name],
        weighting_function=weighting_function,
        gatefilters=False,
        roi_func="dist_beam",
        nb=nb, bsp=bsp, h_factor=np.full(1, h_factor, dtype=np.float32), min_radius=min_radius,
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
        if meta.get("weight_func") != WEIGHT_FUNC:
            print(f"  [caché] weight_func cambió ({meta.get('weight_func')} → {WEIGHT_FUNC}) — recalculando.")
            return None
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
    Evalúa error de valores (sobre voxels válidos en AMBAS grillas)
    y error de máscara (voxels donde una tiene dato y la otra no).
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

    # ── Ubicación de la diferencia máxima ──────────────────────────
    abs_diff = np.abs(diff)
    idx_flat = int(np.argmax(abs_diff))

    # Reconstruir índices (z, y, x) dentro del volumen completo
    valid_indices = np.argwhere(valid)          # (n_valid, ndim)
    max_zyx       = tuple(valid_indices[idx_flat])
    max_val_W     = float(v_W[idx_flat])
    max_val_py    = float(v_py[idx_flat])

    # Percentiles de |diff| para entender la distribución
    pct = np.percentile(abs_diff, [50, 90, 95, 99, 99.9, 100])

    # ── Cross-check con sklearn ──────────────────────────────────
    sk_mse  = mean_squared_error(v_py, v_W)
    sk_rmse = float(np.sqrt(sk_mse))
    sk_mae  = mean_absolute_error(v_py, v_W)
    sk_r2   = r2_score(v_py, v_W)

    # ── Métricas de máscara ──────────────────────────────────────
    both_valid  = valid
    both_masked = mask_W & mask_py
    only_W      = (~mask_W) & mask_py
    only_py     = mask_W & (~mask_py)

    n_both_valid  = int(both_valid.sum())
    n_both_masked = int(both_masked.sum())
    n_only_W      = int(only_W.sum())
    n_only_py     = int(only_py.sum())
    n_masked_W    = int(mask_W.sum())
    n_masked_py   = int(mask_py.sum())

    mask_agree = (n_both_valid + n_both_masked) / n_total * 100
    mask_union = n_both_valid + n_only_W + n_only_py
    mask_iou   = n_both_valid / max(mask_union, 1) * 100

    return dict(
        rmse=rmse, mae=mae, max_diff=max_diff,
        mean_W=mean_W, mean_py=mean_py, rel_err=rel_err,
        n_valid=n_valid, n_total=n_total,
        max_zyx=max_zyx, max_val_W=max_val_W, max_val_py=max_val_py,
        pct_labels=["p50", "p90", "p95", "p99", "p99.9", "p100"],
        pct_values=pct.tolist(),
        sk_rmse=sk_rmse, sk_mae=sk_mae, sk_r2=sk_r2,
        n_both_valid=n_both_valid, n_both_masked=n_both_masked,
        n_only_W=n_only_W, n_only_py=n_only_py,
        n_masked_W=n_masked_W, n_masked_py=n_masked_py,
        mask_agree=mask_agree, mask_iou=mask_iou,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Análisis profundo de outliers
# ══════════════════════════════════════════════════════════════════════════════

def analyze_outliers(grid_W, grid_pyart, grid_shape, grid_limits, top_n=10):
    """
    Analiza en profundidad los voxels con mayor diferencia:
    - Top N outliers con ubicación y valores
    - Vecindario 3x3x3 del peor caso
    - Análisis de máscara alrededor de cada outlier
    """
    mask_W  = np.ma.getmaskarray(grid_W)
    mask_py = np.ma.getmaskarray(grid_pyart)
    valid   = (~mask_W) & (~mask_py)

    # Grilla completa de diferencias (solo donde ambas son válidas)
    diff_3d = np.full(grid_shape, np.nan)
    diff_3d[valid] = grid_W.data[valid].astype(np.float64) - grid_pyart.data[valid].astype(np.float64)

    # Coordenadas en metros para cada eje
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = grid_limits
    nz, ny, nx = grid_shape
    z_coords = np.linspace(zmin, zmax, nz) / 1000  # km
    y_coords = np.linspace(ymin, ymax, ny) / 1000
    x_coords = np.linspace(xmin, xmax, nx) / 1000

    # ── Top N outliers ────────────────────────────────────────────
    abs_diff_flat = np.abs(diff_3d).ravel()
    # Ignorar NaN para el ranking
    abs_diff_flat_nonan = np.where(np.isnan(abs_diff_flat), -1, abs_diff_flat)
    top_indices = np.argsort(abs_diff_flat_nonan)[-top_n:][::-1]

    sep(f"Top {top_n} outliers")
    print(f"  {'#':>3s}  {'z':>3s}  {'y':>3s}  {'x':>3s}  "
          f"{'W (dBZ)':>9s}  {'PyART':>9s}  {'Diff':>9s}  "
          f"{'Dist(km)':>9s}  {'Alt(km)':>8s}  {'Vecinos mask W':>15s}  {'Vecinos mask Py':>15s}")
    print("  " + "─" * 110)

    for rank, flat_idx in enumerate(top_indices, 1):
        zi, yi, xi = np.unravel_index(flat_idx, grid_shape)
        val_w  = float(grid_W.data[zi, yi, xi])
        val_py = float(grid_pyart.data[zi, yi, xi])
        d      = val_w - val_py
        dist   = np.sqrt(x_coords[xi]**2 + y_coords[yi]**2)
        alt    = z_coords[zi]

        # Contar vecinos enmascarados en cubo 3x3x3
        z_sl = slice(max(zi-1, 0), min(zi+2, nz))
        y_sl = slice(max(yi-1, 0), min(yi+2, ny))
        x_sl = slice(max(xi-1, 0), min(xi+2, nx))
        neighborhood_size = (z_sl.stop - z_sl.start) * (y_sl.stop - y_sl.start) * (x_sl.stop - x_sl.start)
        masked_W_count  = int(mask_W[z_sl, y_sl, x_sl].sum())
        masked_py_count = int(mask_py[z_sl, y_sl, x_sl].sum())

        print(f"  {rank:3d}  {zi:3d}  {yi:3d}  {xi:3d}  "
              f"{val_w:9.2f}  {val_py:9.2f}  {d:+9.2f}  "
              f"{dist:9.1f}  {alt:8.1f}  "
              f"{masked_W_count:>3d}/{neighborhood_size:<3d}         "
              f"{masked_py_count:>3d}/{neighborhood_size:<3d}")

    # ── Vecindario detallado del peor caso ─────────────────────────
    worst_flat = top_indices[0]
    zi, yi, xi = np.unravel_index(worst_flat, grid_shape)

    sep("Vecindario 5x5 del peor outlier (mismo nivel z)")
    print(f"  Centro: z={zi}, y={yi}, x={xi}  |  "
          f"pos: x={x_coords[xi]:.1f} km, y={y_coords[yi]:.1f} km, z={z_coords[zi]:.1f} km\n")

    radius = 2  # 5x5
    y_from, y_to = max(yi - radius, 0), min(yi + radius + 1, ny)
    x_from, x_to = max(xi - radius, 0), min(xi + radius + 1, nx)

    def _fmt(val, masked):
        if masked:
            return "   ---   "
        return f"{val:8.2f} "

    # Tabla W
    print("  Valores operador W:")
    header = "        " + "".join(f"  x={j:<4d}  " for j in range(x_from, x_to))
    print(header)
    for j in range(y_from, y_to):
        marker = " ←" if j == yi else "  "
        row = f"  y={j:<4d}" + "".join(
            _fmt(grid_W.data[zi, j, k], mask_W[zi, j, k])
            for k in range(x_from, x_to)
        ) + marker
        print(row)

    print()
    print("  Valores Py-ART:")
    print(header)
    for j in range(y_from, y_to):
        marker = " ←" if j == yi else "  "
        row = f"  y={j:<4d}" + "".join(
            _fmt(grid_pyart.data[zi, j, k], mask_py[zi, j, k])
            for k in range(x_from, x_to)
        ) + marker
        print(row)

    print()
    print("  Diferencia (W − Py-ART):")
    print(header)
    for j in range(y_from, y_to):
        marker = " ←" if j == yi else "  "
        cells = []
        for k in range(x_from, x_to):
            if mask_W[zi, j, k] or mask_py[zi, j, k]:
                cells.append("   ---   ")
            else:
                d = grid_W.data[zi, j, k] - grid_pyart.data[zi, j, k]
                cells.append(f"{d:+8.2f} ")
        row = f"  y={j:<4d}" + "".join(cells) + marker
        print(row)

    # ── Resumen de patrones ────────────────────────────────────────
    sep("Resumen de patrones en outliers")
    top_zyx = [np.unravel_index(idx, grid_shape) for idx in top_indices]
    dists   = [np.sqrt(x_coords[xi]**2 + y_coords[yi]**2) for zi, yi, xi in top_zyx]
    alts    = [z_coords[zi] for zi, yi, xi in top_zyx]
    mask_fracs_W  = []
    mask_fracs_py = []
    for zi, yi, xi in top_zyx:
        z_sl = slice(max(zi-1, 0), min(zi+2, nz))
        y_sl = slice(max(yi-1, 0), min(yi+2, ny))
        x_sl = slice(max(xi-1, 0), min(xi+2, nx))
        n = (z_sl.stop - z_sl.start) * (y_sl.stop - y_sl.start) * (x_sl.stop - x_sl.start)
        mask_fracs_W.append(mask_W[z_sl, y_sl, x_sl].sum() / n * 100)
        mask_fracs_py.append(mask_py[z_sl, y_sl, x_sl].sum() / n * 100)

    print(f"  Distancia al radar : {min(dists):.1f} – {max(dists):.1f} km")
    print(f"  Alturas            : {min(alts):.1f} – {max(alts):.1f} km")
    print(f"  % vecinos mask (W) : {min(mask_fracs_W):.0f}% – {max(mask_fracs_W):.0f}%")
    print(f"  % vecinos mask (Py): {min(mask_fracs_py):.0f}% – {max(mask_fracs_py):.0f}%")
    if min(dists) > 100:
        print("  → Todos los outliers están lejos del radar (>100 km)")
    if min(alts) > 10:
        print("  → Todos los outliers están a gran altura (>10 km)")
    avg_mask = np.mean(mask_fracs_W + mask_fracs_py)
    if avg_mask > 40:
        print(f"  → Alta fracción de vecinos enmascarados ({avg_mask:.0f}% promedio)")
        print("    Confirma: outliers en zona de cobertura parcial/borde")


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

    # Rango de colorbar basado en p99 para que las diferencias sean visibles
    if sl_diff.count() > 0:
        valid_diffs = sl_diff.compressed()
        p99 = float(np.percentile(np.abs(valid_diffs), 99))
        diff_max = max(p99, 0.5)  # mínimo 0.5 para que no sea invisible
    else:
        diff_max = 5.0

    # ── Figura ────────────────────────────────────────────────────
    proj     = ccrs.Mercator()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(22, 6), facecolor="white")

    # 3 paneles con proyección + 1 panel normal para histograma
    ax1 = fig.add_subplot(1, 4, 1, projection=proj)
    ax2 = fig.add_subplot(1, 4, 2, projection=proj)
    ax3 = fig.add_subplot(1, 4, 3, projection=proj)
    ax_hist = fig.add_subplot(1, 4, 4)
    axes = [ax1, ax2, ax3]
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

    # ── Panel 4: histograma de diferencias ──────────────────────
    if sl_diff.count() > 0:
        valid_diffs = sl_diff.compressed()
        ax_hist.hist(valid_diffs, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
        ax_hist.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_hist.set_xlabel("Diferencia W − Py-ART (dBZ)", fontsize=11)
        ax_hist.set_ylabel("Frecuencia", fontsize=11)
        ax_hist.set_title("Distribución de diferencias", fontsize=13, pad=8)

        # Anotar estadísticas
        rmse_sl = float(np.sqrt(np.mean(valid_diffs**2)))
        mae_sl  = float(np.mean(np.abs(valid_diffs)))
        stats_text = (f"RMSE = {rmse_sl:.3f} dBZ\n"
                      f"MAE  = {mae_sl:.3f} dBZ\n"
                      f"N    = {len(valid_diffs):,}")
        ax_hist.text(0.97, 0.95, stats_text, transform=ax_hist.transAxes,
                     fontsize=10, verticalalignment="top", horizontalalignment="right",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        ax_hist.set_yscale("log")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                pad_inches=0.1, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"  [figura] Guardada en: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Diagnóstico ROI asimétrico (gate-centric vs voxel-centric)
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_roi_asymmetry(grid_W, grid_pyart, grid_shape, grid_limits, metrics):
    """
    Para el peor outlier, muestra qué gate eligió W y compara el ROI
    evaluado en la posición del voxel vs la posición del gate.
    Hipótesis: PyART (gate-centric) encuentra gates que W (voxel-centric) no ve
    porque el ROI del gate es mayor que el ROI del voxel.
    """
    sep("Diagnóstico ROI: gate-centric vs voxel-centric")

    if not NC_FILE.exists():
        print("  [skip] Necesita radar para diagnóstico de gates")
        return

    import pyart as _pyart
    from scipy.spatial import cKDTree

    radar = _pyart.io.read(str(NC_FILE))
    field_name, _ = resolve_field(radar, FIELD)
    field_data = radar.fields[field_name]["data"]

    roi_params = ROI_PARAMS_BY_VOLUME.get(VOLUME, ROI_PARAMS_VOL01)
    h_factor, nb, bsp, min_radius = roi_params

    # Coordenadas del peor outlier
    z_idx, y_idx, x_idx = metrics["max_zyx"]
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = grid_limits
    nz, ny, nx = grid_shape

    voxel_x = xmin + (xmax - xmin) * x_idx / max(nx - 1, 1)
    voxel_y = ymin + (ymax - ymin) * y_idx / max(ny - 1, 1)
    voxel_z = zmin + (zmax - zmin) * z_idx / max(nz - 1, 1)

    print(f"  Voxel outlier: z={z_idx}, y={y_idx}, x={x_idx}")
    print(f"  Posición: x={voxel_x/1000:.1f} km, y={voxel_y/1000:.1f} km, z={voxel_z/1000:.1f} km")

    # ROI en la posición del voxel
    from app.services.radar_processing.grid_geometry import calculate_roi_dist_beam
    roi_at_voxel = float(calculate_roi_dist_beam(
        z_coords=voxel_z, y_coords=voxel_y, x_coords=voxel_x,
        h_factor=h_factor, nb=nb, bsp=bsp, min_radius=min_radius,
    ))
    print(f"\n  ROI en posición del VOXEL: {roi_at_voxel:.1f} m ({roi_at_voxel/1000:.2f} km)")

    # Gate coordinates (solo válidos por field mask)
    gate_x = radar.gate_x["data"].ravel()
    gate_y = radar.gate_y["data"].ravel()
    gate_z = radar.gate_z["data"].ravel()

    if np.ma.isMaskedArray(field_data) and field_data.mask is not np.ma.nomask:
        field_valid = ~field_data.mask.ravel()
    else:
        field_valid = np.ones(gate_x.shape[0], dtype=bool)
    toa_valid = gate_z <= TOA
    valid = field_valid & toa_valid

    gx_v = gate_x[valid]
    gy_v = gate_y[valid]
    gz_v = gate_z[valid]
    field_vals = field_data.data.ravel()[valid]

    # Distancia 3D de cada gate válido al voxel
    dx = gx_v - voxel_x
    dy = gy_v - voxel_y
    dz = gz_v - voxel_z
    dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)

    # Gates dentro del ROI del VOXEL (lo que W encuentra)
    in_voxel_roi = dist_3d <= roi_at_voxel
    n_in_voxel_roi = int(in_voxel_roi.sum())

    if n_in_voxel_roi > 0:
        nearest_in_voxel = int(np.argmin(dist_3d[in_voxel_roi]))
        # Mapear al índice correcto
        indices_in_roi = np.where(in_voxel_roi)[0]
        idx_w = indices_in_roi[nearest_in_voxel]
        print(f"\n  ── Lo que W encuentra (voxel-centric, ROI={roi_at_voxel:.0f}m) ──")
        print(f"  Gates dentro del ROI del voxel: {n_in_voxel_roi}")
        print(f"  Gate más cercano: dist={dist_3d[idx_w]:.1f}m, DBZH={field_vals[idx_w]:.2f} dBZ")
        print(f"  Gate pos: x={gx_v[idx_w]/1000:.1f}km, y={gy_v[idx_w]/1000:.1f}km, z={gz_v[idx_w]/1000:.1f}km")
    else:
        print(f"\n  W: 0 gates dentro del ROI del voxel ({roi_at_voxel:.0f}m)")

    # Ahora: ROI evaluado en la posición de cada gate (gate-centric, como PyART)
    roi_at_gates = calculate_roi_dist_beam(
        z_coords=gz_v, y_coords=gy_v, x_coords=gx_v,
        h_factor=h_factor, nb=nb, bsp=bsp, min_radius=min_radius,
    )
    in_gate_roi = dist_3d <= roi_at_gates
    n_in_gate_roi = int(in_gate_roi.sum())

    if n_in_gate_roi > 0:
        nearest_in_gate = int(np.argmin(dist_3d[in_gate_roi]))
        indices_in_gate_roi = np.where(in_gate_roi)[0]
        idx_py = indices_in_gate_roi[nearest_in_gate]
        print(f"\n  ── Lo que PyART encontraría (gate-centric) ──")
        print(f"  Gates cuyo ROI alcanza al voxel: {n_in_gate_roi}")
        print(f"  Gate más cercano: dist={dist_3d[idx_py]:.1f}m, DBZH={field_vals[idx_py]:.2f} dBZ")
        print(f"  Gate pos: x={gx_v[idx_py]/1000:.1f}km, y={gy_v[idx_py]/1000:.1f}km, z={gz_v[idx_py]/1000:.1f}km")
        print(f"  ROI de ese gate: {roi_at_gates[idx_py]:.1f} m")
    else:
        print(f"\n  PyART: 0 gates cuyo ROI alcanza al voxel")

    # Gates que PyART ve pero W no
    only_pyart = in_gate_roi & ~in_voxel_roi
    n_only_pyart = int(only_pyart.sum())
    if n_only_pyart > 0:
        print(f"\n  ── Gates que PyART ve pero W NO ({n_only_pyart} gates) ──")
        idxs = np.where(only_pyart)[0]
        sorted_by_dist = idxs[np.argsort(dist_3d[idxs])]
        for rank, gi in enumerate(sorted_by_dist[:5], 1):
            roi_g = roi_at_gates[gi]
            print(f"    {rank}. dist={dist_3d[gi]:.1f}m, DBZH={field_vals[gi]:.2f}dBZ, "
                  f"ROI_gate={roi_g:.0f}m, z_gate={gz_v[gi]/1000:.1f}km")
        if n_only_pyart > 5:
            print(f"    ... y {n_only_pyart - 5} más")
    else:
        print(f"\n  → Ambos ven exactamente los mismos gates (ROI simétrico aquí)")


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
        max_nb = None if WEIGHT_FUNC != "nearest" else 1
        W = build_W_operator(
            gates_xyz=gates_xyz, voxels_xyz=voxels_xyz, toa=TOA,
            h_factor=h_factor, nb=nb, bsp=bsp, min_radius=min_radius,
            weight_func=WEIGHT_FUNC, max_neighbors=max_nb,
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
        pyart_wf = "Barnes2" if WEIGHT_FUNC in ("Barnes", "Barnes2") else WEIGHT_FUNC
        grid_pyart, t_pyart = grid_with_pyart(
            radar, field_name, grid_shape, grid_limits,
            h_factor, nb, bsp, min_radius, pyart_wf,
        )
        print(f"  t_pyart: {t_pyart:.2f} s")

        meta = {
            "grid_shape": list(grid_shape),
            "grid_limits": [list(lim) for lim in grid_limits],
            "radar_lat": float(radar.latitude["data"][0]),
            "radar_lon": float(radar.longitude["data"][0]),
            "field": field_name,
            "volume": VOLUME,
            "weight_func": WEIGHT_FUNC,
        }
        save_grids_to_cache(grid_W, grid_pyart, meta)

    # ── Métricas de validación ─────────────────────────────────────
    sep("Métricas de validación")
    m = compute_validation_metrics(grid_W, grid_pyart)

    print(f"  Método de peso   : {WEIGHT_FUNC}")
    print(f"  Voxels evaluados : {m['n_valid']:,} / {m['n_total']:,} "
          f"({100*m['n_valid']/m['n_total']:.1f}%)")
    print(f"  RMSE             : {m['rmse']:.4f} dBZ")
    print(f"  MAE              : {m['mae']:.4f} dBZ")
    print(f"  Diferencia máx.  : {m['max_diff']:.4f} dBZ")
    print(f"  Media W          : {m['mean_W']:.4f} dBZ")
    print(f"  Media Py-ART     : {m['mean_py']:.4f} dBZ")
    print(f"  Error relativo   : {m['rel_err']:.4f} %")
    print()
    print(f"  ── Cross-check sklearn ──")
    print(f"  RMSE (sklearn)   : {m['sk_rmse']:.4f} dBZ")
    print(f"  MAE  (sklearn)   : {m['sk_mae']:.4f} dBZ")
    print(f"  R²   (sklearn)   : {m['sk_r2']:.6f}")
    rmse_match = abs(m['rmse'] - m['sk_rmse']) < 1e-10
    mae_match  = abs(m['mae'] - m['sk_mae']) < 1e-10
    print(f"  RMSE coincide    : {'✓' if rmse_match else '✗ DISCREPANCIA'}")
    print(f"  MAE  coincide    : {'✓' if mae_match else '✗ DISCREPANCIA'}")
    print()
    print(f"  ── Métricas de máscara ──")
    print(f"  Masked total W   : {m['n_masked_W']:,}")
    print(f"  Masked total Py  : {m['n_masked_py']:,}")
    print(f"  Válidos en ambas : {m['n_both_valid']:,}")
    print(f"  Masked en ambas  : {m['n_both_masked']:,}")
    print(f"  Solo en W        : {m['n_only_W']:,}")
    print(f"  Solo en PyART    : {m['n_only_py']:,}")
    print(f"  Concordancia     : {m['mask_agree']:.2f}%")
    print(f"  IoU (cobertura)  : {m['mask_iou']:.2f}%")

    sep("Diagnóstico de diferencia máxima")
    z, y, x = m['max_zyx']
    print(f"  Ubicación (z,y,x) : ({z}, {y}, {x})")
    print(f"  Valor W           : {m['max_val_W']:.4f} dBZ")
    print(f"  Valor Py-ART      : {m['max_val_py']:.4f} dBZ")
    print(f"  Diferencia        : {m['max_val_W'] - m['max_val_py']:.4f} dBZ")

    # Posición geográfica aproximada
    nz, ny, nx = grid_shape
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = grid_limits
    pos_z_m = zmin + (zmax - zmin) * z / max(nz - 1, 1)
    pos_y_m = ymin + (ymax - ymin) * y / max(ny - 1, 1)
    pos_x_m = xmin + (xmax - xmin) * x / max(nx - 1, 1)
    dist_horiz_km = np.sqrt(pos_x_m**2 + pos_y_m**2) / 1000
    print(f"  Posición aprox.   : x={pos_x_m/1000:.1f} km, y={pos_y_m/1000:.1f} km, z={pos_z_m/1000:.1f} km")
    print(f"  Distancia al radar: {dist_horiz_km:.1f} km")

    sep("Distribución de |diferencias| (percentiles)")
    for label, val in zip(m['pct_labels'], m['pct_values']):
        print(f"  {label:>6s} : {val:.4f} dBZ")
    print()
    print("  → Si p99.9 es muy bajo y p100 es alto, la diff máxima es")
    print("    un outlier puntual (borde/cobertura parcial).")

    # ── Diagnóstico ROI gate-centric vs voxel-centric ───────────
    diagnose_roi_asymmetry(grid_W, grid_pyart, grid_shape, grid_limits, m)

    # ── Análisis profundo de outliers ─────────────────────────────
    analyze_outliers(grid_W, grid_pyart, grid_shape, grid_limits, top_n=10)

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
┌─ tab:validacion ({WEIGHT_FUNC}) ──────────────────────────────┐
  RMSE                       : {m['rmse']:.3f} dBZ
  Error absoluto medio (MAE) : {m['mae']:.3f} dBZ
  Diferencia máxima          : {m['max_diff']:.2f} dBZ
  Media (operador W)         : {m['mean_W']:.2f} dBZ
  Media (Py-ART)             : {m['mean_py']:.2f} dBZ
  Error relativo             : {m['rel_err']:.2f} %
  R² (coef. determinación)   : {m['sk_r2']:.6f}
  ──────────────────────────────────────────────────────────
  Masked total W             : {m['n_masked_W']:,}
  Masked total PyART         : {m['n_masked_py']:,}
  Concordancia de máscara    : {m['mask_agree']:.2f} %
  IoU de cobertura           : {m['mask_iou']:.2f} %
  Voxels solo en W           : {m['n_only_W']:,}
  Voxels solo en PyART       : {m['n_only_py']:,}
└────────────────────────────────────────────────────────────────┘
""")

    print("─" * 64)
    print("  Rows LaTeX listos para pegar en resultados.tex:")
    print("─" * 64)
    print()
    print(f"% ── tab:validacion ({WEIGHT_FUNC}) ──")
    print(f"RMSE                          & {m['rmse']:.3f}~dBZ \\\\")
    print(f"Error absoluto medio (MAE)    & {m['mae']:.3f}~dBZ \\\\")
    print(f"Diferencia máxima             & {m['max_diff']:.2f}~dBZ \\\\")
    print(f"Media (operador $\\mathbf{{W}}$) & {m['mean_W']:.2f}~dBZ \\\\")
    print(f"Media (Py-ART)                & {m['mean_py']:.2f}~dBZ \\\\")
    print(f"Error relativo                & {m['rel_err']:.2f}~\\% \\\\")
    print(f"$R^2$                         & {m['sk_r2']:.6f} \\\\")
    print(f"\\midrule")
    print(f"Masked total $\\mathbf{{W}}$    & {m['n_masked_W']:,} \\\\")
    print(f"Masked total Py-ART           & {m['n_masked_py']:,} \\\\")
    print(f"Concordancia de máscara       & {m['mask_agree']:.2f}~\\% \\\\")
    print(f"IoU de cobertura              & {m['mask_iou']:.2f}~\\% \\\\")
    print(f"Voxels solo en $\\mathbf{{W}}$  & {m['n_only_W']:,} \\\\")
    print(f"Voxels solo en Py-ART         & {m['n_only_py']:,} \\\\")
    print()
    print("% ── fig:validacion_ppi ──")
    print(f"% Figura guardada en: {fig_path}")
    print(r"% \includegraphics[width=\textwidth]{figures/validation_comparison.png}")


if __name__ == "__main__":
    main()
