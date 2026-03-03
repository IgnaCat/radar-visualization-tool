"""
Benchmark de tiempos de construcción del operador W (grilla 3D)
variando resolución espacial y parámetros de ROI.

Genera gráficos que comparan:
  1. Tiempo vs resolución XY (con ROI fijo)
  2. Tiempo vs min_radius del ROI (con resolución fija)
  3. Heatmap: tiempo en función de resolución × min_radius
  4. Memoria (nnz) del operador W en cada configuración
  5. Tabla resumen con todos los resultados

Usa directamente build_W_operator() del módulo radar_processing.

Uso:
    cd backend
    python experiments/grid_benchmark.py
    python experiments/grid_benchmark.py --nc path/to/file.nc
    python experiments/grid_benchmark.py --quick   # menos combinaciones
"""

import sys
import time
import argparse
import gc
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Agregar backend al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyart
from app.services.radar_processing.grid_builder import (
    get_gate_xyz_coords,
    get_grid_xyz_coords,
)
from app.services.radar_processing.grid_compute import build_W_operator
from app.services.radar_processing.grid_geometry import (
    calculate_grid_points,
    calculate_z_limits,
)

# ──────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_NC = Path(__file__).resolve().parent.parent.parent / "biribiri" / "RMA1_0315_01_20250819T001715Z.nc"

# Parámetros fijos del benchmark (valores por defecto del vol 01)
DEFAULT_ROI = dict(h_factor=0.9, nb=1.2, bsp=1.0, min_radius=700.0)
DEFAULT_RES_XY = 1000   # metros
DEFAULT_RES_Z = 600     # metros
TOA = 12000.0
WEIGHT_FUNC = "Barnes2"
MAX_NEIGHBORS = 30
N_WORKERS = 1            # Secuencial para tiempos reproducibles


@dataclass
class BenchResult:
    """Resultado de un benchmark individual."""
    resolution_xy: float       # metros
    resolution_z: float        # metros
    min_radius: float          # metros
    h_factor: float
    nb: float
    bsp: float
    grid_shape: tuple          # (nz, ny, nx)
    n_voxels: int
    n_gates: int
    build_time_s: float        # seg
    nnz: int                   # elementos no-cero en W
    w_size_mb: float           # MB del operador W
    avg_neighbors: float       # vecinos promedio por voxel


def find_nc_file(user_path: str | None = None) -> Path:
    """Busca un archivo NetCDF para el benchmark."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Archivo no encontrado: {p}")

    # Buscar en ubicaciones conocidas
    candidates = [
        DEFAULT_NC,
        Path(__file__).resolve().parent.parent / "app" / "storage" / "uploads",
    ]
    for c in candidates:
        if c.is_file() and c.suffix == ".nc":
            return c
        if c.is_dir():
            ncs = list(c.rglob("*.nc"))
            if ncs:
                return ncs[0]

    raise FileNotFoundError(
        "No se encontró archivo .nc. Usa --nc <path> para especificarlo."
    )


def compute_grid_params(radar, res_xy, res_z):
    """Calcula grid_shape y grid_limits a partir de la resolución."""
    range_max = float(radar.range["data"][-1])
    z_min, z_max, _ = calculate_z_limits(
        range_max, elevation=0,
        radar_fixed_angles=radar.fixed_angle["data"],
    )
    xy_max = float(range_max)
    grid_limits = ((z_min, z_max), (-xy_max, xy_max), (-xy_max, xy_max))
    grid_shape = calculate_grid_points(
        (z_min, z_max), (-xy_max, xy_max), (-xy_max, xy_max),
        res_xy, res_z,
    )
    return grid_shape, grid_limits


def run_single_benchmark(
    gates_xyz: np.ndarray,
    grid_shape: tuple,
    grid_limits: tuple,
    h_factor: float,
    nb: float,
    bsp: float,
    min_radius: float,
    res_xy: float,
    res_z: float,
    label: str = "",
) -> BenchResult:
    """Ejecuta un benchmark individual: construye W y mide tiempos."""
    nz, ny, nx = grid_shape
    n_voxels = nz * ny * nx
    n_gates = gates_xyz.shape[0]

    print(f"  {label}")
    print(f"    Grid: {grid_shape} = {n_voxels:,} voxels | "
          f"res_xy={res_xy}m, res_z={res_z}m | "
          f"min_r={min_radius:.0f}m, nb={nb}°")

    # Generar coordenadas de la grilla
    voxels_xyz = get_grid_xyz_coords(grid_shape, grid_limits)

    gc.collect()
    t0 = time.perf_counter()

    W = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz,
        toa=TOA,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        volume=None,       # parámetros explícitos, no lookup por volumen
        weight_func=WEIGHT_FUNC,
        max_neighbors=MAX_NEIGHBORS,
        n_workers=N_WORKERS,
    )

    elapsed = time.perf_counter() - t0
    w_size = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1024**2
    avg_n = W.nnz / n_voxels if n_voxels > 0 else 0

    print(f"    → {elapsed:.2f}s | nnz={W.nnz:,} | "
          f"W={w_size:.1f}MB | avg_neighbors={avg_n:.1f}")

    result = BenchResult(
        resolution_xy=res_xy,
        resolution_z=res_z,
        min_radius=min_radius,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        grid_shape=grid_shape,
        n_voxels=n_voxels,
        n_gates=n_gates,
        build_time_s=elapsed,
        nnz=W.nnz,
        w_size_mb=w_size,
        avg_neighbors=avg_n,
    )

    # Liberar W explícitamente
    del W, voxels_xyz
    gc.collect()

    return result


# ──────────────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────────────

def bench_vary_resolution(radar, gates_xyz, resolutions_xy, roi_params):
    """Benchmark variando resolución XY con ROI fijo."""
    results = []
    for res_xy in resolutions_xy:
        grid_shape, grid_limits = compute_grid_params(radar, res_xy, DEFAULT_RES_Z)
        r = run_single_benchmark(
            gates_xyz, grid_shape, grid_limits,
            roi_params["h_factor"], roi_params["nb"],
            roi_params["bsp"], roi_params["min_radius"],
            res_xy, DEFAULT_RES_Z,
            label=f"res_xy={res_xy}m",
        )
        results.append(r)
    return results


def bench_vary_roi(radar, gates_xyz, min_radii, res_xy):
    """Benchmark variando min_radius con resolución fija."""
    grid_shape, grid_limits = compute_grid_params(radar, res_xy, DEFAULT_RES_Z)
    results = []
    for mr in min_radii:
        r = run_single_benchmark(
            gates_xyz, grid_shape, grid_limits,
            DEFAULT_ROI["h_factor"], DEFAULT_ROI["nb"],
            DEFAULT_ROI["bsp"], mr,
            res_xy, DEFAULT_RES_Z,
            label=f"min_radius={mr:.0f}m",
        )
        results.append(r)
    return results


def bench_vary_nb(radar, gates_xyz, nb_values, res_xy):
    """Benchmark variando nb (ancho de haz) con resolución fija."""
    grid_shape, grid_limits = compute_grid_params(radar, res_xy, DEFAULT_RES_Z)
    results = []
    for nb_val in nb_values:
        r = run_single_benchmark(
            gates_xyz, grid_shape, grid_limits,
            DEFAULT_ROI["h_factor"], nb_val,
            DEFAULT_ROI["bsp"], DEFAULT_ROI["min_radius"],
            res_xy, DEFAULT_RES_Z,
            label=f"nb={nb_val}°",
        )
        results.append(r)
    return results


def bench_heatmap(radar, gates_xyz, resolutions_xy, min_radii):
    """Benchmark cruzado: resolución × min_radius."""
    results = []
    total = len(resolutions_xy) * len(min_radii)
    i = 0
    for res_xy in resolutions_xy:
        grid_shape, grid_limits = compute_grid_params(radar, res_xy, DEFAULT_RES_Z)
        for mr in min_radii:
            i += 1
            r = run_single_benchmark(
                gates_xyz, grid_shape, grid_limits,
                DEFAULT_ROI["h_factor"], DEFAULT_ROI["nb"],
                DEFAULT_ROI["bsp"], mr,
                res_xy, DEFAULT_RES_Z,
                label=f"[{i}/{total}] res={res_xy}m × min_r={mr:.0f}m",
            )
            results.append(r)
    return results


# ──────────────────────────────────────────────────────────────────
# Gráficos
# ──────────────────────────────────────────────────────────────────

def plot_time_vs_resolution(results: list[BenchResult]):
    """Gráfico 1: Tiempo de construcción vs resolución XY."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    res = [r.resolution_xy for r in results]
    times = [r.build_time_s for r in results]
    voxels = [r.n_voxels for r in results]
    nnz = [r.nnz for r in results]

    color1 = "C0"
    ax1.plot(res, times, "o-", color=color1, linewidth=2, markersize=8)
    ax1.set_xlabel("Resolución XY (m)", fontsize=11)
    ax1.set_ylabel("Tiempo de construcción (s)", fontsize=11, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_title("Tiempo vs Resolución espacial\n(ROI fijo)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.invert_xaxis()  # menor resolución = más voxels → derecha

    # Eje secundario: n_voxels
    ax1b = ax1.twinx()
    ax1b.plot(res, voxels, "s--", color="C1", alpha=0.7, markersize=6)
    ax1b.set_ylabel("Nº de voxels", fontsize=10, color="C1")
    ax1b.tick_params(axis="y", labelcolor="C1")
    ax1b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    # Anotar grid_shape
    for r in results:
        nz, ny, nx = r.grid_shape
        ax1.annotate(f"{nz}×{ny}×{nx}", xy=(r.resolution_xy, r.build_time_s),
                     xytext=(0, 12), textcoords="offset points", fontsize=7,
                     ha="center", color="gray")

    # Panel 2: nnz (complejidad real)
    ax2.plot(res, nnz, "o-", color="C2", linewidth=2, markersize=8)
    ax2.set_xlabel("Resolución XY (m)", fontsize=11)
    ax2.set_ylabel("Elementos no-cero (nnz)", fontsize=11)
    ax2.set_title("Complejidad del operador W vs Resolución", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.invert_xaxis()
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    fig.tight_layout()
    path = OUTPUT_DIR / "bench_tiempo_vs_resolucion.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)


def plot_time_vs_roi(results: list[BenchResult]):
    """Gráfico 2: Tiempo de construcción vs min_radius."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mr = [r.min_radius for r in results]
    times = [r.build_time_s for r in results]
    avg_n = [r.avg_neighbors for r in results]
    nnz = [r.nnz for r in results]

    ax1.plot(mr, times, "o-", color="C0", linewidth=2, markersize=8)
    ax1.set_xlabel("min_radius (m)", fontsize=11)
    ax1.set_ylabel("Tiempo de construcción (s)", fontsize=11)
    ax1.set_title("Tiempo vs Radio mínimo del ROI\n(resolución fija)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Eje secundario: avg_neighbors
    ax1b = ax1.twinx()
    ax1b.plot(mr, avg_n, "s--", color="C3", alpha=0.7, markersize=6)
    ax1b.set_ylabel("Vecinos promedio por voxel", fontsize=10, color="C3")
    ax1b.tick_params(axis="y", labelcolor="C3")
    ax1b.axhline(y=MAX_NEIGHBORS, linestyle=":", color="gray", alpha=0.5)

    # Panel 2: nnz
    ax2.plot(mr, nnz, "o-", color="C2", linewidth=2, markersize=8)
    ax2.set_xlabel("min_radius (m)", fontsize=11)
    ax2.set_ylabel("Elementos no-cero (nnz)", fontsize=11)
    ax2.set_title("Complejidad W vs min_radius", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    fig.tight_layout()
    path = OUTPUT_DIR / "bench_tiempo_vs_roi.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)


def plot_time_vs_nb(results: list[BenchResult]):
    """Gráfico 3: Tiempo vs nb (ancho de haz)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    nb_vals = [r.nb for r in results]
    times = [r.build_time_s for r in results]
    avg_n = [r.avg_neighbors for r in results]
    nnz = [r.nnz for r in results]

    ax1.plot(nb_vals, times, "o-", color="C0", linewidth=2, markersize=8)
    ax1.set_xlabel("nb — ancho de haz (°)", fontsize=11)
    ax1.set_ylabel("Tiempo de construcción (s)", fontsize=11)
    ax1.set_title("Tiempo vs Ancho de haz (nb)\n(resolución fija)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax1b = ax1.twinx()
    ax1b.plot(nb_vals, avg_n, "s--", color="C3", alpha=0.7, markersize=6)
    ax1b.set_ylabel("Vecinos promedio", fontsize=10, color="C3")
    ax1b.tick_params(axis="y", labelcolor="C3")
    ax1b.axhline(y=MAX_NEIGHBORS, linestyle=":", color="gray", alpha=0.5)

    ax2.plot(nb_vals, nnz, "o-", color="C2", linewidth=2, markersize=8)
    ax2.set_xlabel("nb — ancho de haz (°)", fontsize=11)
    ax2.set_ylabel("nnz", fontsize=11)
    ax2.set_title("Complejidad W vs nb", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    fig.tight_layout()
    path = OUTPUT_DIR / "bench_tiempo_vs_nb.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)


def plot_heatmap(results: list[BenchResult], resolutions_xy, min_radii):
    """Gráfico 4: Heatmap de tiempo resolución × min_radius."""
    nr = len(resolutions_xy)
    nm = len(min_radii)

    time_grid = np.zeros((nr, nm))
    nnz_grid = np.zeros((nr, nm))
    for r in results:
        i = resolutions_xy.index(r.resolution_xy)
        j = min_radii.index(r.min_radius)
        time_grid[i, j] = r.build_time_s
        nnz_grid[i, j] = r.nnz

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap tiempos
    im1 = ax1.imshow(time_grid, aspect="auto", cmap="YlOrRd", origin="lower")
    ax1.set_xticks(range(nm))
    ax1.set_xticklabels([f"{m:.0f}" for m in min_radii], fontsize=8)
    ax1.set_yticks(range(nr))
    ax1.set_yticklabels([f"{r:.0f}" for r in resolutions_xy], fontsize=8)
    ax1.set_xlabel("min_radius (m)", fontsize=10)
    ax1.set_ylabel("Resolución XY (m)", fontsize=10)
    ax1.set_title("Tiempo de construcción (s)", fontsize=12, fontweight="bold")
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.85)
    cbar1.set_label("Tiempo (s)")
    # Anotar valores
    for i in range(nr):
        for j in range(nm):
            ax1.text(j, i, f"{time_grid[i,j]:.1f}s", ha="center", va="center",
                     fontsize=7, color="white" if time_grid[i,j] > time_grid.mean() else "black")

    # Heatmap nnz
    im2 = ax2.imshow(nnz_grid / 1e6, aspect="auto", cmap="YlGnBu", origin="lower")
    ax2.set_xticks(range(nm))
    ax2.set_xticklabels([f"{m:.0f}" for m in min_radii], fontsize=8)
    ax2.set_yticks(range(nr))
    ax2.set_yticklabels([f"{r:.0f}" for r in resolutions_xy], fontsize=8)
    ax2.set_xlabel("min_radius (m)", fontsize=10)
    ax2.set_ylabel("Resolución XY (m)", fontsize=10)
    ax2.set_title("Elementos no-cero (millones)", fontsize=12, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.set_label("nnz (M)")
    for i in range(nr):
        for j in range(nm):
            ax2.text(j, i, f"{nnz_grid[i,j]/1e6:.1f}M", ha="center", va="center",
                     fontsize=7, color="white" if nnz_grid[i,j] > nnz_grid.mean() else "black")

    fig.tight_layout()
    path = OUTPUT_DIR / "bench_heatmap_resolucion_x_roi.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)


def plot_summary_table(all_results: list[BenchResult]):
    """Gráfico 5: Tabla resumen de todos los benchmarks."""
    fig, ax = plt.subplots(figsize=(16, max(4, 0.4 * len(all_results) + 1.5)))
    ax.axis("off")

    headers = ["Res XY (m)", "Res Z (m)", "Grid (nz×ny×nx)", "Voxels",
               "min_r (m)", "nb (°)", "Tiempo (s)", "nnz", "W (MB)", "Avg Neigh"]
    rows = []
    for r in all_results:
        nz, ny, nx = r.grid_shape
        rows.append([
            f"{r.resolution_xy:.0f}",
            f"{r.resolution_z:.0f}",
            f"{nz}×{ny}×{nx}",
            f"{r.n_voxels:,}",
            f"{r.min_radius:.0f}",
            f"{r.nb:.1f}",
            f"{r.build_time_s:.2f}",
            f"{r.nnz:,}",
            f"{r.w_size_mb:.1f}",
            f"{r.avg_neighbors:.1f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Colorear encabezados
    for j, header in enumerate(headers):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternar colores de fila
    for i in range(len(rows)):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("Resumen de benchmarks — Operador W",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    path = OUTPUT_DIR / "bench_tabla_resumen.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✔ {path}")
    plt.close(fig)


def save_results_json(all_results: list[BenchResult]):
    """Guarda resultados en JSON para análisis posterior."""
    data = []
    for r in all_results:
        d = asdict(r)
        d["grid_shape"] = list(d["grid_shape"])
        data.append(d)
    path = OUTPUT_DIR / "bench_results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✔ {path}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark de grilla 3D")
    parser.add_argument("--nc", type=str, default=None,
                        help="Path al archivo NetCDF del radar")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido con menos combinaciones")
    args = parser.parse_args()

    print("=" * 65)
    print("  BENCHMARK: Operador W — Resolución × ROI")
    print("=" * 65)

    # Encontrar y cargar radar
    nc_path = find_nc_file(args.nc)
    print(f"\nArchivo: {nc_path}")
    radar = pyart.io.read(str(nc_path))
    print(f"Radar: {radar.metadata.get('instrument_name', '?')} | "
          f"{radar.nsweeps} sweeps | {radar.nrays} rays × {radar.ngates} gates | "
          f"range_max={radar.range['data'][-1]/1000:.0f} km")

    # Pre-calcular coordenadas de gates (una sola vez)
    gates_xyz = get_gate_xyz_coords(radar)
    print(f"Gates: {gates_xyz.shape[0]:,} puntos 3D\n")

    # Definir rangos de barrido
    if args.quick:
        resolutions_xy = [2000, 1000, 500]
        min_radii = [500.0, 900.0, 2000.0]
        nb_values = [0.8, 1.2, 2.0]
        heatmap_res = [2000, 1000, 500]
        heatmap_mr = [500.0, 900.0, 2000.0]
    else:
        resolutions_xy = [3000, 2000, 1500, 1000, 750, 500]
        min_radii = [300.0, 500.0, 700.0, 900.0, 1500.0, 3000.0]
        nb_values = [0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0]
        heatmap_res = [2000, 1500, 1000, 750, 500]
        heatmap_mr = [500.0, 700.0, 900.0, 1500.0, 3000.0]

    all_results = []

    # ─── Benchmark 1: Resolución ──────────────────────────────────
    print("━" * 65)
    print("  1/4  Variando resolución XY (ROI fijo)")
    print("━" * 65)
    res_results = bench_vary_resolution(radar, gates_xyz, resolutions_xy, DEFAULT_ROI)
    all_results.extend(res_results)

    # ─── Benchmark 2: min_radius ──────────────────────────────────
    print("\n" + "━" * 65)
    print("  2/4  Variando min_radius (resolución fija)")
    print("━" * 65)
    roi_results = bench_vary_roi(radar, gates_xyz, min_radii, DEFAULT_RES_XY)
    all_results.extend(roi_results)

    # ─── Benchmark 3: nb (ancho de haz) ──────────────────────────
    print("\n" + "━" * 65)
    print("  3/4  Variando nb — ancho de haz (resolución fija)")
    print("━" * 65)
    nb_results = bench_vary_nb(radar, gates_xyz, nb_values, DEFAULT_RES_XY)
    all_results.extend(nb_results)

    # ─── Benchmark 4: Heatmap cruzado ─────────────────────────────
    print("\n" + "━" * 65)
    print("  4/4  Heatmap: resolución × min_radius")
    print("━" * 65)
    heatmap_results = bench_heatmap(radar, gates_xyz, heatmap_res, heatmap_mr)
    all_results.extend(heatmap_results)

    # ─── Generar gráficos ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Generando gráficos...")
    print("=" * 65)

    plot_time_vs_resolution(res_results)
    plot_time_vs_roi(roi_results)
    plot_time_vs_nb(nb_results)
    plot_heatmap(heatmap_results, heatmap_res, heatmap_mr)
    plot_summary_table(all_results)
    save_results_json(all_results)

    # ─── Resumen final ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMEN")
    print("=" * 65)
    fastest = min(all_results, key=lambda r: r.build_time_s)
    slowest = max(all_results, key=lambda r: r.build_time_s)
    print(f"  Más rápido: {fastest.build_time_s:.2f}s  "
          f"(res={fastest.resolution_xy}m, min_r={fastest.min_radius}m, "
          f"grid={fastest.grid_shape})")
    print(f"  Más lento:  {slowest.build_time_s:.2f}s  "
          f"(res={slowest.resolution_xy}m, min_r={slowest.min_radius}m, "
          f"grid={slowest.grid_shape})")
    print(f"  Factor:     {slowest.build_time_s/fastest.build_time_s:.1f}x")
    print(f"\n  Gráficos en: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
