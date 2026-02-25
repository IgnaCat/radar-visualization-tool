"""
Visualización del Radio de Influencia (ROI) en función de la distancia al radar.

Genera gráficos que muestran cómo el ROI crece con la distancia horizontal,
variando distintos parámetros del método dist_beam:
  - h_factor: peso de la componente vertical (altura / 20)
  - nb: ancho del haz virtual (grados)
  - bsp: espaciado entre haces (multiplicador)
  - min_radius: radio mínimo garantizado (m)
  - z (altura): nivel de altura fijo del voxel

Fórmula dist_beam:
    ROI = max(h_factor * (z / 20) + dist_xy * tan(nb * bsp * π/180), min_radius)

Uso:
    python experiments/roi_visualization.py
"""

import sys
from pathlib import Path

# Agregar el directorio backend al path para poder importar módulos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from app.services.radar_processing.grid_geometry import calculate_roi_dist_beam
from app.core.constants import ROI_PARAMS_BY_VOLUME


# Configuración general
MAX_RANGE_KM = 240          # Alcance máximo del radar (km)
N_POINTS = 500              # Puntos en el eje de distancia
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Rango de distancias horizontales (metros)
distances_m = np.linspace(0, MAX_RANGE_KM * 1000, N_POINTS)
distances_km = distances_m / 1000.0


def roi_curve(dist_xy_m, z_m, h_factor, nb, bsp, min_radius):
    """Wrapper que calcula ROI para un vector de distancias horizontales."""
    z_arr = np.full_like(dist_xy_m, z_m)
    y_arr = dist_xy_m          # distancia en una sola dirección (Y)
    x_arr = np.zeros_like(dist_xy_m)
    return calculate_roi_dist_beam(
        z_arr, y_arr, x_arr,
        h_factor=h_factor, nb=nb, bsp=bsp, min_radius=min_radius,
    )


def setup_ax(ax, title, xlabel="Distancia horizontal al radar (km)",
             ylabel="Radio de Influencia — ROI (m)"):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))



# 1. ROI por volumen (parámetros reales del proyecto)
def plot_roi_by_volume(z_fixed=2000.0):
    fig, ax = plt.subplots(figsize=(12, 6))
    for vol, (hf, nb, bsp, mr) in sorted(ROI_PARAMS_BY_VOLUME.items()):
        if vol == "03":          # Bird bath tiene escala gigante, lo ponemos aparte
            continue
        roi = roi_curve(distances_m, z_fixed, hf, nb, bsp, mr)
        label = f"Vol {vol}  (h={hf}, nb={nb}°, bsp={bsp}, min_r={mr:.0f}m)"
        ax.plot(distances_km, roi, linewidth=2, label=label)
    setup_ax(ax, f"ROI por volumen de escaneo — z = {z_fixed/1000:.1f} km")
    fig.tight_layout()
    path = OUTPUT_DIR / "roi_por_volumen.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)



# 2. Efecto de la altura (z) con parámetros fijos
def plot_roi_vs_height(h_factor=1.1, nb=1.6, bsp=1.3, min_radius=900.0):
    fig, ax = plt.subplots(figsize=(12, 6))
    z_levels = [500, 1000, 2000, 4000, 8000, 12000]
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.95, len(z_levels)))
    for z_m, color in zip(z_levels, colors):
        roi = roi_curve(distances_m, z_m, h_factor, nb, bsp, min_radius)
        ax.plot(distances_km, roi, linewidth=2, color=color,
                label=f"z = {z_m/1000:.1f} km")
    params_str = f"h_factor={h_factor}, nb={nb}°, bsp={bsp}, min_r={min_radius:.0f}m"
    setup_ax(ax, f"Efecto de la altura (z) sobre el ROI — {params_str}")
    fig.tight_layout()
    path = OUTPUT_DIR / "roi_vs_altura.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)



# 3. Efecto de nb (ancho de haz)
def plot_roi_vs_nb(h_factor=1.1, bsp=1.0, min_radius=700.0, z_fixed=2000.0):
    fig, ax = plt.subplots(figsize=(12, 6))
    nb_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.15, 0.9, len(nb_values)))
    for nb_val, color in zip(nb_values, colors):
        roi = roi_curve(distances_m, z_fixed, h_factor, nb_val, bsp, min_radius)
        ax.plot(distances_km, roi, linewidth=2, color=color,
                label=f"nb = {nb_val}°")
    params_str = f"h_factor={h_factor}, bsp={bsp}, min_r={min_radius:.0f}m, z={z_fixed/1000:.0f}km"
    setup_ax(ax, f"Efecto del ancho de haz (nb) — {params_str}")
    fig.tight_layout()
    path = OUTPUT_DIR / "roi_vs_nb.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)



# 4. Efecto del min_radius
def plot_roi_vs_min_radius(h_factor=1.1, nb=1.6, bsp=1.3, z_fixed=2000.0):
    fig, ax = plt.subplots(figsize=(12, 6))
    min_r_values = [300, 500, 700, 900, 1500, 3000]
    cmap = plt.cm.cool
    colors = cmap(np.linspace(0.1, 0.9, len(min_r_values)))
    for mr, color in zip(min_r_values, colors):
        roi = roi_curve(distances_m, z_fixed, h_factor, nb, bsp, mr)
        ax.plot(distances_km, roi, linewidth=2, color=color,
                label=f"min_radius = {mr:,.0f} m")
    params_str = f"h_factor={h_factor}, nb={nb}°, bsp={bsp}, z={z_fixed/1000:.0f}km"
    setup_ax(ax, f"Efecto del radio mínimo — {params_str}")
    # Resaltar zona donde min_radius domina
    ax.axhline(y=900, color="gray", linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = OUTPUT_DIR / "roi_vs_min_radius.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)



# 5. Descomposición de componentes del ROI
def plot_roi_components(h_factor=1.1, nb=1.6, bsp=1.3, min_radius=900.0,
                        z_fixed=2000.0):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Componente vertical (constante en distancia, depende solo de z)
    vert = h_factor * (z_fixed / 20.0)
    vert_arr = np.full_like(distances_m, vert)

    # Componente horizontal (crece linealmente con distancia)
    beam_angle = nb * bsp * np.pi / 180.0
    horiz = distances_m * np.tan(beam_angle)

    # Suma antes del max
    suma = vert_arr + horiz

    # ROI final (con min_radius)
    roi = np.maximum(suma, min_radius)

    ax.fill_between(distances_km, 0, vert_arr, alpha=0.25, color="C0",
                    label=f"Componente vertical = {vert:.0f} m (constante)")
    ax.fill_between(distances_km, vert_arr, suma, alpha=0.25, color="C1",
                    label="Componente horizontal (∝ distancia)")
    ax.plot(distances_km, roi, linewidth=2.5, color="C3",
            label="ROI final (con min_radius)")
    ax.axhline(y=min_radius, linestyle="--", color="gray", alpha=0.6,
               label=f"min_radius = {min_radius:.0f} m")

    # Marcar punto de cruce donde la curva supera min_radius
    crossover_idx = np.argmax(suma > min_radius)
    if crossover_idx > 0:
        cx = distances_km[crossover_idx]
        ax.axvline(x=cx, linestyle=":", color="C4", alpha=0.6)
        ax.annotate(f"Cruce a {cx:.1f} km", xy=(cx, min_radius),
                    xytext=(cx + 10, min_radius * 1.4),
                    arrowprops=dict(arrowstyle="->", color="C4"),
                    fontsize=9, color="C4")

    params_str = (f"h_factor={h_factor}, nb={nb}°, bsp={bsp}, "
                  f"min_r={min_radius:.0f}m, z={z_fixed/1000:.0f}km")
    setup_ax(ax, f"Descomposición de componentes del ROI — {params_str}")
    fig.tight_layout()
    path = OUTPUT_DIR / "roi_componentes.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)



# 6. Peso (weight) en función de distancia al gate
#    Para un ROI fijo, cómo decae el peso Barnes2 / Cressman
def plot_weight_functions():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    roi_values = [700, 1500, 3000, 6000]
    cmap = plt.cm.tab10

    for ax, (method, formula_label) in zip(axes, [
        ("Barnes2", r"$w = \exp(-d^2 / (R^2/4)) + 10^{-5}$"),
        ("Cressman", r"$w = (R^2 - d^2) / (R^2 + d^2)$"),
    ]):
        for i, R in enumerate(roi_values):
            d = np.linspace(0, R, 300)
            if method == "Barnes2":
                w = np.exp(-d**2 / (R**2 / 4.0)) + 1e-5
            else:  # Cressman
                w = (R**2 - d**2) / (R**2 + d**2)
            ax.plot(d / 1000, w, linewidth=2, color=cmap(i),
                    label=f"ROI = {R:,} m")
        ax.set_title(f"{method}:  {formula_label}", fontsize=11)
        ax.set_xlabel("Distancia al gate (km)", fontsize=10)
        ax.set_ylabel("Peso (w)", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.15)

    fig.suptitle("Funciones de peso por método de interpolación", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "funciones_peso.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✔ {path}")
    plt.close(fig)



# 7. Mapa 2D del ROI sobre la grilla (vista en planta)
def plot_roi_2d_map(h_factor=1.1, nb=1.6, bsp=1.3, min_radius=900.0,
                    z_fixed=2000.0, grid_extent_km=240):
    fig, ax = plt.subplots(figsize=(8, 7))
    n = 300
    coords = np.linspace(-grid_extent_km * 1000, grid_extent_km * 1000, n)
    xx, yy = np.meshgrid(coords, coords)
    zz = np.full_like(xx, z_fixed)

    roi_2d = calculate_roi_dist_beam(
        zz.ravel(), yy.ravel(), xx.ravel(),
        h_factor=h_factor, nb=nb, bsp=bsp, min_radius=min_radius,
    ).reshape(n, n)

    extent_km = grid_extent_km
    im = ax.imshow(
        roi_2d / 1000.0,
        extent=[-extent_km, extent_km, -extent_km, extent_km],
        origin="lower", cmap="magma_r",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="ROI (km)")
    ax.set_xlabel("X (km)", fontsize=10)
    ax.set_ylabel("Y (km)", fontsize=10)
    ax.set_title(
        f"ROI en planta — z={z_fixed/1000:.1f} km\n"
        f"(h={h_factor}, nb={nb}°, bsp={bsp}, min_r={min_radius:.0f}m)",
        fontsize=11, fontweight="bold",
    )
    # Marcar el radar
    ax.plot(0, 0, "w*", markersize=12, markeredgecolor="k", label="Radar")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = OUTPUT_DIR / "roi_mapa_2d.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)



# 8. Comparación ROI lineal simple vs dist_beam
def plot_roi_simple_vs_distbeam(z_fixed=2000.0):
    """Compara el ROI lineal (radar_grid/) vs dist_beam (backend/)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Modelo lineal simple: ROI = max(min_r, dist_3d * beam_factor)
    # beam_factor = tan(1°) ≈ 0.01746
    beam_factor = np.tan(np.radians(1.0))
    min_r_simple = 250.0
    dist_3d = np.sqrt(distances_m**2 + z_fixed**2)
    roi_simple = np.maximum(min_r_simple, dist_3d * beam_factor)
    ax.plot(distances_km, roi_simple, linewidth=2, linestyle="--",
            label=f"Lineal simple (bf=tan(1°), min_r={min_r_simple}m)")

    # dist_beam con vol 01 y vol 02
    for vol in ("01", "02"):
        hf, nb, bsp, mr = ROI_PARAMS_BY_VOLUME[vol]
        roi_db = roi_curve(distances_m, z_fixed, hf, nb, bsp, mr)
        ax.plot(distances_km, roi_db, linewidth=2,
                label=f"dist_beam Vol {vol} (h={hf}, nb={nb}°, bsp={bsp}, min_r={mr}m)")

    setup_ax(ax, f"ROI lineal simple vs dist_beam — z = {z_fixed/1000:.1f} km")
    fig.tight_layout()
    path = OUTPUT_DIR / "roi_simple_vs_distbeam.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)



# 9. Número estimado de gates dentro del ROI vs distancia
def plot_gates_in_roi(h_factor=1.1, nb=1.6, bsp=1.3, min_radius=900.0,
                      z_fixed=2000.0, gate_spacing=250.0, beam_width_deg=1.0):
    """
    Estimación geométrica de cuántos gates caen dentro del ROI en función
    de la distancia. A mayor distancia el ROI crece pero los gates se
    separan angularmente → el conteo no crece tan rápido.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    roi = roi_curve(distances_m, z_fixed, h_factor, nb, bsp, min_radius)

    # Estimación: en un anillo a distancia r, separación angular de gates ≈
    # r * Δθ (con Δθ = beam_width = ancho del haz). Gate spacing radial = gate_spacing.
    # Área del ROI ≈ π * roi²
    # Densidad de gates ≈ 1 / (gate_spacing * r * Δθ_rad)  -> 1 / area del gate
    delta_theta = np.radians(beam_width_deg)
    # Evitar div/0 cerca del radar
    safe_dist = np.maximum(distances_m, gate_spacing)
    gate_area = gate_spacing * safe_dist * delta_theta  # área por gate
    roi_area = np.pi * roi**2
    n_gates_est = roi_area / gate_area

    ax1.plot(distances_km, roi / 1000, linewidth=2, color="C0")
    ax1.set_ylabel("ROI (km)", fontsize=10)
    ax1.set_title(
        f"ROI y gates estimados dentro del ROI\n"
        f"(h={h_factor}, nb={nb}°, bsp={bsp}, min_r={min_radius:.0f}m, "
        f"z={z_fixed/1000:.0f}km, gate_sp={gate_spacing:.0f}m, bw={beam_width_deg}°)",
        fontsize=11, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2.plot(distances_km, n_gates_est, linewidth=2, color="C1",
             label="Gates estimados en ROI")
    ax2.axhline(y=30, linestyle="--", color="gray", alpha=0.7,
                label="max_neighbors = 30 (límite)")
    ax2.set_xlabel("Distancia horizontal al radar (km)", fontsize=10)
    ax2.set_ylabel("Nº estimado de gates", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    path = OUTPUT_DIR / "roi_gates_estimados.png"
    fig.savefig(path, dpi=150)
    print(f"  ✔ {path}")
    plt.close(fig)



def main():
    print("=" * 60)
    print("Generando gráficos de Radio de Influencia (ROI)")
    print("=" * 60)

    print("\n1. ROI por volumen de escaneo")
    plot_roi_by_volume()

    print("\n2. ROI vs altura (z)")
    plot_roi_vs_height()

    print("\n3. ROI vs ancho de haz (nb)")
    plot_roi_vs_nb()

    print("\n4. ROI vs radio mínimo")
    plot_roi_vs_min_radius()

    print("\n5. Descomposición de componentes del ROI")
    plot_roi_components()

    print("\n6. Funciones de peso (Barnes2 / Cressman)")
    plot_weight_functions()

    print("\n7. Mapa 2D del ROI en planta")
    plot_roi_2d_map()

    print("\n8. ROI lineal simple vs dist_beam")
    plot_roi_simple_vs_distbeam()

    print("\n9. Gates estimados dentro del ROI")
    plot_gates_in_roi()

    print("\n" + "=" * 60)
    print(f"Todos los gráficos guardados en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
