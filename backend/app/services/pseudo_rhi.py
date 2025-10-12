import os
import pyart
import copy
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List
from ..core.constants import VARIABLE_UNITS

from ..schemas import RangeFilter
from .radar_common import (
    resolve_field, colormap_for, build_gatefilter,
    safe_range_max_m, get_radar_site, md5_file, limit_line_to_range
)


def calcule_radial_angle(radar_lat, radar_lon, punto_lat, punto_lon):
    """
    Calcula el ángulo en grados desde la latitud y longitud del radar hasta el punto de interés.
    Mide desde Norte en sentido horario.

    Parámetros:
    - radar_lat: Latitud del radar.
    - radar_lon: Longitud del radar.
    - punto_lat: Latitud del punto de interés.
    - punto_lon: Longitud del punto de interés.

    Retorna:
    - radial_angle: Ángulo radial en grados desde el radar al punto de interés.
    """
    # Convertir latitud y longitud a radianes
    radar_lat_rad = np.radians(radar_lat)
    radar_lon_rad = np.radians(radar_lon)
    punto_lat_rad = np.radians(punto_lat)
    punto_lon_rad = np.radians(punto_lon)

    # Calcula la diferencia de longitud
    d_lon = punto_lon_rad - radar_lon_rad

    # Calcula el ángulo usando la fórmula de azimut
    x = np.sin(d_lon) * np.cos(punto_lat_rad)
    y = np.cos(radar_lat_rad) * np.sin(punto_lat_rad) - (np.sin(radar_lat_rad) * np.cos(punto_lat_rad) * np.cos(d_lon))

    azimuth = np.arctan2(x, y)

    # Convierte el ángulo a grados y ajustar el rango [0, 360)
    radial_angle = np.degrees(azimuth)
    radial_angle = (radial_angle + 360) % 360

    return radial_angle


def variable_radar_cross_section(volumen_radar_data, radial_angle, output_path, range_max, variable='DBZH', cmap='viridis', gf=None):
    """
    Función para graficar datos radiales del radar en un perfil a un ángulo dado.
    Esta función grafica en un ángulo dado que esta definido por una latitud y longitud.

    Parámetros:
    - volumen_radar_data: Datos del radar en formato Py-ART.
    - radial_angle: Ángulo del radial para visualizar (de 0 a 360 grados).
    - variable: Variable a graficar (por defecto 'DBZH').
    """
    # Hacemos una copia profunda del objeto volumen_radar_data para no modificar el original
    radar_data_copy_3 = copy.deepcopy(volumen_radar_data)
    # Datos de la variable seleccionada del volumen de radar
    data = radar_data_copy_3.fields[variable]['data']

    # Determinamos los valores mínimos y máximos dinámicamente
    vmin = data.min()
    vmax = data.max()

    # Obtenemos unidades de la variable
    units = VARIABLE_UNITS.get(variable, '')

    # Se realiza gráfico del cross section
    xsect = pyart.util.cross_section_ppi(radar_data_copy_3, [radial_angle], gatefilter=gf)
    display = pyart.graph.RadarDisplay(xsect)  # Crear el display de Py-ART

    # Crear la figura y el subplot
    fig = plt.figure(figsize=[15, 3.5])
    ax2 = plt.subplot(1, 1, 1)

    # Graficar la variable especificada
    display.plot(variable, 0, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax2)
    display.set_limits(xlim=[0, range_max], ylim=[0, 30])

    # Agregar el valor del ángulo radial al gráfico
    ax2.text(0.98, 0.95, f'Ángulo desde el N: {radial_angle:.2f}°',
             horizontalalignment='right', verticalalignment='top',
             transform=ax2.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Distancia al radar (km)', fontsize=14)
    plt.ylabel(f'Altura (km) - {units}', fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_pseudo_rhi_png(
    filepath: str,
    field: str,
    end_lon: float,
    end_lat: float,
    max_length_km: float,
    elevation: int = 0,
    filters : List[RangeFilter] = [],
    output_dir: str = "app/storage/tmp"
):
    
    file_hash = md5_file(filepath)[:12]
    filters_str = "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters]) if filters else "nofilter"
    unique_out_name = f"pseudo_rhi_{field}_{filters_str}_{elevation}_{file_hash}.png"
    out_path = Path(output_dir) / unique_out_name

    if out_path.exists():
        return {"image_url": f"static/tmp/{unique_out_name}", "metadata": None}

    os.makedirs(output_dir, exist_ok=True)
    radar = pyart.io.read(filepath)
    radar = radar.extract_sweeps([elevation]) if elevation < radar.nsweeps else radar.extract_sweeps([0])

    field_name = resolve_field(radar, field)
    site_lon, site_lat, site_alt = get_radar_site(radar)
    range_max_km = safe_range_max_m(radar) / 1000.0

    # Filtros (se aplican por GateFilter para enmascarar fuera de rango)
    gf = build_gatefilter(radar, field_name, filters)

    # Colormap + vmin/vmax
    cmap, vmin, vmax, _ = colormap_for(field)

    # Graficar perfil radial
    variable_radar_cross_section(
        radar,
        calcule_radial_angle(site_lat, site_lon, end_lat, end_lon),
        out_path,
        range_max=min(max_length_km, range_max_km),
        variable=field_name,
        cmap=cmap,
        gf=gf
    )


    return {
        "image_url": f"static/tmp/{unique_out_name}",
        "metadata": {
            "radar_site": {"lon": site_lon, "lat": site_lat, "alt_m": site_alt},
            "field": field.upper(),
            "point": {"lon": end_lon, "lat": end_lat},
        }
    }
