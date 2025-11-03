import os
import pyart
import copy
import numpy as np
import rasterio
from matplotlib import pyplot as plt
# matplotlib.rcParams['pcolormesh.shading'] = 'auto' # evita el error de pcolormesh
from fastapi import HTTPException
from pathlib import Path
from typing import List
from geopy.distance import geodesic
from rasterio.vrt import WarpedVRT
from scipy.interpolate import RegularGridInterpolator
from rasterio.enums import Resampling
from ..core.constants import VARIABLE_UNITS
from ..core.config import settings

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


def variable_radar_cross_section(
        lat, 
        lon, 
        radar_lat,
        radar_lon,
        volumen_radar_data,
        output_path,
        range_max,
        variable='DBZH',
        cmap='viridis',
        gf=None
):
    """
    Función para graficar datos radiales del radar en un perfil a un ángulo dado.
    Esta función grafica en un ángulo dado que esta definido por una latitud y longitud.
    También grafica el perfil de elevación del terreno.

    Parámetros:
    - volumen_radar_data: Datos del radar en formato Py-ART.
    - radial_angle: Ángulo del radial para visualizar (de 0 a 360 grados).
    - variable: Variable a graficar (por defecto 'DBZH').
    """
    ######################################################################
    tif_path = Path("app/storage/data/mosaico_argentina_2.tif")

    radial_angle = calcule_radial_angle(radar_lat, radar_lon, lat, lon)

    # Distancia máxima según el radar (en km)
    radar_range_km = round(volumen_radar_data.ngates * volumen_radar_data.range['meters_between_gates'] / 1000)

    # Defino densidad de muestreo sobre la línea (cada N metros)
    # Podés ajustar este step si querés más/menos detalle
    step_m = 150.0
    num_points = int((radar_range_km * 1000) // step_m) + 1
    # Genero los puntos geodésicos desde el radar hacia el bearing
    lat_points = np.empty(num_points, dtype=np.float64)
    lon_points = np.empty(num_points, dtype=np.float64)
    for i in range(num_points):
        d_km = (i * step_m) / 1000.0
        p = geodesic(kilometers=d_km).destination((radar_lat, radar_lon), radial_angle)
        lat_points[i], lon_points[i] = p.latitude, p.longitude

    # Muestreo del DEM (Digital Elevation Model) solo en esos puntos y en float32 con NaN
    with rasterio.open(tif_path) as src:
        # WarpedVRT reprojecta/ajusta al CRS del raster y lee solo ventanas necesarias
        with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
            coords = list(zip(lon_points, lat_points))  # (x=lon, y=lat)
            # sample devuelve un generador de arrays; lo convertimos y casteamos a float32
            perfil_elevacion = np.fromiter((v[0] for v in vrt.sample(coords)),
                                           dtype=np.float32, count=len(coords))
            # Manejo de nodata -> NaN
            nodata = vrt.nodata
            if nodata is not None:
                mask = perfil_elevacion == nodata
                if mask.any():
                    perfil_elevacion[mask] = np.nan

    # Detectar último índice válido
    last_valid_index = len(perfil_elevacion) - 1
    valid_indices = np.where(~np.isnan(perfil_elevacion))[0]
    if valid_indices.size > 0:
        last_valid_index = valid_indices[-1]
        for i in range(1, last_valid_index + 1):
            if perfil_elevacion[i] < perfil_elevacion[i - 1] - 500:
                last_valid_index = i - 1
                break
        perfil_elevacion = perfil_elevacion[:last_valid_index + 1]
        lat_points = lat_points[:last_valid_index + 1]
        lon_points = lon_points[:last_valid_index + 1]

    # Restar offset y pasar a km
    offset = 439.0423493233697
    perfil_elevacion_km = (perfil_elevacion - offset) / 1000.0

    # Distancias al radar para eje X del perfil
    distances = [geodesic((radar_lat, radar_lon), (lat_points[i], lon_points[i])).km for i in range(len(lat_points))]

    # Empezamos con la parte del pseudo corte
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
    xsect = pyart.util.cross_section_ppi(radar_data_copy_3, [radial_angle])
    display = pyart.graph.RadarDisplay(xsect)  # Crear el display de Py-ART

    # Crear la figura y el subplot
    fig = plt.figure(figsize=[15, 5.5])
    ax2 = plt.subplot(1, 1, 1)

    # Graficar la variable especificada
    display.plot(variable, 0, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax2, mask_outside=True, gatefilter=gf)
    display.set_limits(xlim=[0, range_max], ylim=[-0.5, 30])

    # Grafico el perfil de elevación del terreno
    ax2.plot(distances, perfil_elevacion_km, label=None, color='black', linewidth=2)

    # Marcar el punto de interés con un asterisco
    distancia_interes = geodesic((radar_lat, radar_lon), (lat, lon)).km

    # Interpolar la elevación del punto interés directamente del DEM
    with rasterio.open(tif_path) as src:
        with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
            val = next(vrt.sample([(lon, lat)]))[0]
            elevacion_interes = np.float32(np.nan if (vrt.nodata is not None and val == vrt.nodata) else val)
    if np.isfinite(elevacion_interes):
        elevacion_interes_value = (elevacion_interes - offset) / 1000.0
        ax2.plot(distancia_interes, elevacion_interes_value, 'r*', markersize=15, label='Punto de interés')
    else:
        elevacion_interes_value = None
        print("No se pudo interpolar la elevación del punto de interés.")

    # Texto con el ángulo radial
    ax2.text(0.98, 0.95, f'Ángulo Radial: {radial_angle:.2f}°',
             horizontalalignment='right', verticalalignment='top',
             transform=ax2.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Distancia al radar (km)', fontsize=14)
    plt.ylabel(f'Altura (km) - {units}', fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

from pyproj import Geod
_GEOD = Geod(ellps="WGS84")

def _km_between(lon0, lat0, lon1, lat1):
  _, _, d = _GEOD.inv(lon0, lat0, lon1, lat1)
  return d / 1000.0

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
    points = f"{end_lon}_{end_lat}"
    unique_out_name = f"pseudo_rhi_{field}_{points}_{filters_str}_{elevation}_{file_hash}.png"
    out_path = Path(output_dir) / unique_out_name

    if out_path.exists():
        return {"image_url": f"{settings.BASE_URL}/static/tmp/{unique_out_name}", "metadata": None}

    os.makedirs(output_dir, exist_ok=True)
    radar = pyart.io.read(filepath)
    # radar = radar.extract_sweeps([elevation]) if elevation < radar.nsweeps else radar.extract_sweeps([0])

    field_name, _ = resolve_field(radar, field)
    site_lon, site_lat, site_alt = get_radar_site(radar)
    range_max_km = safe_range_max_m(radar) / 1000.0

    if _km_between(site_lon, site_lat, end_lon, end_lat) > range_max_km:
        print(f"El punto está fuera del alcance ({range_max_km:.1f} km).")
        raise HTTPException(status_code=400, detail=f"El punto está fuera del alcance ({range_max_km:.1f} km).")

    # Filtros (se aplican por GateFilter para enmascarar fuera de rango)
    gf = build_gatefilter(radar, field_name, filters, is_rhi=True)

    # Colormap + vmin/vmax
    cmap, vmin, vmax, _ = colormap_for(field)

    #Graficar perfil radial
    variable_radar_cross_section(
        end_lat, 
        end_lon,
        site_lat, 
        site_lon, 
        radar,
        out_path,
        range_max=range_max_km,
        variable=field_name,
        cmap=cmap,
        gf=gf
    )


    return {
        "image_url": f"{settings.BASE_URL}/static/tmp/{unique_out_name}",
        "metadata": {
            "radar_site": {"lon": site_lon, "lat": site_lat, "alt_m": site_alt},
            "field": field.upper(),
            "point": {"lon": end_lon, "lat": end_lat},
        }
    }
