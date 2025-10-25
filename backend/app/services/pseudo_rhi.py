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
from scipy.interpolate import RegularGridInterpolator
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


def variable_radar_cross_section(lat, lon, radar_lat, radar_lon, volumen_radar_data, output_path, range_max, variable='DBZH', cmap='viridis', gf=None):
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
    tif_path = Path("app/storage/data/DEM_Cba_100_LL.tif")
    with rasterio.open(tif_path) as src:
        elevacion = src.read(1, masked=True)  # Leo el primer haz con máscara para los valores nodata.
                                        #Es una matriz NumPy que contiene los datos de elevación leídos de la banda del raster.
        # Reemplazo los valores "nodata" por NaN si es necesario
        elevacion_filled = elevacion.filled(np.nan)
    ######################################################################
    # Del archivo raster cargado anteriormente se obtiene la elevación de forma geográfica
    transform = src.transform #Obtiene la transformación affine del raster, que es una matriz que convierte las coordenadas de la imagen
                                            #(pixeles) a coordenadas espaciales (geográficas).
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top] #Obtiene los límites espaciales del raster,
                                                      #que son las coordenadas del borde del raster en el sistema de coordenadas del raster.
    ######################################################################

    # Calculo el ángulo radial
    radial_angle = calcule_radial_angle(radar_lat, radar_lon, lat, lon)

    # Obtengo el perfil de elevación del terreno y las distancias desde el radar
    num_points = 10000
    elev_x = np.linspace(transform[0] * 0 + transform[2], transform[0] * elevacion.shape[1] + transform[2], elevacion.shape[1])
    elev_y = np.linspace(transform[4] * 0 + transform[5], transform[4] * elevacion.shape[0] + transform[5], elevacion.shape[0])
    interpolator = RegularGridInterpolator((elev_y, elev_x), elevacion, bounds_error=False, fill_value=None)

    # Destino calculado en base al rango del radar
    radar_range=round(volumen_radar_data.ngates*volumen_radar_data.range['meters_between_gates']/1000)
    destination = geodesic(kilometers=radar_range).destination((radar_lat, radar_lon), radial_angle)
    lat_final = destination.latitude
    lon_final = destination.longitude

    # Puntos de latitud y longitud
    lat_points = np.linspace(radar_lat, lat_final, num_points)
    lon_points = np.linspace(radar_lon, lon_final, num_points)

    points = np.vstack([lat_points, lon_points]).T
    perfil_elevacion = interpolator(points)

    # Detectar último índice válido
    valid_indices = np.where(~np.isnan(perfil_elevacion))[0]
    last_valid_index = num_points - 1  # Inicialmente, el último índice es el último punto calculado

    if valid_indices.size > 0:
        last_valid_index = valid_indices[-1]  # Último índice válido

        # Comprobar cambios abruptos en la elevación (ej. si baja más de 500m)
        for i in range(1, last_valid_index + 1):
            if perfil_elevacion[i] < perfil_elevacion[i - 1] - 500:
                last_valid_index = i - 1  # Cortar antes de este índice
                break

        # Cortar los arrays a los valores válidos
        perfil_elevacion = perfil_elevacion[:last_valid_index + 1]
        lat_points = lat_points[:last_valid_index + 1]
        lon_points = lon_points[:last_valid_index + 1]

    # Restar offset para alinear con la salida del radar y convertir a kilómetros
    offset = 439.0423493233697
    perfil_elevacion -= offset
    perfil_elevacion_km = perfil_elevacion / 1000.0

    # Calculamos las distancias al radar
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
    display.plot(variable, 0, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax2, mask_outside=True)
    display.set_limits(xlim=[0, range_max], ylim=[-0.5, 30])

    # Grafico el perfil de elevación del terreno
    ax2.plot(distances, perfil_elevacion_km, label=None, color='black', linewidth=2)

    # Marcar el punto de interés con un asterisco
    distancia_interes = geodesic((radar_lat, radar_lon), (lat, lon)).km

    # Asegurarse de que la interpolación reciba una lista bidimensional
    elevacion_interes = interpolator([[lat, lon]])
    if elevacion_interes.size > 0 and not np.isnan(elevacion_interes[0]):
        # Aplicar el mismo offset y convertir a kilómetros
        elevacion_interes_value = (elevacion_interes[0] - offset) / 1000.0
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
    gf = build_gatefilter(radar, field_name, filters)

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
