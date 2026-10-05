"""
Proyección de arrays a Web Mercator.

Nota sobre las dos funciones de este módulo — no son intercambiables:

- warp_array_to_mercator: reconstruye su propio Affine transform de origen a
  partir de (radar_lat, radar_lon, x_limits, y_limits), asumiendo tamaño de
  píxel = (max-min)/n. No la usa process_radar_to_cog (queda solo por su
  propio test, test_warping_and_cog.py).
- warp_local_to_mercator: recibe un Affine transform y CRS YA CALCULADOS
  (por build_local_affine_and_crs en radar_common.py, con el offset de medio
  píxel correcto) y solo hace la reproyección. Es la que usa
  process_radar_to_cog. Construir el transform "a mano" de nuevo, como hace
  warp_array_to_mercator, perdería ese ajuste de medio píxel.
"""
import numpy as np
import rasterio.transform
from pyproj import Proj
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from affine import Affine


def warp_local_to_mercator(arr2d, src_transform, src_crs, vmin):
    """
    Reproyecta un array 2D ya georreferenciado (CRS local Azimuthal
    Equidistant) a Web Mercator, usando un Affine transform ya calculado.

    Args:
        arr2d: Array 2D (MaskedArray) en el CRS local del radar
        src_transform: Affine transform del array de origen
        src_crs: CRS del array de origen (string o dict de PROJ)
        vmin: Valor mínimo válido — píxeles por debajo se enmascaran
            (limpia ruido de bordes introducido por la reproyección)

    Returns:
        Tupla (arr_warped, transform_warped, crs_warped)
    """
    ny, nx = arr2d.shape
    dst_crs = "EPSG:3857"

    # Bounds del raster fuente (edge-to-edge, calculados desde el Affine transform)
    src_bounds = rasterio.transform.array_bounds(ny, nx, src_transform)

    # Calcular transform y dimensiones en Web Mercator
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        nx,
        ny,
        left=src_bounds[0],
        bottom=src_bounds[1],
        right=src_bounds[2],
        top=src_bounds[3],
    )

    # Preparar arrays: NaN para datos enmascarados
    src_data = np.ma.filled(arr2d, fill_value=np.nan).astype(np.float32)
    dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    # Reproyectar de Azimuthal Equidistant a Web Mercator
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # Crear MaskedArray y enmascarar ruido de bordes
    arr_warped = np.ma.masked_invalid(dst_data)
    arr_warped = np.ma.masked_less(arr_warped, vmin)

    return arr_warped, dst_transform, dst_crs


def warp_array_to_mercator(data_2d, radar_lat, radar_lon, x_limits, y_limits):
    """
    Warpea un array 2D de coordenadas locales del radar a Web Mercator.
    
    Args:
        data_2d: Array numpy 2D en coordenadas locales del radar
        radar_lat: Latitud del radar
        radar_lon: Longitud del radar
        x_limits: Tupla (x_min, x_max) en metros
        y_limits: Tupla (y_min, y_max) en metros
    
    Returns:
        Tupla (warped_data, transform, crs_string)
        - warped_data: Array numpy 2D proyectado a Web Mercator
        - transform: Affine transform del array warped
        - crs_string: CRS del array warped (Web Mercator)
    """
    # Proyección de origen: Azimuthal Equidistant centrada en el radar
    src_proj = Proj(proj='aeqd', lat_0=radar_lat, lon_0=radar_lon, datum='WGS84')
    # Usar CRS.from_dict en vez de WKT para evitar conflicto de versiones
    # PROJ DB entre osgeo (minor=3) y rasterio (requiere minor>=4)
    src_crs = CRS.from_dict({
        'proj': 'aeqd',
        'lat_0': float(radar_lat),
        'lon_0': float(radar_lon),
        'datum': 'WGS84',
    })
    
    # Calcular transform de origen (coordenadas locales del radar)
    ny, nx = data_2d.shape
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    pixel_size_x = (x_max - x_min) / nx
    pixel_size_y = (y_max - y_min) / ny
    
    src_transform = Affine.translation(x_min, y_max) * Affine.scale(pixel_size_x, -pixel_size_y)
    
    # Calcular bounds en coordenadas geográficas
    left, bottom = src_proj(x_min, y_min, inverse=True)
    right, top = src_proj(x_max, y_max, inverse=True)
    
    # Proyección destino: Web Mercator
    dst_crs = 'EPSG:3857'
    
    # Calcular transformación y dimensiones en Web Mercator
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, nx, ny,
        left=left, bottom=bottom, right=right, top=top
    )
    
    # Crear array de salida
    warped_data = np.full((dst_height, dst_width), np.nan, dtype=data_2d.dtype)
    
    # Warpear datos
    reproject(
        source=data_2d,
        destination=warped_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan
    )
    
    return warped_data, dst_transform, dst_crs
