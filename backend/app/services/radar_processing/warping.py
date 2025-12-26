"""
Proyección de arrays a Web Mercator.
"""
import numpy as np
from pyproj import Proj
from rasterio.warp import calculate_default_transform, reproject, Resampling
from affine import Affine


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
    src_crs = src_proj.to_wkt()
    
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
