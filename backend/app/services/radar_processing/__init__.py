"""
Módulo radar_processing: contiene componentes especializados para procesamiento de radar.

Estructura:
- grid_builder: Construcción y caché de grillas 3D
- grid_compute: Cálculo de operador W para interpolación de grillas 3D
- grid_interpolate: Interpolación de datos de radar a grillas 3D
- field_processor: Colapso de grillas 3D a 2D según producto
- warping: Proyección a Web Mercator
- cog_generator: Generación de COG (Cloud Optimized GeoTIFF)
"""

from .grid_builder import get_or_build_grid3d_with_operator
from .grid_compute import build_W_operator
from .grid_interpolate import apply_operator_to_all_fields
from .field_processor import collapse_grid_to_2d
from .warping import warp_array_to_mercator
from .cog_generator import convert_to_cog, create_cog_from_warped_array

__all__ = [
    'get_or_build_grid3d_with_operator',
    'build_W_operator',
    'apply_operator_to_all_fields',
    'collapse_grid_to_2d',
    'warp_array_to_mercator',
    'convert_to_cog',
    'create_cog_from_warped_array',
]
