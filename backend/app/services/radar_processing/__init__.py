"""
Módulo radar_processing: contiene componentes especializados para procesamiento de radar.

Estructura:
- grid_builder: Construcción y caché de grillas 3D
- field_processor: Colapso de grillas 3D a 2D según producto
- warping: Proyección a Web Mercator
- cog_generator: Generación de COG (Cloud Optimized GeoTIFF)
"""

from .grid_builder import get_or_build_grid3d
from .field_processor import collapse_grid_to_2d
from .warping import warp_array_to_mercator
from .cog_generator import convert_to_cog, create_cog_from_warped_array

__all__ = [
    'get_or_build_grid3d',
    'collapse_grid_to_2d',
    'warp_array_to_mercator',
    'convert_to_cog',
    'create_cog_from_warped_array',
]
