"""
Módulo radar_processing: contiene componentes especializados para procesamiento de radar.

Estructura:
- grid_builder: Construcción y caché de grillas 3D
- grid_compute: Cálculo de operador W para interpolación de grillas 3D
- grid_interpolate: Interpolación de datos de radar a grillas 3D
- field_processor: Colapso de grillas 3D a 2D según producto
- warping: Proyección a Web Mercator
- cog_generator: Generación de COG (Cloud Optimized GeoTIFF)
- grid_geometry: Cálculo de geometría de grillas (límites, resolución, ROI dist_beam)
- product_preparation: Preparación de radar según producto (PPI/CAPPI/COLMAX)
- filter_application: Aplicación de filtros QC y visuales post-grid
"""

from .grid_builder import get_or_build_grid3d_with_operator, get_or_build_W_operator
from .grid_compute import build_W_operator
from .grid_interpolate import apply_operator_to_all_fields, apply_operator
from .field_processor import collapse_grid_to_2d
from .warping import warp_array_to_mercator
from .cog_generator import convert_to_cog, create_cog_from_warped_array
from .grid_geometry import (
    calculate_z_limits,
    calculate_grid_resolution,
    calculate_grid_points,
    calculate_roi_dist_beam,
    beam_height_max_km
)
from .product_preparation import (
    prepare_radar_for_product,
    fill_dbzh_if_needed
)
from .filter_application import (
    separate_filters,
    apply_qc_filters,
    apply_visual_filters
)

__all__ = [
    'get_or_build_grid3d_with_operator',
    'get_or_build_W_operator',
    'build_W_operator',
    'apply_operator_to_all_fields',
    'apply_operator',
    'collapse_grid_to_2d',
    'warp_array_to_mercator',
    'convert_to_cog',
    'create_cog_from_warped_array',
    'calculate_z_limits',
    'calculate_grid_resolution',
    'calculate_grid_points',
    'calculate_roi_dist_beam',
    'beam_height_max_km',
    'prepare_radar_for_product',
    'fill_dbzh_if_needed',
    'separate_filters',
    'apply_qc_filters',
    'apply_visual_filters',
]
