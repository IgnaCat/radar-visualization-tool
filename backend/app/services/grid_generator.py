"""
Generación bajo demanda de grillas 2D para estadísticas.
No cachea resultados - solo para uso en stats cuando el campo no está cacheado.
"""
import pyart
import numpy as np
import pyproj
from typing import Dict, Optional, List
from affine import Affine

from ..core.constants import AFFECTS_INTERP_FIELDS
from .radar_common import (
    resolve_field, 
    safe_range_max_m, 
    normalize_proj_dict,
)
from .grid_geometry import (
    calculate_z_limits,
    calculate_grid_resolution,
    calculate_grid_points,
)
from .product_preparation import (
    prepare_radar_for_product,
    fill_dbzh_if_needed,
)
from .radar_processing import get_or_build_grid3d_with_operator
from .radar_processing import collapse_grid_to_2d
from .filter_application import separate_filters


def generate_grid2d_on_demand(
    radar: pyart.core.Radar,
    field_requested: str,
    product: str,
    file_hash: str,
    radar_name: str,
    estrategia: Optional[str] = None,
    volume: Optional[str] = None,
    elevation: Optional[int] = 0,
    cappi_height: Optional[int] = 4000,
    filters: List = None,
    interp: Optional[str] = "Barnes2",
) -> Dict:
    """
    Genera una grilla 2D bajo demanda SIN cachearla.
    Reutiliza get_or_build_grid3d_with_operator y luego colapsa.
    (Mas q nada utilizada para estadísticas cuando el campo no está cacheado)
    
    Args:
        radar: Objeto radar de PyART
        field_requested: Campo solicitado (ej: 'DBZH')
        product: Tipo de producto ('PPI', 'CAPPI', 'COLMAX')
        file_hash: Hash del archivo para cache key de grid3d
        volume: Volumen del radar
        elevation: Índice de elevación (para PPI)
        cappi_height: Altura CAPPI en metros
        filters: Lista de filtros
        radar_name: Nombre del radar
        estrategia: Estrategia de procesamiento
        interp: Método de interpolación
    
    Returns:
        Dict con: arr (np.ma.MaskedArray), transform (Affine), crs (WKT), qc (dict)
    """
    filters = filters or []
    product_upper = product.upper()
    field_requested_upper = field_requested.upper()
    
    # Resolver nombre del campo en el radar (devuelve tupla: field_name, field_key)
    field_name, field_key = resolve_field(radar, field_requested)
    
    # Preparar radar según producto (rellena DBZH si es CAPPI/COLMAX)
    field_name = fill_dbzh_if_needed(radar, field_name, product)
    radar_to_use, field_to_use = prepare_radar_for_product(
        radar, product, field_name, elevation, cappi_height
    )
    
    # Separar filtros QC
    qc_filters, _ = separate_filters(filters, field_requested_upper)
    
    # Determinar rango máximo del radar usando función segura
    range_max_m = safe_range_max_m(radar)
    
    # Calcular límites y resolución de grilla con todos los parámetros necesarios
    z_min, z_max, elev_deg_used = calculate_z_limits(
        range_max_m=range_max_m,
        elevation=elevation,
        cappi_height=cappi_height,
        radar_fixed_angles=radar.fixed_angle['data']
    )
    z_grid_limits = (z_min, z_max)
    y_grid_limits = (-range_max_m, range_max_m)
    x_grid_limits = (-range_max_m, range_max_m)
    grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits)
    
    # Resolución según volumen
    grid_resolution_xy, grid_resolution_z = calculate_grid_resolution(volume)

    # Calcular puntos de grilla
    z_points, y_points, x_points = calculate_grid_points(
        grid_limits[0], grid_limits[1], grid_limits[2],
        grid_resolution_z, grid_resolution_xy
    )
    grid_shape = (z_points, y_points, x_points)
    
    # Obtener o construir grilla 3D (usa cache 3D multi-campo con TODOS los campos incluyendo QC)
    grid = get_or_build_grid3d_with_operator(
        radar_to_use=radar_to_use,
        file_hash=file_hash,
        radar=radar_name,
        estrategia=estrategia,
        volume=volume,
        range_max_m=range_max_m,
        grid_limits=grid_limits,
        grid_shape=grid_shape,
        grid_resolution_xy=grid_resolution_xy,
        grid_resolution_z=grid_resolution_z,
        weight_func=interp
    )
    
    # Verificar que el campo existe en la grilla
    if field_to_use not in grid.fields:
        raise ValueError(f"Campo '{field_to_use}' no encontrado en grilla 3D")
    
    # Colapsar 3D -> 2D según producto
    if product_upper == "PPI":
        collapse_grid_to_2d(
            grid, field_to_use, "ppi",
            elevation_deg=elev_deg_used
        )
    elif product_upper == "CAPPI":
        collapse_grid_to_2d(
            grid, field_to_use, "cappi",
            target_height_m=cappi_height
        )
    elif product_upper == "COLMAX":
        collapse_grid_to_2d(
            grid, field_to_use, "colmax"
        )
    else:
        raise ValueError(f"Producto '{product}' no soportado")
    
    # Extraer array 2D colapsado
    arr2d = grid.fields[field_to_use]['data'][0, :, :]  # (ny, nx)
    arr2d = np.ma.masked_invalid(arr2d)

    # Obtener grid_origin para normalize_proj_dict
    grid_origin = (
        float(radar_to_use.latitude['data'][0]),
        float(radar_to_use.longitude['data'][0]),
    )
    
    # Construir transform (Affine)
    x = grid.x['data'].astype(float)
    y = grid.y['data'].astype(float)
    ny, nx = arr2d.shape
    dx = float(np.mean(np.diff(x))) if x.size > 1 else (x_grid_limits[1]-x_grid_limits[0]) / max(nx-1, 1)
    dy = float(np.mean(np.diff(y))) if y.size > 1 else (y_grid_limits[1]-y_grid_limits[0]) / max(ny-1, 1)
    xmin = float(x.min()) if x.size else x_grid_limits[0]
    ymax = float(y.max()) if y.size else y_grid_limits[1]

    # CRS de la grilla (normalizado)
    transform = Affine.translation(xmin - dx/2, ymax + dy/2) * Affine.scale(dx, -dy)
    proj_dict_norm = normalize_proj_dict(grid, grid_origin)
    crs_wkt = pyproj.CRS.from_dict(proj_dict_norm).to_wkt()
    
    # Recopilar campos QC que YA ESTÁN en la grilla 3D cacheada (no recalcular)
    qc_dict = {}
    for qc_field_name in AFFECTS_INTERP_FIELDS:
        # Los campos en la grilla ya están con sus nombres reales, buscar directamente
        if qc_field_name == field_to_use:
            continue
        if qc_field_name not in grid.fields:
            continue
            
        # Colapsar el campo QC con el mismo método que el campo principal
        if product_upper == "PPI":
            collapse_grid_to_2d(
                grid, qc_field_name, "ppi",
                elevation_deg=elev_deg_used
            )
        elif product_upper == "CAPPI":
            collapse_grid_to_2d(
                grid, qc_field_name, "cappi",
                target_height_m=cappi_height
            )
        elif product_upper == "COLMAX":
            collapse_grid_to_2d(
                grid, qc_field_name, "colmax"
            )
        
        qc_arr = grid.fields[qc_field_name]['data'][0, :, :]
        qc_dict[qc_field_name] = np.ma.masked_invalid(qc_arr)
    
    return {
        "arr": arr2d,
        "transform": transform,
        "crs": crs_wkt,
        "qc": qc_dict,
    }
