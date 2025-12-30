"""
Construcción y caché de grillas 3D de radar.
"""
import numpy as np
import pyart

from ...core.cache import GRID3D_CACHE
from ...core.constants import AFFECTS_INTERP_FIELDS
from ..radar_common import (
    build_gatefilter,
    qc_signature,
    grid3d_cache_key,
)
from ..grid_geometry import calculate_grid_points


def get_or_build_grid3d(
    radar_to_use: pyart.core.Radar,
    file_hash: str,
    volume: str | None,
    qc_filters,
    z_grid_limits: tuple,
    y_grid_limits: tuple,
    x_grid_limits: tuple,
    grid_resolution_xy: float,
    grid_resolution_z: float,
    session_id: str | None = None,
) -> pyart.core.Grid:
    """
    Función para obtener o construir una grilla 3D multi-campo cacheada.
    CAMBIO: Ya no recibe field_to_use - griddea TODOS los campos disponibles.
    
    Args:
        radar_to_use: Objeto radar de PyART
        file_hash: Hash del archivo para cache key
        volume: Volumen del radar (afecta resolución)
        qc_filters: Filtros QC a aplicar durante interpolación
        z_grid_limits: Límites en Z (altura) de la grilla
        y_grid_limits: Límites en Y de la grilla
        x_grid_limits: Límites en X de la grilla
        grid_resolution_xy: Resolución horizontal en metros
        grid_resolution_z: Resolución vertical en metros
        session_id: Identificador de sesión para aislar cache
    
    Returns:
        pyart.core.Grid con la grilla 3D multi-campo construida o recuperada de cache
    """
    # Generar cache key sin field_to_use
    qc_sig = qc_signature(qc_filters)
    cache_key = grid3d_cache_key(
        file_hash=file_hash,
        volume=volume,
        qc_sig=qc_sig,
        grid_res_xy=grid_resolution_xy,
        grid_res_z=grid_resolution_z,
        z_top_m=z_grid_limits[1],
        session_id=session_id,
    )
    
    # Verificar cache 3D
    pkg_cached = GRID3D_CACHE.get(cache_key)
    
    if pkg_cached is not None:
        # Reconstruir Grid multi-campo desde cache
        fields_dict = {}
        for fname, fdata in pkg_cached["fields"].items():
            field_dict = fdata["metadata"].copy()
            field_dict['data'] = fdata["data"]
            fields_dict[fname] = field_dict
        
        # Crear Grid con TODOS los campos cacheados
        grid = pyart.core.Grid(
            time={
                'data': np.array([0]),
                'units': 'seconds since 2000-01-01T00:00:00Z',
                'calendar': 'gregorian',
                'standard_name': 'time'
            },
            fields=fields_dict,
            metadata={'instrument_name': 'RADAR'},
            origin_latitude={'data': radar_to_use.latitude['data']},
            origin_longitude={'data': radar_to_use.longitude['data']},
            origin_altitude={'data': radar_to_use.altitude['data']},
            x={'data': pkg_cached["x"]},
            y={'data': pkg_cached["y"]},
            z={'data': pkg_cached["z"]},
        )
        grid.projection = pkg_cached["projection"]
        return grid
    
    # Construir grilla 3D con TODOS los campos del radar
    all_fields = list(radar_to_use.fields.keys())
    
    # Usar primer campo disponible para gatefilter (aplica a todos)
    first_field = all_fields[0] if all_fields else None
    gf = build_gatefilter(radar_to_use, first_field, qc_filters, is_rhi=False)
    
    grid_origin = (
        float(radar_to_use.latitude['data'][0]),
        float(radar_to_use.longitude['data'][0]),
    )
    
    range_max_m = (y_grid_limits[1] - y_grid_limits[0]) / 2
    constant_roi = max(
        grid_resolution_xy * 1.5,
        800 + (range_max_m / 100000) * 400
    )
    
    # Calcular puntos de grilla usando la función del módulo grid_geometry
    z_points, y_points, x_points = calculate_grid_points(
        z_grid_limits, y_grid_limits, x_grid_limits,
        grid_resolution_z, grid_resolution_xy
    )
    
    # CAMBIO: Gridear TODOS los campos disponibles en el radar
    fields_for_grid = all_fields
    
    grid = pyart.map.grid_from_radars(
        radar_to_use,
        grid_shape=(z_points, y_points, x_points),
        grid_limits=(z_grid_limits, y_grid_limits, x_grid_limits),
        gridding_algo="map_gates_to_grid",
        grid_origin=grid_origin,
        fields=fields_for_grid,
        weighting_function='nearest',
        gatefilters=gf,
        roi_func="constant",
        constant_roi=constant_roi,
    )
    grid.to_xarray()
    
    # Cachear TODOS los campos en estructura multi-campo
    fields_to_cache = {}
    for fname in grid.fields.keys():
        fields_to_cache[fname] = {
            "data": grid.fields[fname]['data'].copy(),
            "metadata": {k: v for k, v in grid.fields[fname].items() if k != 'data'}
        }
    
    pkg_to_cache = {
        "fields": fields_to_cache,  # Dict con todos los campos
        "x": grid.x['data'].copy(),
        "y": grid.y['data'].copy(),
        "z": grid.z['data'].copy(),
        "projection": dict(getattr(grid, "projection", {}) or {}),
    }
    GRID3D_CACHE[cache_key] = pkg_to_cache
    
    return grid
