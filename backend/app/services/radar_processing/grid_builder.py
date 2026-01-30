"""
Construcción y caché de grillas 3D de radar usando operador disperso W.
"""
import time
import logging
import numpy as np
import pyart
from scipy.sparse import csr_matrix

from ...core.cache import (
    W_OPERATOR_CACHE,
    save_w_operator_to_disk,
    load_w_operator_from_disk,
    try_cache_w_operator_in_ram,
    get_w_operator_size_mb,
    W_OPERATOR_SESSION_INDEX,
    W_OPERATOR_REF_COUNT
)
from ...core.constants import AFFECTS_INTERP_FIELDS
from ..radar_common import w_operator_cache_key
from .grid_compute import build_W_operator
from .grid_interpolate import apply_operator_to_all_fields
from .filter_application import build_gatefilter_for_gridding
from .product_preparation import prepare_radar_for_product

logger = logging.getLogger(__name__)


def get_gate_xyz_coords(radar, edges=False):
    """Calcular coordenadas cartesianas (x,y,z) de todos los gates del radar.
    
    Args:
        radar: objeto pyart.core.Radar
        edges: si True, calcula usando bordes. Default False.
    
    Returns:
        xyz: np.ndarray shape (radar.nrays * radar.ngates, 3) en metros
    """
    # Asegurar que existan / estén calculados.
    if edges:
        # Para edges, pyart típicamente requiere computarlo por sweep.
        # (porque gate_x "normal" suele representar centros).
        xs, ys, zs = [], [], []
        for s in range(radar.nsweeps):
            gx, gy, gz = radar.get_gate_x_y_z(s, edges=True, filter_transitions=False)
            xs.append(gx.ravel()); ys.append(gy.ravel()); zs.append(gz.ravel())
        x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    else:
        # Centros: usar los atributos (coinciden con field_data)
        x = radar.gate_x["data"].ravel()
        y = radar.gate_y["data"].ravel()
        z = radar.gate_z["data"].ravel()

    xyz = np.column_stack((x, y, z))

    # Chequeo de consistencia
    expected = radar.nrays * radar.ngates
    if xyz.shape[0] != expected:
        raise ValueError(
            f"Gate XYZ size mismatch: got {xyz.shape[0]}, expected {expected}. "
            "Puede haber transiciones filtradas o un orden de rays distinto."
        )

    return xyz

def get_grid_xyz_coords(grid_shape, grid_limits, dtype=np.float32):
    """
    Genera coordenadas (x,y,z) de todos los voxels (centros) de una grilla 3D regular.
    
    Args:
        grid_shape: tuple (nz, ny, nx)
        grid_limits: tuple ((z_min, z_max), (y_min, y_max), (x_min, x_max))
        dtype: dtype de salida (float32 recomendado)
    
    Returns:
        np.ndarray: (Nvoxels, 3) coordenadas [x, y, z] en metros
    """
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = grid_limits
    nz, ny, nx = grid_shape

    if nz <= 0 or ny <= 0 or nx <= 0:
        raise ValueError(f"grid_shape inválido: {grid_shape}")

    def axis_coords(vmin, vmax, n):
        # Si hay un solo punto, lo ponemos en el centro del intervalo
        if n == 1:
            return np.array([(vmin + vmax) / 2.0], dtype=dtype)
        return np.linspace(vmin, vmax, n, dtype=dtype)

    z = axis_coords(zmin, zmax, nz)  # (nz,)
    y = axis_coords(ymin, ymax, ny)  # (ny,)
    x = axis_coords(xmin, xmax, nx)  # (nx,)

    # Construcción sin meshgrid gigante:
    # Orden 'ij' (z,y,x): el índice lineal i recorre primero x, luego y, luego z.
    N = nz * ny * nx

    X = np.tile(x, nz * ny)
    Y = np.tile(np.repeat(y, nx), nz)
    Z = np.repeat(z, ny * nx)

    xyz = np.column_stack((X, Y, Z)).astype(dtype, copy=False)

    # Check de consistencia
    if xyz.shape != (N, 3):
        raise RuntimeError(f"xyz shape inesperado: {xyz.shape}, esperado {(N, 3)}")

    return xyz


def get_or_build_W_operator(
    radar_to_use: pyart.core.Radar,
    radar: str,
    estrategia: str,
    volumen: str,
    grid_shape: tuple,
    grid_limits: tuple,
    h_factor: float = 0.8,
    nb: float = 1.1,
    bsp: float = 0.9,
    min_radius: float = 300.0,
    toa: float = 12000.0,
    weight_func: str = 'Barnes2',
    max_neighbors: int | None = None,
    session_id: str | None = None,
) -> csr_matrix:
    """
    Obtiene operador W desde caché (RAM o disco) o lo construye.
    
    Flujo:
    1. Verifica cache RAM (compartida)
    2. Si no está, verifica cache disco y carga a RAM
    3. Si no existe, construye, guarda en disco y RAM
    
    Args:
        radar_to_use: Objeto radar PyART (para obtener coordenadas gates)
        radar: Código del radar (ej: RMA1)
        estrategia: Estrategia de escaneo (ej: 0315)
        volumen: Número de volumen (ej: 01)
        grid_shape: (nz, ny, nx)
        grid_limits: ((z_min, z_max), (y_min, y_max), (x_min, x_max))
        h_factor: Escalado de altura (default 0.8)
        nb: Ancho de haz virtual en grados (default 1.0)
        bsp: Espaciado entre haces (default 0.8)
        min_radius: Radio mínimo en metros (default 300.0)
        weight_func: Función de ponderación
        max_neighbors: Máximo número de vecinos
    
    Returns:
        scipy.sparse.csr_matrix: Operador W
    """
    # Generar cache key (sin session_id - compartido globalmente)
    try:
        cache_key = w_operator_cache_key(
            radar=radar,
            estrategia=estrategia,
            volumen=volumen,
            grid_shape=grid_shape,
            grid_limits=grid_limits,
            h_factor=h_factor,
            nb=nb,
            bsp=bsp,
            min_radius=min_radius,
            weight_func=weight_func,
            max_neighbors=max_neighbors,
        )
    except Exception as e:
        logger.error(f"Error generando cache_key: {e}")
        raise
    
    # 1. Verificar cache RAM
    cached_pkg = W_OPERATOR_CACHE.get(cache_key)
    if cached_pkg is not None:
        logger.info(f"Operador W recuperado de cache RAM: {cache_key[:16]}...")
        return cached_pkg["W"]
    
    # 2. Verificar cache disco
    try:
        disk_result = load_w_operator_from_disk(cache_key)
        if disk_result is not None:
            W, metadata = disk_result
            logger.info(f"Operador W cargado desde disco: {cache_key[:16]}...")
            
            # Intentar guardar en cache RAM
            try_cache_w_operator_in_ram(cache_key, W, metadata, session_id)
            
            return W
    except Exception as e:
        logger.warning(f"Error cargando operador W desde disco, construyendo nuevo: {e}")
    
    # 3. Construir operador W
    logger.info(f"Construyendo operador W: {radar}_{estrategia}_{volumen}")
    
    gates_xyz = get_gate_xyz_coords(radar_to_use, edges=False)
    voxels_xyz = get_grid_xyz_coords(grid_shape, grid_limits)
    
    W = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz,
        toa=toa,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        weight_func=weight_func,
        max_neighbors=max_neighbors,
        n_workers=3,  # Usa cpu_count() - 1 automáticamente
        temp_dir=None,   # Crea directorio temporal automáticamente
        dtype_idx=np.int64,
    )
    
    # Metadata para referencia
    metadata = {
        "radar": radar,
        "estrategia": estrategia,
        "volumen": volumen,
        "grid_shape": grid_shape,
        "grid_limits": grid_limits,
        "h_factor": h_factor,
        "nb": nb,
        "bsp": bsp,
        "min_radius": min_radius,
        "weight_func": weight_func,
        "max_neighbors": max_neighbors,
        "nnz": W.nnz,
        "shape": W.shape,
        "created_at": time.time(),
    }
    
    # Siempre guardar en disco
    w_size_mb = get_w_operator_size_mb(W)
    save_w_operator_to_disk(cache_key, W, metadata)
    logger.info(f"Operador W guardado en disco ({w_size_mb:.2f} MB)")
    
    # Intentar guardar en cache RAM si no es muy grande
    try_cache_w_operator_in_ram(cache_key, W, metadata, session_id)
    
    return W


def get_or_build_grid3d_with_operator(
    radar_to_use: pyart.core.Radar,
    file_hash: str,
    radar: str,
    estrategia: str,
    volume: str,
    toa: float,
    grid_limits: tuple, 
    grid_shape: tuple,
    grid_resolution_xy: float,
    grid_resolution_z: float,
    weight_func: str = 'Barnes2',
    qc_filters: list | None = None,
    session_id: str | None = None,
) -> pyart.core.Grid:
    """
    Construye grilla 3D usando operador disperso W con caché persistente.
    
    Args:
        radar_to_use: Objeto radar de PyART
        file_hash: Hash del archivo para identificación
        radar: Código del radar (ej: RMA1)
        estrategia: Estrategia de escaneo (ej: 0315)
        volume: Volumen del radar
        grid_limits: Límites de la grilla ((z_min, z_max), (y_min, y_max), (x_min, x_max))
        grid_shape: (nz, ny, nx)
        grid_resolution_xy: Resolución horizontal en metros
        grid_resolution_z: Resolución vertical en metros
        weight_func: Función de ponderación para el operador W
        qc_filters: Lista de RangeFilter con filtros QC (ej. RHOHV) para aplicar durante interpolación
        session_id: Identificador de sesión
    
    Returns:
        pyart.core.Grid con la grilla 3D multi-campo construida
    """
    # Calcular parámetros dist_beam
    # h_factor: escalado de altura estándar
    h_factor = 0.8
    # nb: ancho de haz en grados
    nb = 1.1
    # bsp: espaciado entre haces
    bsp = 0.9
    # min_radius: radio mínimo en metros
    min_radius = 300.0
    
    # Obtener operador W (con caché completo: RAM -> Disco -> Build)
    W = get_or_build_W_operator(
        radar_to_use=radar_to_use,
        radar=radar,
        estrategia=estrategia,
        volumen=volume,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        toa=toa,
        weight_func=weight_func,
        max_neighbors=None,
        session_id=session_id,
    )
    
    # Construir GateFilter desde qc_filters
    gatefilter = build_gatefilter_for_gridding(radar_to_use, qc_filters)
    
    # Preparar dict de filtros por campo (mismo filtro para todos los campos)
    additional_filters = {}
    if gatefilter is not None:
        # Aplicar el mismo gatefilter a todos los campos
        for field_name in radar_to_use.fields.keys():
            additional_filters[field_name] = [gatefilter]
    
    # Aplicar operador W a todos los campos disponibles
    fields_dict = apply_operator_to_all_fields(
        radar=radar_to_use,
        W=W,
        grid_shape=grid_shape,
        handle_mask=True,
        additional_filters=additional_filters
    )
    
    # Generar coordenadas de la grilla
    x_coords = np.linspace(grid_limits[2][0], grid_limits[2][1], grid_shape[2]).astype(np.float32)
    y_coords = np.linspace(grid_limits[1][0], grid_limits[1][1], grid_shape[1]).astype(np.float32)
    z_coords = np.linspace(grid_limits[0][0], grid_limits[0][1], grid_shape[0]).astype(np.float32)
    
    # Crear Grid PyART
    grid_origin = (
        float(radar_to_use.latitude['data'][0]),
        float(radar_to_use.longitude['data'][0]),
    )
    metadata = dict(getattr(radar_to_use, "metadata", {}) or {})
    metadata.setdefault("instrument_name", metadata.get("instrument_name", "RADAR"))
    
    grid = pyart.core.Grid(
        time={
            'data': np.array([0]),
            'units': 'seconds since 2000-01-01T00:00:00Z',
            'calendar': 'gregorian',
            'standard_name': 'time'
        },
        fields=fields_dict,
        metadata=metadata,
        origin_latitude={'data': radar_to_use.latitude['data']},
        origin_longitude={'data': radar_to_use.longitude['data']},
        origin_altitude={'data': radar_to_use.altitude['data']},
        x={'data': x_coords},
        y={'data': y_coords},
        z={'data': z_coords},
    )
    
    # Agregar proyección
    grid.projection = {
        'proj': 'pyart_aeqd',
        'lat_0': grid_origin[0],
        'lon_0': grid_origin[1],
        '_include_lon_0_lat_0': True,
    }
    
    return grid
