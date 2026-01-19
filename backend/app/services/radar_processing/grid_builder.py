"""
Construcción y caché de grillas 3D de radar usando operador disperso W.
"""
import time
import logging
import numpy as np
import pyart
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from ...core.cache import (
    W_OPERATOR_CACHE,
    save_w_operator_to_disk,
    load_w_operator_from_disk
)
from ...core.constants import AFFECTS_INTERP_FIELDS
from ..radar_common import (
    build_gatefilter,
    qc_signature,
    w_operator_cache_key,
)
from ..grid_geometry import (
    calculate_grid_points
)

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

    
def compute_weights(distances, roi, method='Barnes'):
    """
    Calcula pesos de interpolación según la distancia y el método.
    
    Args:
        distances: np.ndarray de distancias en metros
        roi: Radio de influencia en metros
        method: 'Barnes', 'Cressman', o 'nearest'
    
    Returns:
        np.ndarray: Pesos (sin normalizar)
    """
    distances = np.asarray(distances, dtype=np.float32)
    roi = float(roi)

    if distances.size == 0:
        return distances  # array vacío

    if roi <= 0:
        return np.zeros_like(distances, dtype=np.float32)

    if method == 'Barnes':
         # Barnes: w = exp(-d²/(2σ²)) con σ = ROI/2
        sigma = roi / 2.0
        # sigma > 0 ya garantizado por roi > 0
        return np.exp(-(distances * distances) / (2.0 * sigma * sigma)).astype(np.float32)
    
    elif method == 'Barnes2':
        # Py-ART: weights = exp(-dist2 / (r2/4)) + 1e-5
        r2 = roi * roi
        dist2 = distances * distances
        w = np.exp(-dist2 / (r2 / 4.0)) + 1e-5
        return w.astype(np.float32)

    elif method == 'Cressman':
        # Cressman: w = (ROI² - d²) / (ROI² + d²)
        r2 = roi * roi
        dist2 = distances * distances
        w = (r2 - dist2) / (r2 + dist2 + 1e-12)   # epsilon para estabilidad
        w = np.clip(w, 0.0, 1.0)  # evita negativos
        return w.astype(np.float32)

    elif method == 'nearest':
        # Nearest: solo el más cercano tiene peso 1
        weights = np.zeros_like(distances, dtype=np.float32)
        weights[int(np.argmin(distances))] = 1.0
        return weights

    else:
        raise ValueError(f"Método desconocido: {method}")


def build_W_operator(
    gates_xyz,
    voxels_xyz,
    constant_roi,
    weight_func="Barnes",
    max_neighbors=None,
    chunk_size=50_000,
    dtype_val=np.float32,
    dtype_idx=np.int32,
):
    """
    Construye operador disperso W (CSR: Compressed Sparse Row) que mapea gates -> voxels.

    W[i, j] = peso (sin normalizar) del gate j para el voxel i

    gates_xyz: (Ngates, 3) float
    voxels_xyz: (Nvoxels, 3) float
    constant_roi: float (metros)
    weight_func: 'Barnes', 'Cressman', 'nearest'
    max_neighbors: int o None
        - None: usa TODOS los gates dentro del ROI (query_ball_point)
        - int: usa hasta K vecinos dentro del ROI (query con upper bound)
    chunk_size: voxels procesados por bloque para reducir RAM
    dtype_val: dtype para los pesos
    dtype_idx: dtype para índices

    Returns:
        W: scipy.sparse.csr_matrix shape (Nvoxels, Ngates)
    """
    t_start = time.time()
    gates_xyz = np.asarray(gates_xyz, dtype=np.float32)
    voxels_xyz = np.asarray(voxels_xyz, dtype=np.float32)

    ngates = gates_xyz.shape[0]
    nvoxels = voxels_xyz.shape[0]
    roi = float(constant_roi)

    logger.info(f"Construyendo operador W: {nvoxels} voxels, {ngates} gates, ROI={roi:.1f}m")

    # Construir KDTree de gates (una sola vez)
    tree = cKDTree(gates_xyz)

    # Vamos a acumular COO (Coordinate format) por chunks en arrays (mucho mejor que listas gigantes)
    rows_chunks = []
    cols_chunks = []
    vals_chunks = []

    voxels_with_data = 0
    total_neighbors = 0

    # Para cada voxel, buscar gates vecinos
    for start in range(0, nvoxels, chunk_size):
        end = min(start + chunk_size, nvoxels)
        V = voxels_xyz[start:end]  # (m, 3)
        m = V.shape[0]

        # ==== Caso A: TODOS los vecinos en ROI ====
        if max_neighbors is None:
           # Query ball: todos los gates dentro del ROI
            neigh_lists = tree.query_ball_point(V, r=roi)

            # contamos nnz (no ceros) del chunk para prealocar
            # dice cuántos pares (voxel,gate) vamos a agregar en este bloque
            nnz_chunk = sum(len(lst) for lst in neigh_lists)
            if nnz_chunk == 0:
                continue

            # Prealocar arrays del tamaño justo
            rows = np.empty(nnz_chunk, dtype=dtype_idx)
            cols = np.empty(nnz_chunk, dtype=dtype_idx)
            vals = np.empty(nnz_chunk, dtype=dtype_val)

            pos = 0
            for local_i, idxs in enumerate(neigh_lists):
                if not idxs:
                    continue

                i_global = start + local_i
                idxs = np.asarray(idxs, dtype=np.int64)

                # Calcular distancias
                gates_subset = gates_xyz[idxs]  # (k,3)
                d = np.linalg.norm(gates_subset - V[local_i], axis=1).astype(np.float32)

                # Calcular pesos (sin normalizar)
                w = compute_weights(d, roi, weight_func).astype(dtype_val)

                k = idxs.size
                rows[pos:pos+k] = i_global
                cols[pos:pos+k] = idxs.astype(dtype_idx, copy=False)
                vals[pos:pos+k] = w

                pos += k
                voxels_with_data += 1
                total_neighbors += k

            # por si pos < nnz_chunk (si hubo voxels vacíos)
            rows = rows[:pos]
            cols = cols[:pos]
            vals = vals[:pos]

        # ==== Caso B: hasta K vecinos dentro de ROI ====
        # (Limitar número de vecinos si se especifica)
        else:
            K = int(max_neighbors)
            if K <= 0:
                raise ValueError("max_neighbors debe ser > 0 o None")

            dist, idx = tree.query(V, k=K, distance_upper_bound=roi)

            # normalizar shapes para K=1
            if K == 1:
                dist = dist.reshape(-1, 1)
                idx = idx.reshape(-1, 1)

            # Filtrar entradas inválidas (cuando no hay vecino, idx == ngates y dist=inf)
            valid = np.isfinite(dist) & (idx < ngates)
            nnz_chunk = int(valid.sum())
            if nnz_chunk == 0:
                continue

            # indices COO
            # filas: repetimos cada voxel por la cantidad de vecinos válidos
            rows = np.repeat(np.arange(start, end, dtype=dtype_idx), valid.sum(axis=1).astype(np.int32))
            cols = idx[valid].astype(dtype_idx, copy=False)

            # pesos: si nearest -> todos 1, si no -> por distancia
            d = dist[valid].astype(np.float32, copy=False)
            if weight_func == "nearest":
                vals = np.ones_like(d, dtype=dtype_val)
            else:
                vals = compute_weights(d, roi, weight_func).astype(dtype_val)

            # stats
            voxels_with_data += int((valid.sum(axis=1) > 0).sum())
            total_neighbors += nnz_chunk

        rows_chunks.append(rows)
        cols_chunks.append(cols)
        vals_chunks.append(vals)

    if len(rows_chunks) == 0:
        # No hubo vecinos para ningún voxel
        logger.warning("Operador W vacío (sin vecinos)")
        return csr_matrix((nvoxels, ngates), dtype=dtype_val)

    # Concatenar chunks
    rows_all = np.concatenate(rows_chunks)
    cols_all = np.concatenate(cols_chunks)
    vals_all = np.concatenate(vals_chunks)

    # Construir CSR
    W = csr_matrix((vals_all, (rows_all, cols_all)), shape=(nvoxels, ngates), dtype=dtype_val)
    # Limpieza útil
    W.sum_duplicates()
    W.sort_indices()

    t_elapsed = time.time() - t_start
    logger.info(
        f"Operador W construido: {W.nnz} elementos no-cero, "
        f"{voxels_with_data}/{nvoxels} voxels con datos, "
        f"tiempo={t_elapsed:.2f}s"
    )

    return W


def get_or_build_W_operator(
    radar_to_use: pyart.core.Radar,
    radar: str,
    estrategia: str,
    volumen: str,
    grid_shape: tuple,
    grid_limits: tuple,
    constant_roi: float,
    weight_func: str = 'Barnes2',
    max_neighbors: int | None = None,
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
        constant_roi: Radio de influencia en metros
        weight_func: Función de ponderación
        max_neighbors: Máximo número de vecinos
    
    Returns:
        scipy.sparse.csr_matrix: Operador W
    """
    # Generar cache key (sin session_id - compartido globalmente)
    cache_key = w_operator_cache_key(
        radar=radar,
        estrategia=estrategia,
        volumen=volumen,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        constant_roi=constant_roi,
        weight_func=weight_func,
        max_neighbors=max_neighbors,
    )
    
    # 1. Verificar cache RAM
    cached_pkg = W_OPERATOR_CACHE.get(cache_key)
    if cached_pkg is not None:
        logger.info(f"Operador W recuperado de cache RAM: {cache_key[:16]}...")
        return cached_pkg["W"]
    
    # 2. Verificar cache disco
    disk_result = load_w_operator_from_disk(cache_key)
    if disk_result is not None:
        W, metadata = disk_result
        logger.info(f"Operador W cargado desde disco: {cache_key[:16]}...")
        
        # Guardar en cache RAM para próximos usos
        W_OPERATOR_CACHE[cache_key] = {
            "W": W,
            "metadata": metadata
        }
        
        return W
    
    # 3. Construir operador W
    logger.info(f"Construyendo operador W: {radar}_{estrategia}_{volumen}")
    
    gates_xyz = get_gate_xyz_coords(radar_to_use, edges=False)
    voxels_xyz = get_grid_xyz_coords(grid_shape, grid_limits)
    
    W = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz,
        constant_roi=constant_roi,
        weight_func=weight_func,
        max_neighbors=max_neighbors
    )
    
    # Metadata para referencia
    metadata = {
        "radar": radar,
        "estrategia": estrategia,
        "volumen": volumen,
        "grid_shape": grid_shape,
        "grid_limits": grid_limits,
        "constant_roi": constant_roi,
        "weight_func": weight_func,
        "max_neighbors": max_neighbors,
        "nnz": W.nnz,
        "shape": W.shape,
        "created_at": time.time(),
    }
    
    # Guardar en disco
    save_w_operator_to_disk(cache_key, W, metadata)
    
    # Guardar en cache RAM
    W_OPERATOR_CACHE[cache_key] = {
        "W": W,
        "metadata": metadata
    }
    
    return W


def apply_operator(W, field_data, grid_shape, handle_mask=True):
    """
    Aplica operador W a datos de campo del radar.
    
    Args:
        W: scipy.sparse.csr_matrix (Nvoxels, Ngates) (CSR: Compressed Sparse Row)
        field_data: np.ma.MaskedArray de shape (nrays, ngates)
        grid_shape: tuple (nz, ny, nx) para reshape final
        handle_mask: Si True, normaliza por gates válidos
    
    Returns:
        np.ma.MaskedArray: Grilla 3D de shape (nz, ny, nx)
    """
    # Aplanar field_data a vector 1D
    # NOTA: si field_data es MaskedArray, conviene separar valores (.data) y máscara (.mask)
    if np.ma.isMaskedArray(field_data):
        g = field_data.data.ravel()  # valores puros (sin máscara)
        field_mask = field_data.mask  # máscara 2D
    else:
        g = np.asarray(field_data).ravel()
        field_mask = None
    
    if handle_mask:
        # Crear vector de máscara (1 = válido, 0 = enmascarado)
        if field_mask is not None:
            mask_valid = (~field_mask).astype(float).ravel()
        else:
            mask_valid = np.ones_like(g, dtype=float)
        
        # Reemplazar masked values con 0 para no afectar la suma
        g_filled = np.where(mask_valid, g, 0.0)

        # Denominador: suma de pesos solo para gates válidos
        den = W @ mask_valid
    else:
        g_filled = np.asarray(g, dtype=float)   # si no hay máscara, g ya es válido
        den = W @ np.ones_like(g_filled, dtype=float)   # suma de pesos por voxel
        
    # Numerador: suma ponderada de valores
    num = W @ g_filled
    
    # Evitar división por cero
    den = np.where(den > 1e-10, den, np.nan)
    
    # Resultado normalizado
    v = num / den
    
    # Crear masked array (marcar donde no hay datos)
    v_masked = np.ma.masked_invalid(v)
    
    # Reshape a 3D
    grid3d = v_masked.reshape(grid_shape)
    
    return grid3d


def get_or_build_grid3d_with_operator(
    radar_to_use: pyart.core.Radar,
    file_hash: str,
    radar: str,
    estrategia: str,
    volume: str,
    range_max_m: float,
    grid_limits: tuple, 
    grid_shape: tuple,
    grid_resolution_xy: float,
    grid_resolution_z: float,
    weight_func: str = 'Barnes2',
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
        range_max_m: Rango máximo en metros (afecta ROI)
        grid_limits: Límites de la grilla ((z_min, z_max), (y_min, y_max), (x_min, x_max))
        grid_shape: (nz, ny, nx)
        grid_resolution_xy: Resolución horizontal en metros
        grid_resolution_z: Resolución vertical en metros
        weight_func: Función de ponderación para el operador W
        session_id: Identificador de sesión
    
    Returns:
        pyart.core.Grid con la grilla 3D multi-campo construida
    """
    # Calcular ROI
    constant_roi = max(
        grid_resolution_xy * 1.5,
        800 + (range_max_m / 100000) * 400
    )
    
    # Obtener operador W (con caché completo: RAM -> Disco -> Build)
    W = get_or_build_W_operator(
        radar_to_use=radar_to_use,
        radar=radar,
        estrategia=estrategia,
        volumen=volume,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        constant_roi=constant_roi,
        weight_func=weight_func,
        max_neighbors=None,
    )
    
    # Aplicar a todos los campos disponibles
    all_fields = list(radar_to_use.fields.keys())
    
    fields_dict = {}
    for field_name in all_fields:
        field_data = radar_to_use.fields[field_name]['data']
        
        # Aplicar operador W
        grid3d_field = apply_operator(W, field_data, grid_shape, handle_mask=True)
        
        # Guardar en formato PyART
        field_dict = {
            'data': grid3d_field[np.newaxis, :, :, :],  # Agregar dimensión temporal
            'long_name': radar_to_use.fields[field_name].get('long_name', field_name),
            'units': radar_to_use.fields[field_name].get('units', ''),
            'standard_name': radar_to_use.fields[field_name].get('standard_name', field_name),
            '_FillValue': -9999.0,
        }
        fields_dict[field_name] = field_dict
    
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
