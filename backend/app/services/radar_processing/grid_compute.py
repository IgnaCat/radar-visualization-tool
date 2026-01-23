"""
Módulo para cálculo de grillas de radar usando operadores dispersos W.
Construye el operador W que mapea gates de radar a voxels de grilla 3D
usando procesamiento opcional.
"""

import time
import gc
import os
import tempfile
import logging
import pyart
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from typing import Tuple

from .grid_geometry import calculate_roi_dist_beam

logger = logging.getLogger(__name__)

def compute_weights(distances, roi, method='Barnes2'):
    """
    Calcula pesos de interpolación según la distancia y el método.
    
    Args:
        distances: np.ndarray de distancias en metros
        roi: Radio de influencia en metros (puede ser escalar o array del mismo tamaño que distances)
        method: 'Barnes', 'Cressman', o 'nearest'
    
    Returns:
        np.ndarray: Pesos (sin normalizar)
    """
    distances = np.asarray(distances, dtype=np.float32)
    roi = np.asarray(roi, dtype=np.float32)

    if distances.size == 0:
        return distances  # array vacío

    # Manejar ROI escalar o array
    if roi.ndim == 0:  # escalar
        roi_val = float(roi)
        if roi_val <= 0:
            return np.zeros_like(distances, dtype=np.float32)
    else:  # array
        if np.any(roi <= 0):
            result = np.zeros_like(distances, dtype=np.float32)
            valid_roi = roi > 0
            if not valid_roi.any():
                return result

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


def _process_single_level(args) -> Tuple[int, int, str]:
    """
    Worker function para procesar un único nivel Z (procesamiento paralelo).
    
    Construye KD-tree para gates válidos y encuentra todos los mapeos gate-to-grid
    para una slice horizontal de la grilla.
    
    Args:
        args: tuple con (iz, z_coord, grid_y_2d, grid_x_2d, gate_x, gate_y, gate_z,
                         gate_valid_mask, h_factor, nb, bsp, min_radius,
                         weight_func, temp_dir, dtype_val, dtype_idx)
    
    Returns:
        (iz, n_pairs, temp_file): índice nivel, número de pares, archivo temporal
    """
    (iz, z_coord, grid_y_2d, grid_x_2d, gate_x, gate_y, gate_z,
     gate_valid_mask, h_factor, nb, bsp, min_radius,
     weight_func, temp_dir, dtype_val, dtype_idx) = args
    
    n_points = grid_y_2d.shape[0]
    
    # Filter gates by mask dentro del workers
    valid_indices = np.where(gate_valid_mask)[0]
    gate_x_valid = gate_x[gate_valid_mask]
    gate_y_valid = gate_y[gate_valid_mask]
    gate_z_valid = gate_z[gate_valid_mask]
    ngates = len(valid_indices)
    
    # Build KD-tree from valid gates only
    gate_coords = np.column_stack([gate_x_valid, gate_y_valid, gate_z_valid]).astype('float64')
    tree = cKDTree(gate_coords)
    del gate_coords
    gc.collect()
    
    grid_z_level = np.full(n_points, z_coord, dtype='float64')
    
    # ROI para este nivel usando dist_beam
    roi = calculate_roi_dist_beam(
        z_coords=grid_z_level,
        y_coords=grid_y_2d,
        x_coords=grid_x_2d,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        radar_offset=(0, 0, 0)
    )
    
    # Storage para este nivel (listas temporales)
    level_indices = []
    level_weights = []
    level_indptr = [0]
    
    for i in range(n_points):
        gx, gy, gz_pt = grid_x_2d[i], grid_y_2d[i], grid_z_level[i]
        r = roi[i]
        r2 = r * r
        
        point = np.array([gx, gy, gz_pt], dtype='float64')
        candidate_indices_local = tree.query_ball_point(point, r)
        
        if not candidate_indices_local:
            level_indptr.append(level_indptr[-1])
            continue
        
        candidate_indices_local = np.array(candidate_indices_local, dtype='int32')
        candidate_indices_global = valid_indices[candidate_indices_local]
        
        # Calcular distancias
        dx = gate_x_valid[candidate_indices_local] - gx
        dy = gate_y_valid[candidate_indices_local] - gy
        dz = gate_z_valid[candidate_indices_local] - gz_pt
        d2 = dx*dx + dy*dy + dz*dz
        
        # Doble validación ROI (como compute.py)
        mask = d2 < r2
        if not np.any(mask):
            level_indptr.append(level_indptr[-1])
            continue
        
        final_indices = candidate_indices_global[mask]
        final_d2 = d2[mask]
        
        # Calcular pesos
        d = np.sqrt(final_d2).astype('float32')
        w = compute_weights(d, r, weight_func).astype(dtype_val)
        
        level_indices.extend(final_indices)
        level_weights.extend(w)
        level_indptr.append(level_indptr[-1] + final_indices.shape[0])
    
    # Guardar nivel a archivo temporal
    temp_file = os.path.join(temp_dir, f'geometry_level_{iz}.npz')
    np.savez(
        temp_file,
        indptr=np.array(level_indptr, dtype=dtype_idx),
        gate_indices=np.array(level_indices, dtype=dtype_idx),
        weights=np.array(level_weights, dtype=dtype_val)
    )
    
    n_pairs = len(level_indices)
    
    # Limpiar memoria del worker
    del tree, level_indices, level_weights, level_indptr
    gc.collect()
    
    return iz, n_pairs, temp_file


def build_W_operator(
    gates_xyz,
    voxels_xyz,
    toa=12000,
    h_factor=1.0,
    nb=1.5,
    bsp=1.0,
    min_radius=800.0,
    weight_func="Barnes2",
    max_neighbors=None,
    n_workers=None,
    temp_dir=None,
    dtype_val=np.float32,
    dtype_idx=np.int32,
):
    """
    Construye operador disperso W (CSR: Compressed Sparse Row) que mapea gates -> voxels.
    
    OPERADOR W con PROCESAMIENTO PARALELO:
    Este operador se construye con z_max = TOA para servir a TODOS los sweeps.
    Usa filtro TOA para excluir gates (ecos no-meteorológicos).
    Procesa niveles Z en paralelo para mayor eficiencia.
    
    Args:
    gates_xyz: (Ngates, 3) float - coordenadas (x,y,z) de todos los gates
    voxels_xyz: (Nvoxels, 3) float - coordenadas (x,y,z) de todos los voxels
    toa: float, Top Of Atmosphere en metros (default 12000)
         Límite físico para excluir ecos no-meteorológicos sobre tropopausa
    h_factor: float, escalado de altura (default 1.0)
    nb: float, ancho de haz virtual en grados (default 1.5° para radares meteorológicos)
    bsp: float, espaciado entre haces (default 1.0)
    min_radius: float, radio mínimo en metros (default 800.0)
    weight_func: 'Barnes', 'Barnes2', 'Cressman', 'nearest'
    max_neighbors: int o None
        - None: usa TODOS los gates dentro del ROI (procesamiento paralelo por nivel Z)
        - int: NO IMPLEMENTADO para parallel (usa modo secuencial original)
    n_workers: int o None
        - None: cpu_count() - 1
        - 1: procesamiento secuencial
        - >1: procesamiento paralelo por niveles Z
    temp_dir: str o None (path a directorio temporal para archivos intermedios)
        Si None, se crea uno automáticamente
    dtype_val: dtype para los pesos
    dtype_idx: dtype para índices

    Returns:
        W: scipy.sparse.csr_matrix shape (Nvoxels, Ngates_total)
        valid_indices: np.ndarray mapeo de índices filtrados a originales

    Notes:
    Memory usage during computation is managed by:
    - Processing one z-level at a time
    - Writing intermediate results to temp files
    - Using multiprocessing to parallelize across levels
    """
    t_start = time.time()
    
    # Configurar workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # Configurar directorio temporal
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='w_operator_')
        cleanup_temp = True
    else:
        cleanup_temp = False
        if not os.path.isdir(temp_dir):
            raise ValueError(f"temp_dir no existe: {temp_dir}")
    
    gates_xyz = np.asarray(gates_xyz, dtype=np.float32)
    voxels_xyz = np.asarray(voxels_xyz, dtype=np.float32)

    ngates_total = gates_xyz.shape[0]
    nvoxels = voxels_xyz.shape[0]
    
    # Crear máscara TOA (filtrado se hace dentro del worker)
    gate_z = gates_xyz[:, 2]
    toa_mask = gate_z <= toa
    n_valid_gates = toa_mask.sum()
    n_excluded = ngates_total - n_valid_gates
    logger.info(f"Filtro TOA ({toa/1000:.1f}km): {n_valid_gates:,}/{ngates_total:,} gates válidos ({n_excluded:,} excluidos)")
    
    # Inferir grid_shape desde voxels_xyz
    # Necesitamos conocer (nz, ny, nx) para procesamiento por niveles
    # Asumimos que voxels están ordenados en Z-Y-X
    z_coords = np.unique(voxels_xyz[:, 2])
    y_coords = np.unique(voxels_xyz[:, 1])
    x_coords = np.unique(voxels_xyz[:, 0])
    
    nz = len(z_coords)
    ny = len(y_coords)
    nx = len(x_coords)
    
    if nz * ny * nx != nvoxels:
        # Si no es una grilla regular, caer a modo secuencial
        print(f"Grilla no regular detectada, usando modo secuencial...")
        n_workers = 1
    
    logger.info(f"Construyendo operador W: {nvoxels:,} voxels, {n_valid_gates:,} gates válidos")
    logger.info(f"  ROI dist_beam: h_factor={h_factor}, nb={nb}°, bsp={bsp}, min_radius={min_radius}m")
    logger.info(f"  Workers: {n_workers} (procesamiento {'paralelo' if n_workers > 1 else 'secuencial'})")


    # MODO PARALELO POR NIVELES Z
    if max_neighbors is not None:
        raise ValueError("max_neighbors no está implementado. Use None para procesamiento completo.")
    
    if nz * ny * nx != nvoxels:
        raise ValueError(f"Grilla no regular detectada: {nz}*{ny}*{nx}={nz*ny*nx} != {nvoxels}")
    
    logger.info(f"\nProcesando {nz} niveles Z con {n_workers} worker(s)...")
        
    # Preparar grid 2D (Y-X plane)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    grid_y_2d = yy.ravel().astype('float64')
    grid_x_2d = xx.ravel().astype('float64')
    
    # Extraer coordenadas individuales de gates (arrays 1D)
    gate_x = gates_xyz[:, 0]
    gate_y = gates_xyz[:, 1]
    gate_z = gates_xyz[:, 2]
    
    # Preparar argumentos para cada nivel Z
    # Pasar arrays 1D, máscara y parámetros dist_beam
    args_list = [
        (iz, z_coord, grid_y_2d, grid_x_2d, gate_x, gate_y, gate_z,
            toa_mask, h_factor, nb, bsp, min_radius,
            weight_func, temp_dir, dtype_val, dtype_idx)
        for iz, z_coord in enumerate(z_coords)
    ]
    
    # Procesar niveles en paralelo
    if n_workers == 1:
        # Secuencial
        results = []
        for args in args_list:
            result = _process_single_level(args)
            results.append(result)
    else:
        # Paralelo
        with Pool(n_workers) as pool:
            results = []
            for result in pool.imap_unordered(_process_single_level, args_list):
                results.append(result)
    
    # Ordenar resultados por nivel
    results.sort(key=lambda x: x[0])
    
    # Calcular tamaño total para PRE-ALLOCATION
    total_pairs = sum(r[1] for r in results)
    total_grid_points = ny * nx * nz
    
    logger.info(f"\nMerging {nz} niveles ({total_pairs:,} total pares)...")
    
    # PRE-ALLOCATION: arrays con tamaño exacto conocido
    final_indptr = np.zeros(total_grid_points + 1, dtype=dtype_idx)
    final_indices = np.empty(total_pairs, dtype=dtype_idx)
    final_weights = np.empty(total_pairs, dtype=dtype_val)
    
    # Llenar arrays pre-alocados nivel por nivel
    pair_offset = 0
    point_offset = 0
    
    for iz, n_pairs, temp_file in results:
        # Cargar datos del nivel
        data = np.load(temp_file)
        level_indptr = data['indptr']
        level_indices = data['gate_indices']
        level_weights = data['weights']
        
        n_level_points = len(level_indptr) - 1
        n_level_pairs = len(level_indices)
        
        # Copiar índices y pesos directamente a arrays pre-alocados
        if n_level_pairs > 0:
            final_indices[pair_offset:pair_offset + n_level_pairs] = level_indices
            final_weights[pair_offset:pair_offset + n_level_pairs] = level_weights
        
        # Construir indptr para este nivel con offset adecuado
        for j in range(n_level_points):
            final_indptr[point_offset + j + 1] = pair_offset + level_indptr[j + 1]
        
        # Actualizar offsets
        pair_offset += n_level_pairs
        point_offset += n_level_points
        
        # GESTIÓN DE MEMORIA: cerrar archivo y limpiar referencias
        data.close()
        del data, level_indptr, level_indices, level_weights
        
        # Eliminar archivo temporal inmediatamente
        os.remove(temp_file)
        
        # Forzar garbage collection después de cada nivel
        gc.collect()
    
    logger.info("Merge completo.")
    
    # Limpiar directorio temporal si fue creado automáticamente
    if cleanup_temp:
        try:
            os.rmdir(temp_dir)
        except:
            pass
    
    # Construir CSR directamente desde arrays pre-alocados
    W = csr_matrix((final_weights, final_indices, final_indptr),
                    shape=(nvoxels, ngates_total), dtype=dtype_val)
    
    # Limpieza final
    del final_weights, final_indices, final_indptr
    gc.collect()
    
    voxels_with_data = nvoxels  # Aproximación (no calculamos exacto en modo paralelo)
    total_neighbors = total_pairs

    t_elapsed = time.time() - t_start
    avg_neighbors = total_neighbors / voxels_with_data if voxels_with_data > 0 else 0.0
    size_mb = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1024**2
    logger.info(
        f"Operador W construido:\n"
        f"  - {W.nnz:,} elementos no-cero\n"
        f"  - {voxels_with_data:,}/{nvoxels:,} voxels con datos ({100*voxels_with_data/nvoxels:.1f}%)\n"
        f"  - Vecinos promedio: {avg_neighbors:.1f}\n"
        f"  - Memoria: {size_mb:.2f} MB\n"
        f"  - Tiempo: {t_elapsed:.2f}s"
    )

    return W