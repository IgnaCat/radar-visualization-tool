"""
Script para validar construcción de operador W disperso de manera paralelas.

Probando construcción de operador W con procesamiento paralelo por niveles Z
Comparación con PyART para validación
Diferencia con el secuencial, ademas de cambiar build_W_operator, usamos roi=dist_beam

Este script:
1. Lee un archivo NetCDF de radar
2. Construye geometría de gates (coordenadas cartesianas)
3. Define geometría de grilla cartesiana 3D
4. Construye operador W usando KDTree + función de pesos
5. Aplica W a un campo (DBZH)
6. Compara con PyART tradicional (grid_from_radars)
7. Reporta métricas de validación (RMSE, tiempo, sparsity)

Uso:
    python backend/experiments/test_sparse_operator.py
"""

import sys
import time
import gc
import os
import tempfile
import numpy as np
import pyart
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.radar_common import resolve_field, safe_range_max_m
from backend.app.services.grid_geometry import calculate_roi_dist_beam, calculate_grid_resolution, calculate_z_limits, calculate_grid_points
from app.utils.helpers import extract_metadata_from_filename

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
    
    OPERADOR W UNIVERSAL con PROCESAMIENTO PARALELO:
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
    print(f"Filtro TOA ({toa/1000:.1f}km): {n_valid_gates:,}/{ngates_total:,} gates válidos ({n_excluded:,} excluidos)")
    
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
    
    print(f"Construyendo operador W UNIVERSAL: {nvoxels:,} voxels, {n_valid_gates:,} gates válidos")
    print(f"  ROI dist_beam: h_factor={h_factor}, nb={nb}°, bsp={bsp}, min_radius={min_radius}m")
    print(f"  Workers: {n_workers} (procesamiento {'paralelo' if n_workers > 1 else 'secuencial'})")

    # ============================================================================
    # MODO PARALELO POR NIVELES Z
    # ============================================================================
    if max_neighbors is not None:
        raise ValueError("max_neighbors no está implementado. Use None para procesamiento completo.")
    
    if nz * ny * nx != nvoxels:
        raise ValueError(f"Grilla no regular detectada: {nz}*{ny}*{nx}={nz*ny*nx} != {nvoxels}")
    
    print(f"\nProcesando {nz} niveles Z con {n_workers} worker(s)...")
        
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
            print(f"  Nivel {result[0]}: {result[1]:,} pares")
            results.append(result)
    else:
        # Paralelo
        with Pool(n_workers) as pool:
            results = []
            for result in pool.imap_unordered(_process_single_level, args_list):
                print(f"  Nivel {result[0]}: {result[1]:,} pares")
                results.append(result)
    
    # Ordenar resultados por nivel
    results.sort(key=lambda x: x[0])
    
    # Calcular tamaño total para PRE-ALLOCATION
    total_pairs = sum(r[1] for r in results)
    total_grid_points = ny * nx * nz
    
    print(f"\nMerging {nz} niveles ({total_pairs:,} total pares)...")
    
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
    
    print("Merge completo.")
    
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
    print(
        f"Operador W construido:\n"
        f"  - {W.nnz:,} elementos no-cero\n"
        f"  - {voxels_with_data:,}/{nvoxels:,} voxels con datos ({100*voxels_with_data/nvoxels:.1f}%)\n"
        f"  - Vecinos promedio: {avg_neighbors:.1f}\n"
        f"  - Memoria: {size_mb:.2f} MB\n"
        f"  - Tiempo: {t_elapsed:.2f}s"
    )

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


def slice_W_operator(W_full, grid_shape_full, z_levels_needed, grid_shape_sliced):
    """
    Extrae un sub-operador W con solo los niveles Z necesarios.
    
    Permite reutilizar un operador W universal extrayendo solo los niveles
    verticales relevantes para un sweep específico.
    
    Args:
        W_full: csr_matrix completo (nvoxels_full, ngates)
        grid_shape_full: tuple (nz_full, ny, nx) de la grilla completa
        z_levels_needed: int, número de niveles Z a extraer desde el fondo
        grid_shape_sliced: tuple (nz_sliced, ny, nx) de la grilla resultante
    
    Returns:
        W_sliced: csr_matrix (nvoxels_sliced, ngates)
        
        El slicing es instantáneo (solo extrae filas del CSR)
    """
    nz_full, ny, nx = grid_shape_full
    nz_sliced, ny_check, nx_check = grid_shape_sliced
    
    if ny != ny_check or nx != nx_check:
        raise ValueError(f"Grid XY dimensions must match: ({ny},{nx}) vs ({ny_check},{nx_check})")
    
    if z_levels_needed > nz_full:
        raise ValueError(f"Cannot extract {z_levels_needed} levels from grid with {nz_full} levels")
    
    if z_levels_needed != nz_sliced:
        raise ValueError(f"z_levels_needed ({z_levels_needed}) must match nz in grid_shape_sliced ({nz_sliced})")
    
    # Número de voxels por nivel Z
    voxels_per_level = ny * nx
    
    # Índices de filas a extraer (primeros z_levels_needed niveles)
    # En orden Z-Y-X: nivel 0 = filas [0, voxels_per_level)
    #                 nivel 1 = filas [voxels_per_level, 2*voxels_per_level)
    #                 ...
    row_end = z_levels_needed * voxels_per_level
    
    # Slicing de CSR es eficiente (solo copia punteros, no recalcula)
    W_sliced = W_full[:row_end, :]
    
    print(f"\nSlicing operador W:")
    print(f"  - De {W_full.shape[0]:,} voxels ({nz_full} niveles Z) → {W_sliced.shape[0]:,} voxels ({z_levels_needed} niveles Z)")
    print(f"  - Conserva {W_sliced.nnz:,} elementos no-cero")
    print(f"  - Memoria: {(W_sliced.data.nbytes + W_sliced.indices.nbytes + W_sliced.indptr.nbytes)/1024**2:.2f} MB")
    
    return W_sliced


def grid_with_pyart(radar, field_name, grid_shape, grid_limits, h_factor, nb, bsp, min_radius, weighting_function='nearest'):
    """
    Gridding tradicional usando PyART (referencia).
    
    Args:
        radar: pyart.core.Radar
        field_name: Nombre del campo a gridear
        grid_shape: tuple (nz, ny, nx)
        grid_limits: tuple ((z_min, z_max), (y_min, y_max), (x_min, x_max))
        constant_roi: Radio de influencia
    
    Returns:
        np.ma.MaskedArray: Grilla 3D de shape (1, nz, ny, nx) (PyART agrega dim tiempo)
    """
    grid_origin = (
        float(radar.latitude['data'][0]),
        float(radar.longitude['data'][0]),
    )
    
    t0 = time.time()
    
    grid = pyart.map.grid_from_radars(
        radar,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        gridding_algo="map_to_grid",
        grid_origin=grid_origin,
        fields=[field_name],
        weighting_function=weighting_function,
        gatefilters=None,
        roi_func="dist_beam",
        nb=nb,
        bsp=bsp,
        h_factor=h_factor,
        min_radius=min_radius,
    )
    
    t_pyart = time.time() - t0
    
    # PyART retorna shape variable según si es 3D o 4D
    field_data = grid.fields[field_name]['data']
    if field_data.ndim == 4:
        grid3d = field_data[0, :, :, :]  # (1, nz, ny, nx) -> (nz, ny, nx)
    else:
        grid3d = field_data  # Ya es (nz, ny, nx)
    
    return grid3d, t_pyart


def main():
    print(f"="*80)
    print(f"Validación: Operador W con Procesamiento Paralelo")
    print(f"="*80)

    # Archivo NetCDF de prueba
    nc_path = Path(__file__).parent.parent.parent / "biribiri" / "RMA1_0315_01_20250819T001715Z.nc"
    radar = pyart.io.read(str(nc_path))

    # Parámetros de grilla
    elevation = 0
    cappi_height = None
    interp = "Barnes2"
    field_name, _ = resolve_field(radar, "DBZH")
    radar_name, strategy, volume, _ = extract_metadata_from_filename(str(nc_path))

    # Parámetros comunes
    toa = 12000  # 12 km - límite físico para filtrado de ecos no-meteorológicos
    grid_res_xy, grid_res_z = calculate_grid_resolution(volume)
    range_max_m = safe_range_max_m(radar)
    
    # Calcular parámetros dist_beam
    h_factor = 1.0
    nb = 1.5
    bsp = 1.0
    min_radius = 300.0

    # ==================== CONFIGURACIÓN DE GRILLA ====================
    print(f"\n[1] CONFIGURACIÓN DE GRILLA")
    print(f"    TOA (filtrado): {toa/1000:.1f} km")
    print(f"    Elevación: {radar.fixed_angle['data'][elevation]:.2f}°\n")
    
    # Grilla universal: z_max = TOA (no depende del sweep)
    z_grid_limits = (0.0, toa)
    y_grid_limits = (-range_max_m, range_max_m)
    x_grid_limits = (-range_max_m, range_max_m)
    
    z_points, y_points, x_points = calculate_grid_points(
        z_grid_limits, y_grid_limits, x_grid_limits,
        grid_res_z, grid_res_xy
    )
    
    grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits)
    grid_shape = (z_points, y_points, x_points)
    
    print(f"    Grid shape: {grid_shape}")
    print(f"    Grid limits Z: {z_grid_limits[0]/1000:.1f} - {z_grid_limits[1]/1000:.1f} km")
    print(f"    Total voxels: {np.prod(grid_shape):,}")

    # ==================== CONSTRUIR OPERADOR W ====================
    print(f"\n[2] CONSTRUCCIÓN OPERADOR W")
    
    # Coordenadas cartesianas de gates
    gates_xyz = get_gate_xyz_coords(radar)
    print(f"    Gates totales: {gates_xyz.shape[0]:,}")

    # Coordenadas cartesianas de voxels
    voxels_xyz = get_grid_xyz_coords(grid_shape, grid_limits)
    print(f"    Voxels: {voxels_xyz.shape[0]:,}\n")

    # Construir operador W
    t0 = time.time()
    W = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz,
        toa=toa,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        weight_func=interp,
        max_neighbors=None,
        n_workers=3,  # Usa cpu_count() - 1 automáticamente
        temp_dir=None,   # Crea directorio temporal automáticamente
    )
    t_build_w = time.time() - t0
    
    print(f"\n    ✓ Operador W construido en {t_build_w:.2f}s")

    # ==================== APLICAR OPERADOR ====================
    print(f"\n[3] APLICACIÓN DEL OPERADOR W")
    
    field_data = radar.fields[field_name]['data']

    t0 = time.time()
    grid3d_sparse = apply_operator(W, field_data, grid_shape, handle_mask=True)
    t_apply = time.time() - t0
    
    print(f"    Completado en {t_apply:.2f}s")
    print(f"    Shape resultado: {grid3d_sparse.shape}")
    print(f"    Voxels válidos: {(~grid3d_sparse.mask).sum():,} / {grid3d_sparse.size:,} "
          f"({100 * (~grid3d_sparse.mask).sum() / grid3d_sparse.size:.1f}%)")

    # ==================== COMPARAR CON PYART ====================
    print(f"\n[4] VALIDACIÓN VS PYART (referencia)")
    
    grid3d_pyart, t_pyart = grid_with_pyart(
        radar=radar,
        field_name=field_name,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        weighting_function=interp,
    )
    
    print(f"    PyART completado en {t_pyart:.2f}s")
    
    # ==================== MÉTRICAS DE COMPARACIÓN ====================
    print(f"\n[5] MÉTRICAS DE COMPARACIÓN")
    print(f"="*80)
    
    # Calcular diferencias solo en voxels válidos de ambos
    mask_both_valid = (~grid3d_sparse.mask) & (~grid3d_pyart.mask)
    n_valid = mask_both_valid.sum()
    
    if n_valid > 0:
        sparse_valid = grid3d_sparse.data[mask_both_valid]
        pyart_valid = grid3d_pyart.data[mask_both_valid]
        
        diff = sparse_valid - pyart_valid
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        max_diff = np.max(np.abs(diff))
        relative_error = rmse / (np.abs(pyart_valid.mean()) + 1e-10)
        
        print(f"\nDataset: {radar_name}_{strategy}_{volume}")
        print(f"Field: {field_name}, Sweep: {elevation}")
        print(f"\nVoxels válidos: {n_valid:,} ({100*n_valid/grid3d_sparse.size:.1f}% del total)")
        print(f"\nComparación Operador W vs PyART:")
        print(f"  - RMSE: {rmse:.6f}")
        print(f"  - MAE:  {mae:.6f}")
        print(f"  - Max diff: {max_diff:.6f}")
        print(f"  - Mean (W Operator): {sparse_valid.mean():.2f}")
        print(f"  - Mean (PyART):      {pyart_valid.mean():.2f}")
        print(f"  - Error relativo: {relative_error*100:.4f}%")
        
        # Criterio de éxito
        tolerance = 0.01  # 1% de diferencia relativa
        if relative_error < tolerance:
            print(f"\n✓ VALIDACIÓN EXITOSA")
        else:
            print(f"\nError relativo sobre tolerancia (>{tolerance*100}%)")
    else:
        print(f"\nNo hay voxels válidos en común para comparar")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()