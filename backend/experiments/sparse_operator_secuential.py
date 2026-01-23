"""
Script para validar construcción de operador W disperso de manera secuencial.

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
import numpy as np
import pyart
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

# Agregar path al módulo app para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.radar_common import resolve_field, safe_range_max_m
from backend.app.services.radar_processing import calculate_grid_resolution


def get_gate_xyz_coords(radar):
    """Calcular coordenadas cartesianas (x,y,z) de los gates del radar.
    
    Args:
        radar: Objeto pyart.core.Radar
    
    Returns:
        np.ndarray: (Ngates_total_all_sweeps, 3) coordenadas [x, y, z] en metros
    """
    # Procesar TODOS los sweeps (como PyART hace en grid_from_radars)
    all_gate_x = []
    all_gate_y = []
    all_gate_z = []
    
    for sweep_num in range(radar.nsweeps):
        gate_x, gate_y, gate_z = radar.get_gate_x_y_z(sweep_num)
        all_gate_x.append(gate_x.ravel())
        all_gate_y.append(gate_y.ravel())
        all_gate_z.append(gate_z.ravel())
    
    # Concatenar todos los sweeps
    gate_x_all = np.concatenate(all_gate_x)
    gate_y_all = np.concatenate(all_gate_y)
    gate_z_all = np.concatenate(all_gate_z)
    
    # Stack a (Ngates_total, 3)
    xyz = np.stack([gate_x_all, gate_y_all, gate_z_all], axis=1)
    
    return xyz


def get_grid_xyz_coords(grid_shape, grid_limits, dtype=np.float32):
    """
    Genera coordenadas cartesianas de voxels usando la MISMA convención que PyART.
    
    PyART en map_to_grid calcula:
        x = x_start + x_step * ix  (donde ix = 0, 1, ..., nx-1)
    
    Args:
        grid_shape: tuple (nz, ny, nx)
        grid_limits: tuple ((z_min, z_max), (y_min, y_max), (x_min, x_max))
    
    Returns:
        np.ndarray: (Nvoxels, 3) coordenadas [x, y, z] en metros
    """
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = grid_limits
    nz, ny, nx = grid_shape

    def axis_coords(vmin, vmax, n):
        if n == 1:
            return np.array([(vmin + vmax) / 2.0], dtype=dtype)
        return np.linspace(vmin, vmax, n, dtype=dtype)

    z = axis_coords(zmin, zmax, nz)
    y = axis_coords(ymin, ymax, ny)
    x = axis_coords(xmin, xmax, nx)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(dtype, copy=False)

def get_grid_xyz_coords_optim(grid_shape, grid_limits, dtype=np.float32):
    """
    Genera coordenadas (x,y,z) de TODOS los voxels (centros) de una grilla 3D regular.

    Convención: para cada eje, los centros están espaciados uniformemente y
    los extremos incluyen los límites:
        coord[k] = min + (max-min)/(n-1) * k     si n>1
    (equivalente a np.linspace(min, max, n))

    Args:
        grid_shape: (nz, ny, nx)
        grid_limits: ((zmin, zmax), (ymin, ymax), (xmin, xmax)) en metros
        dtype: dtype de salida (float32 recomendado)

    Returns:
        xyz: array (Nvoxels, 3) con filas [x, y, z], orden compatible con
             reshape(grid_shape) usando indexing='ij' (z-major).
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

    # X cambia más rápido
    X = np.tile(x, nz * ny)
    # Y cambia cada bloque de nx
    Y = np.tile(np.repeat(y, nx), nz)
    # Z cambia cada bloque de (ny*nx)
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
    if method == 'Barnes':
        # Barnes: w = exp(-d²/(2σ²)) con σ = ROI/2
        sigma = roi / 2.0
        return np.exp(-(distances**2) / (2 * sigma**2))
    
    elif method == 'Cressman':
        # Cressman: w = (ROI² - d²) / (ROI² + d²)
        roi_sq = roi**2
        d_sq = distances**2
        return (roi_sq - d_sq) / (roi_sq + d_sq)
    
    elif method == 'nearest':
        # Nearest: solo el más cercano tiene peso 1
        weights = np.zeros_like(distances)
        if len(distances) > 0:
            weights[np.argmin(distances)] = 1.0
        return weights
    
    else:
        raise ValueError(f"Método desconocido: {method}")


def build_W_operator(gates_xyz, voxels_xyz, constant_roi, weight_func='Barnes', 
                     max_neighbors=None, verbose=True):
    """
    Construye operador disperso W que mapea gates → voxels.
    
    W[i, j] = peso del gate j para el voxel i
    
    Args:
        gates_xyz: (Ngates, 3) coordenadas cartesianas de gates
        voxels_xyz: (Nvoxels, 3) coordenadas cartesianas de voxels
        constant_roi: Radio de influencia en metros
        weight_func: 'Barnes', 'Cressman', o 'nearest'
        max_neighbors: Límite de vecinos por voxel (None = sin límite)
        verbose: Mostrar progreso
    
    Returns:
        scipy.sparse.csr_matrix: Operador W de shape (Nvoxels, Ngates)
    """
    ngates = gates_xyz.shape[0]
    nvoxels = voxels_xyz.shape[0]
    
    if verbose:
        print(f"[*] Construyendo operador W:")
        print(f"   Gates: {ngates:,}")
        print(f"   Voxels: {nvoxels:,}")
        print(f"   ROI: {constant_roi:.0f} m")
        print(f"   Método: {weight_func}")
    
    # Construir KDTree de gates (una sola vez)
    t0 = time.time()
    tree = cKDTree(gates_xyz)
    t_kdtree = time.time() - t0
    
    if verbose:
        print(f"   KDTree construido en {t_kdtree:.2f}s")
    
    # Listas para construcción COO (después convertimos a CSR)
    row_list = []
    col_list = []
    val_list = []
    
    # Para cada voxel, buscar gates vecinos
    t0 = time.time()
    voxels_with_data = 0
    total_neighbors = 0
    
    for i, voxel_xyz in enumerate(voxels_xyz):
        # Query ball: todos los gates dentro del ROI
        indices = tree.query_ball_point(voxel_xyz, r=constant_roi)
        
        if len(indices) == 0:
            continue  # Voxel sin datos
        
        # Limitar número de vecinos si se especifica
        if max_neighbors and len(indices) > max_neighbors:
            # Tomar los más cercanos
            dists = np.linalg.norm(gates_xyz[indices] - voxel_xyz, axis=1)
            closest = np.argsort(dists)[:max_neighbors]
            indices = [indices[k] for k in closest]
        
        # Calcular distancias
        gates_subset = gates_xyz[indices]
        distances = np.linalg.norm(gates_subset - voxel_xyz, axis=1)
        
        # Calcular pesos (sin normalizar)
        weights = compute_weights(distances, constant_roi, weight_func)
        
        # Agregar a listas COO
        row_list.extend([i] * len(indices))
        col_list.extend(indices)
        val_list.extend(weights)
        
        voxels_with_data += 1
        total_neighbors += len(indices)
        
        # Progress bar
        if verbose and (i + 1) % 10000 == 0:
            pct = 100 * (i + 1) / nvoxels
            print(f"   Progreso: {pct:.1f}% ({i+1:,}/{nvoxels:,} voxels)")
    
    t_query = time.time() - t0
    
    # Construir matriz CSR
    W = csr_matrix((val_list, (row_list, col_list)), shape=(nvoxels, ngates))
    
    # Estadísticas
    nnz = W.nnz
    sparsity = (1 - nnz / (nvoxels * ngates)) * 100
    avg_neighbors = total_neighbors / voxels_with_data if voxels_with_data > 0 else 0
    
    if verbose:
        print(f"   Búsqueda completada en {t_query:.2f}s")
        print(f"\n[*] Estadísticas del operador W:")
        print(f"   Shape: {W.shape}")
        print(f"   Entradas no-cero: {nnz:,}")
        print(f"   Sparsity: {sparsity:.2f}%")
        print(f"   Voxels con datos: {voxels_with_data:,} ({100*voxels_with_data/nvoxels:.1f}%)")
        print(f"   Vecinos promedio: {avg_neighbors:.1f}")
        print(f"   Tamaño estimado: {(W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1024**2:.1f} MB")
    
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
    g = field_data.ravel()  # (Ngates_total,)
    
    if handle_mask:
        # Crear vector de máscara (1 = válido, 0 = enmascarado)
        if np.ma.is_masked(field_data):
            mask_valid = (~field_data.mask).astype(float).ravel()
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


def grid_with_pyart(radar, field_name, grid_shape, grid_limits, constant_roi, weighting_function='nearest'):
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
        gridding_algo="map_gates_to_grid",
        grid_origin=grid_origin,
        fields=[field_name],
        weighting_function=weighting_function,
        gatefilters=None,
        roi_func="constant",
        constant_roi=constant_roi,
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
    """Script principal de validación."""
    
    print("=" * 70)
    print("Validación de Operador Disperso para Gridding de Radar")
    print("=" * 70)
    
    # ==================== CONFIGURACIÓN ====================
    
    # Archivo NetCDF de prueba
    nc_path = Path(__file__).parent.parent.parent / "biribiri" / "RMA1_0315_01_20250819T001715Z.nc"
    
    if not nc_path.exists():
        print(f"Error: Archivo no encontrado: {nc_path}")
        return
    
    print(f"\n Archivo: {nc_path.name}")
    
    # Parámetros de grilla (simplificados para prueba rápida)
    volume = "03"  # Volumen con resolución media (300m)
    grid_res_xy, grid_res_z = calculate_grid_resolution(volume)
    
    print(f"\n Configuración:")
    print(f"   Volumen: {volume}")
    print(f"   Resolución XY: {grid_res_xy} m")
    print(f"   Resolución Z: {grid_res_z} m")
    
    # ==================== LEER RADAR ====================
    
    print(f"\n Leyendo radar...")
    t0 = time.time()
    radar = pyart.io.read(str(nc_path))
    t_read = time.time() - t0
    
    print(f"   Radar leído en {t_read:.2f}s")
    print(f"   Nrays: {radar.nrays}")
    print(f"   Ngates: {radar.ngates}")
    print(f"   Campos: {list(radar.fields.keys())}")
    
    # Resolver campo DBZH
    field_name, field_key = resolve_field(radar, "DBZH")
    print(f"   Campo a usar: {field_name}")
    
    # ==================== GEOMETRÍA ====================
    
    # Determinar límites de grilla
    range_max_m = safe_range_max_m(radar)
    
    # Grilla reducida para prueba rápida (ajustar según necesidad)
    # z: 0 - 10km, xy: ±100km (ajustable)
    z_min, z_max = 0, 10000
    grid_limits = (
        (z_min, z_max),
        (-range_max_m, range_max_m),
        (-range_max_m, range_max_m)
    )
    
    # Calcular shape de grilla
    nz = int((z_max - z_min) / grid_res_z)
    ny = int((2 * range_max_m) / grid_res_xy)
    nx = int((2 * range_max_m) / grid_res_xy)
    grid_shape = (nz, ny, nx)
    
    # ROI (Radio de Influencia) - Aumentado para asegurar cobertura
    constant_roi = max(
        grid_res_xy * 3.0,  # Aumentado de 1.5 a 3.0
        800 + (range_max_m / 100000) * 400
    )
    
    print(f"\n Geometría de grilla:")
    print(f"   Shape: {grid_shape} (Z, Y, X)")
    print(f"   Límites X: [{grid_limits[2][0]:.0f}, {grid_limits[2][1]:.0f}] m")
    print(f"   Límites Y: [{grid_limits[1][0]:.0f}, {grid_limits[1][1]:.0f}] m")
    print(f"   Límites Z: [{grid_limits[0][0]:.0f}, {grid_limits[0][1]:.0f}] m")
    print(f"   Voxels totales: {nz * ny * nx:,}")
    print(f"   Gates totales: {radar.nrays * radar.ngates:,}")
    print(f"   ROI: {constant_roi:.0f} m")
    
    # ==================== CONSTRUIR OPERADOR W ====================
    
    print(f"\n" + "=" * 70)
    print("FASE 1: Construcción de Operador W")
    print("=" * 70)
    
    # Coordenadas cartesianas de gates
    print(f"\n  Calculando coordenadas de gates...")
    t0 = time.time()
    gates_xyz = get_gate_xyz_coords(radar)
    t_gates = time.time() - t0
    print(f"   Completado en {t_gates:.2f}s")
    print(f"   Shape: {gates_xyz.shape}")
    
    # Coordenadas cartesianas de voxels
    print(f"\n  Calculando coordenadas de voxels...")
    t0 = time.time()
    voxels_xyz = get_grid_xyz_coords(grid_shape, grid_limits)
    t_voxels = time.time() - t0
    print(f"   Completado en {t_voxels:.2f}s")
    print(f"   Shape: {voxels_xyz.shape}")
    
    # Construir operador W
    print(f"\n")
    t0 = time.time()
    W = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz,
        constant_roi=constant_roi,
        weight_func='nearest',
        verbose=True
    )
    t_build_w = time.time() - t0
    
    print(f"\n   Tiempo total construcción W: {t_build_w:.2f}s")
    
    # ==================== APLICAR OPERADOR ====================
    
    print(f"\n" + "=" * 70)
    print("FASE 2: Aplicación de Operador W")
    print("=" * 70)
    
    field_data = radar.fields[field_name]['data']
    print(f"\n[*] Datos del campo '{field_name}':")
    print(f"   Shape: {field_data.shape}")
    print(f"   Tipo: {type(field_data)}")
    print(f"   Enmascarado: {np.ma.is_masked(field_data)}")
    if np.ma.is_masked(field_data):
        print(f"   Gates válidos: {(~field_data.mask).sum():,} / {field_data.size:,} "
              f"({100 * (~field_data.mask).sum() / field_data.size:.1f}%)")
    
    print(f"\n  Aplicando operador W...")
    t0 = time.time()
    grid3d_sparse = apply_operator(W, field_data, grid_shape, handle_mask=True)
    t_apply = time.time() - t0
    
    print(f"   Completado en {t_apply:.2f}s")
    print(f"   Shape resultado: {grid3d_sparse.shape}")
    print(f"   Voxels válidos: {(~grid3d_sparse.mask).sum():,} / {grid3d_sparse.size:,} "
          f"({100 * (~grid3d_sparse.mask).sum() / grid3d_sparse.size:.1f}%)")
    
    # ==================== COMPARAR CON PYART ====================
    
    print(f"\n" + "=" * 70)
    print("FASE 3: Validación vs PyART")
    print("=" * 70)
    
    print(f"\n  Ejecutando PyART tradicional...")
    grid3d_pyart, t_pyart = grid_with_pyart(
        radar=radar,
        field_name=field_name,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        constant_roi=constant_roi,
        weighting_function='nearest',
    )
    
    print(f"   Completado en {t_pyart:.2f}s")
    print(f"   Shape resultado: {grid3d_pyart.shape}")
    print(f"   Voxels válidos: {(~grid3d_pyart.mask).sum():,} / {grid3d_pyart.size:,} "
          f"({100 * (~grid3d_pyart.mask).sum() / grid3d_pyart.size:.1f}%)")
    
    # ==================== MÉTRICAS DE COMPARACIÓN ====================
    
    print(f"\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    
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
        
        print(f"\n  Métricas de comparación ({n_valid:,} voxels válidos):")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE:  {mae:.6f}")
        print(f"   Max diff: {max_diff:.6f}")
        print(f"   Mean (Sparse): {sparse_valid.mean():.2f}")
        print(f"   Mean (PyART):  {pyart_valid.mean():.2f}")
        
        # Criterio de éxito
        tolerance = 0.01  # 1% de diferencia relativa
        relative_error = rmse / (np.abs(pyart_valid.mean()) + 1e-10)
        
        if relative_error < tolerance:
            print(f"\ VALIDACIÓN EXITOSA (error relativo: {relative_error*100:.4f}%)")
        else:
            print(f"\n  Error relativo alto: {relative_error*100:.4f}%")
    else:
        print(f"\n No hay voxels válidos en común para comparar")
    
    print(f"\n   Tiempos de ejecución:")
    print(f"   Construcción W: {t_build_w:.2f}s")
    print(f"   Aplicación W:   {t_apply:.2f}s")
    print(f"   PyART:          {t_pyart:.2f}s")
    print(f"   Speedup (solo aplicación): {t_pyart / t_apply:.2f}x")
    print(f"   Speedup (total primera vez): {t_pyart / (t_build_w + t_apply):.2f}x")
    
    print(f"\n  Tamaño del operador W:")
    w_size_mb = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1024**2
    print(f"   {w_size_mb:.2f} MB")
    
    print(f"\n  Nota: La construcción de W se hace UNA SOLA VEZ y se reutiliza.")
    print(f"   En ejecuciones subsecuentes con W cacheado, el speedup sería:")
    print(f"   {t_pyart / t_apply:.2f}x ")
    
    print(f"\n" + "=" * 70)
    print("PASO 1.1 COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
