import sys
import time
import numpy as np
import pyart
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.radar_common import resolve_field, safe_range_max_m
from app.services.grid_geometry import calculate_grid_resolution, calculate_z_limits, calculate_grid_points
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
    verbose=True,
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
    t_total0 = time.time()

    gates_xyz = np.asarray(gates_xyz, dtype=np.float32)
    voxels_xyz = np.asarray(voxels_xyz, dtype=np.float32)

    ngates = gates_xyz.shape[0]
    nvoxels = voxels_xyz.shape[0]
    roi = float(constant_roi)

    if verbose:
        print(f"[*] Construyendo operador W (CSR)")
        print(f"   Gates:  {ngates:,}")
        print(f"   Voxels: {nvoxels:,}")
        print(f"   ROI:    {roi:.1f} m")
        print(f"   Método: {weight_func}")
        print(f"   max_neighbors: {max_neighbors}")
        print(f"   chunk_size:    {chunk_size:,}")

    # Construir KDTree de gates (una sola vez)
    t0 = time.time()
    tree = cKDTree(gates_xyz)
    if verbose:
        print(f"   KDTree construido en {time.time()-t0:.2f}s")

    # Vamos a acumular COO (Coordinate format) por chunks en arrays (mucho mejor que listas gigantes)
    rows_chunks = []
    cols_chunks = []
    vals_chunks = []

    voxels_with_data = 0
    total_neighbors = 0

    t_query0 = time.time()
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

        if verbose:
            done = end
            if done % (chunk_size * 5) == 0 or done == nvoxels:
                pct = 100.0 * done / nvoxels
                print(f"   Progreso: {pct:.1f}% ({done:,}/{nvoxels:,} voxels)")

    t_query = time.time() - t_query0

    if len(rows_chunks) == 0:
        # No hubo vecinos para ningún voxel
        if verbose:
            print("No se encontraron vecinos para ningún voxel. W será todo ceros.")
        return csr_matrix((nvoxels, ngates), dtype=dtype_val)

    # Concatenar chunks
    t0 = time.time()
    rows_all = np.concatenate(rows_chunks)
    cols_all = np.concatenate(cols_chunks)
    vals_all = np.concatenate(vals_chunks)
    if verbose:
        print(f"   Concatenación COO en {time.time()-t0:.2f}s")
        print(f"   nnz total: {vals_all.size:,}")

    # Construir CSR
    t0 = time.time()
    W = csr_matrix((vals_all, (rows_all, cols_all)), shape=(nvoxels, ngates), dtype=dtype_val)
    # Limpieza útil
    W.sum_duplicates()
    W.sort_indices()
    t_csr = time.time() - t0

    # Stats
    nnz = W.nnz
    sparsity = (1.0 - nnz / (nvoxels * ngates)) * 100.0
    avg_neighbors = total_neighbors / voxels_with_data if voxels_with_data > 0 else 0.0
    size_mb = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1024**2

    if verbose:
        print(f"\n[*] Búsqueda vecinos (total) en {t_query:.2f}s")
        print(f"[*] CSR armado en {t_csr:.2f}s")
        print(f"\n[*] Estadísticas W:")
        print(f"   Shape: {W.shape}")
        print(f"   nnz: {nnz:,}")
        print(f"   Sparsity: {sparsity:.4f}%")
        print(f"   Voxels con datos: {voxels_with_data:,} ({100*voxels_with_data/nvoxels:.2f}%)")
        print(f"   Vecinos promedio (solo voxels con datos): {avg_neighbors:.2f}")
        print(f"   Tamaño (data+indices+indptr): {size_mb:.2f} MB")
        print(f"   Tiempo total build_W: {time.time()-t_total0:.2f}s")

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
        gridding_algo="map_to_grid",
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
    print(f"Validación de Operador Disperso W para Gridding de Radar\n")

    # Archivo NetCDF de prueba
    nc_path = Path(__file__).parent.parent.parent / "biribiri" / "RMA1_0315_01_20250819T001715Z.nc"
    radar = pyart.io.read(str(nc_path))

    # Parámetros de grilla
    elevation = 0
    cappi_height = None
    interp = "Barnes2"
    field_name, _ = resolve_field(radar, "DBZH")
    radar,strategy,volume,_ = extract_metadata_from_filename(str(nc_path))

    # Calcular shape de grilla y límites
    grid_res_xy, grid_res_z = calculate_grid_resolution(volume)
    range_max_m = safe_range_max_m(radar)
    z_min, z_max, elev_deg = calculate_z_limits(
        range_max_m, elevation, cappi_height, radar.fixed_angle['data']
    )
    z_grid_limits = (z_min, z_max)
    y_grid_limits = (-range_max_m, range_max_m)
    x_grid_limits = (-range_max_m, range_max_m)
    z_points, y_points, x_points = calculate_grid_points(
        z_grid_limits, y_grid_limits, x_grid_limits,
        grid_res_z, grid_res_xy
    )

    grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits)
    grid_shape = (z_points, y_points, x_points)
    constant_roi = max(
        grid_res_xy * 1.5,
        800 + (range_max_m / 100000) * 400
    )

    # ==================== CONSTRUIR OPERADOR W ====================
    # Coordenadas cartesianas de gates
    print(f"\n  Calculando coordenadas de gates...")
    gates_xyz = get_gate_xyz_coords(radar)
    print(f"   Shape: {gates_xyz.shape}")

    # Coordenadas cartesianas de voxels
    print(f"  Calculando coordenadas de voxels...")
    voxels_xyz = get_grid_xyz_coords(grid_shape, grid_limits)
    print(f"   Shape: {voxels_xyz.shape}")

    # Construir operador W
    print(f"\n")
    print("Creacion de Operador W")
    t0 = time.time()
    W = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz,
        constant_roi=constant_roi,
        weight_func=interp,
        verbose=False
    )
    t_build_w = time.time() - t0
    
    print(f"\n   Tiempo total construcción W: {t_build_w:.2f}s")


    # ==================== APLICAR OPERADOR ====================
    
    print(f"\n")
    print("Aplicación de Operador W")
    
    field_data = radar.fields[field_name]['data']

    t0 = time.time()
    grid3d_sparse = apply_operator(W, field_data, grid_shape, handle_mask=True)
    t_apply = time.time() - t0
    
    print(f"   Completado en {t_apply:.2f}s")
    print(f"   Shape resultado: {grid3d_sparse.shape}")
    print(f"   Voxels válidos: {(~grid3d_sparse.mask).sum():,} / {grid3d_sparse.size:,} "
          f"({100 * (~grid3d_sparse.mask).sum() / grid3d_sparse.size:.1f}%)")
    

    # ==================== COMPARAR CON PYART ====================
    
    print(f"\n")
    print("Validación vs PyART")
    
    grid3d_pyart, t_pyart = grid_with_pyart(
        radar=radar,
        field_name=field_name,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        constant_roi=constant_roi,
        weighting_function=interp,
    )
    
    print(f"   Completado en {t_pyart:.2f}s")
    
    # ==================== MÉTRICAS DE COMPARACIÓN ====================
    
    print(f"\n")
    print("RESULTADOS FINALES")
    
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
        print(f"Field: {field_name}, Volume: {volume}, Radar: {radar}, Estrategia: {strategy}")
        print(f"   RMSE (Root Mean Square Error): {rmse:.6f}")
        print(f"   MAE (Mean Absolute Error):  {mae:.6f}")
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



if __name__ == "__main__":
    main()