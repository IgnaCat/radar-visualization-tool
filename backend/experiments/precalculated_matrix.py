import sys
import time
import numpy as np
import pyart
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.radar_common import resolve_field, safe_range_max_m
from app.services.grid_geometry import calculate_roi_dist_beam, calculate_grid_resolution, calculate_z_limits, calculate_grid_points
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
    chunk_size=10_000,
    dtype_val=np.float32,
    dtype_idx=np.int32,
):
    """
    Construye operador disperso W (CSR: Compressed Sparse Row) que mapea gates -> voxels.
    
    OPERADOR W UNIVERSAL:
    Este operador se construye con z_max = TOA para servir a TODOS los sweeps.
    Usa filtro TOA para excluir gates (ecos no-meteorológicos).

    W[i, j] = peso (sin normalizar) del gate j para el voxel i

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
        - None: usa TODOS los gates dentro del ROI (query_ball_point)
        - int: usa hasta K vecinos dentro del ROI (query con upper bound)
    chunk_size: voxels procesados por bloque para reducir RAM
    dtype_val: dtype para los pesos
    dtype_idx: dtype para índices

    Returns:
        W: scipy.sparse.csr_matrix shape (Nvoxels, Ngates_total)
    """
    t_start = time.time()
    gates_xyz = np.asarray(gates_xyz, dtype=np.float32)
    voxels_xyz = np.asarray(voxels_xyz, dtype=np.float32)

    ngates_total = gates_xyz.shape[0]
    nvoxels = voxels_xyz.shape[0]
    
    # Filtro TOA
    # Esto elimina ecos no-meteorológicos y reduce tamaño del KDTree
    gate_z = gates_xyz[:, 2]
    toa_mask = gate_z <= toa
    valid_indices = np.where(toa_mask)[0]
    gates_xyz_filtered = gates_xyz[toa_mask]
    ngates = gates_xyz_filtered.shape[0]
    
    n_excluded = ngates_total - ngates
    print(f"Filtro TOA ({toa/1000:.1f}km): {ngates:,}/{ngates_total:,} gates válidos ({n_excluded:,} excluidos)")
    
    # Calcular centro del radar (asumiendo que los gates están centrados en el origen)
    radar_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    print(f"Construyendo operador W UNIVERSAL: {nvoxels:,} voxels, {ngates:,} gates")
    print(f"  ROI dist_beam: h_factor={h_factor}, nb={nb}°, bsp={bsp}, min_radius={min_radius}m")

    # Construir KDTree solo con gates filtrados
    tree = cKDTree(gates_xyz_filtered)

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
            # PASO 1: Calcular ROI máximo del chunk

            # Calculamos el ROI de TODOS los voxels del chunk simultáneamente (vectorizado).
            # Luego tomamos el ROI más grande del chunk para hacer UNA SOLA búsqueda en el KDTree.
            # Esto evita hacer N búsquedas individuales (una por voxel).

            # IMPORTANTE: Algunos voxels obtendrán gates FUERA de su ROI real (ej: voxel con 
            # ROI=850m recibirá gates hasta 2200m). Esto se corrige en el PASO 2.
            
            z_vals = V[:, 0] - radar_offset[0]
            y_vals = V[:, 1] - radar_offset[1]
            x_vals = V[:, 2] - radar_offset[2]
            
            # Calcular ROI para cada voxel usando dist_beam
            roi_vals = calculate_roi_dist_beam(
                z_coords=z_vals,
                y_coords=y_vals, 
                x_coords=x_vals,
                h_factor=h_factor,
                nb=nb,
                bsp=bsp,
                min_radius=min_radius,
                radar_offset=(0, 0, 0)  # Ya aplicamos offset arriba
            )
            
            max_roi_chunk = float(roi_vals.max())
            
            # Búsqueda grupal con manejo de MemoryError
            # Si el chunk es muy grande o el ROI es grande, puede fallar por memoria
            try:
                neigh_lists = tree.query_ball_point(V, r=max_roi_chunk)
            except MemoryError:
                # Sub-dividir el chunk en partes más pequeñas
                print(f"    ⚠️  MemoryError en chunk [{start}:{end}], subdividiendo...")
                sub_chunk_size = m // 4  # Dividir en 4 partes
                if sub_chunk_size < 1:
                    sub_chunk_size = 1
                
                neigh_lists = []
                for sub_start in range(0, m, sub_chunk_size):
                    sub_end = min(sub_start + sub_chunk_size, m)
                    V_sub = V[sub_start:sub_end]
                    
                    # Calcular ROI máximo del sub-chunk
                    roi_sub = roi_vals[sub_start:sub_end]
                    max_roi_sub = float(roi_sub.max())
                    
                    try:
                        sub_neigh = tree.query_ball_point(V_sub, r=max_roi_sub)
                        neigh_lists.extend(sub_neigh)
                    except MemoryError:
                        # Si aún falla, procesar de a uno
                        print(f"      ⚠️  MemoryError en sub-chunk, procesando voxel por voxel...")
                        for i in range(sub_end - sub_start):
                            vox = V_sub[i:i+1]
                            roi = float(roi_sub[i])
                            try:
                                n = tree.query_ball_point(vox, r=roi)
                                neigh_lists.extend(n)
                            except:
                                neigh_lists.append([])  # Voxel sin vecinos

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
                
                # PASO 2: Filtrado fino por voxel individual

                # Ahora procesamos CADA voxel individualmente con su ROI específico.
                # Calculamos el ROI real de este voxel y descartamos gates que están
                # fuera de su ROI (aunque estén dentro del max_roi_chunk).

                # Esto garantiza PRECISIÓN: cada voxel usa solo gates dentro de su ROI real.
                
                voxel = V[local_i]
                
                # Calcular ROI específico para ESTE voxel
                roi = float(calculate_roi_dist_beam(
                    z_coords=voxel[0] - radar_offset[0],
                    y_coords=voxel[1] - radar_offset[1],
                    x_coords=voxel[2] - radar_offset[2],
                    h_factor=h_factor,
                    nb=nb,
                    bsp=bsp,
                    min_radius=min_radius,
                    radar_offset=(0, 0, 0)  # Ya aplicamos offset arriba
                ))

                # Calcular distancias euclidianas gate → voxel
                gates_subset = gates_xyz_filtered[idxs]  # (k,3)
                d = np.linalg.norm(gates_subset - voxel, axis=1).astype(np.float32)
                
                # Filtrar: solo gates dentro del ROI REAL de este voxel
                within_roi = d <= roi
                if not within_roi.any():
                    continue
                
                idxs = idxs[within_roi]
                d = d[within_roi]

                # Calcular pesos (sin normalizar) usando el ROI específico
                w = compute_weights(d, roi, weight_func).astype(dtype_val)

                k = idxs.size
                rows[pos:pos+k] = i_global
                # Mapear índices locales (del array filtrado) a índices globales originales
                cols[pos:pos+k] = valid_indices[idxs].astype(dtype_idx, copy=False)
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
            
            # Calcular ROI para todos los voxels del chunk
            z_vals = V[:, 0] - radar_offset[0]
            y_vals = V[:, 1] - radar_offset[1]
            x_vals = V[:, 2] - radar_offset[2]
            
            roi_vals = calculate_roi_dist_beam(
                z_coords=z_vals,
                y_coords=y_vals,
                x_coords=x_vals,
                h_factor=h_factor,
                nb=nb,
                bsp=bsp,
                min_radius=min_radius,
                radar_offset=(0, 0, 0)
            )
            
            max_roi_chunk = float(roi_vals.max())

            dist, idx = tree.query(V, k=K, distance_upper_bound=max_roi_chunk)

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
            # Mapear índices locales a globales
            cols = valid_indices[idx[valid]].astype(dtype_idx, copy=False)

            # pesos: si nearest -> todos 1, si no -> por distancia
            d = dist[valid].astype(np.float32, copy=False)
            if weight_func == "nearest":
                vals = np.ones_like(d, dtype=dtype_val)
            else:
                # Para pesos, usar el ROI específico de cada voxel
                voxel_indices = np.repeat(np.arange(m), valid.sum(axis=1).astype(np.int32))
                voxel_rois = roi_vals[voxel_indices].astype(np.float32)
                vals = compute_weights(d, voxel_rois, weight_func).astype(dtype_val)

            # stats
            voxels_with_data += int((valid.sum(axis=1) > 0).sum())
            total_neighbors += nnz_chunk

        rows_chunks.append(rows)
        cols_chunks.append(cols)
        vals_chunks.append(vals)

    if len(rows_chunks) == 0:
        # No hubo vecinos para ningún voxel
        print("Operador W vacío (sin vecinos)")
        return csr_matrix((nvoxels, ngates_total), dtype=dtype_val), valid_indices

    # Concatenar chunks
    rows_all = np.concatenate(rows_chunks)
    cols_all = np.concatenate(cols_chunks)
    vals_all = np.concatenate(vals_chunks)

    # Construir CSR con shape completa (nvoxels x ngates_total originales)
    W = csr_matrix((vals_all, (rows_all, cols_all)), shape=(nvoxels, ngates_total), dtype=dtype_val)
    # Limpieza útil
    W.sum_duplicates()
    W.sort_indices()

    t_elapsed = time.time() - t_start
    avg_neighbors = total_neighbors / voxels_with_data if voxels_with_data > 0 else 0.0
    size_mb = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1024**2
    print(
        f"Operador W UNIVERSAL construido:\n"
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
    
    SLICING INTELIGENTE:
    Permite reutilizar un operador W universal extrayendo solo los niveles
    verticales relevantes para un sweep específico.
    
    Args:
        W_full: csr_matrix completo (nvoxels_full, ngates)
        grid_shape_full: tuple (nz_full, ny, nx) de la grilla completa
        z_levels_needed: int, número de niveles Z a extraer desde el fondo
        grid_shape_sliced: tuple (nz_sliced, ny, nx) de la grilla resultante
    
    Returns:
        W_sliced: csr_matrix (nvoxels_sliced, ngates)
    
    Ejemplo:
        W_full shape: (13*400*400=2,080,000 voxels, 3.5M gates) para z_max=12km
        z_levels_needed: 10 (primeros 10 niveles, 0-9km)
        W_sliced shape: (10*400*400=1,600,000 voxels, 3.5M gates)
        
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
    print(f"Validación: Operador W Universal + Slicing Inteligente")
    print(f"="*80)

    # Archivo NetCDF de prueba
    nc_path = Path(__file__).parent.parent.parent / "biribiri" / "RMA1_0315_01_20250819T001715Z.nc"
    radar = pyart.io.read(str(nc_path))

    # Parámetros de grilla
    elevation = 0  # Sweep bajo para demostrar slicing
    cappi_height = None
    interp = "Barnes2"
    field_name, _ = resolve_field(radar, "DBZH")
    radar_name, strategy, volume, _ = extract_metadata_from_filename(str(nc_path))

    # Parámetros comunes
    toa = 12000  # 12 km - límite físico universal
    grid_res_xy, grid_res_z = calculate_grid_resolution(volume)
    range_max_m = safe_range_max_m(radar)
    
    # Calcular parámetros dist_beam
    h_factor = 1.0
    nb = 1.5
    bsp = 1.0
    min_radius = 300.0

    # ==================== GRILLA UNIVERSAL (z_max = TOA) ====================
    print(f"\n[1] CONFIGURACIÓN GRILLA UNIVERSAL")
    print(f"    TOA (Top Of Atmosphere): {toa/1000:.1f} km")
    print(f"    Este operador sirve para TODOS los sweeps\n")
    
    # Grilla universal: z_max = TOA (no depende del sweep)
    z_grid_limits_universal = (0.0, toa)
    y_grid_limits = (-range_max_m, range_max_m)
    x_grid_limits = (-range_max_m, range_max_m)
    
    z_points_universal, y_points, x_points = calculate_grid_points(
        z_grid_limits_universal, y_grid_limits, x_grid_limits,
        grid_res_z, grid_res_xy
    )
    
    grid_limits_universal = (z_grid_limits_universal, y_grid_limits, x_grid_limits)
    grid_shape_universal = (z_points_universal, y_points, x_points)
    
    print(f"    Grid shape universal: {grid_shape_universal}")
    print(f"    Total voxels: {np.prod(grid_shape_universal):,}")

    # ==================== CONSTRUIR OPERADOR W UNIVERSAL ====================
    print(f"\n[2] CONSTRUCCIÓN OPERADOR W UNIVERSAL")
    print(f"    Este paso se hace UNA SOLA VEZ y sirve para todos los sweeps\n")
    
    # Coordenadas cartesianas de gates
    gates_xyz = get_gate_xyz_coords(radar)
    print(f"    Gates totales: {gates_xyz.shape[0]:,}")

    # Coordenadas cartesianas de voxels (GRILLA UNIVERSAL)
    voxels_xyz_universal = get_grid_xyz_coords(grid_shape_universal, grid_limits_universal)
    print(f"    Voxels grilla universal: {voxels_xyz_universal.shape[0]:,}\n")

    # Construir operador W UNIVERSAL
    t0 = time.time()
    W_universal, valid_gate_indices = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz_universal,
        toa=toa,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        weight_func=interp,
        max_neighbors=None,
        chunk_size=5_000,
    )
    t_build_w = time.time() - t0
    
    print(f"\n    ✓ Operador W universal listo en {t_build_w:.2f}s")

    # ==================== SLICING PARA SWEEP ESPECÍFICO ====================
    print(f"\n[3] SLICING PARA SWEEP {elevation} (elevación {radar.fixed_angle['data'][elevation]:.2f}°)")
    print(f"    Extraemos solo los niveles Z necesarios para este sweep\n")
    
    # Calcular altura máxima del haz para este sweep específico
    z_min_sweep, z_max_sweep, elev_deg = calculate_z_limits(
        range_max_m, elevation, cappi_height, radar.fixed_angle['data']
    )
    
    # Limitar por TOA (no puede ser mayor que la grilla universal)
    z_max_sweep = min(z_max_sweep, toa)
    
    z_grid_limits_sweep = (z_min_sweep, z_max_sweep)
    z_points_sweep, _, _ = calculate_grid_points(
        z_grid_limits_sweep, y_grid_limits, x_grid_limits,
        grid_res_z, grid_res_xy
    )
    
    grid_limits_sweep = (z_grid_limits_sweep, y_grid_limits, x_grid_limits)
    grid_shape_sweep = (z_points_sweep, y_points, x_points)
    
    print(f"    Altura haz: 0 - {z_max_sweep/1000:.1f} km")
    print(f"    Niveles Z necesarios: {z_points_sweep} (de {z_points_universal} disponibles)")
    print(f"    Grid shape sweep: {grid_shape_sweep}")
    
    # Hacer slicing del operador W universal
    W_sliced = slice_W_operator(
        W_full=W_universal,
        grid_shape_full=grid_shape_universal,
        z_levels_needed=z_points_sweep,
        grid_shape_sliced=grid_shape_sweep
    )

    # ==================== APLICAR OPERADOR SLICED ====================
    print(f"\n[4] APLICACIÓN DEL OPERADOR W SLICED")
    
    field_data = radar.fields[field_name]['data']

    t0 = time.time()
    grid3d_sparse = apply_operator(W_sliced, field_data, grid_shape_sweep, handle_mask=True)
    t_apply = time.time() - t0
    
    print(f"    Completado en {t_apply:.2f}s")
    print(f"    Shape resultado: {grid3d_sparse.shape}")
    print(f"    Voxels válidos: {(~grid3d_sparse.mask).sum():,} / {grid3d_sparse.size:,} "
          f"({100 * (~grid3d_sparse.mask).sum() / grid3d_sparse.size:.1f}%)")

    # ==================== COMPARAR CON PYART ====================
    print(f"\n[5] VALIDACIÓN VS PYART (referencia)")
    
    grid3d_pyart, t_pyart = grid_with_pyart(
        radar=radar,
        field_name=field_name,
        grid_shape=grid_shape_sweep,
        grid_limits=grid_limits_sweep,
        h_factor=h_factor,
        nb=nb,
        bsp=bsp,
        min_radius=min_radius,
        weighting_function=interp,
    )
    
    print(f"    PyART completado en {t_pyart:.2f}s")
    
    # ==================== MÉTRICAS DE COMPARACIÓN ====================
    print(f"\n[6] MÉTRICAS DE COMPARACIÓN")
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
        print(f"Field: {field_name}, Sweep: {elevation} ({elev_deg:.2f}°)")
        print(f"\nVoxels válidos: {n_valid:,} ({100*n_valid/grid3d_sparse.size:.1f}% del total)")
        print(f"\nComparación W Universal + Slicing vs PyART:")
        print(f"  - RMSE: {rmse:.6f}")
        print(f"  - MAE:  {mae:.6f}")
        print(f"  - Max diff: {max_diff:.6f}")
        print(f"  - Mean (W+Slice): {sparse_valid.mean():.2f}")
        print(f"  - Mean (PyART):   {pyart_valid.mean():.2f}")
        print(f"  - Error relativo: {relative_error*100:.4f}%")
        
        # Criterio de éxito
        tolerance = 0.01  # 1% de diferencia relativa
        if relative_error < tolerance:
            print(f"\n✓ VALIDACIÓN EXITOSA")
        else:
            print(f"\n⚠ Error relativo sobre tolerancia (>{tolerance*100}%)")
    else:
        print(f"\n⚠ No hay voxels válidos en común para comparar")
    
    # ==================== RESUMEN DE VENTAJAS ====================
    print(f"\n{'='*80}")
    print(f"VENTAJAS DEL OPERADOR W UNIVERSAL + SLICING:")
    print(f"="*80)
    print(f"\n1. UN SOLO operador W sirve para TODOS los sweeps")
    print(f"   - Construcción: {t_build_w:.2f}s (una sola vez)")
    print(f"   - Aplicación: {t_apply:.2f}s (slicing instantáneo + aplicación)")
    print(f"\n2. Cache eficiente:")
    print(f"   - W universal se guarda UNA vez en disco")
    print(f"   - Slicing en memoria es instantáneo (<0.01s)")
    print(f"\n3. Flexibilidad:")
    print(f"   - Sweep 0 (bajo): usa {z_points_sweep}/{z_points_universal} niveles")
    print(f"   - Sweep N (alto): usa más niveles, mismo operador")
    print(f"\n4. Memoria controlada:")
    print(f"   - W universal: {(W_universal.data.nbytes + W_universal.indices.nbytes + W_universal.indptr.nbytes)/1024**2:.1f} MB")
    print(f"   - W sliced: {(W_sliced.data.nbytes + W_sliced.indices.nbytes + W_sliced.indptr.nbytes)/1024**2:.1f} MB")
    print(f"   - Slicing NO copia datos, solo ajusta punteros")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()