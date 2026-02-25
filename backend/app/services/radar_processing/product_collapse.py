"""
Colapso de grillas 3D a 2D según producto (PPI, CAPPI, COLMAX).
"""
import numpy as np
from scipy import ndimage
from .grid_geometry import compute_beam_height


def fill_grid3d_holes_inplace(data3d, max_distance=3):
    """
    Rellena huecos pequeños en una grilla 3D usando nearest neighbor horizontal.
    
    Para cada nivel Z, rellena pixeles enmascarados con el valor del pixel válido más cercano
    en el plano horizontal (X-Y). Esto reduce los huecos causados por interpolación
    polar→cartesiano sin regenerar toda la matriz W.
    
    Args:
        data3d: np.ma.MaskedArray de shape (nz, ny, nx)
        max_distance: Distancia máxima en pixeles para buscar vecinos (default: 3)
    
    Modifica data3d in-place.
    """
    if not np.ma.is_masked(data3d):
        return  # No hay máscaras, nada que hacer
    
    nz, ny, nx = data3d.shape
    total_filled = 0
    
    # Procesar cada nivel Z independientemente
    for iz in range(nz):
        level = data3d[iz, :, :]
        
        if not np.ma.is_masked(level):
            continue
        
        mask = level.mask
        if not np.any(mask):
            continue  # Este nivel no tiene huecos
        
        # Usar distance_transform_edt para encontrar pixeles válidos más cercanos
        distances, indices = ndimage.distance_transform_edt(
            mask, 
            return_distances=True,
            return_indices=True
        )
        
        # Solo rellenar huecos dentro de max_distance pixeles
        fill_mask = (distances > 0) & (distances <= max_distance)
        
        if np.any(fill_mask):
            # indices tiene shape (2, ny, nx) con [y_indices, x_indices]
            y_nearest = indices[0]
            x_nearest = indices[1]
            
            # Rellenar con valores de pixeles válidos más cercanos
            filled_values = level.data[y_nearest, x_nearest]
            data3d.data[iz, fill_mask] = filled_values[fill_mask]
            data3d.mask[iz, fill_mask] = False
            total_filled += np.sum(fill_mask)
    
    if total_filled > 0:
        print(f"[fill_grid3d_holes] Rellenados {total_filled} pixeles (max_distance={max_distance})")


def collapse_grid_to_2d(grid, field, product, *,
                        elevation_deg=None,       # para PPI
                        target_height_m=None,     # para CAPPI
                        vmin=-30.0):
    """
    Convierte la grilla 3D a 2D según el producto:
      - "ppi": sigue el haz del sweep con elevación `elevation_deg`
      - "cappi": toma el nivel z más cercano a `target_height_m`
      - "colmax": toma el máximo en columna
    
    Args:
        grid: pyart.core.Grid con datos 3D
        field: Nombre del campo a colapsar
        product: Tipo de producto ('ppi', 'cappi', 'colmax')
        elevation_deg: Ángulo de elevación para PPI (requerido si product='ppi')
        target_height_m: Altura objetivo para CAPPI (requerido si product='cappi')
        vmin: Valor mínimo para enmascarado
    
    Modifica grid in-place, colapsando el campo 3D a 2D.
    """
    data3d = grid.fields[field]['data']
    product = product.lower()
    
    # PyART Grid puede tener dimensión temporal (time, z, y, x)
    # Eliminar dimensión temporal si existe
    if data3d.ndim == 4:
        data3d = data3d[0, :, :, :]  # Tomar primer (y único) timestep
    
    z = grid.z['data']
    x = grid.x['data']
    y = grid.y['data']
    ny, nx = len(y), len(x)

    if data3d.ndim == 2:  # ya llegó 2D (raro)
        arr2d = data3d
    else:
        if product == "ppi":
            # PPI: sigue el haz a elevación constante con interpolación lineal
            assert elevation_deg is not None
            arr2d = collapse_ppi(data3d, z, x, y, elevation_deg)
            
        elif product == "cappi":
            # CAPPI: slice horizontal a altura constante con interpolación lineal
            assert target_height_m is not None
            arr2d = collapse_cappi(data3d, z, target_height_m)
            
        elif product == "colmax":
            # COLMAX: máximo en cada columna vertical
            arr2d = collapse_colmax(data3d)
            
        else:
            raise ValueError("Producto inválido")
    
    # Enmascarar valores fuera de rango según el campo
    if field in ["filled_DBZH", "DBZH", "DBZV", "DBZHF", "composite_reflectivity", "cappi"]:
        arr2d = np.ma.masked_less_equal(arr2d, vmin)
    elif field in ["KDP", "ZDR"]:
        arr2d = np.ma.masked_less(arr2d, vmin)

    # Lo escribimos como un único nivel
    grid.fields[field]['data'] = arr2d[np.newaxis, ...]   # (1,ny,nx)
    grid.fields[field]['_FillValue'] = -9999.0
    grid.z['data'] = np.array([0.0], dtype=float)


def collapse_cappi(data3d, z_coords, target_height_m):
    """
    Extrae un slice horizontal a altura constante desde la grilla 3D.
    
    Si la altura coincide con un nivel Z de la grilla, toma ese nivel.
    Si no coincide, interpola linealmente entre los dos niveles más cercanos.
    
    Args:
        data3d: Array 3D (nz, ny, nx) con los datos
        z_coords: Array 1D con coordenadas Z de la grilla (en metros)
        target_height_m: Altura objetivo en metros
    
    Returns:
        Array 2D (ny, nx) con el CAPPI
    """
    nz = data3d.shape[0]
    
    # Chequear si la altura pedida coincide exactamente con un nivel de la grilla
    matches = np.isclose(z_coords, target_height_m, rtol=1e-6)
    if np.any(matches):
        # Coincide exactamente - tomar ese nivel directo
        z_idx = np.where(matches)[0][0]
        return data3d[z_idx, :, :]
    
    # No coincide - interpolar linealmente entre niveles superior e inferior
    z_min, z_max = z_coords[0], z_coords[-1]
    z_step = (z_max - z_min) / (nz - 1) if nz > 1 else 1.0
    
    # Calcular posición fraccionaria en la grilla
    z_frac = (target_height_m - z_min) / z_step
    z_low = int(np.floor(z_frac))
    z_high = z_low + 1
    
    # Manejar casos fuera de rango
    if z_low < 0:
        return data3d[0, :, :]  # Debajo de la grilla - usar nivel más bajo
    if z_high >= nz:
        return data3d[nz - 1, :, :]  # Arriba de la grilla - usar nivel más alto
    
    # Interpolar entre niveles inferior y superior
    # Qué tanto peso darle al nivel superior o inferior
    weight_high = z_frac - z_low
    weight_low = 1.0 - weight_high
    
    val_low = data3d[z_low, :, :]
    val_high = data3d[z_high, :, :]
    
    # Interpolación lineal: valor = peso_bajo * val_bajo + peso_alto * val_alto
    return weight_low * val_low + weight_high * val_high


def collapse_colmax(data3d):
    """
    Calcula el máximo en cada columna vertical (COLMAX).
    
    Para cada pixel (y,x), toma el valor máximo entre todos los niveles Z.
    
    Args:
        data3d: Array 3D (nz, ny, nx) con los datos
    
    Returns:
        Array 2D (ny, nx) con el máximo de cada columna
    """
    # np.nanmax ignora NaN y funciona con masked arrays
    return np.nanmax(data3d, axis=0)


def collapse_ppi(data3d, z_coords, x_coords, y_coords, elevation_deg):
    """
    Extrae un PPI a elevación constante desde la grilla 3D con interpolación lineal.
    
    Para cada punto (x,y) en la grilla:
    1. Calcula la distancia horizontal al radar
    2. Calcula la altura del haz a esa distancia usando Earth curvature (4/3 modelo)
    3. Interpola linealmente entre los niveles Z para obtener el valor
    
    Esto resuelve el problema de "puntos transparentes" que ocurría con nearest neighbor.
    La interpolación lineal suaviza la mezcla inherente de sweeps en la grilla cartesiana.
    
    Args:
        data3d: Array 3D (nz, ny, nx) con los datos
        z_coords: Array 1D con coordenadas Z (alturas) en metros
        x_coords: Array 1D con coordenadas X en metros
        y_coords: Array 1D con coordenadas Y en metros
        elevation_deg: Ángulo de elevación en grados
    
    Returns:
        Array 2D (ny, nx) con el PPI interpolado
    """
    nz, ny, nx = data3d.shape
    z_min, z_max = z_coords[0], z_coords[-1]
    
    # Rellenar huecos pequeños en grilla 3D antes de colapsar
    # Esto mejora cobertura en niveles Z bajos que tienen pocos datos (0.5-18%)
    #fill_grid3d_holes_inplace(data3d, max_distance=10)
    
    # Crear meshgrid de coordenadas horizontales (indexing='ij' para match con products.py)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Distancia horizontal de cada pixel al radar (asumido en origen)
    horizontal_dist = np.sqrt(xx**2 + yy**2)
    
    # Calcular altura del haz con curvatura terrestre (modelo 4/3)
    target_z = compute_beam_height(horizontal_dist, elevation_deg, radar_altitude=0.0)
    
    # Interpolación lineal entre niveles Z
    z_step = (z_max - z_min) / (nz - 1) if nz > 1 else 1.0
    
    # Inicializar resultado
    result = np.full((ny, nx), np.nan, dtype='float32')
    
    # Calcular posición fraccionaria en grilla Z para cada pixel
    z_frac = (target_z - z_min) / z_step
    z_low = np.floor(z_frac).astype(int)
    z_high = z_low + 1
    
    # Pesos para interpolación
    weight_high = z_frac - z_low
    weight_low = 1.0 - weight_high
    
    # Identificar regiones fuera de grilla ANTES de clipear
    below_grid = target_z < z_min
    above_grid = target_z > z_max
    
    # Clipear índices para indexación segura
    z_low_safe = np.clip(z_low, 0, nz - 1)
    z_high_safe = np.clip(z_high, 0, nz - 1)
    
    # Crear índices para indexación avanzada
    y_indices, x_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    # Obtener valores en niveles inferior y superior
    val_low = data3d[z_low_safe, y_indices, x_indices]
    val_high = data3d[z_high_safe, y_indices, x_indices]
    
    # Propagar máscaras correctamente durante interpolación
    if np.ma.is_masked(val_low) or np.ma.is_masked(val_high):
        # Crear máscara combinada: enmascarar si CUALQUIERA está enmascarado
        combined_mask = np.ma.getmaskarray(val_low) | np.ma.getmaskarray(val_high)
        
        # Interpolar datos (ignorando máscaras temporalmente)
        result_data = weight_low * np.ma.filled(val_low, 0) + weight_high * np.ma.filled(val_high, 0)
        
        # Crear resultado como masked array con máscara combinada
        result = np.ma.array(result_data, mask=combined_mask, dtype='float32')
    else:
        # Interpolar normalmente si no hay máscaras
        result = weight_low * val_low + weight_high * val_high
    
    # Enmascarar puntos fuera de grilla
    # Esto previene el "anillo" de datos incorrectos
    if np.ma.is_masked(result):
        # Si es masked array, actualizar la máscara
        result.mask[below_grid] = True
        result.mask[above_grid] = True
    else:
        # Si no es masked, convertir a masked array
        mask = below_grid | above_grid
        result = np.ma.array(result, mask=mask, dtype='float32')
    
    return result