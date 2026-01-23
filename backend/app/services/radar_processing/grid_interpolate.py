"""
Módulo para interpolación de datos de radar en grillas 3D usando operadores dispersos W.
"""

import numpy as np
import pyart

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


def apply_operator_to_all_fields(radar, W, grid_shape, handle_mask=True):
    """
    Aplica operador W a todos los campos del radar y devuelve dict formateado para PyART Grid.
    
    Args:
        radar: pyart.core.Radar con múltiples campos
        W: scipy.sparse.csr_matrix operador de interpolación
        grid_shape: tuple (nz, ny, nx)
        handle_mask: Si True, maneja máscaras en los datos
    
    Returns:
        dict: Diccionario de campos formateados para pyart.core.Grid
              {field_name: {'data': array, 'long_name': str, 'units': str, ...}}
    """
    all_fields = list(radar.fields.keys())
    fields_dict = {}
    
    for field_name in all_fields:
        field_data = radar.fields[field_name]['data']
        
        # Aplicar operador W
        grid3d_field = apply_operator(W, field_data, grid_shape, handle_mask=handle_mask)
        
        # Guardar en formato PyART con dimensión temporal
        field_dict = {
            'data': grid3d_field[np.newaxis, :, :, :],  # Agregar dimensión temporal
            'long_name': radar.fields[field_name].get('long_name', field_name),
            'units': radar.fields[field_name].get('units', ''),
            'standard_name': radar.fields[field_name].get('standard_name', field_name),
            '_FillValue': -9999.0,
        }
        fields_dict[field_name] = field_dict
    
    return fields_dict