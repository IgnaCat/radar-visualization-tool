"""
Módulo para interpolación de datos de radar en grillas 3D usando operadores dispersos W.
"""

import numpy as np
import pyart
from typing import Optional, List, Dict

def apply_operator(W, field_data, grid_shape, handle_mask=True, additional_filters=None):
    """
    Aplica operador W a datos de campo del radar.
    
    Args:
        W: scipy.sparse.csr_matrix (Nvoxels, Ngates) (CSR: Compressed Sparse Row)
        field_data: np.ma.MaskedArray de shape (nrays, ngates)
        grid_shape: tuple (nz, ny, nx) para reshape final
        handle_mask: Si True, normaliza por gates válidos
        additional_filters: Optional[List[pyart.filters.GateFilter]] - Filtros (ej. RHOHV) 
                            para aplicar antes de interpolar
    
    Returns:
        np.ma.MaskedArray: Grilla 3D de shape (nz, ny, nx)
    """
    # Normalizar filtros a lista
    if additional_filters is None:
        additional_filters = []
    elif not isinstance(additional_filters, list):
        additional_filters = [additional_filters]
    
    # Aplanar field_data a vector 1D
    # NOTA: si field_data es MaskedArray, conviene separar valores (.data) y máscara (.mask)
    if np.ma.isMaskedArray(field_data):
        g = field_data.data.ravel()  # valores puros (sin máscara)
        field_mask = field_data.mask.ravel()  # máscara 2D aplanada
    else:
        g = np.asarray(field_data).ravel()
        field_mask = np.zeros(g.shape, dtype=bool)
    
    # Combinar con filtros adicionales (ej. RHOHV QC)
    for gf in additional_filters:
        if hasattr(gf, 'gate_excluded'):
            field_mask = field_mask | gf.gate_excluded.ravel()
    
    if handle_mask:
        # Crear vector de máscara (1 = válido, 0 = enmascarado)
        mask_valid = (~field_mask).astype(float)
        
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


def apply_operator_to_all_fields(radar, W, grid_shape, handle_mask=True, additional_filters=None):
    """
    Aplica operador W a todos los campos del radar y devuelve dict formateado para PyART Grid.
    
    Args:
        radar: pyart.core.Radar con múltiples campos
        W: scipy.sparse.csr_matrix operador de interpolación
        grid_shape: tuple (nz, ny, nx)
        handle_mask: Si True, maneja máscaras en los datos
        additional_filters: Optional[Dict[str, List[pyart.filters.GateFilter]]] - 
                            Dict con filtros por campo. Si es None o un campo no está,
                            se usa lista vacía para ese campo.
    
    Returns:
        dict: Diccionario de campos formateados para pyart.core.Grid
              {field_name: {'data': array, 'long_name': str, 'units': str, ...}}
    """
    if additional_filters is None:
        additional_filters = {}
    
    all_fields = list(radar.fields.keys())
    fields_dict = {}
    
    for field_name in all_fields:
        field_data = radar.fields[field_name]['data']
        
        # Obtener filtros específicos para este campo
        field_filters = additional_filters.get(field_name, None)
        
        # Aplicar operador W
        grid3d_field = apply_operator(W, field_data, grid_shape, handle_mask=handle_mask, 
                                       additional_filters=field_filters)
        
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