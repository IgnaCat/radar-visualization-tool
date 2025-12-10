"""
Aplicación de filtros sobre grillas 2D.
Separa filtros QC (campos auxiliares como RHOHV) de filtros visuales (mismo campo).
"""

import numpy as np
from typing import List
from ..core.constants import AFFECTS_INTERP_FIELDS
from ..schemas import RangeFilter


def separate_filters(filters: List[RangeFilter], field_to_use: str) -> tuple[List, List]:
    """
    Separa filtros en QC (afectan interpolación, ej RHOHV) vs visuales (mismo campo).
    
    Args:
        filters: Lista de filtros a aplicar
        field_to_use: Nombre del campo principal siendo procesado
    
    Returns:
        Tupla (qc_filters, visual_filters):
            - qc_filters: Filtros sobre campos QC (RHOHV, ZDR, etc)
            - visual_filters: Filtros sobre el mismo campo principal
    """
    qc_filters = []
    visual_filters = []
    
    for f in (filters or []):
        ffield = str(getattr(f, "field", "") or "").upper()
        if ffield in AFFECTS_INTERP_FIELDS:
            qc_filters.append(f)
        else:
            visual_filters.append(f)
    
    return qc_filters, visual_filters


def apply_visual_filters(
    arr2d: np.ma.array,
    visual_filters: List[RangeFilter],
    field_to_use: str
) -> np.ma.array:
    """
    Aplica filtros visuales de rango (sobre el mismo campo) como máscaras post-grid.
    
    Regla: cualquier filtro cuyo .field == field_to_use se aplica como máscara 2D.
    
    Args:
        arr2d: Array 2D enmascarado (datos de grilla)
        visual_filters: Lista de filtros a aplicar
        field_to_use: Nombre del campo principal
    
    Returns:
        Array 2D con máscaras adicionales aplicadas
    """
    masked = np.ma.array(arr2d, copy=True)
    dyn_mask = np.zeros(masked.shape, dtype=bool)
    
    for f in (visual_filters or []):
        ffield = getattr(f, "field", None)
        if not ffield:
            continue
        if str(ffield).upper() == str(field_to_use).upper():
            fmin = getattr(f, "min", None)
            fmax = getattr(f, "max", None)
            if fmin is not None:
                # Excepción especial para RHOHV con umbrales muy bajos
                if (fmin <= 0.3 and field_to_use == "RHOHV"):
                    continue
                else:
                    dyn_mask |= (masked < float(fmin))
            if fmax is not None:
                dyn_mask |= (masked > float(fmax))
    
    masked.mask = np.ma.getmaskarray(masked) | dyn_mask
    return masked


def apply_qc_filters(
    arr2d: np.ma.array,
    qc_filters: List[RangeFilter],
    qc_dict: dict
) -> np.ma.array:
    """
    Aplica filtros QC (campos auxiliares) como máscaras post-grid.
    
    Los filtros QC usan campos como RHOHV, ZDR, etc. para enmascarar datos
    del campo principal sin regridding.
    
    Args:
        arr2d: Array 2D enmascarado (datos de grilla)
        qc_filters: Lista de filtros QC a aplicar
        qc_dict: Diccionario {field_name: arr2d_qc} con campos QC colapsados
    
    Returns:
        Array 2D con máscaras QC adicionales aplicadas
    """
    masked = np.ma.array(arr2d, copy=True)
    
    for f in qc_filters:
        qf = str(getattr(f, "field", "") or "").upper()
        q2d = qc_dict.get(qf)
        if q2d is None:
            continue
        
        qmask = np.zeros(masked.shape, dtype=bool)
        fmin = getattr(f, "min", None)
        fmax = getattr(f, "max", None)
        
        if fmin is not None:
            qmask |= (q2d < float(fmin))
        if fmax is not None:
            qmask |= (q2d > float(fmax))
        
        masked.mask = np.ma.getmaskarray(masked) | qmask
    
    return masked
