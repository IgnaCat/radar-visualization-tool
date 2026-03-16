"""
Aplicación de filtros sobre grillas 2D.
Separa filtros QC (campos auxiliares como RHOHV) de filtros visuales (mismo campo).
"""

import numpy as np
import pyart
import logging
from typing import List, Optional
from ...core.constants import AFFECTS_INTERP_FIELDS
from ...models import RangeFilter

logger = logging.getLogger(__name__)


def separate_filters(
    filters: List[RangeFilter], field_to_use: str
) -> tuple[List, List]:
    """
    Separa filtros en QC (afectan interpolación, ej RHOHV) vs visuales (mismo campo).

    Regla clave: Si el filtro es para el campo principal (field_to_use), SIEMPRE es visual,
    incluso si el campo está en AFFECTS_INTERP_FIELDS.

    Args:
        filters: Lista de filtros a aplicar
        field_to_use: Nombre del campo principal siendo procesado

    Returns:
        Tupla (qc_filters, visual_filters):
            - qc_filters: Filtros sobre campos QC (RHOHV, ZDR, etc) que NO son el campo principal
            - visual_filters: Filtros sobre el mismo campo principal, o campos no-QC
    """
    qc_filters = []
    visual_filters = []
    field_to_use_upper = str(field_to_use or "").upper()

    for f in filters or []:
        ffield = str(getattr(f, "field", "") or "").upper()

        # Si el filtro es para el campo principal, SIEMPRE es visual
        if ffield == field_to_use_upper:
            visual_filters.append(f)
        # Si es un campo QC pero NO es el principal, es QC
        elif ffield in AFFECTS_INTERP_FIELDS:
            qc_filters.append(f)
        # Otros campos son visuales
        else:
            visual_filters.append(f)

    return qc_filters, visual_filters


def apply_visual_filters(
    arr2d: np.ma.array, visual_filters: List[RangeFilter], field_to_use: str
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

    for f in visual_filters or []:
        ffield = getattr(f, "field", None)
        if not ffield:
            continue
        if str(ffield).upper() == str(field_to_use).upper():
            fmin = getattr(f, "min", None)
            fmax = getattr(f, "max", None)
            if fmin is not None:
                dyn_mask |= masked < float(fmin)
            if fmax is not None:
                dyn_mask |= masked > float(fmax)

    masked.mask = np.ma.getmaskarray(masked) | dyn_mask
    return masked


def apply_qc_filters(
    arr_warped: np.ma.array, qc_filters: List[RangeFilter], qc_warped_dict: dict
) -> np.ma.array:
    """
    Aplica filtros QC (campos auxiliares) como máscaras post-warp sobre arrays warped.

    Los filtros QC usan campos como RHOHV, ZDR, etc. para enmascarar datos
    del campo principal comparando arrays warped.

    Args:
        arr_warped: Array principal warped (no se modifica directamente)
        qc_filters: Lista de filtros QC a aplicar
        qc_warped_dict: Diccionario {field_name: arr_warped_qc} con campos QC warped

    Returns:
        Array principal con máscaras QC adicionales aplicadas
    """
    masked = np.ma.array(arr_warped, copy=True)

    for f in qc_filters:
        qf = str(getattr(f, "field", "") or "").upper()
        q_warped = qc_warped_dict.get(qf)
        if q_warped is None:
            logger.warning(f"Campo QC '{qf}' no encontrado en qc_warped_dict")
            continue

        logger.info(
            f"Aplicando filtro QC {qf}: min={getattr(f, 'min', None)}, max={getattr(f, 'max', None)}"
        )
        logger.info(
            f"Tipo de q_warped para {qf}: {type(q_warped)}, shape: {getattr(q_warped, 'shape', 'N/A')}"
        )
        q_warped_copy = np.asarray(np.ma.filled(q_warped, np.nan), dtype=np.float32)
        logger.info(
            f"q_warped {qf}: min={np.nanmin(q_warped_copy)}, max={np.nanmax(q_warped_copy)}, mean={np.nanmean(q_warped_copy)}"
        )
        logger.info(
            f"q_warped {qf}: masked={np.ma.is_masked(q_warped)}, finite_pixels={np.sum(np.isfinite(q_warped_copy))}"
        )

        qmask = np.zeros(masked.shape, dtype=bool)
        fmin = getattr(f, "min", None)
        fmax = getattr(f, "max", None)

        # Tratar NaN/inf del campo QC como no confiables: se enmascaran siempre.
        qmask |= ~np.isfinite(q_warped_copy)

        if fmin is not None:
            qmask |= q_warped_copy < float(fmin)
        if fmax is not None:
            qmask |= q_warped_copy > float(fmax)

        logger.info(f"qmask para {qf}: True en {np.sum(qmask)} posiciones")

        masked.mask = np.ma.getmaskarray(masked) | qmask
        logger.info(f"Filtro QC aplicado: {np.sum(qmask)} píxeles enmascarados")

    return masked


def build_gatefilter_for_gridding(
    radar: pyart.core.Radar, qc_filters: Optional[List[RangeFilter]] = None
) -> Optional[pyart.filters.GateFilter]:
    """
    Construye un GateFilter para usar durante la interpolación de grilla.

    Aplica exclude_transition() y filtros de rango sobre campos QC (RHOHV, etc.)
    que afectan la interpolación.

    Args:
        radar: Objeto radar PyART
        qc_filters: Lista de RangeFilter con filtros QC

    Returns:
        GateFilter configurado o None si no hay filtros
    """
    if not qc_filters:
        return None

    gatefilter = pyart.filters.GateFilter(radar)

    # Exclude transition gates
    try:
        gatefilter.exclude_transition()
    except Exception:
        pass

    # Aplicar cada filtro QC
    for f in qc_filters:
        fld = getattr(f, "field", None)
        if not fld or fld not in radar.fields:
            continue

        fmin = getattr(f, "min", None)
        fmax = getattr(f, "max", None)

        try:
            if fmin is not None:
                gatefilter.exclude_below(fld, float(fmin))
            if fmax is not None:
                gatefilter.exclude_above(fld, float(fmax))
        except Exception as e:
            logger.warning(f"Error aplicando filtro {fld} durante gridding: {e}")

    return gatefilter


def build_gatefilter_for_visual(
    radar: pyart.core.Radar,
    visual_filters: Optional[List[RangeFilter]] = None,
    field_to_use: Optional[str] = None,
) -> Optional[pyart.filters.GateFilter]:
    """
    Construye un GateFilter desde filtros visuales (rango sobre el campo principal).

    Solo aplica filtros cuyo campo coincide con field_to_use, para excluir
    gates fuera de rango antes de la interpolación (más preciso que filtrar post-grid).

    Args:
        radar: Objeto radar PyART
        visual_filters: Lista de RangeFilter con filtros de rango visual
        field_to_use: Nombre resuelto del campo principal (ej. "DBZH")

    Returns:
        GateFilter configurado o None si no hay filtros aplicables
    """
    if not visual_filters or not field_to_use:
        return None

    if field_to_use not in radar.fields:
        return None

    applicable = [
        f
        for f in visual_filters
        if str(getattr(f, "field", "") or "").upper() == str(field_to_use).upper()
    ]
    if not applicable:
        return None

    gatefilter = pyart.filters.GateFilter(radar)

    for f in applicable:
        fmin = getattr(f, "min", None)
        fmax = getattr(f, "max", None)

        try:
            if fmin is not None:
                # Excepción especial para RHOHV con umbrales muy bajos
                if fmin <= 0.3 and field_to_use == "RHOHV":
                    continue
                else:
                    gatefilter.exclude_below(field_to_use, float(fmin))
            if fmax is not None:
                gatefilter.exclude_above(field_to_use, float(fmax))
        except Exception as e:
            logger.warning(
                f"Error aplicando filtro visual {field_to_use} durante gridding: {e}"
            )

    return gatefilter
