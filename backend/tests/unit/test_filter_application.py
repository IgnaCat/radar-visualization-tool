"""
Tests para app.services.radar_processing.filter_application — filtrado post-grid.

Qué testea este archivo:
1. separate_filters: Clasifica filtros en QC vs visuales.
   - Un filtro sobre RHOHV (que está en AFFECTS_INTERP_FIELDS) → va a qc_filters.
   - Un filtro sobre DBZH (campo principal) → va a visual_filters.
   - Lista vacía o None → no explota.

2. apply_visual_filters: Aplica máscaras de rango sobre el array 2D.
   - Valores fuera del rango [min, max] se enmascaran.
   - Excepción especial: RHOHV con min <= 0.3 NO se aplica (bypass).

3. apply_qc_filters: Aplica máscaras de campos auxiliares (RHOHV, etc).
   - Usa un diccionario {campo: array2d} de campos QC.
   - Si el valor del campo QC está fuera de rango → enmascara el dato principal.
"""

import pytest
import numpy as np
from app.models import RangeFilter
from app.services.radar_processing.filter_application import (
    separate_filters,
    apply_visual_filters,
    apply_qc_filters,
)


# ═══════════════════════════════════════════════════════════════════
# separate_filters
# ═══════════════════════════════════════════════════════════════════

class TestSeparateFilters:
    """
    separate_filters divide filtros en dos categorías:
    - qc_filters: campos QC como RHOHV (en AFFECTS_INTERP_FIELDS)
    - visual_filters: todo lo demás (incluido el campo principal)
    """

    def test_rhohv_va_a_qc(self, sample_range_filter_rhohv):
        """RHOHV está en AFFECTS_INTERP_FIELDS → va a qc_filters."""
        qc, vis = separate_filters([sample_range_filter_rhohv], "DBZH")
        assert len(qc) == 1
        assert len(vis) == 0
        assert qc[0].field == "RHOHV"

    def test_dbzh_va_a_visual(self, sample_range_filter_dbzh):
        """DBZH no es campo QC → va a visual_filters."""
        qc, vis = separate_filters([sample_range_filter_dbzh], "DBZH")
        assert len(qc) == 0
        assert len(vis) == 1
        assert vis[0].field == "DBZH"

    def test_mezcla_qc_y_visual(
        self, sample_range_filter_rhohv, sample_range_filter_dbzh
    ):
        """Con ambos tipos, se separan correctamente."""
        filters = [sample_range_filter_rhohv, sample_range_filter_dbzh]
        qc, vis = separate_filters(filters, "DBZH")
        assert len(qc) == 1
        assert len(vis) == 1

    def test_lista_vacia(self):
        """Lista vacía no explota, devuelve dos listas vacías."""
        qc, vis = separate_filters([], "DBZH")
        assert qc == []
        assert vis == []

    def test_none_como_filtros(self):
        """None como input no explota (el código usa `filters or []`)."""
        qc, vis = separate_filters(None, "DBZH")
        assert qc == []
        assert vis == []

    def test_campo_no_reconocido_va_a_visual(self):
        """Un campo inventado que no es QC va a visual_filters."""
        f = RangeFilter(field="CAMPO_RARO", min=0, max=100)
        qc, vis = separate_filters([f], "DBZH")
        assert len(qc) == 0
        assert len(vis) == 1


# ═══════════════════════════════════════════════════════════════════
# apply_visual_filters
# ═══════════════════════════════════════════════════════════════════

class TestApplyVisualFilters:
    """
    apply_visual_filters enmascara valores del array 2D que caen
    fuera del rango [min, max] del filtro, SOLO si el filtro es
    para el mismo campo que se está visualizando.
    """

    def test_mascara_valores_fuera_de_rango(self):
        """Valores < min o > max se enmascaran."""
        arr = np.ma.array([5.0, 15.0, 25.0, 35.0, 45.0])
        f = RangeFilter(field="DBZH", min=10.0, max=40.0)
        result = apply_visual_filters(arr, [f], "DBZH")
        
        # 5.0 < 10.0 → enmascarado, 45.0 > 40.0 → enmascarado
        assert result.mask[0] == True   # 5.0 fuera
        assert result.mask[1] == False  # 15.0 dentro
        assert result.mask[2] == False  # 25.0 dentro
        assert result.mask[3] == False  # 35.0 dentro
        assert result.mask[4] == True   # 45.0 fuera

    def test_no_modifica_array_original(self):
        """apply_visual_filters hace copy=True, no toca el original."""
        arr = np.ma.array([5.0, 15.0, 25.0])
        f = RangeFilter(field="DBZH", min=10.0, max=20.0)
        result = apply_visual_filters(arr, [f], "DBZH")
        
        # El original no tiene máscaras
        assert not np.any(np.ma.getmaskarray(arr))
        # El resultado sí
        assert result.mask[0] == True
        assert result.mask[2] == True

    def test_filtro_de_otro_campo_no_aplica(self):
        """Si el filtro es para ZDR pero visualizo DBZH, no se aplica."""
        arr = np.ma.array([5.0, 15.0, 25.0])
        f = RangeFilter(field="ZDR", min=10.0, max=20.0)
        result = apply_visual_filters(arr, [f], "DBZH")
        
        # Ningún valor enmascarado porque el filtro no es para este campo
        assert not np.any(np.ma.getmaskarray(result))

    def test_excepcion_rhohv_min_bajo(self):
        """
        Caso especial: RHOHV con min <= 0.3 se ignora (bypass).
        Esto es por diseño — umbrales QC muy bajos no tienen sentido como 
        filtro visual porque dejarían pasar ruido.
        """
        arr = np.ma.array([0.1, 0.5, 0.8, 0.95])
        f = RangeFilter(field="RHOHV", min=0.2, max=1.0)  # min=0.2 ≤ 0.3
        result = apply_visual_filters(arr, [f], "RHOHV")
        
        # El min NO se aplica (bypass), solo max
        assert result.mask[0] == False  # 0.1 normalmente sería filtrado, pero bypass
        assert result.mask[1] == False
        assert result.mask[2] == False
        assert result.mask[3] == False

    def test_rhohv_min_alto_si_aplica(self):
        """RHOHV con min > 0.3 SÍ se aplica normalmente."""
        arr = np.ma.array([0.1, 0.5, 0.8, 0.95])
        f = RangeFilter(field="RHOHV", min=0.7, max=1.0)  # min=0.7 > 0.3
        result = apply_visual_filters(arr, [f], "RHOHV")
        
        assert result.mask[0] == True   # 0.1 < 0.7
        assert result.mask[1] == True   # 0.5 < 0.7
        assert result.mask[2] == False  # 0.8 >= 0.7
        assert result.mask[3] == False  # 0.95 >= 0.7

    def test_solo_max_sin_min(self):
        """Filtro con solo max, sin min explícito (min=0 default)."""
        arr = np.ma.array([5.0, 15.0, 25.0])
        f = RangeFilter(field="DBZH", min=0, max=20.0)
        result = apply_visual_filters(arr, [f], "DBZH")
        
        assert result.mask[2] == True  # 25 > 20

    def test_lista_vacia_no_cambia_nada(self):
        """Sin filtros, el array queda igual."""
        arr = np.ma.array([5.0, 15.0, 25.0])
        result = apply_visual_filters(arr, [], "DBZH")
        np.testing.assert_array_equal(result.data, arr.data)

    def test_array_2d(self):
        """Funciona con arrays 2D (como las grillas reales)."""
        arr = np.ma.array([[5.0, 15.0], [25.0, 35.0]])
        f = RangeFilter(field="DBZH", min=10.0, max=30.0)
        result = apply_visual_filters(arr, [f], "DBZH")
        
        assert result.mask[0, 0] == True   # 5 < 10
        assert result.mask[0, 1] == False  # 15 ok
        assert result.mask[1, 0] == False  # 25 ok
        assert result.mask[1, 1] == True   # 35 > 30


# ═══════════════════════════════════════════════════════════════════
# apply_qc_filters
# ═══════════════════════════════════════════════════════════════════

class TestApplyQcFilters:
    """
    apply_qc_filters usa campos auxiliares (como RHOHV) para enmascarar
    el campo principal. Si RHOHV < 0.7 en un pixel, se enmascara DBZH
    en ese mismo pixel.
    
    Esto es diferente a visual_filters porque:
    - Visual: filtra valores del MISMO campo.
    - QC: filtra valores de OTRO campo (auxiliar).
    """

    def test_enmascara_por_rhohv_bajo(self):
        """Si RHOHV < 0.7, se enmascara el dato principal."""
        arr_dbzh = np.ma.array([10.0, 20.0, 30.0, 40.0])
        arr_rhohv = np.ma.array([0.5, 0.8, 0.3, 0.95])
        
        f = RangeFilter(field="RHOHV", min=0.7, max=1.0)
        qc_dict = {"RHOHV": arr_rhohv}
        
        result = apply_qc_filters(arr_dbzh, [f], qc_dict)
        
        assert result.mask[0] == True   # RHOHV=0.5 < 0.7
        assert result.mask[1] == False  # RHOHV=0.8 ok
        assert result.mask[2] == True   # RHOHV=0.3 < 0.7
        assert result.mask[3] == False  # RHOHV=0.95 ok

    def test_campo_qc_no_en_dict(self):
        """Si el campo QC no está en qc_dict, no enmascara nada."""
        arr = np.ma.array([10.0, 20.0, 30.0])
        f = RangeFilter(field="RHOHV", min=0.7, max=1.0)
        qc_dict = {}  # vacío
        
        result = apply_qc_filters(arr, [f], qc_dict)
        assert not np.any(np.ma.getmaskarray(result))

    def test_no_modifica_original(self):
        """copy=True: el array original queda intacto."""
        arr = np.ma.array([10.0, 20.0, 30.0])
        arr_qc = np.ma.array([0.5, 0.8, 0.95])
        f = RangeFilter(field="RHOHV", min=0.7, max=1.0)
        
        result = apply_qc_filters(arr, [f], {"RHOHV": arr_qc})
        assert not np.any(np.ma.getmaskarray(arr))  # original no tocado

    def test_multiples_filtros_qc(self):
        """Varios filtros QC se acumulan (OR de máscaras)."""
        arr = np.ma.array([10.0, 20.0, 30.0, 40.0])
        arr_rhohv = np.ma.array([0.5, 0.8, 0.9, 0.95])
        arr_zdr = np.ma.array([1.0, -3.0, 2.0, 5.0])  # hipotético
        
        f1 = RangeFilter(field="RHOHV", min=0.7, max=1.0)
        f2 = RangeFilter(field="ZDR", min=-2.0, max=4.0)
        
        qc_dict = {"RHOHV": arr_rhohv, "ZDR": arr_zdr}
        result = apply_qc_filters(arr, [f1, f2], qc_dict)
        
        # Pixel 0: RHOHV=0.5 < 0.7 → enmascarado
        # Pixel 1: ZDR=-3.0 < -2.0 → enmascarado
        # Pixel 2: ambos ok
        # Pixel 3: ZDR=5.0 > 4.0 → enmascarado
        assert result.mask[0] == True
        assert result.mask[1] == True
        assert result.mask[2] == False
        assert result.mask[3] == True

    def test_filtro_solo_max(self):
        """Filtro QC con solo max."""
        arr = np.ma.array([10.0, 20.0, 30.0])
        arr_qc = np.ma.array([0.5, 0.8, 1.1])
        f = RangeFilter(field="RHOHV", min=0, max=1.0)
        
        result = apply_qc_filters(arr, [f], {"RHOHV": arr_qc})
        assert result.mask[2] == True  # 1.1 > 1.0
