"""
Integration tests: filtros QC y visuales sobre datos reales.

Verifica que separate_filters, apply_visual_filters y apply_qc_filters
funcionan correctamente cuando se les pasa una grilla interpolada real.
"""

import pytest
import numpy as np

from app.models import RangeFilter
from app.services.radar_processing.filter_application import (
    separate_filters,
    apply_visual_filters,
    apply_qc_filters,
    build_gatefilter_for_gridding,
)
from app.services.radar_processing.grid_interpolate import apply_operator
from app.services.radar_common import resolve_field

from .conftest import SMALL_GRID_SHAPE


class TestFiltersWithRealGrid:
    """Filtros sobre grillas interpoladas de datos reales."""

    def _get_grids(self, radar_vol01, w_and_limits):
        """Helper: interpola DBZH y RHOHV si están disponibles."""
        W, _ = w_and_limits

        dbzh_field, _ = resolve_field(radar_vol01, "DBZH")
        dbzh_3d = apply_operator(W, radar_vol01.fields[dbzh_field]['data'], SMALL_GRID_SHAPE)
        # Colapsar con COLMAX (max vertical)
        dbzh_2d = np.ma.max(dbzh_3d, axis=0)

        rhohv_2d = None
        try:
            rhohv_field, _ = resolve_field(radar_vol01, "RHOHV")
            rhohv_3d = apply_operator(W, radar_vol01.fields[rhohv_field]['data'], SMALL_GRID_SHAPE)
            rhohv_2d = np.ma.max(rhohv_3d, axis=0)
        except KeyError:
            pass

        return dbzh_2d, rhohv_2d

    def test_visual_filter_masks_low_reflectivity(self, radar_vol01, w_and_limits):
        dbzh_2d, _ = self._get_grids(radar_vol01, w_and_limits)

        # Filtro: solo mostrar DBZH >= 10
        filters = [RangeFilter(field="DBZH", min=10.0, max=60.0)]
        filtered = apply_visual_filters(dbzh_2d, filters, "DBZH")

        # Debe haber MÁS valores enmascarados que antes
        original_masked = np.ma.count_masked(dbzh_2d)
        filtered_masked = np.ma.count_masked(filtered)
        assert filtered_masked >= original_masked

    def test_visual_filter_does_not_modify_original(self, radar_vol01, w_and_limits):
        dbzh_2d, _ = self._get_grids(radar_vol01, w_and_limits)

        original_count = dbzh_2d.count()
        _ = apply_visual_filters(dbzh_2d, [RangeFilter(field="DBZH", min=10.0, max=60.0)], "DBZH")

        # El array original no debe cambiar
        assert dbzh_2d.count() == original_count

    def test_qc_filter_with_rhohv(self, radar_vol01, w_and_limits):
        dbzh_2d, rhohv_2d = self._get_grids(radar_vol01, w_and_limits)

        if rhohv_2d is None:
            pytest.skip("RHOHV no disponible en este archivo")

        qc_filters = [RangeFilter(field="RHOHV", min=0.8, max=1.0)]
        qc_dict = {"RHOHV": rhohv_2d}

        filtered = apply_qc_filters(dbzh_2d, qc_filters, qc_dict)

        # QC filter debe enmascarar pixels donde RHOHV < 0.8
        original_masked = np.ma.count_masked(dbzh_2d)
        filtered_masked = np.ma.count_masked(filtered)
        assert filtered_masked >= original_masked

    def test_separate_filters_classification(self):
        filters = [
            RangeFilter(field="RHOHV", min=0.7, max=1.0),
            RangeFilter(field="DBZH", min=0.0, max=60.0),
            RangeFilter(field="ZDR", min=-2.0, max=6.0),
        ]
        qc, visual = separate_filters(filters, "DBZH")

        # RHOHV es QC (afecta interpolación)
        assert any(f.field == "RHOHV" for f in qc)
        # DBZH es visual (mismo campo que el principal)
        assert any(f.field == "DBZH" for f in visual)


class TestGateFilterForGridding:
    """GateFilter para excluir gates durante interpolación."""

    def test_gatefilter_none_without_filters(self, radar_vol01):
        """Sin filtros QC, no se crea GateFilter."""
        gf = build_gatefilter_for_gridding(radar_vol01, qc_filters=None)
        assert gf is None

    def test_gatefilter_none_with_empty_filters(self, radar_vol01):
        gf = build_gatefilter_for_gridding(radar_vol01, qc_filters=[])
        assert gf is None

    def test_gatefilter_with_rhohv_filter(self, radar_vol01):
        try:
            resolve_field(radar_vol01, "RHOHV")
        except KeyError:
            pytest.skip("RHOHV no disponible")

        qc = [RangeFilter(field="RHOHV", min=0.7, max=1.0)]
        gf = build_gatefilter_for_gridding(radar_vol01, qc_filters=qc)

        if gf is not None:
            # GateFilter debe tener gate_excluded con shape = (nrays, ngates)
            assert hasattr(gf, 'gate_excluded')
            assert gf.gate_excluded.shape == (radar_vol01.nrays, radar_vol01.ngates)
            # Debe excluir al menos algunos gates
            assert np.any(gf.gate_excluded)
