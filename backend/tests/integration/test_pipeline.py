"""
Integration tests: pipeline completo de interpolación y colapso.

Verifica el flujo: radar → W operator → grid 3D → collapse 2D
Estos tests son más lentos (construyen W) pero verifican la integración real.
"""

import pytest
import numpy as np
import pyart

from app.services.radar_processing.grid_interpolate import (
    apply_operator,
    apply_operator_to_all_fields,
)
from app.services.radar_processing.product_collapse import (
    collapse_grid_to_2d,
    collapse_cappi,
    collapse_colmax,
    collapse_ppi,
)
from app.services.radar_common import resolve_field

from .conftest import SMALL_GRID_SHAPE


class TestWOperatorBuild:
    """Construcción del operador W con datos reales."""

    def test_w_operator_is_sparse(self, w_and_limits):
        W, _ = w_and_limits
        from scipy.sparse import issparse
        assert issparse(W)

    def test_w_operator_shape(self, w_and_limits):
        W, _ = w_and_limits
        nz, ny, nx = SMALL_GRID_SHAPE
        expected_rows = nz * ny * nx  # un row por voxel
        assert W.shape[0] == expected_rows

    def test_w_operator_has_nonzero_entries(self, w_and_limits):
        W, _ = w_and_limits
        assert W.nnz > 0, "Operador W no tiene entradas (interpolación vacía)"

    def test_w_operator_weights_nonnegative(self, w_and_limits):
        W, _ = w_and_limits
        assert W.min() >= 0, "Pesos negativos en el operador W"

    def test_w_operator_columns_match_gates(self, w_and_limits, radar_vol01):
        W, _ = w_and_limits
        expected_cols = radar_vol01.nrays * radar_vol01.ngates
        assert W.shape[1] == expected_cols


class TestApplyOperator:
    """Aplicación del operador W a campos reales."""

    def test_interpolate_dbzh_produces_3d_grid(self, radar_vol01, w_and_limits):
        W, _ = w_and_limits
        field_to_use, _ = resolve_field(radar_vol01, "DBZH")
        field_data = radar_vol01.fields[field_to_use]['data']

        grid3d = apply_operator(W, field_data, SMALL_GRID_SHAPE)

        assert grid3d.shape == SMALL_GRID_SHAPE
        assert np.ma.isMaskedArray(grid3d)

    def test_interpolated_values_in_valid_range(self, radar_vol01, w_and_limits):
        W, _ = w_and_limits
        field_to_use, _ = resolve_field(radar_vol01, "DBZH")
        field_data = radar_vol01.fields[field_to_use]['data']

        grid3d = apply_operator(W, field_data, SMALL_GRID_SHAPE)
        valid = grid3d.compressed()  # solo valores no-masked

        if len(valid) > 0:
            assert valid.min() > -60, f"Valor interpolado extremo bajo: {valid.min()}"
            assert valid.max() < 100, f"Valor interpolado extremo alto: {valid.max()}"

    def test_grid_has_valid_and_masked_values(self, radar_vol01, w_and_limits):
        W, _ = w_and_limits
        field_to_use, _ = resolve_field(radar_vol01, "DBZH")
        field_data = radar_vol01.fields[field_to_use]['data']

        grid3d = apply_operator(W, field_data, SMALL_GRID_SHAPE)

        n_valid = grid3d.count()
        n_total = grid3d.size
        # Debería tener datos válidos pero no estar 100% llena
        assert n_valid > 0, "Grid totalmente vacía"
        assert n_valid < n_total, "Grid sin valores enmascarados"

    def test_apply_to_all_fields(self, radar_vol01, w_and_limits):
        W, _ = w_and_limits
        fields_dict = apply_operator_to_all_fields(
            radar=radar_vol01, W=W, grid_shape=SMALL_GRID_SHAPE
        )

        assert isinstance(fields_dict, dict)
        assert len(fields_dict) > 0

        # Cada campo debe tener la forma correcta (con dim temporal)
        for name, field in fields_dict.items():
            assert 'data' in field
            assert field['data'].shape == (1, *SMALL_GRID_SHAPE)


class TestCollapseWithRealData:
    """Colapso 3D→2D usando grillas interpoladas de datos reales."""

    def _get_grid3d(self, radar_vol01, w_and_limits):
        W, grid_limits = w_and_limits
        field_to_use, _ = resolve_field(radar_vol01, "DBZH")
        field_data = radar_vol01.fields[field_to_use]['data']
        grid3d = apply_operator(W, field_data, SMALL_GRID_SHAPE)
        z_coords = np.linspace(grid_limits[0][0], grid_limits[0][1], SMALL_GRID_SHAPE[0])
        x_coords = np.linspace(grid_limits[2][0], grid_limits[2][1], SMALL_GRID_SHAPE[2])
        y_coords = np.linspace(grid_limits[1][0], grid_limits[1][1], SMALL_GRID_SHAPE[1])
        return grid3d, z_coords, x_coords, y_coords

    def test_collapse_cappi_real_data(self, radar_vol01, w_and_limits):
        grid3d, z_coords, _, _ = self._get_grid3d(radar_vol01, w_and_limits)

        # CAPPI a 3000m
        arr2d = collapse_cappi(grid3d, z_coords, target_height_m=3000)

        nz, ny, nx = SMALL_GRID_SHAPE
        assert arr2d.shape == (ny, nx)
        assert np.ma.isMaskedArray(arr2d)

    def test_collapse_colmax_real_data(self, radar_vol01, w_and_limits):
        grid3d, _, _, _ = self._get_grid3d(radar_vol01, w_and_limits)

        arr2d = collapse_colmax(grid3d)

        nz, ny, nx = SMALL_GRID_SHAPE
        assert arr2d.shape == (ny, nx)

    def test_collapse_ppi_real_data(self, radar_vol01, w_and_limits):
        grid3d, z_coords, x_coords, y_coords = self._get_grid3d(radar_vol01, w_and_limits)

        elev_deg = float(radar_vol01.fixed_angle['data'][0])
        arr2d = collapse_ppi(grid3d, z_coords, x_coords, y_coords, elev_deg)

        nz, ny, nx = SMALL_GRID_SHAPE
        assert arr2d.shape == (ny, nx)

    def test_colmax_gte_cappi(self, radar_vol01, w_and_limits):
        """COLMAX (máximo) debe ser >= CAPPI (slice) en cada pixel."""
        grid3d, z_coords, _, _ = self._get_grid3d(radar_vol01, w_and_limits)

        cappi_2d = collapse_cappi(grid3d, z_coords, target_height_m=3000)
        colmax_2d = collapse_colmax(grid3d)

        # En pixels donde ambos tienen datos, COLMAX >= CAPPI
        both_valid = ~cappi_2d.mask & ~colmax_2d.mask
        if np.any(both_valid):
            assert np.all(
                colmax_2d[both_valid] >= cappi_2d[both_valid] - 0.01
            ), "COLMAX debería ser >= CAPPI (es el máximo vertical)"
