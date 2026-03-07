"""
Integration tests: construcción de grillas 3D.

Verifica que la cadena completa de interpolación funciona:
  radar → gate coords → grid coords → W operator → grid 3D
"""

import pytest
import numpy as np
import pyart

from app.services.radar_processing.grid_builder import (
    get_gate_xyz_coords,
    get_grid_xyz_coords,
    get_roi_params_for_volume,
)
from app.services.radar_processing.grid_geometry import (
    calculate_z_limits,
    calculate_grid_resolution,
    calculate_grid_points,
)
from app.services.radar_common import safe_range_max_m, resolve_field


class TestGateCoords:
    """Coordenadas de gates del radar con datos reales."""

    def test_gate_xyz_shape(self, radar_vol01):
        xyz = get_gate_xyz_coords(radar_vol01)
        expected = radar_vol01.nrays * radar_vol01.ngates
        assert xyz.shape == (expected, 3)

    def test_gate_xyz_types(self, radar_vol01):
        xyz = get_gate_xyz_coords(radar_vol01)
        assert xyz.dtype in (np.float32, np.float64)

    def test_gate_z_positive(self, radar_vol01):
        xyz = get_gate_xyz_coords(radar_vol01)
        # z (columna 2) debería ser mayormente ≥ 0 (sobre el suelo)
        z = xyz[:, 2]
        assert z.min() >= -100, f"Altura mínima de gate inesperada: {z.min()}"

    def test_gate_xy_symmetric(self, radar_vol01):
        xyz = get_gate_xyz_coords(radar_vol01)
        x, y = xyz[:, 0], xyz[:, 1]
        # Los gates deben extenderse en todas las direcciones desde el radar
        assert x.min() < -10_000, "Gates no se extienden hacia el oeste"
        assert x.max() > 10_000, "Gates no se extienden hacia el este"
        assert y.min() < -10_000, "Gates no se extienden hacia el sur"
        assert y.max() > 10_000, "Gates no se extienden hacia el norte"


class TestGridCoords:
    """Coordenadas de la grilla cartesiana."""

    def test_grid_xyz_shape(self):
        grid_shape = (20, 100, 100)
        grid_limits = ((0, 12000), (-150000, 150000), (-150000, 150000))

        xyz = get_grid_xyz_coords(grid_shape, grid_limits)
        N = 20 * 100 * 100
        assert xyz.shape == (N, 3)

    def test_grid_xyz_bounds(self):
        grid_shape = (10, 50, 50)
        grid_limits = ((0, 10000), (-100000, 100000), (-100000, 100000))

        xyz = get_grid_xyz_coords(grid_shape, grid_limits)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        assert x.min() == pytest.approx(-100000, rel=0.01)
        assert x.max() == pytest.approx(100000, rel=0.01)
        assert z.min() == pytest.approx(0, abs=1)
        assert z.max() == pytest.approx(10000, rel=0.01)

    def test_grid_coords_float32(self):
        grid_shape = (5, 10, 10)
        grid_limits = ((0, 5000), (-50000, 50000), (-50000, 50000))

        xyz = get_grid_xyz_coords(grid_shape, grid_limits, dtype=np.float32)
        assert xyz.dtype == np.float32


class TestGridGeometryWithRealData:
    """Cálculos de geometría usando datos reales para verificar consistencia."""

    def test_calculate_z_limits_ppi(self, radar_vol01):
        range_max = safe_range_max_m(radar_vol01)
        fixed_angles = radar_vol01.fixed_angle['data']
        elevation_idx = 0

        z_min, z_max, elev_deg = calculate_z_limits(
            range_max, elevation_idx, 4000, fixed_angles
        )

        assert z_min >= 0
        assert z_max > z_min
        assert elev_deg >= 0

    def test_grid_resolution_per_volume(self):
        res_01_xy, res_01_z = calculate_grid_resolution("01")
        res_03_xy, res_03_z = calculate_grid_resolution("03")

        # Vol 03 tiene mayor resolución (menor valor en metros)
        assert res_03_xy < res_01_xy

    def test_grid_points_reasonable_size(self, radar_vol01):
        range_max = safe_range_max_m(radar_vol01)
        res_xy, res_z = calculate_grid_resolution("01")

        z_pts, y_pts, x_pts = calculate_grid_points(
            (0, 12000), (-range_max, range_max), (-range_max, range_max),
            res_xy, res_z
        )

        # Verificar que las dimensiones son razonables
        assert 5 <= z_pts <= 100, f"z_pts fuera de rango: {z_pts}"
        assert 50 <= y_pts <= 2000, f"y_pts fuera de rango: {y_pts}"
        assert 50 <= x_pts <= 2000, f"x_pts fuera de rango: {x_pts}"
        # y_pts y x_pts deben ser iguales (grilla cuadrada xy)
        assert y_pts == x_pts


class TestROIParams:
    """Parámetros de ROI por volumen."""

    def test_roi_params_vol01(self):
        h, nb, bsp, min_r = get_roi_params_for_volume("01")
        assert h > 0 and nb > 0 and bsp > 0 and min_r > 0

    def test_roi_params_vol03_larger_roi(self):
        _, _, _, min_r_01 = get_roi_params_for_volume("01")
        _, _, _, min_r_03 = get_roi_params_for_volume("03")
        # Vol 03 debería tener ROI más grande
        assert min_r_03 > min_r_01

    def test_roi_unknown_volume_uses_default(self):
        default = get_roi_params_for_volume(None)
        vol01 = get_roi_params_for_volume("01")
        assert default == vol01
