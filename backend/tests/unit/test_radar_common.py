"""Tests for radar_common utility functions."""
import json
import numpy as np
import pytest
from unittest.mock import MagicMock

from app.services.radar_common import (
    stable_hash,
    _roundf,
    _stable,
    _hash_of,
    md5_file,
    w_operator_cache_key,
    grid2d_cache_key,
    qc_signature,
    filters_affect_interpolation,
    collapse_field_3d_to_2d,
    get_radar_site,
    safe_range_max_m,
)


class TestStableHash:
    def test_same_input_same_hash(self):
        assert stable_hash({"a": 1}) == stable_hash({"a": 1})

    def test_key_order_irrelevant(self):
        assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})

    def test_different_values_different_hash(self):
        assert stable_hash({"a": 1}) != stable_hash({"a": 2})


class TestRoundf:
    def test_rounds_to_6_decimals(self):
        assert _roundf(1.123456789) == 1.123457

    def test_custom_precision(self):
        assert _roundf(1.555, 1) == 1.6


class TestStable:
    def test_sorts_dict_keys(self):
        result = _stable({"z": 1, "a": 2})
        assert list(result.keys()) == ["a", "z"]

    def test_rounds_floats(self):
        assert _stable(1.123456789) == 1.123457

    def test_converts_tuples_to_lists(self):
        assert _stable((1, 2, 3)) == [1, 2, 3]

    def test_nested_structures(self):
        result = _stable({"x": (1.1111111, {"b": 2, "a": 1})})
        assert isinstance(result["x"], list)
        assert list(result["x"][1].keys()) == ["a", "b"]


class TestHashOf:
    def test_deterministic(self):
        assert _hash_of({"key": "val"}) == _hash_of({"key": "val"})

    def test_order_independent(self):
        assert _hash_of({"b": 2, "a": 1}) == _hash_of({"a": 1, "b": 2})


class TestWOperatorCacheKey:
    def test_includes_radar_name(self):
        key = w_operator_cache_key(
            radar="RMA1", estrategia="0301", volumen="01",
            grid_shape=(25, 201, 201),
            grid_limits=((0, 25000), (-100000, 100000), (-100000, 100000)),
        )
        assert "RMA1" in key
        assert "0301" in key

    def test_different_radars_different_keys(self):
        common = dict(
            estrategia="0301", volumen="01",
            grid_shape=(25, 201, 201),
            grid_limits=((0, 25000), (-100000, 100000), (-100000, 100000)),
        )
        k1 = w_operator_cache_key(radar="RMA1", **common)
        k2 = w_operator_cache_key(radar="AR8", **common)
        assert k1 != k2


class TestQcSignature:
    def test_empty_filters(self):
        sig = qc_signature([])
        assert sig == ()

    def test_none_filters(self):
        sig = qc_signature(None)
        assert sig == ()

    def test_with_filters_matching(self):
        f = MagicMock()
        f.field = "RHOHV"
        f.min = 0.7
        f.max = None
        sig = qc_signature([f])
        assert len(sig) > 0
        assert sig[0][0] == "RHOHV"


class TestFiltersAffectInterpolation:
    def test_no_filters(self):
        assert filters_affect_interpolation([], "DBZH") is False

    def test_none_filters(self):
        assert filters_affect_interpolation(None, "DBZH") is False

    def test_qc_filter_on_non_qc_field(self):
        f = MagicMock()
        f.field = "RHOHV"
        assert filters_affect_interpolation([f], "DBZH") is True

    def test_qc_filter_on_qc_field(self):
        f = MagicMock()
        f.field = "RHOHV"
        assert filters_affect_interpolation([f], "RHOHV") is False

    def test_cross_field_filter(self):
        f = MagicMock()
        f.field = "ZDR"
        assert filters_affect_interpolation([f], "DBZH") is True


class TestMd5File:
    def test_consistent_hash(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        h1 = md5_file(str(p))
        h2 = md5_file(str(p))
        assert h1 == h2
        assert len(h1) == 32

    def test_different_content_different_hash(self, tmp_path):
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"aaa")
        p2.write_bytes(b"bbb")
        assert md5_file(str(p1)) != md5_file(str(p2))


class TestGrid2dCacheKey:
    def test_deterministic(self):
        common = dict(file_hash="abc", product_upper="PPI", field_to_use="DBZH",
                      elevation=0, cappi_height=None, volume="01",
                      interp="Barnes2", qc_sig=())
        assert grid2d_cache_key(**common) == grid2d_cache_key(**common)

    def test_different_fields_different_keys(self):
        common = dict(file_hash="abc", product_upper="PPI", elevation=0,
                      cappi_height=None, volume="01", interp="Barnes2", qc_sig=())
        k1 = grid2d_cache_key(field_to_use="DBZH", **common)
        k2 = grid2d_cache_key(field_to_use="VRAD", **common)
        assert k1 != k2

    def test_session_isolation(self):
        common = dict(file_hash="abc", product_upper="PPI", field_to_use="DBZH",
                      elevation=0, cappi_height=None, volume="01",
                      interp="Barnes2", qc_sig=())
        k1 = grid2d_cache_key(session_id="sess1", **common)
        k2 = grid2d_cache_key(session_id="sess2", **common)
        assert k1 != k2


class TestGetRadarSite:
    def test_extracts_coordinates(self):
        radar = MagicMock()
        radar.latitude = {"data": np.array([-31.4])}
        radar.longitude = {"data": np.array([-64.2])}
        radar.altitude = {"data": np.array([730.0])}
        lon, lat, alt = get_radar_site(radar)
        assert lat == pytest.approx(-31.4)
        assert lon == pytest.approx(-64.2)
        assert alt == pytest.approx(730.0)


class TestSafeRangeMaxM:
    def test_rounds_up_to_nearest_20km(self):
        radar = MagicMock()
        radar.range = {"data": np.array([0, 500, 116580.0])}
        result = safe_range_max_m(radar)
        assert result == 120000.0

    def test_exact_multiple_stays(self):
        radar = MagicMock()
        radar.range = {"data": np.array([0, 240000.0])}
        result = safe_range_max_m(radar)
        assert result == 240000.0


class TestCollapseField3dTo2d:
    def test_colmax(self):
        data3d = np.ma.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[3, 1], [2, 9]],
        ])
        result = collapse_field_3d_to_2d(data3d, "colmax")
        np.testing.assert_array_equal(result, [[5, 6], [7, 9]])

    def test_cappi(self):
        z_levels = np.array([1000, 2000, 3000])
        data3d = np.ma.array([
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
            [[90, 99], [88, 77]],
        ])
        result = collapse_field_3d_to_2d(data3d, "cappi",
                                         z_levels=z_levels, target_height_m=2000)
        np.testing.assert_array_equal(result, [[50, 60], [70, 80]])

    def test_4d_input_removes_time(self):
        data4d = np.ma.array([[[[1, 2], [3, 4]]]])  # (1, 1, 2, 2)
        result = collapse_field_3d_to_2d(data4d, "colmax")
        assert result.shape == (2, 2)

    def test_2d_input_passthrough(self):
        data2d = np.ma.array([[1, 2], [3, 4]])
        result = collapse_field_3d_to_2d(data2d, "ppi")
        np.testing.assert_array_equal(result, data2d)
