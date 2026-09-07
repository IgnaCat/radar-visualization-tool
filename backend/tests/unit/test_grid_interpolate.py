"""Tests for grid_interpolate.apply_operator."""
import numpy as np
import scipy.sparse as sp
import pytest

from app.services.radar_processing.grid_interpolate import apply_operator


def _simple_W(nvoxels, ngates, connections):
    """Build a tiny W matrix from (voxel, gate, weight) triples."""
    rows, cols, data = zip(*connections)
    return sp.csr_matrix((data, (rows, cols)), shape=(nvoxels, ngates))


class TestApplyOperator:
    def test_uniform_weights_returns_mean(self):
        W = _simple_W(1, 3, [(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0)])
        field = np.ma.array([10.0, 20.0, 30.0]).reshape(1, 3)
        result = apply_operator(W, field, grid_shape=(1, 1, 1))
        assert result.shape == (1, 1, 1)
        np.testing.assert_allclose(result[0, 0, 0], 20.0)

    def test_weighted_average(self):
        W = _simple_W(1, 2, [(0, 0, 3.0), (0, 1, 1.0)])
        field = np.ma.array([10.0, 30.0]).reshape(1, 2)
        result = apply_operator(W, field, grid_shape=(1, 1, 1))
        np.testing.assert_allclose(result[0, 0, 0], 15.0)  # (3*10+1*30)/4

    def test_masked_gates_excluded(self):
        W = _simple_W(1, 3, [(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0)])
        field = np.ma.array([10.0, 999.0, 30.0], mask=[False, True, False]).reshape(1, 3)
        result = apply_operator(W, field, grid_shape=(1, 1, 1))
        np.testing.assert_allclose(result[0, 0, 0], 20.0)  # (10+30)/2, gate 1 excluded

    def test_all_masked_returns_masked(self):
        W = _simple_W(1, 2, [(0, 0, 1.0), (0, 1, 1.0)])
        field = np.ma.array([10.0, 20.0], mask=[True, True]).reshape(1, 2)
        result = apply_operator(W, field, grid_shape=(1, 1, 1))
        assert result[0, 0, 0] is np.ma.masked

    def test_no_mask_handling(self):
        W = _simple_W(1, 2, [(0, 0, 1.0), (0, 1, 1.0)])
        field = np.ma.array([10.0, 20.0]).reshape(1, 2)
        result = apply_operator(W, field, grid_shape=(1, 1, 1), handle_mask=False)
        np.testing.assert_allclose(result[0, 0, 0], 15.0)

    def test_multiple_voxels(self):
        W = _simple_W(4, 2, [
            (0, 0, 1.0),               # voxel 0 ← gate 0
            (1, 1, 1.0),               # voxel 1 ← gate 1
            (2, 0, 0.5), (2, 1, 0.5),  # voxel 2 ← both
            (3, 0, 0.0),               # voxel 3 ← no real weight
        ])
        field = np.ma.array([10.0, 30.0]).reshape(1, 2)
        result = apply_operator(W, field, grid_shape=(2, 2, 1))
        np.testing.assert_allclose(result[0, 0, 0], 10.0)
        np.testing.assert_allclose(result[0, 1, 0], 30.0)
        np.testing.assert_allclose(result[1, 0, 0], 20.0)
        assert result[1, 1, 0] is np.ma.masked  # weight=0 → den≈0 → masked

    def test_nomask_field(self):
        W = _simple_W(1, 2, [(0, 0, 1.0), (0, 1, 1.0)])
        field = np.ma.array([10.0, 20.0])
        field.mask = np.ma.nomask
        result = apply_operator(W, field.reshape(1, 2), grid_shape=(1, 1, 1))
        np.testing.assert_allclose(result[0, 0, 0], 15.0)

    def test_plain_ndarray_input(self):
        W = _simple_W(1, 2, [(0, 0, 1.0), (0, 1, 1.0)])
        field = np.array([10.0, 20.0]).reshape(1, 2)
        result = apply_operator(W, field, grid_shape=(1, 1, 1))
        np.testing.assert_allclose(result[0, 0, 0], 15.0)
