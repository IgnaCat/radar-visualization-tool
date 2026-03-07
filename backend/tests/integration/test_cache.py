"""
Integration tests: cache de operador W y grillas 2D.

Verifica que los mecanismos de cache funcionan correctamente:
- Cache keys son deterministas con datos reales
- Guardar/cargar operador W en disco
- GRID2D_CACHE almacena y recupera correctamente
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from app.services.radar_common import (
    md5_file,
    w_operator_cache_key,
    grid2d_cache_key,
    qc_signature,
)
from app.services.radar_processing.grid_builder import get_roi_params_for_volume
from app.core.cache import (
    save_w_operator_to_disk,
    load_w_operator_from_disk,
    get_w_operator_cache_path,
    GRID2D_CACHE,
)


class TestFileHash:
    """Hash MD5 de archivos reales."""

    def test_md5_produces_hex_string(self, vol01_filepath):
        h = md5_file(vol01_filepath)
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex = 32 chars
        # Solo caracteres hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_md5_deterministic(self, vol01_filepath):
        h1 = md5_file(vol01_filepath)
        h2 = md5_file(vol01_filepath)
        assert h1 == h2

    def test_md5_different_files(self, vol01_filepath, vol03_filepath):
        h1 = md5_file(vol01_filepath)
        h3 = md5_file(vol03_filepath)
        assert h1 != h3


class TestCacheKeyConsistency:
    """Cache keys con parámetros derivados de datos reales."""

    def test_w_operator_key_deterministic(self):
        key1 = w_operator_cache_key(
            radar="RMA1", estrategia="0315", volumen="01",
            grid_shape=(20, 300, 300),
            grid_limits=((0, 12000), (-240000, 240000), (-240000, 240000)),
            weight_func="Barnes2", max_neighbors=30,
        )
        key2 = w_operator_cache_key(
            radar="RMA1", estrategia="0315", volumen="01",
            grid_shape=(20, 300, 300),
            grid_limits=((0, 12000), (-240000, 240000), (-240000, 240000)),
            weight_func="Barnes2", max_neighbors=30,
        )
        assert key1 == key2

    def test_w_operator_key_changes_with_volume(self):
        args = dict(
            radar="RMA1", estrategia="0315",
            grid_shape=(20, 300, 300),
            grid_limits=((0, 12000), (-240000, 240000), (-240000, 240000)),
            weight_func="Barnes2", max_neighbors=30,
        )
        key_01 = w_operator_cache_key(volumen="01", **args)
        key_03 = w_operator_cache_key(volumen="03", **args)
        assert key_01 != key_03

    def test_grid2d_key_different_products(self):
        common = dict(
            file_hash="abc123",
            field_to_use="DBZH",
            elevation=0,
            cappi_height=None,
            volume="01",
            interp="Barnes2",
            qc_sig=(),
        )
        key_ppi = grid2d_cache_key(product_upper="PPI", **common)
        key_colmax = grid2d_cache_key(product_upper="COLMAX", **common)
        assert key_ppi != key_colmax


class TestWOperatorDiskPersistence:
    """Guardado y carga de operador W desde disco."""

    def test_save_and_load_w_operator(self):
        """Crea un W sintético, guarda en disco, carga y compara."""
        from scipy.sparse import csr_matrix, random as sp_random

        # Crear W sintético (100 voxels, 500 gates)
        W_original = sp_random(100, 500, density=0.05, format='csr', dtype=np.float32)
        metadata = {"radar": "TEST", "volumen": "01", "shape": W_original.shape}

        test_key = "test_disk_persistence_key_12345"

        try:
            # Guardar
            save_w_operator_to_disk(test_key, W_original, metadata)

            # Verificar archivo existe
            cache_path = get_w_operator_cache_path(test_key)
            assert cache_path.exists()

            # Cargar
            result = load_w_operator_from_disk(test_key)
            assert result is not None

            W_loaded, meta_loaded = result

            # Comparar matrices
            assert W_loaded.shape == W_original.shape
            assert W_loaded.nnz == W_original.nnz
            diff = (W_loaded - W_original)
            assert diff.nnz == 0, "Matrices W difieren después de save/load"

            # Comparar metadata
            assert meta_loaded["radar"] == "TEST"
            assert meta_loaded["volumen"] == "01"
        finally:
            # Limpiar archivo de test
            cache_path = get_w_operator_cache_path(test_key)
            for f in cache_path.parent.glob(f"{test_key}*"):
                f.unlink(missing_ok=True)

    def test_load_nonexistent_returns_none(self):
        result = load_w_operator_from_disk("nonexistent_key_xyz_999")
        assert result is None


class TestGrid2DCache:
    """Cache de grillas 2D (in-memory LRU)."""

    def test_cache_store_and_retrieve(self):
        """Guardar y recuperar un paquete en GRID2D_CACHE."""
        test_key = "test_integ_cache_key_abc"
        arr = np.random.rand(50, 50).astype(np.float32)
        pkg = {
            "arr": np.ma.array(arr),
            "crs": "EPSG:3857",
            "transform": None,
        }

        try:
            GRID2D_CACHE[test_key] = pkg
            retrieved = GRID2D_CACHE.get(test_key)
            assert retrieved is not None
            assert np.array_equal(retrieved["arr"], pkg["arr"])
        finally:
            GRID2D_CACHE.pop(test_key, None)

    def test_cache_miss_returns_none(self):
        result = GRID2D_CACHE.get("definitely_not_cached_xyz")
        assert result is None
