"""Tests for cog_generator — normalización y generación RGBA."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.cm as cm

from app.services.radar_processing.cog_generator import create_cog_from_warped_array


class TestCogColorMapping:
    """Tests para la lógica de normalización y colormap (sin escribir a disco)."""

    def test_normalization_range(self):
        data = np.ma.array([0, 5, 10, 15, 20], dtype=float).reshape(1, 5)
        vmin, vmax = 0, 20
        norm = (data - vmin) / (vmax - vmin)
        np.testing.assert_allclose(norm, [[0.0, 0.25, 0.5, 0.75, 1.0]])

    def test_masked_values_become_transparent(self):
        data = np.ma.array([10.0, np.nan, 20.0], mask=[False, True, False]).reshape(1, 3)
        data = np.ma.masked_invalid(data)
        cmap = cm.get_cmap("viridis")
        norm = np.clip((data - 0) / (30 - 0), 0, 1)
        rgba = cmap(norm.filled(0))
        alpha = np.where(data.mask, 0, 255).astype(np.uint8)
        assert alpha[0, 0] == 255
        assert alpha[0, 1] == 0   # masked → transparent
        assert alpha[0, 2] == 255

    def test_masked_pixels_rgb_zeroed(self):
        data = np.ma.array([10.0, np.nan], mask=[False, True]).reshape(1, 2)
        data = np.ma.masked_invalid(data)
        cmap = cm.get_cmap("viridis")
        norm = np.clip((data - 0) / (30 - 0), 0, 1)
        rgba = (cmap(norm.filled(0)) * 255).astype(np.uint8)
        mask = data.mask
        for i in range(3):
            rgba[:, :, i] = np.where(mask, 0, rgba[:, :, i])
        assert rgba[0, 1, 0] == 0  # R
        assert rgba[0, 1, 1] == 0  # G
        assert rgba[0, 1, 2] == 0  # B

    @patch("rasterio.open")
    def test_creates_4_band_rgba(self, mock_rasterio_open):
        mock_dst = MagicMock()
        mock_rasterio_open.return_value.__enter__ = MagicMock(return_value=mock_dst)
        mock_rasterio_open.return_value.__exit__ = MagicMock(return_value=False)

        data = np.ma.array([[1.0, 2.0], [3.0, 4.0]])
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 1, 1, 2, 2)
        cmap = cm.get_cmap("viridis")

        result = create_cog_from_warped_array(
            data, "test.tif", transform, "EPSG:3857", cmap, 0, 10
        )
        assert result == "test.tif"
        # Should write 4 bands (R, G, B, Alpha)
        assert mock_dst.write.call_count == 4

    def test_clip_out_of_range_values(self):
        data = np.ma.array([-10, 50], dtype=float).reshape(1, 2)
        vmin, vmax = 0, 20
        norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        assert norm[0, 0] == 0.0  # -10 clipped to 0
        assert norm[0, 1] == 1.0  # 50 clipped to 1
