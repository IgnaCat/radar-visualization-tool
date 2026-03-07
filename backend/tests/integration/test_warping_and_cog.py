"""
Integration tests: warping y generación de COG.

Verifica: grid 2D → warp a Web Mercator → COG (GeoTIFF).
Usa datos reales warpeados para verificar que el archivo de
salida es un GeoTIFF válido en EPSG:3857.

NOTA: Estos tests requieren que PROJ y rasterio estén correctamente
configurados (misma versión de PROJ DB). En entornos con conflictos
GDAL/PROJ (ej. pip rasterio + conda GDAL) se skipean automáticamente.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

try:
    import rasterio
    from rasterio.crs import CRS
    # Verificar que PROJ funciona correctamente
    _test_crs = CRS.from_epsg(3857)
    PROJ_AVAILABLE = True
except Exception:
    PROJ_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PROJ_AVAILABLE,
    reason="PROJ/rasterio no configurados correctamente (conflicto de versiones PROJ DB)"
)

from app.services.radar_processing.warping import warp_array_to_mercator
from app.services.radar_processing.cog_generator import create_cog_from_warped_array
from app.services.radar_common import safe_range_max_m, colormap_for


class TestWarpToMercator:
    """Reproyección de array 2D (AzEq) a Web Mercator."""

    def test_warp_synthetic_grid(self, radar_vol01):
        """Warp de una grilla sintética centrada en la posición del radar."""
        lat = float(radar_vol01.latitude['data'][0])
        lon = float(radar_vol01.longitude['data'][0])

        # Grilla sintética 100x100 con un patrón circular
        ny, nx = 100, 100
        y = np.linspace(-50000, 50000, ny)
        x = np.linspace(-50000, 50000, nx)
        xx, yy = np.meshgrid(x, y)
        data_2d = np.sqrt(xx ** 2 + yy ** 2) / 1000  # distancia en km

        warped, transform, crs = warp_array_to_mercator(
            data_2d, lat, lon,
            x_limits=(-50000, 50000),
            y_limits=(-50000, 50000)
        )

        assert warped.ndim == 2
        assert warped.shape[0] > 0
        assert warped.shape[1] > 0
        assert not np.all(np.isnan(warped)), "Array warped totalmente NaN"
        assert crs == 'EPSG:3857'

    def test_warp_preserves_data_range(self, radar_vol01):
        lat = float(radar_vol01.latitude['data'][0])
        lon = float(radar_vol01.longitude['data'][0])

        ny, nx = 80, 80
        data_2d = np.random.uniform(0, 70, (ny, nx)).astype(np.float32)

        warped, _, _ = warp_array_to_mercator(
            data_2d, lat, lon,
            x_limits=(-80000, 80000),
            y_limits=(-80000, 80000)
        )

        valid = warped[~np.isnan(warped)]
        if len(valid) > 0:
            # Valores válidos deben estar en el rango original
            assert valid.min() >= -1
            assert valid.max() <= 75


class TestCOGGeneration:
    """Generación de Cloud Optimized GeoTIFF."""

    def test_create_cog_produces_valid_file(self, radar_vol01):
        """Genera un COG desde una grilla warpeada y verifica su validez."""
        lat = float(radar_vol01.latitude['data'][0])
        lon = float(radar_vol01.longitude['data'][0])

        # Crear datos sintéticos y warpear
        ny, nx = 80, 80
        data_2d = np.random.uniform(-10, 60, (ny, nx)).astype(np.float32)

        warped, transform, crs = warp_array_to_mercator(
            data_2d, lat, lon,
            x_limits=(-80000, 80000),
            y_limits=(-80000, 80000)
        )

        # Obtener colormap para DBZH
        cmap, vmin, vmax, _ = colormap_for("DBZH")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.tif"

            result = create_cog_from_warped_array(
                data_warped=warped,
                output_path=output_path,
                transform=transform,
                crs=crs,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            assert output_path.exists(), "COG no fue creado"
            assert output_path.stat().st_size > 0, "COG vacío"

            # Verificar con rasterio
            with rasterio.open(output_path) as src:
                assert src.crs.to_epsg() == 3857, f"CRS incorrecto: {src.crs}"
                assert src.count == 4, f"Esperado 4 bandas (RGBA), got {src.count}"
                assert src.dtypes[0] == 'uint8', f"Tipo inesperado: {src.dtypes[0]}"
                assert src.height > 0
                assert src.width > 0

    def test_cog_alpha_channel(self, radar_vol01):
        """Verifica que el canal alpha enmascara correctamente NaN."""
        lat = float(radar_vol01.latitude['data'][0])
        lon = float(radar_vol01.longitude['data'][0])

        # Datos con NaN (simulan zonas sin datos del radar)
        ny, nx = 60, 60
        data_2d = np.full((ny, nx), np.nan, dtype=np.float32)
        # Solo un cuadrante tiene datos
        data_2d[:30, :30] = np.random.uniform(0, 50, (30, 30))

        warped, transform, crs = warp_array_to_mercator(
            data_2d, lat, lon,
            x_limits=(-60000, 60000),
            y_limits=(-60000, 60000)
        )

        cmap, vmin, vmax, _ = colormap_for("DBZH")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "alpha_test.tif"

            create_cog_from_warped_array(
                data_warped=warped,
                output_path=output_path,
                transform=transform,
                crs=crs,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            with rasterio.open(output_path) as src:
                alpha = src.read(4)  # Banda 4 = alpha
                # Debe haber pixels transparentes (alpha=0) y opacos (alpha=255)
                assert np.any(alpha == 0), "No hay pixels transparentes"
                assert np.any(alpha == 255), "No hay pixels opacos"

    def test_cog_different_colormaps(self, radar_vol01):
        """Verifica que distintos colormaps generan COGs distintos."""
        lat = float(radar_vol01.latitude['data'][0])
        lon = float(radar_vol01.longitude['data'][0])

        data_2d = np.random.uniform(0, 50, (50, 50)).astype(np.float32)
        warped, transform, crs = warp_array_to_mercator(
            data_2d, lat, lon,
            x_limits=(-50000, 50000),
            y_limits=(-50000, 50000)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = {}
            for field in ["DBZH", "RHOHV"]:
                cmap, vmin, vmax, _ = colormap_for(field)
                path = Path(tmpdir) / f"test_{field}.tif"

                create_cog_from_warped_array(
                    data_warped=warped,
                    output_path=path,
                    transform=transform,
                    crs=crs,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax
                )
                files[field] = path

            # Los dos COGs deben existir pero tener contenido RGB diferente
            assert files["DBZH"].exists()
            assert files["RHOHV"].exists()

            with rasterio.open(files["DBZH"]) as a, rasterio.open(files["RHOHV"]) as b:
                rgb_a = a.read(1)
                rgb_b = b.read(1)
                assert not np.array_equal(rgb_a, rgb_b), "Colormaps diferentes deberían generar RGB diferente"
