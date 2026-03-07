"""
Integration tests: lectura de radar y resolución de campos.

Verifica que PyART lee correctamente los archivos NetCDF reales
y que la resolución de campos (aliases) funciona con datos reales.
"""

import pytest
import numpy as np
import pyart

from app.services.radar_common import resolve_field, safe_range_max_m, get_radar_site, colormap_for


class TestRadarRead:
    """Lectura de archivos NetCDF reales con PyART."""

    def test_read_vol01_creates_radar_object(self, radar_vol01):
        assert isinstance(radar_vol01, pyart.core.Radar)

    def test_radar_has_fields(self, radar_vol01):
        assert len(radar_vol01.fields) > 0

    def test_radar_has_sweeps(self, radar_vol01):
        assert radar_vol01.nsweeps > 0

    def test_radar_has_gates(self, radar_vol01):
        assert radar_vol01.ngates > 0
        assert radar_vol01.nrays > 0

    def test_radar_has_coordinates(self, radar_vol01):
        lat = float(radar_vol01.latitude['data'][0])
        lon = float(radar_vol01.longitude['data'][0])
        # RMA1 está en Córdoba, Argentina: lat ~ -31, lon ~ -64
        assert -35 < lat < -28, f"Latitud inesperada: {lat}"
        assert -67 < lon < -60, f"Longitud inesperada: {lon}"

    def test_radar_has_fixed_angle(self, radar_vol01):
        angles = radar_vol01.fixed_angle['data']
        assert len(angles) == radar_vol01.nsweeps
        # Elevaciones positivas (radar apunta hacia arriba)
        assert all(a >= 0 for a in angles)


class TestResolveFieldReal:
    """resolve_field con objetos radar reales."""

    def test_resolve_dbzh(self, radar_vol01):
        field_to_use, field_key = resolve_field(radar_vol01, "DBZH")
        assert field_key == "DBZH"
        # El campo resuelto debe existir en el radar
        assert field_to_use in radar_vol01.fields

    def test_resolve_rhohv(self, radar_vol01):
        try:
            field_to_use, field_key = resolve_field(radar_vol01, "RHOHV")
            assert field_to_use in radar_vol01.fields
        except KeyError:
            pytest.skip("RHOHV no disponible en este archivo")

    def test_resolve_zdr(self, radar_vol01):
        try:
            field_to_use, field_key = resolve_field(radar_vol01, "ZDR")
            assert field_to_use in radar_vol01.fields
        except KeyError:
            pytest.skip("ZDR no disponible en este archivo")

    def test_resolve_inexistente_raises(self, radar_vol01):
        with pytest.raises(KeyError):
            resolve_field(radar_vol01, "CAMPO_INEXISTENTE")

    def test_dbzh_data_has_valid_range(self, radar_vol01):
        field_to_use, _ = resolve_field(radar_vol01, "DBZH")
        data = radar_vol01.fields[field_to_use]['data']
        # DBZH válido debe estar en rango -40 a 80 dBZ
        valid = data.compressed()  # solo valores no-masked
        if len(valid) > 0:
            assert valid.min() > -60, f"DBZH mínimo demasiado bajo: {valid.min()}"
            assert valid.max() < 100, f"DBZH máximo demasiado alto: {valid.max()}"


class TestRadarSiteInfo:
    """Info del sitio del radar con datos reales."""

    def test_get_radar_site(self, radar_vol01):
        lon, lat, alt = get_radar_site(radar_vol01)
        assert -67 < lon < -60
        assert -35 < lat < -28
        assert 0 < alt < 3000  # altitud en metros, razonable

    def test_safe_range_max(self, radar_vol01):
        range_max = safe_range_max_m(radar_vol01)
        # Alcance de radar meteorológico: entre 50 km y 500 km
        assert 50_000 < range_max < 500_000, f"Alcance inesperado: {range_max}m"


class TestColormapResolution:
    """Resolución de colormaps con campos del radar."""

    def test_colormap_for_dbzh(self):
        cmap, vmin, vmax, cmap_key = colormap_for("DBZH")
        assert vmin == -30.0
        assert vmax == 70.0
        assert callable(cmap)  # matplotlib colormap

    def test_colormap_for_rhohv(self):
        cmap, vmin, vmax, cmap_key = colormap_for("RHOHV")
        assert vmin == 0.0
        assert vmax == 1.0

    def test_colormap_for_override(self):
        cmap, vmin, vmax, cmap_key = colormap_for("DBZH", override_cmap="pyart_HomeyerRainbow")
        assert cmap_key == "pyart_HomeyerRainbow"


class TestVolume03Differences:
    """Verifica que volumen 03 (bird bath) difiere del 01."""

    def test_vol03_has_different_elevations(self, radar_vol01, radar_vol03):
        angles_01 = radar_vol01.fixed_angle['data']
        angles_03 = radar_vol03.fixed_angle['data']
        # Vol 03 tiene elevaciones más altas o distintas
        assert not np.array_equal(angles_01, angles_03)

    def test_vol03_has_fewer_gates_or_different_range(self, radar_vol01, radar_vol03):
        # Vol 03 suele tener alcance más corto
        range01 = safe_range_max_m(radar_vol01)
        range03 = safe_range_max_m(radar_vol03)
        # No necesariamente menor, pero distinto
        assert range01 != range03 or radar_vol01.ngates != radar_vol03.ngates
