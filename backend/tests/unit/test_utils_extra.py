"""
Tests para funciones utilitarias adicionales — elevation_profile y radar_common.

Qué testea este archivo:
1. haversine_distance: Distancia entre dos puntos geográficos usando fórmula de Haversine.
   - Validada contra distancia conocida Córdoba → Buenos Aires ≈ 650 km.
   - Distancia de un punto a sí mismo = 0.

2. interpolate_line_points: Genera puntos intermedios entre coordenadas.
   - Más points_per_km → más puntos generados.
   - Siempre incluye el último punto.

3. limit_line_to_range: Limita el largo de una línea geodésica.
   - Si la línea ya es más corta que el máximo → no cambia.
   - Si es más larga → la recorta al máximo.
"""

import pytest
import numpy as np
from app.services.elevation_profile import (
    haversine_distance,
    interpolate_line_points,
)
from app.services.radar_common import limit_line_to_range


# ═══════════════════════════════════════════════════════════════════
# haversine_distance
# ═══════════════════════════════════════════════════════════════════

class TestHaversineDistance:
    """
    Distancia en km entre dos puntos usando fórmula de Haversine.
    Usa radio terrestre = 6371 km.
    """

    def test_mismo_punto_distancia_cero(self):
        """Un punto a sí mismo tiene distancia 0."""
        d = haversine_distance(-64.0, -31.0, -64.0, -31.0)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_cordoba_buenos_aires(self):
        """
        Córdoba (-31.42, -64.18) → Buenos Aires (-34.60, -58.38).
        Distancia real ≈ 645-700 km.
        """
        d = haversine_distance(-64.18, -31.42, -58.38, -34.60)
        assert 600 < d < 750, f"Distancia Córdoba-BsAs fuera de rango: {d} km"

    def test_simetria(self):
        """La distancia es simétrica: d(A,B) = d(B,A)."""
        d1 = haversine_distance(-64.0, -31.0, -58.0, -34.0)
        d2 = haversine_distance(-58.0, -34.0, -64.0, -31.0)
        assert d1 == pytest.approx(d2, rel=1e-10)

    def test_distancia_siempre_positiva(self):
        """La distancia siempre es >= 0."""
        d = haversine_distance(0, 0, 180, 0)  # antípodas
        assert d > 0

    def test_distancia_maxima_antipodas(self):
        """Antípodas = máxima distancia ≈ π * 6371 ≈ 20015 km."""
        d = haversine_distance(0, 0, 180, 0)
        assert 19_900 < d < 20_100


# ═══════════════════════════════════════════════════════════════════
# interpolate_line_points
# ═══════════════════════════════════════════════════════════════════

class TestInterpolateLinePoints:
    """
    Genera puntos intermedios entre coordenadas para mayor resolución.
    """

    def test_dos_puntos_genera_interpolados(self):
        coords = [
            {"lat": -31.0, "lon": -64.0},
            {"lat": -32.0, "lon": -64.0},
        ]
        result = interpolate_line_points(coords, points_per_km=1)
        
        # ~111 km entre 1° lat → debería generar ~111 puntos
        assert len(result) > 10

    def test_incluye_ultimo_punto(self):
        """Siempre incluye el punto final."""
        coords = [
            {"lat": -31.0, "lon": -64.0},
            {"lat": -32.0, "lon": -64.0},
        ]
        result = interpolate_line_points(coords, points_per_km=1)
        
        assert result[-1]["lat"] == pytest.approx(-32.0)
        assert result[-1]["lon"] == pytest.approx(-64.0)

    def test_mas_points_per_km_mas_puntos(self):
        coords = [
            {"lat": -31.0, "lon": -64.0},
            {"lat": -31.5, "lon": -64.0},
        ]
        r1 = interpolate_line_points(coords, points_per_km=1)
        r10 = interpolate_line_points(coords, points_per_km=10)
        
        assert len(r10) > len(r1)

    def test_un_solo_punto(self):
        """Con un solo punto, devuelve ese punto."""
        coords = [{"lat": -31.0, "lon": -64.0}]
        result = interpolate_line_points(coords)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════
# limit_line_to_range
# ═══════════════════════════════════════════════════════════════════

class TestLimitLineToRange:
    """
    Limita una línea geodésica a una distancia máxima.
    Si la línea es más corta → no cambia.
    Si es más larga → la recorta al máximo.
    """

    def test_linea_corta_no_cambia(self):
        """Una línea de ~111km con max 200km no se modifica."""
        lon0, lat0 = -64.0, -31.0
        lon1, lat1 = -64.0, -32.0  # ~111 km
        
        lon_r, lat_r, length_km = limit_line_to_range(
            lon0, lat0, lon1, lat1, max_len_km=200.0
        )
        
        assert lon_r == pytest.approx(lon1, abs=0.001)
        assert lat_r == pytest.approx(lat1, abs=0.001)
        assert length_km == pytest.approx(111.0, rel=0.05)

    def test_linea_larga_se_recorta(self):
        """Una línea de ~111km con max 50km se recorta."""
        lon0, lat0 = -64.0, -31.0
        lon1, lat1 = -64.0, -32.0  # ~111 km
        
        lon_r, lat_r, length_km = limit_line_to_range(
            lon0, lat0, lon1, lat1, max_len_km=50.0
        )
        
        assert length_km == pytest.approx(50.0, rel=0.01)
        # El punto resultante está entre el inicio y el final
        assert lat_r > lat1  # más cerca de lat0 (-31)
        assert lat_r < lat0

    def test_devuelve_max_len_km_exacto(self):
        """Cuando se recorta, la longitud devuelta es exactamente max_len_km."""
        _, _, length_km = limit_line_to_range(
            -64.0, -31.0, -64.0, -35.0, max_len_km=100.0
        )
        assert length_km == pytest.approx(100.0, rel=0.001)
