"""
Tests para app.services.radar_processing.grid_geometry — geometría de grillas.

Qué testea este archivo:
1. beam_height_max_km: Calcula la altura máxima del haz dada una distancia y elevación.
   - Usa la ecuación con radio terrestre efectivo Re = 8.49e6 m.
   - Resultados validados contra valores conocidos de la literatura.

2. compute_beam_height: Calcula la altura del haz con curvatura terrestre (modelo 4/3).
   - Acepta arrays numpy (vectorizado).
   - Para distancias cortas, se asemeja a r*sin(θ).
   - Para distancias largas, la curvatura terrestre agrega altura significativa.

3. calculate_grid_resolution: Resolución XY según volumen.
   - Volumen 03 → 300m (alta resolución, bird bath).
   - Otros volúmenes → 1000m (resolución estándar).

4. calculate_grid_points: Calcula cantidad de puntos en cada eje.
   - Resultado directamente proporcional al tamaño de la grilla / resolución.

5. calculate_roi_dist_beam: Radio de Influencia (ROI) para interpolación.
   - Crece con la distancia al radar (simula ensanchamiento del haz).
   - Tiene un radio mínimo para evitar voxels vacíos cerca del radar.
"""

import pytest
import math
import numpy as np
from app.services.radar_processing.grid_geometry import (
    beam_height_max_km,
    compute_beam_height,
    calculate_z_limits,
    calculate_grid_resolution,
    calculate_grid_points,
    calculate_roi_dist_beam,
)


# ═══════════════════════════════════════════════════════════════════
# beam_height_max_km
# ═══════════════════════════════════════════════════════════════════

class TestBeamHeightMaxKm:
    """
    Calcula h = r*sin(θ) + r²/(2*Re) + h_antena, en km.
    Re = 8.49e6 m (radio terrestre efectivo).
    """

    def test_elevacion_cero(self):
        """
        Con elevación 0°, sin(0)=0, la altura es solo por curvatura terrestre:
        h = 0 + r²/(2*Re)
        Con r=240km: h = (240000)²/(2*8.49e6) = 3393m ≈ 3.39 km
        """
        h = beam_height_max_km(240_000, 0.0)
        assert 3.0 < h < 4.0, f"Esperado ~3.4 km, obtenido {h}"

    def test_elevacion_positiva(self):
        """Con elevación positiva, h crece (componente sin(θ) + curvatura)."""
        h_0 = beam_height_max_km(240_000, 0.0)
        h_5 = beam_height_max_km(240_000, 5.0)
        h_10 = beam_height_max_km(240_000, 10.0)
        
        assert h_5 > h_0, "5° debe dar más altura que 0°"
        assert h_10 > h_5, "10° debe dar más altura que 5°"

    def test_rango_corto(self):
        """Con rango corto (10km), la curvatura es despreciable."""
        h = beam_height_max_km(10_000, 0.0)
        # r²/(2*Re) = (10000)²/(2*8.49e6) ≈ 5.9 m = 0.006 km
        assert h < 0.1, f"Para 10km a 0°, la altura debe ser mínima: {h}"

    def test_con_altura_antena(self):
        """La altura de la antena se suma directamente."""
        h_sin = beam_height_max_km(100_000, 5.0, antenna_alt_m=0)
        h_con = beam_height_max_km(100_000, 5.0, antenna_alt_m=500)
        
        np.testing.assert_allclose(h_con - h_sin, 0.5, atol=0.01)

    def test_resultado_positivo(self):
        """La altura siempre debe ser positiva (o cero con r=0)."""
        for elev in [0, 1, 5, 10, 20]:
            for rango in [10_000, 50_000, 100_000, 240_000]:
                h = beam_height_max_km(rango, elev)
                assert h >= 0, f"Altura negativa para r={rango}, elev={elev}"


# ═══════════════════════════════════════════════════════════════════
# compute_beam_height
# ═══════════════════════════════════════════════════════════════════

class TestComputeBeamHeight:
    """
    Versión vectorizada con modelo 4/3 de radio terrestre.
    Usa la fórmula completa:
    h = sqrt(r² + (ke*Re)² + 2*r*ke*Re*sin(θ)) - ke*Re + h0
    """

    def test_altura_cero_en_origen(self):
        """A distancia 0 del radar, la altura es 0 (o la del radar)."""
        h = compute_beam_height(np.array([0.0]), 5.0, radar_altitude=0.0)
        np.testing.assert_allclose(h, 0.0, atol=1.0)

    def test_crece_con_distancia(self):
        """La altura crece monótonamente con la distancia."""
        distancias = np.array([10_000, 50_000, 100_000, 200_000], dtype=float)
        h = compute_beam_height(distancias, 2.0)
        
        for i in range(1, len(h)):
            assert h[i] > h[i-1], f"h[{i}]={h[i]} no es mayor que h[{i-1}]={h[i-1]}"

    def test_crece_con_elevacion(self):
        """A misma distancia, más elevación → más altura."""
        d = np.array([100_000.0])
        h_low = compute_beam_height(d, 1.0)
        h_high = compute_beam_height(d, 10.0)
        assert h_high > h_low

    def test_acepta_escalar(self):
        """Funciona con un solo valor (no solo arrays)."""
        h = compute_beam_height(np.array([50_000.0]), 5.0)
        assert h.shape == (1,)
        assert h[0] > 0

    def test_radar_altitude_se_suma(self):
        """La altitud del radar se suma al resultado."""
        d = np.array([50_000.0])
        h_0 = compute_beam_height(d, 5.0, radar_altitude=0)
        h_500 = compute_beam_height(d, 5.0, radar_altitude=500)
        np.testing.assert_allclose(h_500 - h_0, 500.0, atol=1.0)

    def test_distancia_100km_elevacion_1_grado(self):
        """
        Valor de referencia: a 100km con 1° de elevación,
        la altura del haz es ~2.4 km (con curvatura).
        (Doviak & Zrnić, tabla estándar)
        """
        h = compute_beam_height(np.array([100_000.0]), 1.0)
        assert 1.5 < h[0] / 1000 < 3.5, f"Valor fuera de rango esperado: {h[0]/1000} km"


# ═══════════════════════════════════════════════════════════════════
# calculate_grid_resolution
# ═══════════════════════════════════════════════════════════════════

class TestCalculateGridResolution:
    """
    La resolución depende del volumen:
    - '03' (bird bath): 300m horizontal (alta resolución, alcance corto)
    - Cualquier otro: 1000m horizontal (resolución estándar)
    - Z siempre: 600m (resolución fina para transectos suaves)
    """

    def test_volumen_03_alta_resolucion(self):
        xy, z = calculate_grid_resolution('03')
        assert xy == 300
        assert z == 600

    def test_volumen_01_resolucion_estandar(self):
        xy, z = calculate_grid_resolution('01')
        assert xy == 1000
        assert z == 600

    def test_volumen_02(self):
        xy, z = calculate_grid_resolution('02')
        assert xy == 1000

    def test_volumen_04(self):
        xy, z = calculate_grid_resolution('04')
        assert xy == 1000

    def test_volumen_none(self):
        """Sin volumen, usa resolución estándar."""
        xy, z = calculate_grid_resolution(None)
        assert xy == 1000

    def test_z_siempre_600(self):
        """La resolución Z no cambia nunca."""
        for vol in ['01', '02', '03', '04', None]:
            _, z = calculate_grid_resolution(vol)
            assert z == 600


# ═══════════════════════════════════════════════════════════════════
# calculate_grid_points
# ═══════════════════════════════════════════════════════════════════

class TestCalculateGridPoints:
    """
    Calcula cuántos puntos de grilla caben dados límites y resolución.
    """

    def test_caso_simple(self):
        """100km de rango con 1km de resolución → ~201 puntos por eje."""
        nz, ny, nx = calculate_grid_points(
            z_limits=(0, 10_000),       # 10km de alto
            y_limits=(-100_000, 100_000),  # 200km de ancho
            x_limits=(-100_000, 100_000),
            resolution_xy=1000,
            resolution_z=600
        )
        
        # Z: ceil(10000/600) + 1 = 17 + 1 = 18
        assert nz == 18
        # Y: (100000 - (-100000)) / 1000 + 1 = 201
        assert ny == 201
        # X: igual que Y
        assert nx == 201

    def test_volumen_03_mas_puntos(self):
        """Con resolución 300m, hay más puntos que con 1000m."""
        nz, ny_300, nx_300 = calculate_grid_points(
            z_limits=(0, 10_000),
            y_limits=(-50_000, 50_000),
            x_limits=(-50_000, 50_000),
            resolution_xy=300,
            resolution_z=600
        )
        _, ny_1000, nx_1000 = calculate_grid_points(
            z_limits=(0, 10_000),
            y_limits=(-50_000, 50_000),
            x_limits=(-50_000, 50_000),
            resolution_xy=1000,
            resolution_z=600
        )
        
        assert ny_300 > ny_1000
        assert nx_300 > nx_1000

    def test_resultado_positivo(self):
        """Siempre debe devolver al menos 1 punto por eje."""
        nz, ny, nx = calculate_grid_points(
            z_limits=(0, 600),
            y_limits=(0, 1000),
            x_limits=(0, 1000),
            resolution_xy=1000,
            resolution_z=600
        )
        assert nz >= 1
        assert ny >= 1
        assert nx >= 1


# ═══════════════════════════════════════════════════════════════════
# calculate_z_limits
# ═══════════════════════════════════════════════════════════════════

class TestCalculateZLimits:
    """
    Calcula los límites verticales (z_min, z_max) de la grilla.
    z_min siempre es 0. z_max depende de la elevación y rango.
    """

    def test_z_min_siempre_cero(self):
        z_min, z_max, _ = calculate_z_limits(
            240_000, elevation=0,
            radar_fixed_angles=np.array([0.5, 1.0, 2.0])
        )
        assert z_min == 0.0

    def test_devuelve_elevacion_en_grados(self):
        """Devuelve el ángulo de elevación usado."""
        _, _, elev = calculate_z_limits(
            240_000, elevation=1,
            radar_fixed_angles=np.array([0.5, 1.5, 3.0])
        )
        assert elev == 1.5

    def test_z_max_crece_con_elevacion(self):
        """Mayor elevación → z_max más alto."""
        angles = np.array([0.5, 2.0, 5.0, 10.0])
        _, z_max_low, _ = calculate_z_limits(240_000, elevation=0,
                                             radar_fixed_angles=angles)
        _, z_max_high, _ = calculate_z_limits(240_000, elevation=3,
                                              radar_fixed_angles=angles)
        assert z_max_high > z_max_low

    def test_sin_fixed_angles_lanza_error(self):
        """Sin radar_fixed_angles, lanza ValueError."""
        with pytest.raises(ValueError, match="radar_fixed_angles requerido"):
            calculate_z_limits(240_000, elevation=0, radar_fixed_angles=None)


# ═══════════════════════════════════════════════════════════════════
# calculate_roi_dist_beam
# ═══════════════════════════════════════════════════════════════════

class TestCalculateROIDistBeam:
    """
    Radio de Influencia (ROI) usando el método dist_beam.
    
    Fórmula: ROI = max(h_factor*(z/20) + dist_xy*tan(nb*bsp*π/180), min_radius)
    
    El ROI simula el ensanchamiento del haz con la distancia.
    """

    def test_roi_minimo_en_origen(self):
        """En el origen (0,0,0), el ROI es el mínimo."""
        roi = calculate_roi_dist_beam(0.0, 0.0, 0.0, min_radius=300.0)
        assert roi == pytest.approx(300.0)

    def test_roi_crece_con_distancia(self):
        """Más lejos del radar → mayor ROI."""
        roi_cerca = calculate_roi_dist_beam(1000.0, 0.0, 1000.0)
        roi_lejos = calculate_roi_dist_beam(1000.0, 0.0, 50_000.0)
        assert roi_lejos > roi_cerca

    def test_roi_crece_con_altura(self):
        """Más alto → mayor ROI (componente vertical)."""
        roi_bajo = calculate_roi_dist_beam(0.0, 1000.0, 1000.0)
        roi_alto = calculate_roi_dist_beam(10_000.0, 1000.0, 1000.0)
        assert roi_alto > roi_bajo

    def test_roi_nunca_menor_que_min_radius(self):
        """El ROI nunca es menor que min_radius."""
        for z in [0, 100, 500]:
            for y in [0, 100]:
                for x in [0, 100]:
                    roi = calculate_roi_dist_beam(
                        float(z), float(y), float(x), min_radius=500.0
                    )
                    assert roi >= 500.0

    def test_acepta_arrays(self):
        """Funciona con arrays numpy (vectorizado)."""
        z = np.array([0.0, 5000.0, 20_000.0])
        y = np.array([0.0, 30_000.0, 100_000.0])
        x = np.array([0.0, 30_000.0, 100_000.0])
        
        roi = calculate_roi_dist_beam(z, y, x)
        assert roi.shape == (3,)
        # Cada ROI sucesivo debe ser mayor (con valores suficientemente
        # grandes para superar el min_radius en todos los casos)
        assert roi[2] > roi[1] > roi[0]

    def test_parametros_volumen_03(self):
        """Volumen 03 tiene ROI mucho más grande (bird bath)."""
        # Parámetros de vol 03: (5.0, 3.0, 2.5, 15000.0)
        roi_vol03 = calculate_roi_dist_beam(
            5000.0, 10_000.0, 10_000.0,
            h_factor=5.0, nb=3.0, bsp=2.5, min_radius=15_000.0
        )
        # Parámetros de vol 01: (0.9, 1.2, 1.0, 700.0)
        roi_vol01 = calculate_roi_dist_beam(
            5000.0, 10_000.0, 10_000.0,
            h_factor=0.9, nb=1.2, bsp=1.0, min_radius=700.0
        )
        
        assert roi_vol03 > roi_vol01, "Vol 03 debería tener ROI mucho mayor que vol 01"
