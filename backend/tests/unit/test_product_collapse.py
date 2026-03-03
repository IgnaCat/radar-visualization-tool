"""
Tests para app.services.radar_processing.product_collapse — colapso 3D → 2D.

Qué testea este archivo:
1. collapse_cappi: Slice horizontal a altura constante.
   - Si la altura coincide con un nivel Z exacto → toma ese nivel directo.
   - Si está entre dos niveles → interpola linealmente.
   - Si está por debajo/arriba de la grilla → usa el nivel más cercano.

2. collapse_colmax: Máximo en columna vertical.
   - Para cada pixel (y,x), toma el valor máximo entre todos los niveles Z.

3. collapse_ppi: Sigue el haz del radar a elevación constante.
   - Calcula la altura del haz con curvatura terrestre (4/3 modelo).
   - Interpola linealmente entre niveles Z.
   - Enmascara puntos fuera de la grilla vertical.
"""

import pytest
import numpy as np
from app.services.radar_processing.product_collapse import (
    collapse_cappi,
    collapse_colmax,
    collapse_ppi,
)


# ═══════════════════════════════════════════════════════════════════
# collapse_cappi
# ═══════════════════════════════════════════════════════════════════

class TestCollapseCAPPI:
    """
    CAPPI (Constant Altitude Plan Position Indicator):
    Corta la grilla 3D a una altura fija.
    
    Con z_coords = [0, 1000, 2000, 3000] y datos donde nivel_i = (i+1)*10:
    - Nivel 0 (0m):    todos 10.0
    - Nivel 1 (1000m): todos 20.0
    - Nivel 2 (2000m): todos 30.0
    - Nivel 3 (3000m): todos 40.0
    """

    def test_altura_exacta_en_nivel(self, simple_3d_array, z_coords_4levels):
        """Si pido exactamente 2000m, toma el nivel 2 directo (valor=30)."""
        result = collapse_cappi(simple_3d_array, z_coords_4levels, 2000.0)
        
        assert result.shape == (5, 6)
        # Nivel 2 tiene valor 30.0 en todos los pixeles
        np.testing.assert_allclose(result, 30.0)

    def test_altura_exacta_nivel_0(self, simple_3d_array, z_coords_4levels):
        """Nivel más bajo (0m) → valor 10.0."""
        result = collapse_cappi(simple_3d_array, z_coords_4levels, 0.0)
        np.testing.assert_allclose(result, 10.0)

    def test_interpolacion_entre_niveles(self, simple_3d_array, z_coords_4levels):
        """
        A 500m (entre nivel 0 y nivel 1):
        - z_frac = (500 - 0) / 1000 = 0.5
        - weight_low = 0.5, weight_high = 0.5
        - resultado = 0.5 * 10.0 + 0.5 * 20.0 = 15.0
        """
        result = collapse_cappi(simple_3d_array, z_coords_4levels, 500.0)
        np.testing.assert_allclose(result, 15.0)

    def test_interpolacion_un_cuarto(self, simple_3d_array, z_coords_4levels):
        """
        A 1250m (entre nivel 1=1000m y nivel 2=2000m):
        - z_frac = (1250 - 0) / 1000 = 1.25
        - z_low=1, z_high=2
        - weight_high = 0.25, weight_low = 0.75
        - resultado = 0.75 * 20.0 + 0.25 * 30.0 = 22.5
        """
        result = collapse_cappi(simple_3d_array, z_coords_4levels, 1250.0)
        np.testing.assert_allclose(result, 22.5)

    def test_debajo_de_grilla(self, simple_3d_array, z_coords_4levels):
        """
        Altura negativa (debajo de la grilla):
        → Devuelve el nivel más bajo (nivel 0, valor 10.0).
        """
        result = collapse_cappi(simple_3d_array, z_coords_4levels, -500.0)
        np.testing.assert_allclose(result, 10.0)

    def test_arriba_de_grilla(self, simple_3d_array, z_coords_4levels):
        """
        Altura 5000m (arriba de la grilla que llega hasta 3000m):
        → Devuelve el nivel más alto (nivel 3, valor 40.0).
        """
        result = collapse_cappi(simple_3d_array, z_coords_4levels, 5000.0)
        np.testing.assert_allclose(result, 40.0)

    def test_shape_resultado(self, simple_3d_array, z_coords_4levels):
        """El resultado siempre es 2D con shape (ny, nx)."""
        result = collapse_cappi(simple_3d_array, z_coords_4levels, 1000.0)
        assert result.ndim == 2
        assert result.shape == (5, 6)

    def test_con_masked_array(self, z_coords_4levels):
        """Funciona con masked arrays (datos faltantes)."""
        data = np.ma.array(
            np.ones((4, 3, 3)) * 10.0,
            mask=np.zeros((4, 3, 3), dtype=bool)
        )
        # Enmascarar un pixel en nivel 0
        data.mask[0, 1, 1] = True
        
        result = collapse_cappi(data, z_coords_4levels, 0.0)
        # El pixel enmascarado sigue enmascarado
        assert np.ma.is_masked(result[1, 1])


# ═══════════════════════════════════════════════════════════════════
# collapse_colmax
# ═══════════════════════════════════════════════════════════════════

class TestCollapseCOLMAX:
    """
    COLMAX (Column Maximum): Para cada columna vertical, toma el valor máximo.
    Es el producto más simple — equivale a np.nanmax(data, axis=0).
    """

    def test_maximo_por_columna(self, simple_3d_array):
        """
        Con datos crecientes por nivel [10,20,30,40],
        el máximo siempre es el nivel más alto: 40.0.
        """
        result = collapse_colmax(simple_3d_array)
        np.testing.assert_allclose(result, 40.0)

    def test_valores_distintos_por_columna(self):
        """Cada columna puede tener un máximo diferente."""
        data = np.ma.array([
            [[1.0, 5.0],    # Z=0
             [3.0, 2.0]],
            [[4.0, 1.0],    # Z=1
             [2.0, 6.0]],
        ])
        result = collapse_colmax(data)
        expected = np.array([[4.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_shape_resultado(self, simple_3d_array):
        """Resultado es 2D (ny, nx)."""
        result = collapse_colmax(simple_3d_array)
        assert result.shape == (5, 6)

    def test_con_nan(self):
        """NaN se ignora (usa np.nanmax)."""
        data = np.ma.array([
            [[1.0, np.nan], [3.0, 2.0]],
            [[4.0, 1.0], [np.nan, 6.0]],
        ])
        result = collapse_colmax(data)
        assert result[0, 0] == 4.0  # max(1, 4) = 4
        assert result[0, 1] == 1.0  # max(nan, 1) = 1
        assert result[1, 1] == 6.0  # max(2, 6) = 6

    def test_valores_negativos(self):
        """Funciona con reflectividades negativas (dBZ)."""
        data = np.ma.array([
            [[-30.0, -10.0]],
            [[-20.0, -5.0]],
        ])
        result = collapse_colmax(data)
        np.testing.assert_array_equal(result, [[-20.0, -5.0]])


# ═══════════════════════════════════════════════════════════════════
# collapse_ppi
# ═══════════════════════════════════════════════════════════════════

class TestCollapsePPI:
    """
    PPI (Plan Position Indicator): Sigue el haz del radar a elevación constante.
    
    Para cada pixel (x,y), calcula la altura del haz usando el modelo 4/3
    de radio terrestre y luego interpola entre niveles Z.
    
    Nota: Al ser físicamente más complejo que CAPPI, los tests verifican
    propiedades generales en vez de valores exactos.
    """

    def test_shape_resultado(self, simple_3d_array, z_coords_4levels, xy_coords_5x6):
        """Resultado es 2D con shape (ny, nx)."""
        x, y = xy_coords_5x6
        result = collapse_ppi(simple_3d_array, z_coords_4levels, x, y, 1.0)
        assert result.shape == (5, 6)

    def test_elevation_cero_cerca_del_radar(self):
        """
        Con elevación 0° y coordenadas cerca del radar (distancia pequeña),
        la altura del haz es ~0m → debería tomar valores del nivel más bajo.
        """
        data = np.ma.array(np.ones((4, 3, 3)) * 0.0)
        data[0, :, :] = 10.0  # Nivel 0 (más bajo) = 10
        data[1, :, :] = 20.0
        data[2, :, :] = 30.0
        data[3, :, :] = 40.0
        
        z = np.array([0.0, 1000.0, 2000.0, 3000.0])
        # Coordenadas muy cerca del radar (100m)
        x = np.array([-100.0, 0.0, 100.0])
        y = np.array([-100.0, 0.0, 100.0])
        
        result = collapse_ppi(data, z, x, y, 0.5)  # 0.5° de elevación
        
        # Cerca del radar con elevación baja, debe estar cerca del nivel 0
        center = result[1, 1]
        if not np.ma.is_masked(center):
            # El valor debe estar cerca de 10.0 (nivel más bajo)
            assert center < 15.0, f"Esperado < 15 cerca del radar, obtenido {center}"

    def test_elevacion_alta_valores_mayores(self):
        """
        Con elevación alta, el haz sube rápido → toma valores de niveles
        más altos que con elevación baja.
        """
        data = np.ma.array(np.zeros((4, 3, 3)))
        for iz in range(4):
            data[iz, :, :] = (iz + 1) * 10.0
        
        z = np.array([0.0, 2000.0, 4000.0, 6000.0])
        x = np.array([-5000.0, 0.0, 5000.0])
        y = np.array([-5000.0, 0.0, 5000.0])
        
        result_low = collapse_ppi(data, z, x, y, 1.0)   # 1° elevación
        result_high = collapse_ppi(data, z, x, y, 10.0)  # 10° elevación
        
        # Con elevación alta, los valores promedio deben ser mayores
        # (toma niveles Z más altos)
        mean_low = np.nanmean(result_low)
        mean_high = np.nanmean(result_high)
        
        # Si hay datos válidos en ambos, el de mayor elevación debería
        # tener valores más altos (o al menos iguales)
        if not np.isnan(mean_low) and not np.isnan(mean_high):
            assert mean_high >= mean_low, (
                f"Elevación alta debería dar valores >= que baja: "
                f"high={mean_high}, low={mean_low}"
            )

    def test_resultado_es_masked_array(self, simple_3d_array, z_coords_4levels, xy_coords_5x6):
        """El resultado siempre es un masked array (para enmascarar fuera de grilla)."""
        x, y = xy_coords_5x6
        result = collapse_ppi(simple_3d_array, z_coords_4levels, x, y, 1.0)
        assert isinstance(result, np.ma.MaskedArray)

    def test_simetria_respecto_al_origen(self):
        """
        Con datos simétricos y elevación constante, el resultado debe ser
        simétrico respecto al radar (origen).
        """
        data = np.ma.array(np.ones((4, 5, 5)) * 20.0)
        z = np.array([0.0, 1000.0, 2000.0, 3000.0])
        # Coordenadas perfectamente simétricas
        x = np.array([-2000.0, -1000.0, 0.0, 1000.0, 2000.0])
        y = np.array([-2000.0, -1000.0, 0.0, 1000.0, 2000.0])
        
        result = collapse_ppi(data, z, x, y, 2.0)
        
        # Con datos uniformes, el resultado también debería ser ~uniforme
        valid = result.compressed()  # solo valores no-enmascarados
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 20.0, atol=1e-3)
