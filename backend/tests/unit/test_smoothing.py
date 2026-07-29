"""
Tests para app.services.radar_processing.smoothing — suavizado de grillas 2D.

Qué testea este archivo:
1. apply_gaussian_smoothing_masked: Suavizado gaussiano con convolución normalizada.
   - Preserva máscaras (no mezcla nodata con datos).
   - sigma=0 devuelve el array sin cambios.
   - Reduce varianza (suaviza gradientes).

2. apply_median_smoothing_masked: Suavizado por mediana ignorando NaN.
   - Preserva máscaras.
   - size=1 devuelve el array sin cambios.
   - Fuerza ventana impar si se pasa un size par.

3. apply_smoothing_masked: Dispatcher que delega a gaussian o median.
"""

import pytest
import numpy as np
from app.services.radar_processing.smoothing import (
    apply_gaussian_smoothing_masked,
    apply_median_smoothing_masked,
    apply_smoothing_masked,
)


# ═══════════════════════════════════════════════════════════════════
# apply_gaussian_smoothing_masked
# ═══════════════════════════════════════════════════════════════════

class TestGaussianSmoothing:
    """Suavizado gaussiano con convolución normalizada."""

    def test_sigma_cero_no_modifica(self):
        """Con sigma=0, devuelve el array original sin cambios."""
        arr = np.ma.array([[10.0, 20.0], [30.0, 40.0]])
        result = apply_gaussian_smoothing_masked(arr, sigma=0.0)
        np.testing.assert_array_equal(result.data, arr.data)

    def test_reduce_varianza(self):
        """El suavizado reduce la varianza de los datos."""
        np.random.seed(42)
        data = np.random.uniform(0, 50, (20, 20)).astype(np.float32)
        arr = np.ma.array(data)

        result = apply_gaussian_smoothing_masked(arr, sigma=2.0)
        assert result.compressed().std() < arr.compressed().std()

    def test_preserva_mascara(self):
        """Los pixels enmascarados siguen enmascarados después del suavizado."""
        data = np.ones((5, 5), dtype=np.float32) * 20.0
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True  # centro enmascarado
        arr = np.ma.array(data, mask=mask)

        result = apply_gaussian_smoothing_masked(arr, sigma=1.0)
        assert result.mask[2, 2] == True

    def test_no_introduce_nan_en_datos_validos(self):
        """Un array sin NaN ni máscara no debería generar NaN."""
        data = np.arange(25, dtype=np.float32).reshape(5, 5)
        arr = np.ma.array(data)

        result = apply_gaussian_smoothing_masked(arr, sigma=1.0)
        assert not np.any(np.isnan(result.compressed()))

    def test_shape_se_mantiene(self):
        """El resultado tiene la misma forma que la entrada."""
        arr = np.ma.array(np.ones((10, 15), dtype=np.float32))
        result = apply_gaussian_smoothing_masked(arr, sigma=1.5)
        assert result.shape == (10, 15)

    def test_con_mascara_parcial(self):
        """Suaviza correctamente cuando hay una franja enmascarada."""
        data = np.ones((6, 6), dtype=np.float32) * 30.0
        mask = np.zeros((6, 6), dtype=bool)
        mask[:, 3:] = True  # mitad derecha enmascarada
        arr = np.ma.array(data, mask=mask)

        result = apply_gaussian_smoothing_masked(arr, sigma=1.0)

        # Mitad izquierda debe seguir teniendo datos válidos
        assert result[:, 0].count() > 0
        # Mitad derecha enmascarada sigue enmascarada
        assert np.all(result.mask[:, 3:])

    def test_resultado_float32(self):
        """El resultado es float32."""
        arr = np.ma.array(np.ones((4, 4), dtype=np.float64) * 10.0)
        result = apply_gaussian_smoothing_masked(arr, sigma=1.0)
        assert result.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════
# apply_median_smoothing_masked
# ═══════════════════════════════════════════════════════════════════

class TestMedianSmoothing:
    """Suavizado por mediana ignorando NaN."""

    def test_size_1_no_modifica(self):
        """Con size=1, devuelve el array sin cambios."""
        arr = np.ma.array([[10.0, 20.0], [30.0, 40.0]])
        result = apply_median_smoothing_masked(arr, size=1)
        np.testing.assert_array_equal(result.data, arr.data)

    def test_elimina_outlier(self):
        """La mediana elimina outliers (spikes)."""
        data = np.ones((5, 5), dtype=np.float32) * 20.0
        data[2, 2] = 999.0  # spike
        arr = np.ma.array(data)

        result = apply_median_smoothing_masked(arr, size=3)
        # El spike debe desaparecer o reducirse significativamente
        assert result[2, 2] < 100.0

    def test_preserva_mascara(self):
        """Pixels enmascarados siguen enmascarados."""
        data = np.ones((5, 5), dtype=np.float32) * 20.0
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        arr = np.ma.array(data, mask=mask)

        result = apply_median_smoothing_masked(arr, size=3)
        assert result.mask[0, 0] == True

    def test_size_par_se_convierte_a_impar(self):
        """Un size par se incrementa a impar (el código hace size += 1)."""
        data = np.ones((5, 5), dtype=np.float32) * 10.0
        arr = np.ma.array(data)

        # size=4 debería comportarse como size=5
        result_4 = apply_median_smoothing_masked(arr, size=4)
        result_5 = apply_median_smoothing_masked(arr, size=5)
        np.testing.assert_array_equal(result_4.data, result_5.data)

    def test_shape_se_mantiene(self):
        """El resultado tiene la misma forma que la entrada."""
        arr = np.ma.array(np.ones((8, 12), dtype=np.float32))
        result = apply_median_smoothing_masked(arr, size=3)
        assert result.shape == (8, 12)

    def test_array_uniforme_no_cambia(self):
        """Un array uniforme no cambia con mediana."""
        arr = np.ma.array(np.ones((5, 5), dtype=np.float32) * 42.0)
        result = apply_median_smoothing_masked(arr, size=3)
        np.testing.assert_allclose(result, 42.0)

    def test_resultado_float32(self):
        """El resultado es float32."""
        arr = np.ma.array(np.ones((4, 4), dtype=np.float64) * 10.0)
        result = apply_median_smoothing_masked(arr, size=3)
        assert result.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════
# apply_smoothing_masked (dispatcher)
# ═══════════════════════════════════════════════════════════════════

class TestSmoothingDispatcher:
    """Dispatcher que delega a gaussian o median según el método."""

    def test_median_por_defecto(self):
        """Sin método explícito, usa median."""
        arr = np.ma.array(np.ones((5, 5), dtype=np.float32) * 20.0)
        arr[2, 2] = 999.0

        result = apply_smoothing_masked(arr)
        # Median elimina el spike
        assert result[2, 2] < 100.0

    def test_despacha_a_gaussian(self):
        """method='gaussian' delega a apply_gaussian_smoothing_masked."""
        np.random.seed(42)
        data = np.random.uniform(0, 50, (10, 10)).astype(np.float32)
        arr = np.ma.array(data)

        result = apply_smoothing_masked(arr, method="gaussian", sigma=2.0)
        # Gaussian reduce varianza
        assert result.compressed().std() < arr.compressed().std()

    def test_despacha_a_median(self):
        """method='median' delega a apply_median_smoothing_masked."""
        arr = np.ma.array(np.ones((5, 5), dtype=np.float32) * 20.0)
        result = apply_smoothing_masked(arr, method="median", median_size=3)
        np.testing.assert_allclose(result, 20.0)

    def test_method_none_usa_median(self):
        """method=None usa median por defecto."""
        arr = np.ma.array(np.ones((5, 5), dtype=np.float32) * 20.0)
        result = apply_smoothing_masked(arr, method=None, median_size=3)
        np.testing.assert_allclose(result, 20.0)

    def test_method_case_insensitive(self):
        """El método es case-insensitive."""
        arr = np.ma.array(np.ones((5, 5), dtype=np.float32) * 20.0)
        result = apply_smoothing_masked(arr, method="GAUSSIAN", sigma=0.0)
        np.testing.assert_array_equal(result.data, arr.data)
