"""
Tests para app.services.radar_processing.grid_compute — cálculo de pesos de interpolación.

Qué testea este archivo:
compute_weights: Calcula los pesos de interpolación según distancia y ROI.
   - Barnes2: w = exp(-d²/(ROI²/4)) + 1e-5 (implementación Py-ART)
   - Barnes: w = exp(-d²/(2σ²)) con σ = ROI/2
   - Cressman: w = (ROI² - d²)/(ROI² + d²), clipeado a [0,1]
   - nearest: solo el más cercano tiene peso 1
"""

import pytest
import numpy as np
from app.services.radar_processing.grid_compute import compute_weights


class TestComputeWeightsBarnes2:
    """
    Barnes2 (default de Py-ART):
    w = exp(-d² / (ROI²/4)) + 1e-5
    
    Propiedades:
    - d=0 → w = exp(0) + 1e-5 = 1.00001
    - d=ROI → w = exp(-4) + 1e-5 ≈ 0.0183
    - d >> ROI → w → 1e-5 (offset para estabilidad numérica)
    """

    def test_peso_maximo_en_distancia_cero(self):
        """A distancia 0, el peso es ~1.0 (máximo)."""
        w = compute_weights(np.array([0.0]), roi=np.float32(1000.0), method='Barnes2')
        assert w[0] == pytest.approx(1.0, abs=0.001)

    def test_peso_decrece_con_distancia(self):
        """Los pesos decrecen monótonamente con la distancia."""
        distances = np.array([0.0, 100.0, 500.0, 1000.0, 2000.0])
        w = compute_weights(distances, roi=np.float32(1000.0), method='Barnes2')
        
        for i in range(1, len(w)):
            assert w[i] < w[i-1], f"w[{i}]={w[i]} no es menor que w[{i-1}]={w[i-1]}"

    def test_peso_positivo_siempre(self):
        """El offset 1e-5 asegura que el peso nunca es exactamente 0."""
        distances = np.array([10_000.0, 100_000.0])  # muy lejos
        w = compute_weights(distances, roi=np.float32(1000.0), method='Barnes2')
        assert np.all(w > 0)

    def test_peso_en_roi(self):
        """A distancia = ROI, el peso es exp(-4) + 1e-5 ≈ 0.0183."""
        w = compute_weights(np.array([1000.0]), roi=np.float32(1000.0), method='Barnes2')
        expected = np.exp(-4.0) + 1e-5
        assert w[0] == pytest.approx(expected, rel=0.01)

    def test_array_vacio(self):
        """Array vacío devuelve array vacío (no explota)."""
        w = compute_weights(np.array([]), roi=np.float32(1000.0), method='Barnes2')
        assert w.size == 0

    def test_roi_array(self):
        """ROI puede ser un array (distinto ROI por punto)."""
        distances = np.array([500.0, 500.0], dtype=np.float32)
        roi = np.array([1000.0, 2000.0], dtype=np.float32)
        w = compute_weights(distances, roi=roi, method='Barnes2')
        
        # Mismo d=500, pero ROI 2000 → más peso que ROI 1000
        assert w[1] > w[0]


class TestComputeWeightsCressman:
    """
    Cressman: w = (R² - d²) / (R² + d²)
    
    Propiedades:
    - d=0 → w = 1.0
    - d=R → w = 0.0
    - d>R → w se clipea a 0.0
    """

    def test_peso_1_en_origen(self):
        w = compute_weights(np.array([0.0]), roi=np.float32(1000.0), method='Cressman')
        assert w[0] == pytest.approx(1.0, abs=0.001)

    def test_peso_0_en_roi(self):
        """A distancia = ROI exacto, el peso es ~0."""
        w = compute_weights(np.array([1000.0]), roi=np.float32(1000.0), method='Cressman')
        assert w[0] == pytest.approx(0.0, abs=0.01)

    def test_peso_clipeado_a_cero(self):
        """Más allá del ROI, el peso es 0 (no negativo)."""
        w = compute_weights(np.array([2000.0]), roi=np.float32(1000.0), method='Cressman')
        assert w[0] == 0.0

    def test_pesos_en_rango_0_1(self):
        """Todos los pesos Cressman están en [0, 1]."""
        distances = np.linspace(0, 2000, 100).astype(np.float32)
        w = compute_weights(distances, roi=np.float32(1000.0), method='Cressman')
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)


class TestComputeWeightsNearest:
    """
    Nearest: Solo el punto más cercano tiene peso 1, el resto 0.
    """

    def test_solo_mas_cercano(self):
        distances = np.array([500.0, 100.0, 800.0])
        w = compute_weights(distances, roi=np.float32(1000.0), method='nearest')
        
        assert w[1] == 1.0  # el más cercano (d=100)
        assert w[0] == 0.0
        assert w[2] == 0.0


class TestComputeWeightsBarnes:
    """
    Barnes clásico: w = exp(-d²/(2σ²)) con σ = ROI/2.
    Similar a Barnes2 pero sin el offset 1e-5.
    """

    def test_peso_1_en_origen(self):
        w = compute_weights(np.array([0.0]), roi=np.float32(1000.0), method='Barnes')
        assert w[0] == pytest.approx(1.0, abs=0.001)

    def test_decrece_con_distancia(self):
        distances = np.array([0.0, 500.0, 1000.0])
        w = compute_weights(distances, roi=np.float32(1000.0), method='Barnes')
        assert w[0] > w[1] > w[2]


class TestComputeWeightsEdgeCases:
    """Casos borde y errores."""

    def test_metodo_invalido_lanza_error(self):
        """Método desconocido lanza ValueError."""
        with pytest.raises(ValueError, match="Método desconocido"):
            compute_weights(np.array([0.0]), roi=np.float32(1000.0), method='invalido')

    def test_tipo_resultado_float32(self):
        """El resultado siempre es float32 (eficiencia de memoria)."""
        w = compute_weights(
            np.array([0.0, 500.0]), roi=np.float32(1000.0), method='Barnes2'
        )
        assert w.dtype == np.float32
