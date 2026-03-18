"""
Tests para ProcessingOrchestrator — validación y filtrado de archivos.

Qué testea este archivo:
1. validate_request: Valida los parámetros de una solicitud de procesamiento.
   - Producto válido vs inválido.
   - Altura fuera de rango.
   - Elevación negativa.
   - Filepaths vacíos.
   - Más de 3 radares seleccionados.

2. filter_by_volumes: Filtra archivos por volúmenes seleccionados.
   - Solo pasa archivos cuyo volumen está en la lista seleccionada.
   - Volumen 03 + PPI es inválido (genera warning y se omite).
   - Sin volúmenes seleccionados → pasan todos.

3. filter_by_radars: Filtra archivos por radares seleccionados.
   - Solo pasa archivos cuyo radar está en la lista.
   - Sin radares seleccionados → pasan todos.
"""

import pytest
from unittest.mock import patch
from app.models import ProcessRequest
from app.core.constants import TOA
from app.services.orchestrators.processing_orchestrator import ProcessingOrchestrator


# ═══════════════════════════════════════════════════════════════════
# validate_request
# ═══════════════════════════════════════════════════════════════════


class TestValidateRequest:
    """
    Valida los parámetros críticos del ProcessRequest.
    Lanza ValueError para problemas graves, retorna warnings para leves.
    """

    def _make_request(self, **overrides):
        """Helper para crear un ProcessRequest con defaults válidos."""
        defaults = dict(
            filepaths=["RMA1_0315_01_20250819T001715Z.nc"],
            product="PPI",
            fields=["DBZH"],
            height=4000,
            elevation=0,
            selectedVolumes=["01"],
            selectedRadars=[],
        )
        defaults.update(overrides)
        return ProcessRequest(**defaults)

    def test_request_valido(self):
        """Un request bien formado no lanza excepción."""
        req = self._make_request()
        warnings = ProcessingOrchestrator.validate_request(req)
        assert isinstance(warnings, list)

    def test_producto_invalido(self):
        """Producto que no está en ALLOWED_PRODUCTS lanza ValueError."""
        req = self._make_request(product="INVALIDO")
        with pytest.raises(ValueError, match="Producto.*no permitido"):
            ProcessingOrchestrator.validate_request(req)

    def test_productos_validos(self):
        """Todos los productos permitidos pasan la validación."""
        for prod in ["PPI", "CAPPI", "COLMAX", "RHI"]:
            req = self._make_request(product=prod)
            ProcessingOrchestrator.validate_request(req)  # no lanza

    def test_altura_negativa(self):
        """Altura < 0 es rechazada por Pydantic (ge=0)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._make_request(height=-100)

    def test_altura_excesiva(self):
        """Altura > TOA es rechazada por Pydantic (le=TOA)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._make_request(height=TOA + 1)

    def test_elevacion_negativa(self):
        """Elevación negativa es rechazada por Pydantic (ge=0)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._make_request(elevation=-1)

    def test_filepaths_vacio(self):
        """Sin archivos es rechazado por Pydantic (min_items=1)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._make_request(filepaths=[])

    def test_mas_de_3_radares(self):
        """Más de 3 radares seleccionados lanza ValueError."""
        req = self._make_request(selectedRadars=["RMA1", "RMA2", "RMA3", "RMA4"])
        with pytest.raises(ValueError, match="3 radares"):
            ProcessingOrchestrator.validate_request(req)

    def test_3_radares_es_valido(self):
        """Exactamente 3 radares es válido."""
        req = self._make_request(selectedRadars=["RMA1", "RMA2", "RMA3"])
        ProcessingOrchestrator.validate_request(req)  # no lanza


# ═══════════════════════════════════════════════════════════════════
# filter_by_volumes
# ═══════════════════════════════════════════════════════════════════


class TestFilterByVolumes:
    """
    Filtra archivos por volúmenes seleccionados.
    Regla especial: volumen 03 + PPI es inválido.
    """

    def test_filtra_por_volumen(self):
        """Solo pasa archivos cuyo volumen está en la lista."""
        filepaths = [
            "RMA1_0315_01_20250819T001715Z.nc",
            "RMA1_0315_02_20250819T001715Z.nc",
            "RMA1_0315_03_20250819T001715Z.nc",
        ]
        filtered, warnings = ProcessingOrchestrator.filter_by_volumes(
            filepaths, ["01", "02"], "PPI"
        )
        assert len(filtered) == 2
        assert "RMA1_0315_01_20250819T001715Z.nc" in filtered
        assert "RMA1_0315_02_20250819T001715Z.nc" in filtered

    def test_volumen_03_ppi_se_omite(self):
        """Volumen 03 con producto PPI genera warning y se omite."""
        filepaths = ["RMA1_0315_03_20250819T001715Z.nc"]
        filtered, warnings = ProcessingOrchestrator.filter_by_volumes(
            filepaths, ["03"], "PPI"
        )
        assert len(filtered) == 0
        assert any("03" in w and "PPI" in w for w in warnings)

    def test_volumen_03_cappi_si_pasa(self):
        """Volumen 03 con CAPPI sí pasa (restricción solo para PPI)."""
        filepaths = ["RMA1_0315_03_20250819T001715Z.nc"]
        filtered, warnings = ProcessingOrchestrator.filter_by_volumes(
            filepaths, ["03"], "CAPPI"
        )
        assert len(filtered) == 1

    def test_sin_volumenes_seleccionados_pasan_todos(self):
        """Sin volúmenes seleccionados, pasan todos los archivos."""
        filepaths = [
            "RMA1_0315_01_20250819T001715Z.nc",
            "RMA1_0315_02_20250819T001715Z.nc",
        ]
        filtered, warnings = ProcessingOrchestrator.filter_by_volumes(
            filepaths, [], "PPI"
        )
        assert len(filtered) == 2

    def test_genera_warnings_para_omitidos(self):
        """Los archivos omitidos generan warnings informativos."""
        filepaths = [
            "RMA1_0315_01_20250819T001715Z.nc",
            "RMA1_0315_02_20250819T001715Z.nc",
        ]
        filtered, warnings = ProcessingOrchestrator.filter_by_volumes(
            filepaths, ["01"], "PPI"
        )
        assert len(filtered) == 1
        assert len(warnings) >= 1  # al menos un warning por el vol 02 omitido


# ═══════════════════════════════════════════════════════════════════
# filter_by_radars
# ═══════════════════════════════════════════════════════════════════


class TestFilterByRadars:
    """Filtra archivos por radares seleccionados."""

    def test_filtra_por_radar(self):
        """Solo pasa archivos del radar seleccionado."""
        filepaths = [
            "RMA1_0315_01_20250819T001715Z.nc",
            "RMA3_0315_01_20250819T001715Z.nc",
        ]
        filtered, warnings = ProcessingOrchestrator.filter_by_radars(
            filepaths, ["RMA1"]
        )
        assert len(filtered) == 1
        assert "RMA1" in filtered[0]

    def test_sin_radares_seleccionados_pasan_todos(self):
        """Sin radares seleccionados, pasan todos."""
        filepaths = [
            "RMA1_0315_01_20250819T001715Z.nc",
            "RMA3_0315_01_20250819T001715Z.nc",
        ]
        filtered, warnings = ProcessingOrchestrator.filter_by_radars(filepaths, [])
        assert len(filtered) == 2

    def test_multiples_radares(self):
        """Se pueden seleccionar múltiples radares."""
        filepaths = [
            "RMA1_0315_01_20250819T001715Z.nc",
            "RMA3_0315_01_20250819T001715Z.nc",
            "RMA7_0315_01_20250819T001715Z.nc",
        ]
        filtered, warnings = ProcessingOrchestrator.filter_by_radars(
            filepaths, ["RMA1", "RMA7"]
        )
        assert len(filtered) == 2

    def test_genera_warnings_para_omitidos(self):
        filepaths = [
            "RMA1_0315_01_20250819T001715Z.nc",
            "RMA3_0315_01_20250819T001715Z.nc",
        ]
        filtered, warnings = ProcessingOrchestrator.filter_by_radars(
            filepaths, ["RMA1"]
        )
        assert len(warnings) >= 1
