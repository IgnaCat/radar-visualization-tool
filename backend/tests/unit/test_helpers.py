"""
Tests para app.utils.helpers — funciones utilitarias puras.

Qué testea este archivo:
1. extract_metadata_from_filename: Parsea nombres de archivos NetCDF con regex.
   - Caso feliz: nombre con formato correcto → extrae radar, estrategia, volumen, timestamp.
   - Casos borde: extensión incorrecta, formato roto, paths con directorios.

2. extract_volume_from_filename: Wrapper que solo devuelve el volumen.

3. should_animate: Decide si un conjunto de resultados forma una animación.
   - Mismo radar + timestamps cercanos → True.
   - Radares distintos o timestamps muy separados → False.
   - Lista vacía o con 1 solo elemento → False (no tiene sentido animar 1 frame).
"""

import pytest
from datetime import datetime
from app.utils.helpers import (
    extract_metadata_from_filename,
    extract_volume_from_filename,
    should_animate,
)


# ═══════════════════════════════════════════════════════════════════
# extract_metadata_from_filename
# ═══════════════════════════════════════════════════════════════════

class TestExtractMetadataFromFilename:
    """
    Testea el parsing del nombre de archivo con formato:
    {RADAR}_{ESTRATEGIA}_{VOLUMEN}_{TIMESTAMP}Z.nc
    
    Ejemplo real: RMA1_0315_01_20250819T001715Z.nc
    """

    def test_nombre_valido_completo(self):
        """Caso feliz: nombre bien formado devuelve las 4 partes."""
        radar, est, vol, ts = extract_metadata_from_filename(
            "RMA1_0315_01_20250819T001715Z.nc"
        )
        assert radar == "RMA1"
        assert est == "0315"
        assert vol == "01"
        assert ts == datetime(2025, 8, 19, 0, 17, 15)

    def test_otro_radar(self):
        """Funciona con otros radares (RMA3, RMA7, etc)."""
        radar, est, vol, ts = extract_metadata_from_filename(
            "RMA3_0303_02_20221209T230832Z.nc"
        )
        assert radar == "RMA3"
        assert est == "0303"
        assert vol == "02"
        assert ts == datetime(2022, 12, 9, 23, 8, 32)

    def test_volumen_03(self):
        """Volumen 03 (bird bath) se parsea correctamente."""
        radar, est, vol, _ = extract_metadata_from_filename(
            "RMA1_0315_03_20250819T001715Z.nc"
        )
        assert vol == "03"

    def test_con_path_completo(self):
        """Si viene con path, extrae solo el nombre."""
        radar, est, vol, ts = extract_metadata_from_filename(
            "C:/uploads/session123/RMA1_0315_01_20250819T001715Z.nc"
        )
        assert radar == "RMA1"
        assert ts == datetime(2025, 8, 19, 0, 17, 15)

    def test_nombre_invalido_devuelve_nones(self):
        """Nombre que no matchea el patrón devuelve 4 Nones."""
        radar, est, vol, ts = extract_metadata_from_filename("archivo_random.nc")
        assert radar is None
        assert est is None
        assert vol is None
        assert ts is None

    def test_extension_diferente(self):
        """El regex no depende de la extensión, matchea por el patrón interno."""
        radar, est, vol, ts = extract_metadata_from_filename(
            "RMA1_0315_01_20250819T001715Z.BUFR"
        )
        # El regex busca el patrón en el nombre, la extensión no importa
        assert radar == "RMA1"

    def test_nombre_vacio(self):
        """String vacío no explota, devuelve Nones."""
        radar, est, vol, ts = extract_metadata_from_filename("")
        assert radar is None

    def test_solo_radar_sin_resto(self):
        """Nombre parcial no matchea."""
        radar, est, vol, ts = extract_metadata_from_filename("RMA1_0315.nc")
        assert radar is None


# ═══════════════════════════════════════════════════════════════════
# extract_volume_from_filename
# ═══════════════════════════════════════════════════════════════════

class TestExtractVolumeFromFilename:
    """
    Wrapper que devuelve solo el volumen como string.
    Importante para filter_by_volumes en el orchestrator.
    """

    def test_devuelve_volumen_como_string(self):
        vol = extract_volume_from_filename("RMA1_0315_01_20250819T001715Z.nc")
        assert vol == "01"

    def test_volumen_03(self):
        vol = extract_volume_from_filename("RMA1_0315_03_20250819T001715Z.nc")
        assert vol == "03"

    def test_nombre_invalido_devuelve_none(self):
        vol = extract_volume_from_filename("basura.nc")
        assert vol is None


# ═══════════════════════════════════════════════════════════════════
# should_animate
# ═══════════════════════════════════════════════════════════════════

class TestShouldAnimate:
    """
    Decide si un conjunto de resultados deben animarse como GIF.
    Reglas:
    - Necesita >= 2 resultados.
    - Todos del mismo radar.
    - Timestamps consecutivos dentro de max_minutes_diff.
    """

    def test_lista_vacia_no_anima(self):
        assert should_animate([]) is False

    def test_un_solo_resultado_no_anima(self):
        results = [{"source_file": "RMA1_0315_01_20250819T001715Z.nc"}]
        assert should_animate(results) is False

    def test_mismo_radar_cercanos_si_anima(self):
        """Dos archivos del mismo radar separados por 5 min → animar."""
        results = [
            {"source_file": "RMA1_0315_01_20250819T001715Z.nc"},
            {"source_file": "RMA1_0315_01_20250819T002215Z.nc"},  # +5 min
        ]
        assert should_animate(results) is True

    def test_radares_distintos_no_anima(self):
        """RMA1 y RMA3 no se pueden animar juntos."""
        results = [
            {"source_file": "RMA1_0315_01_20250819T001715Z.nc"},
            {"source_file": "RMA3_0315_01_20250819T001715Z.nc"},
        ]
        assert should_animate(results) is False

    def test_timestamps_muy_separados_no_anima(self):
        """Archivos del mismo radar pero separados > 30 min → no animar."""
        results = [
            {"source_file": "RMA1_0315_01_20250819T001715Z.nc"},
            {"source_file": "RMA1_0315_01_20250819T010000Z.nc"},  # +42 min
        ]
        assert should_animate(results) is False

    def test_sin_source_file_no_anima(self):
        """Si falta source_file, retorna False."""
        results = [
            {"other_key": "value"},
            {"other_key": "value2"},
        ]
        assert should_animate(results) is False

    def test_max_minutes_custom(self):
        """Se puede cambiar la tolerancia de tiempo."""
        results = [
            {"source_file": "RMA1_0315_01_20250819T001715Z.nc"},
            {"source_file": "RMA1_0315_01_20250819T010000Z.nc"},  # +42 min
        ]
        # Con 30 min (default) no anima, con 60 sí
        assert should_animate(results, max_minutes_diff=30) is False
        assert should_animate(results, max_minutes_diff=60) is True

    def test_tres_archivos_secuenciales(self):
        """Tres archivos cercanos secuencialmente → animan."""
        results = [
            {"source_file": "RMA1_0315_01_20250819T001700Z.nc"},
            {"source_file": "RMA1_0315_01_20250819T002200Z.nc"},
            {"source_file": "RMA1_0315_01_20250819T002700Z.nc"},
        ]
        assert should_animate(results) is True
