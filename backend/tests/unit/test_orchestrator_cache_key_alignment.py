"""
Tests de alineación de cache keys entre process y endpoints de stats/pixel.
"""

from app.models import RangeFilter
from app.services.radar_common import grid2d_cache_key
from app.services.orchestrators.pixel_orchestrator import PixelOrchestrator
from app.services.orchestrators.stats_orchestrator import StatsOrchestrator


def test_stats_cache_key_matches_process_semantics(monkeypatch):
    monkeypatch.setattr(
        "app.services.orchestrators.stats_orchestrator.md5_file",
        lambda _path: "abc123abc123abc123abc123",
    )

    key = StatsOrchestrator.generate_cache_key(
        filepath="/tmp/fake.nc",
        product="PPI",
        field="DBZH",
        elevation=0,
        cappi_height=4000,
        volume="01",
        filters=[RangeFilter(field="RHOHV", min=0.8, max=1.0)],
        session_id="sess-1",
        weight_func="nearest",
        max_neighbors=1,
    )

    expected = grid2d_cache_key(
        file_hash="abc123abc123",
        product_upper="PPI",
        field_to_use="DBZH",
        elevation=0,
        cappi_height=None,
        volume="01",
        interp="nearest",
        qc_sig=None,
        max_neighbors=1,
        session_id="sess-1",
    )

    assert key == expected


def test_pixel_cache_key_matches_process_semantics(monkeypatch):
    monkeypatch.setattr(
        "app.services.orchestrators.pixel_orchestrator.md5_file",
        lambda _path: "def456def456def456def456",
    )

    key = PixelOrchestrator.generate_cache_key(
        filepath="/tmp/fake.nc",
        product="CAPPI",
        field="DBZH",
        elevation=0,
        cappi_height=3000,
        volume="03",
        filters=[RangeFilter(field="RHOHV", min=0.7, max=1.0)],
        session_id="sess-2",
        weight_func="nearest",
        max_neighbors=10,
    )

    expected = grid2d_cache_key(
        file_hash="def456def456",
        product_upper="CAPPI",
        field_to_use="DBZH",
        elevation=None,
        cappi_height=3000,
        volume="03",
        interp="nearest",
        qc_sig=None,
        max_neighbors=10,
        session_id="sess-2",
    )

    assert key == expected


def test_stats_field_candidates_include_aliases():
    candidates = StatsOrchestrator.get_field_candidates("DBZH")
    upper_candidates = {c.upper() for c in candidates}

    assert "DBZH" in upper_candidates
    assert "CORRECTED_REFLECTIVITY_HORIZONTAL" in upper_candidates


def test_pixel_field_candidates_include_aliases():
    candidates = PixelOrchestrator.get_field_candidates("DBZH")
    upper_candidates = {c.upper() for c in candidates}

    assert "DBZH" in upper_candidates
    assert "CORRECTED_REFLECTIVITY_HORIZONTAL" in upper_candidates
