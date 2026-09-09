# backend/tests/unit/test_file_ingest.py
"""Tests para el helper build_file_entry (usado por /admin/demo/load)."""
from pathlib import Path

from app.services.file_ingest import build_file_entry

DEMO_NC = Path(__file__).resolve().parents[1] / "data" / "RMA1_0315_01_20250819T001715Z.nc"


def test_build_file_entry_real_netcdf():
    entry, volume, radar = build_file_entry(DEMO_NC)
    assert entry["filepath"] == "RMA1_0315_01_20250819T001715Z.nc"
    assert entry["filename"] == "RMA1_0315_01_20250819T001715Z.nc"
    assert entry["size_bytes"] > 0
    assert isinstance(entry["metadata"], dict)
    assert "fields_present" in entry["metadata"]
    assert radar == "RMA1"
    assert volume == "01"


def test_build_file_entry_overrides():
    entry, _, _ = build_file_entry(DEMO_NC, filepath="renamed.nc", filename="Original.nc")
    assert entry["filepath"] == "renamed.nc"
    assert entry["filename"] == "Original.nc"
    # size sale siempre del archivo en disco
    assert entry["size_bytes"] > 0
