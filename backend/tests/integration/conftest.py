"""
Fixtures para tests de integración.

Estos fixtures trabajan con archivos NetCDF reales y objetos PyART.
Son más lentos que los unit tests pero verifican que los componentes
funcionan correctamente juntos con datos reales.
"""

import pytest
import numpy as np
import pyart
from pathlib import Path

from app.services.radar_processing.grid_builder import (
    get_gate_xyz_coords,
    get_grid_xyz_coords,
)
from app.services.radar_processing.grid_compute import build_W_operator
from app.services.radar_common import safe_range_max_m

# ─── Rutas a archivos de prueba ─────────────────────────────────────
VOLUMENES_DIR = Path(__file__).parent.parent / "data" / "volumenes"

# Archivos disponibles por volumen:
#   RMA1_0315_01_20250830T203606Z.nc  (vol 01)
#   RMA1_0315_02_20250830T204059Z.nc  (vol 02)
#   RMA1_0315_03_20250830T203520Z.nc  (vol 03 - bird bath)
#   RMA1_0315_04_20250830T203546Z.nc  (vol 04)

VOL01_FILE = VOLUMENES_DIR / "RMA1_0315_01_20250830T203606Z.nc"
VOL02_FILE = VOLUMENES_DIR / "RMA1_0315_02_20250830T204059Z.nc"
VOL03_FILE = VOLUMENES_DIR / "RMA1_0315_03_20250830T203520Z.nc"
VOL04_FILE = VOLUMENES_DIR / "RMA1_0315_04_20250830T203546Z.nc"


def _check_test_data():
    """Verifica que los archivos de prueba existen."""
    if not VOLUMENES_DIR.exists():
        pytest.skip(f"Directorio de datos no encontrado: {VOLUMENES_DIR}")
    if not VOL01_FILE.exists():
        pytest.skip(f"Archivo de prueba no encontrado: {VOL01_FILE}")


# ─── Fixtures de lectura de radar ───────────────────────────────────

@pytest.fixture(scope="session")
def radar_vol01():
    """
    Objeto radar PyART leído desde archivo real, volumen 01.
    scope="session" → se lee UNA sola vez para todos los tests.
    """
    _check_test_data()
    return pyart.io.read(str(VOL01_FILE))


@pytest.fixture(scope="session")
def radar_vol03():
    """Objeto radar PyART, volumen 03 (bird bath - alta resolución vertical)."""
    _check_test_data()
    if not VOL03_FILE.exists():
        pytest.skip(f"Archivo vol03 no encontrado: {VOL03_FILE}")
    return pyart.io.read(str(VOL03_FILE))


@pytest.fixture(scope="session")
def vol01_filepath():
    """Path absoluto al archivo de volumen 01."""
    _check_test_data()
    return str(VOL01_FILE)


@pytest.fixture(scope="session")
def vol03_filepath():
    """Path absoluto al archivo de volumen 03."""
    _check_test_data()
    if not VOL03_FILE.exists():
        pytest.skip(f"Archivo vol03 no encontrado: {VOL03_FILE}")
    return str(VOL03_FILE)


# ─── Grilla pequeña y operador W (compartidos) ─────────────────────

SMALL_GRID_SHAPE = (10, 50, 50)


@pytest.fixture(scope="session")
def w_and_limits(radar_vol01):
    """
    Construye una grilla pequeña y su operador W.
    scope="session" → se construye UNA sola vez para todos los integration tests.
    """
    range_max = safe_range_max_m(radar_vol01, round_to_km=20)
    extent = min(range_max, 150_000)
    grid_limits = (
        (0, 12000),
        (-extent, extent),
        (-extent, extent),
    )

    gates_xyz = get_gate_xyz_coords(radar_vol01)
    voxels_xyz = get_grid_xyz_coords(SMALL_GRID_SHAPE, grid_limits)

    W = build_W_operator(
        gates_xyz=gates_xyz,
        voxels_xyz=voxels_xyz,
        toa=12000.0,
        h_factor=0.9,
        nb=1.2,
        bsp=1.0,
        min_radius=700.0,
        volume="01",
        weight_func="Barnes2",
        max_neighbors=30,
    )

    return W, grid_limits
