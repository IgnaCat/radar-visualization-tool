"""
Fixtures compartidos para todos los tests del backend.

Conceptos clave:
- Un fixture es una función que prepara datos o estado que tus tests necesitan.
- pytest los inyecta automáticamente cuando un test los pide por nombre en sus argumentos.
- Scope: "session" = se ejecuta una sola vez por corrida de pytest.
         "function" = se ejecuta una vez por cada test (default).
"""

import pytest
import numpy as np
from app.models import RangeFilter


# ─── Fixtures para filtros ──────────────────────────────────────────
@pytest.fixture
def sample_range_filter_rhohv():
    """Filtro QC típico: RHOHV entre 0.7 y 1.0."""
    return RangeFilter(field="RHOHV", min=0.7, max=1.0)


@pytest.fixture
def sample_range_filter_dbzh():
    """Filtro visual típico: DBZH entre -10 y 60."""
    return RangeFilter(field="DBZH", min=-10.0, max=60.0)


@pytest.fixture
def sample_range_filter_zdr():
    """Filtro sobre ZDR."""
    return RangeFilter(field="ZDR", min=-2.0, max=6.0)


# ─── Fixtures para arrays numpy ────────────────────────────────────
@pytest.fixture
def simple_3d_array():
    """
    Grilla 3D chica (4 niveles Z, 5 filas Y, 6 columnas X).
    Los valores crecen con Z para que sea fácil verificar colapsos.
    
    Nivel 0: todos 10.0
    Nivel 1: todos 20.0
    Nivel 2: todos 30.0
    Nivel 3: todos 40.0
    """
    data = np.zeros((4, 5, 6), dtype=np.float32)
    for iz in range(4):
        data[iz, :, :] = (iz + 1) * 10.0
    return np.ma.array(data)


@pytest.fixture
def z_coords_4levels():
    """Coordenadas Z para 4 niveles: 0m, 1000m, 2000m, 3000m."""
    return np.array([0.0, 1000.0, 2000.0, 3000.0])


@pytest.fixture
def xy_coords_5x6():
    """Coordenadas X (6 puntos) e Y (5 puntos) centradas en el origen."""
    x = np.linspace(-5000, 5000, 6)
    y = np.linspace(-4000, 4000, 5)
    return x, y
