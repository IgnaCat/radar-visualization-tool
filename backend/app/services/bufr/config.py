"""
Configuración del módulo BUFR local.

Resuelve la ruta a bufr_resources (tablas BUFR + librería C) sin depender
de radarlib.config.
"""

from __future__ import annotations

import os
from pathlib import Path

# Ruta por defecto: junto a este archivo en bufr_resources/
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_BUFR_RESOURCES = _THIS_DIR / "bufr_resources"

# Permitir override via variable de entorno
BUFR_RESOURCES_PATH: str = os.environ.get(
    "BUFR_RESOURCES_PATH",
    str(_DEFAULT_BUFR_RESOURCES),
)
