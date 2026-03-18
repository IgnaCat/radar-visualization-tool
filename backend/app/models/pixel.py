"""
Modelos para consulta de valores en píxeles individuales.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

from .common import RangeFilter


class RadarPixelRequest(BaseModel):
    """Request para obtener valor en un píxel específico."""

    filepath: str
    product: str
    field: str
    height: Optional[int] = Field(
        default=4000,
        ge=0,
        le=12000,
        description="Altura en metros (0-12000). Default 4000m",
    )
    elevation: Optional[int] = Field(
        default=0,
        ge=0,
        le=12,
        description="Ángulo de elevación en grados (0-12). Default 0",
    )
    filters: Optional[List[RangeFilter]] = Field(default=[], min_items=0)
    lat: float
    lon: float
    session_id: Optional[str] = Field(
        default=None, description="Identificador único de sesión"
    )
    weight_func: Optional[str] = Field(
        default="nearest",
        description="Función de ponderación (default 'nearest'): 'nearest', 'Barnes2', 'Barnes', 'Cressman'",
    )
    max_neighbors: Optional[int] = Field(
        default=1,
        ge=1,
        le=500,
        description="Máximo número de vecinos para interpolación",
    )


class RadarPixelResponse(BaseModel):
    """Respuesta de valor en píxel."""

    value: Optional[float] = None
    masked: bool = False
    row: Optional[int] = None
    col: Optional[int] = None
    message: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
