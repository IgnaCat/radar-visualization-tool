"""
Modelos para estadísticas sobre áreas de radar.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

from .common import RangeFilter


class RadarStatsRequest(BaseModel):
    """Request para calcular estadísticas sobre un polígono."""
    polygon_geojson: Dict[str, Any] = Field(..., description="Polígono GeoJSON en EPSG:4326")
    filepath: Optional[str] = None
    product: str
    field: str
    height: Optional[int] = Field(
        default=4000, ge=0, le=12000,
        description="Altura en metros (0-12000). Default 4000m"
    )
    elevation: Optional[int] = Field(
        default=0, ge=0, le=12,
        description="Ángulo de elevación en grados (0-12). Default 0"
    )
    filters: Optional[List[RangeFilter]] = Field(default=[], min_items=0)
    session_id: Optional[str] = Field(
        default=None,
        description="Identificador único de sesión"
    )
    colormap_overrides: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Mapeo de campo a colormap personalizado, ej. {'DBZH': 'grc_th2'}"
    )

    @validator("filepath")
    def validate_filepath(cls, v):
        if v in (None, "", "undefined"):
            raise ValueError("Debe enviarse un filepath válido (no 'undefined').")
        return v


class StatsResult(BaseModel):
    """Resultado de estadísticas calculadas."""
    min: float
    max: float
    mean: float
    median: float
    std: float
    count: int
    valid_pct: float


class RadarStatsResponse(BaseModel):
    """Respuesta de estadísticas de radar."""
    stats: Optional[StatsResult] = None
    noCoverage: bool = False
    reason: Optional[str] = None
