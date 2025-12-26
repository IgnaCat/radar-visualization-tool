"""
Modelos para generación de pseudo RHI (transectos verticales).
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path

from .common import RangeFilter


class PseudoRHIRequest(BaseModel):
    """Request para generación de pseudo RHI."""
    filepaths: List[str] = Field(..., min_items=1)
    field: str = Field(..., description="Campo a visualizar, ej. DBZH")
    end_lon: float
    end_lat: float
    start_lon: Optional[float] = None
    start_lat: Optional[float] = None
    max_length_km: Optional[float] = Field(
        default=240.0, 
        description="Longitud máxima del gráfico en km (0.5 - 300). Default 240 km"
    )
    max_height_km: Optional[float] = Field(
        default=20.0, ge=0.5, le=30.0, 
        description="Altura máxima del gráfico en km (0.5 - 30). Default 20 km"
    )
    elevation: Optional[int] = Field(
        default=0, ge=0, le=12,
        description="Ángulo de elevación en grados (0-12). Default 0"
    )
    filters: Optional[List[RangeFilter]] = Field(default=[], min_items=0)
    png_width_px: int = 900
    png_height_px: int = 500
    colormap_overrides: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Mapeo de campo a colormap personalizado, ej. {'DBZH': 'grc_th2'}"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Identificador único de sesión"
    )


class PseudoRHIResponse(BaseModel):
    """Respuesta de pseudo RHI."""
    image_url: str
    metadata: Optional[dict] = None
    timestamp: Optional[datetime] = None
    source_file: Optional[Path] = None
