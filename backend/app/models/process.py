"""
Modelos para el procesamiento de archivos de radar.
"""
from pydantic import BaseModel, Field, field_serializer
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path

from .common import RangeFilter


class ProcessRequest(BaseModel):
    """Request para procesamiento de archivos de radar."""
    filepaths: List[str] = Field(..., min_items=1)
    product: str = Field(..., description="Producto a procesar, ej PPI")
    fields: List[str] = Field(..., min_items=1, description="Campos a procesar")
    height: Optional[int] = Field(
        default=4000, ge=0, le=12000,
        description="Altura en metros (0-12000). Default 4000m"
    )
    elevation: Optional[int] = Field(
        default=0, ge=0, le=12,
        description="Ángulo de elevación en grados (0-12). Default 0"
    )
    filters: Optional[List[RangeFilter]] = Field(default=[], min_items=0)
    selectedVolumes: Optional[List[str]]
    selectedRadars: Optional[List[str]]
    colormap_overrides: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Mapeo de campo a colormap personalizado, ej. {'DBZH': 'grc_th2'}"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Identificador único de sesión para aislar archivos y cache"
    )


class LayerResult(BaseModel):
    """Resultado de una capa procesada."""
    image_url: str
    tilejson_url: str
    metadata: Optional[dict] = None
    field: Optional[str] = None
    order: Optional[int] = None
    bounds: Optional[List[List[float]]] = Field(
        default=None, min_length=2, max_length=2,
        description="BBox [[west, south], [east, north]]"
    )
    source_file: Optional[Path] = None
    timestamp: Optional[datetime] = None

    @field_serializer("timestamp", when_used="json")
    def _check_ts(self, v: Optional[datetime]) -> Optional[str]:
        return v.isoformat() if v else None
    
    @field_serializer("source_file", when_used="json")
    def _check_path(self, v: Optional[Path]) -> Optional[str]:
        return str(v) if v else None


class RadarProcessResult(BaseModel):
    """Resultados de procesamiento por radar."""
    radar: str
    animation: bool
    outputs: List[List[LayerResult]]  # frames x layers


class ProcessResponse(BaseModel):
    """Respuesta final del procesamiento."""
    results: List[RadarProcessResult]
    product: str
    warnings: Optional[List[str]] = []
