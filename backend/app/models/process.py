"""
Modelos para el procesamiento de archivos de radar.
"""

from pydantic import BaseModel, Field, field_serializer
from typing import Literal
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path

from .common import RangeFilter
from ..core.constants import TOA, DEFAULT_WEIGHT_FUNC, DEFAULT_MAX_NEIGHBORS


class ImageSmoothing(BaseModel):
    """Configuración de suavizado opcional para la imagen final."""

    enabled: bool = Field(
        default=False,
        description="Activa/desactiva suavizado en la imagen final",
    )
    method: Literal["gaussian", "median"] = Field(
        default="median",
        description="Método de suavizado: 'gaussian' o 'median'",
    )
    sigma: float = Field(
        default=0.8,
        ge=0.0,
        le=5.0,
        description="Intensidad de suavizado Gaussiano (desvío estándar en píxeles)",
    )
    median_size: int = Field(
        default=3,
        ge=1,
        le=15,
        description="Tamaño de ventana para suavizado de mediana (impar recomendado)",
    )
    only_when_nearest: bool = Field(
        default=True,
        description="Si es true, aplica suavizado solo cuando weight_func='nearest'",
    )


class ProcessRequest(BaseModel):
    """Request para procesamiento de archivos de radar."""

    filepaths: List[str] = Field(..., min_items=1)
    product: str = Field(..., description="Producto a procesar, ej PPI")
    fields: List[str] = Field(..., min_items=1, description="Campos a procesar")
    height: Optional[int] = Field(
        default=4000,
        ge=0,
        le=TOA,
        description=f"Altura en metros (0-{TOA}). Default 4000m",
    )
    elevation: Optional[int] = Field(
        default=0,
        ge=0,
        le=12,
        description="Ángulo de elevación en grados (0-12). Default 0",
    )
    filters: Optional[List[RangeFilter]] = Field(default=[], min_items=0)
    filters_per_field: Optional[Dict[str, List[RangeFilter]]] = Field(
        default=None,
        description="Filtros por campo. Si se especifica, sobreescribe 'filters' para ese campo. Ej: {'DBZH': [{'field': 'RHOHV', 'min': 0.8}]}",
    )
    selectedVolumes: Optional[List[str]]
    selectedRadars: Optional[List[str]]
    colormap_overrides: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapeo de campo a colormap personalizado, ej. {'DBZH': 'grc_th2'}",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Identificador único de sesión para aislar archivos y cache",
    )
    weight_func: Optional[str] = Field(
        default=DEFAULT_WEIGHT_FUNC,
        description="Función de ponderación: 'Barnes2', 'Barnes', 'Cressman', 'nearest'",
    )
    max_neighbors: Optional[int] = Field(
        default=DEFAULT_MAX_NEIGHBORS,
        ge=1,
        le=500,
        description=f"Máximo número de vecinos para interpolación (default {DEFAULT_MAX_NEIGHBORS})",
    )
    smoothing: Optional[ImageSmoothing] = Field(
        default=None,
        description="Configuración opcional de suavizado visual de imagen",
    )


class LayerResult(BaseModel):
    """Resultado de una capa procesada."""

    image_url: str
    tilejson_url: str
    metadata: Optional[dict] = None
    field: Optional[str] = None
    order: Optional[int] = None
    bounds: Optional[List[List[float]]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="BBox [[west, south], [east, north]]",
    )
    source_file: Optional[Path] = None
    timestamp: Optional[datetime] = None
    colormap: Optional[str] = Field(
        default=None,
        description="Colormap usado para esta capa (ej: 'grc_th', 'pyart_NWSRef')",
    )

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
