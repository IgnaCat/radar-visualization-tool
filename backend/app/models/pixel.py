"""
Modelos para consulta de valores en píxeles individuales.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

from .common import RangeFilter
from ..core.constants import TOA, DEFAULT_WEIGHT_FUNC, DEFAULT_MAX_NEIGHBORS


class RadarPixelDebugResponse(BaseModel):
    """Información de depuración para comparar click visible vs pixel científico."""

    render_zoom: Optional[int] = None
    snap_aplicado: bool = False
    lon_original: Optional[float] = None
    lat_original: Optional[float] = None
    lon_consulta: Optional[float] = None
    lat_consulta: Optional[float] = None
    pixel_global_x: Optional[int] = None
    pixel_global_y: Optional[int] = None
    tile_x: Optional[int] = None
    tile_y: Optional[int] = None
    pixel_en_tile_x: Optional[int] = None
    pixel_en_tile_y: Optional[int] = None
    col_f: Optional[float] = None
    row_f: Optional[float] = None
    col_int: Optional[int] = None
    row_int: Optional[int] = None
    width_raster: Optional[int] = None
    height_raster: Optional[int] = None
    crs_consulta: Optional[str] = None


class RadarPixelRequest(BaseModel):
    """Request para obtener valor en un píxel específico."""

    filepath: str
    product: str
    field: str
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
    lat: float
    lon: float
    session_id: Optional[str] = Field(
        default=None, description="Identificador único de sesión"
    )
    weight_func: Optional[str] = Field(
        default=DEFAULT_WEIGHT_FUNC,
        description="Función de ponderación: 'Barnes2', 'Barnes', 'Cressman', 'nearest'",
    )
    max_neighbors: Optional[int] = Field(
        default=DEFAULT_MAX_NEIGHBORS,
        ge=1,
        le=500,
        description="Máximo número de vecinos para interpolación",
    )
    render_zoom: Optional[int] = Field(
        default=None,
        ge=0,
        le=30,
        description=(
            "Zoom entero del mapa al momento del click. "
            "Si se informa, el backend alinea la consulta al pixel visible "
            "de la grilla WebMercatorQuad"
        ),
    )
    render_native_zoom: Optional[int] = Field(
        default=None,
        ge=0,
        le=30,
        description=(
            "Max native zoom real de la capa renderizada. "
            "Si se informa, el backend evita snapear a un zoom mÃ¡s fino "
            "que el soportado por el raster visible"
        ),
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
    debug: Optional[RadarPixelDebugResponse] = None
