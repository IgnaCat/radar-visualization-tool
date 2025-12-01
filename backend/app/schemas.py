from pydantic import BaseModel, Field, field_serializer, validator
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from pathlib import Path


class RangeFilter(BaseModel):
    field: str = Field(..., description="Nombre del campo")
    type: Literal["range"] = "range"
    min: Optional[float] = Field(default=0, description="Límite inferior (inclusivo)")
    max: Optional[float] = Field(default=1, description="Límite superior (inclusivo)")

class ProcessRequest(BaseModel):
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

class LayerResult(BaseModel):
    image_url: str    # si radar_processor devuelve PNG/Geotiff/URL
    tilejson_url: str
    metadata: Optional[dict] = None
    field: Optional[str] = None
    order: Optional[int] = None  # para ordenar capas
    bounds: Optional[List[List[float]]] = Field(
        default=None, min_length=2, max_length=2,
        description="BBox [[west, south], [east, north]]"
    )
    source_file: Optional[Path] = None
    timestamp: Optional[datetime] = None

    # para garantizar string ISO al serializar JSON
    @field_serializer("timestamp", when_used="json")
    def _ckeck_ts(self, v: Optional[datetime]) -> Optional[str]:
        return v.isoformat() if v else None
    
    @field_serializer("source_file", when_used="json")
    def _check_path(self, v: Optional[Path]) -> Optional[str]:
        return str(v) if v else None

# Resultados por radar
class RadarProcessResult(BaseModel):
    radar: str
    animation: bool
    outputs: List[List[LayerResult]]  # frames x layers

# Respuesta final del process request
class ProcessResponse(BaseModel):
    results: List[RadarProcessResult]
    product: str
    warnings: Optional[List[str]] = []


class CleanupRequest(BaseModel):
    uploads: list[str] = []
    cogs: list[str] = []
    delete_cache: bool = False


class PseudoRHIRequest(BaseModel):
    filepaths: List[str] = Field(..., min_items=1)
    field: str = Field(..., description="Campo a visualizar, ej. DBZH")
    end_lon: float       # Coordenadas punto de interés
    end_lat: float
    start_lon: Optional[float] = None   # Coordenadas opcionales de inicio
    start_lat: Optional[float] = None
    max_length_km: Optional[float] = Field(default=240.0, description="Longitud máxima del gráfico en km (0.5 - 300). Default 240 km")
    max_height_km: Optional[float] = Field(default=20.0, ge=0.5, le=30.0, description="Altura máxima del gráfico en km (0.5 - 30). Default 20 km")
    elevation: Optional[int] = Field(
        default=0, ge=0, le=12,
        description="Ángulo de elevación en grados (0-12). Default 0"
    )
    filters: Optional[List[RangeFilter]] = Field(default=[], min_items=0)
    png_width_px: int = 900
    png_height_px: int = 500

class PseudoRHIResponse(BaseModel):
    image_url: str
    metadata: Optional[dict] = None
    timestamp: Optional[datetime] = None
    source_file: Optional[Path] = None

class RadarStatsRequest(BaseModel):
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

    @validator("filepath")
    def validate_filepath(cls, v):
        if v in (None, "", "undefined"):
            raise ValueError("Debe enviarse un filepath válido (no 'undefined').")
        return v

class StatsResult(BaseModel):
    min: float
    max: float
    mean: float
    median: float
    std: float
    count: int
    valid_pct: float

class RadarStatsResponse(BaseModel):
    stats: Optional[StatsResult] = None
    noCoverage: bool = False
    reason: Optional[str] = None

class RadarPixelRequest(BaseModel):
    filepath: str
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
    lat: float                   # click del usuario (EPSG:4326)
    lon: float

class RadarPixelResponse(BaseModel):
    value: Optional[float] = None
    masked: bool = False
    row: Optional[int] = None
    col: Optional[int] = None
    message: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

class Coordinate(BaseModel):
    lat: float = Field(..., description="Latitud en grados")
    lon: float = Field(..., description="Longitud en grados")

class ElevationProfileRequest(BaseModel):
    coordinates: List[Coordinate] = Field(..., min_items=2, description="Lista de coordenadas que forman la línea")
    interpolate: Optional[bool] = Field(default=True, description="Interpolar puntos adicionales")
    points_per_km: Optional[int] = Field(default=10, ge=1, le=100, description="Puntos por kilómetro al interpolar")

class ProfilePoint(BaseModel):
    distance: float = Field(..., description="Distancia acumulada en km")
    elevation: Optional[float] = Field(..., description="Elevación en metros")
    lat: float
    lon: float

class ElevationProfileResponse(BaseModel):
    profile: List[ProfilePoint] = Field(..., description="Lista de puntos del perfil")
    # total_distance: float = Field(..., description="Distancia total en km")
    # min_elevation: float = Field(..., description="Elevación mínima en m")
    # max_elevation: float = Field(..., description="Elevación máxima en m")