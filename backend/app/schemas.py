from pydantic import BaseModel, AnyHttpUrl, Field, field_serializer
from typing import List, Optional, Any
from datetime import datetime
from pathlib import Path

class ProcessRequest(BaseModel):
    filepaths: List[str] = Field(..., min_items=1)
    product: str = Field(..., description="Producto a procesar")
    height: Optional[int] = Field(
        default=4000, ge=0, le=12000,
        description="Altura en metros (0-12000). Default 4000m"
    )
    elevation: Optional[int] = Field(
        default=0, ge=0, le=12,
        description="Ángulo de elevación en grados (0-12). Default 0"
    )

class ProcessOutput(BaseModel):
    image_url: str    # si radar_processor devuelve PNG/Geotiff/URL
    tilejson_url: str
    metadata: Optional[dict] = None
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

class ProcessResponse(BaseModel):
    animation: bool
    outputs: List[ProcessOutput]
    product: str

class CleanupRequest(BaseModel):
    uploads: list[str] = []
    cogs: list[str] = []
    delete_cache: bool = False


