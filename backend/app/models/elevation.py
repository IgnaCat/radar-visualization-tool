"""
Modelos para perfiles de elevación topográfica.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class Coordinate(BaseModel):
    """Coordenada geográfica."""
    lat: float = Field(..., description="Latitud en grados")
    lon: float = Field(..., description="Longitud en grados")


class ElevationProfileRequest(BaseModel):
    """Request para perfil de elevación."""
    coordinates: List[Coordinate] = Field(
        ..., min_items=2, 
        description="Lista de coordenadas que forman la línea"
    )
    interpolate: Optional[bool] = Field(
        default=True, 
        description="Interpolar puntos adicionales"
    )
    points_per_km: Optional[int] = Field(
        default=10, ge=1, le=100, 
        description="Puntos por kilómetro al interpolar"
    )


class ProfilePoint(BaseModel):
    """Punto en el perfil de elevación."""
    distance: float = Field(..., description="Distancia acumulada en km")
    elevation: Optional[float] = Field(..., description="Elevación en metros")
    lat: float
    lon: float


class ElevationProfileResponse(BaseModel):
    """Respuesta de perfil de elevación."""
    profile: List[ProfilePoint] = Field(..., description="Lista de puntos del perfil")
