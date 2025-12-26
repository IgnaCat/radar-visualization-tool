"""
DEPRECATED: Este módulo mantiene imports por compatibilidad.
Use 'from app.models import ...' en su lugar.
"""
from .models import (
    # Common
    RangeFilter,
    # Process
    ProcessRequest,
    ProcessResponse,
    LayerResult,
    RadarProcessResult,
    # Cleanup
    CleanupRequest,
    # Pseudo RHI
    PseudoRHIRequest,
    PseudoRHIResponse,
    # Stats
    RadarStatsRequest,
    RadarStatsResponse,
    StatsResult,
    # Pixel
    RadarPixelRequest,
    RadarPixelResponse,
    # Elevation
    Coordinate,
    ElevationProfileRequest,
    ProfilePoint,
    ElevationProfileResponse,
)

__all__ = [
    'RangeFilter',
    'ProcessRequest',
    'ProcessResponse',
    'LayerResult',
    'RadarProcessResult',
    'CleanupRequest',
    'PseudoRHIRequest',
    'PseudoRHIResponse',
    'RadarStatsRequest',
    'RadarStatsResponse',
    'StatsResult',
    'RadarPixelRequest',
    'RadarPixelResponse',
    'Coordinate',
    'ElevationProfileRequest',
    'ProfilePoint',
    'ElevationProfileResponse',
]