"""
Modelos de dominio de la aplicación.
Divididos por responsabilidad funcional.
"""
from .common import RangeFilter
from .process import ProcessRequest, ProcessResponse, LayerResult, RadarProcessResult
from .cleanup import CleanupRequest, FileCleanupRequest
from .pseudo_rhi import PseudoRHIRequest, PseudoRHIResponse
from .stats import RadarStatsRequest, RadarStatsResponse, StatsResult
from .pixel import RadarPixelRequest, RadarPixelResponse
from .elevation import Coordinate, ElevationProfileRequest, ProfilePoint, ElevationProfileResponse

__all__ = [
    # Common
    'RangeFilter',
    # Process
    'ProcessRequest',
    'ProcessResponse',
    'LayerResult',
    'RadarProcessResult',
    # Cleanup
    'CleanupRequest',
    'FileCleanupRequest',
    # Pseudo RHI
    'PseudoRHIRequest',
    'PseudoRHIResponse',
    # Stats
    'RadarStatsRequest',
    'RadarStatsResponse',
    'StatsResult',
    # Pixel
    'RadarPixelRequest',
    'RadarPixelResponse',
    # Elevation
    'Coordinate',
    'ElevationProfileRequest',
    'ProfilePoint',
    'ElevationProfileResponse',
]
