"""
Router para generar perfiles de elevación topográficos.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from ..schemas import ElevationProfileRequest, ElevationProfileResponse
from ..services.elevation_profile import extract_elevation_profile
from ..core.config import settings

router = APIRouter(prefix="/stats", tags=["stats"])


@router.post("/elevation_profile", response_model=ElevationProfileResponse)
async def get_elevation_profile(payload: ElevationProfileRequest):
    """
    Genera un perfil de elevación a partir de una línea de coordenadas.
    Utiliza el DEM (Digital Elevation Model) de Argentina para extraer las elevaciones.
    
    Args:
        payload: Coordenadas de la línea y parámetros de interpolación
    
    Returns:
        Perfil de elevación con distancias y elevaciones
    """
    
    # Validar que tengamos al menos 2 coordenadas
    if len(payload.coordinates) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Se requieren al menos 2 coordenadas para generar un perfil"
        )
    
    # Path al DEM de Argentina
    dem_path = Path(settings.DATA_DIR) / "mosaico_argentina_2.tif"
    
    if not dem_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se encuentra el archivo DEM de Argentina"
        )
    
    # Convertir coordenadas de Pydantic a diccionarios
    coordinates = [{"lat": coord.lat, "lon": coord.lon} for coord in payload.coordinates]
    
    try:
        # Ejecutar el procesamiento en threadpool para no bloquear
        result = await run_in_threadpool(
            extract_elevation_profile,
            coordinates=coordinates,
            dem_path=dem_path,
            interpolate=payload.interpolate,
            points_per_km=payload.points_per_km
        )
        
        return ElevationProfileResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"Error generando perfil de elevación: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando perfil de elevación: {str(e)}"
        )
