from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..models import RadarPixelRequest, RadarPixelResponse
from ..services.orchestrators import PixelOrchestrator

router = APIRouter(prefix="/stats", tags=["radar-pixel"])


@router.post("/pixel", response_model=RadarPixelResponse)
async def pixel_stat(payload: RadarPixelRequest):
    """
    Obtiene el valor del radar en un píxel específico (coordenadas lat/lon).
    Usa interpolación bilinear cuando es posible, nearest neighbor en bordes.
    """
    try:
        # Ejecutar en threadpool (bloqueante pero seguro)
        response = await run_in_threadpool(
            PixelOrchestrator.process_pixel_request,
            payload
        )
        return response
    except ValueError as e:
        # Errores de validación o datos no disponibles
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Errores inesperados
        print(f"Error consultando píxel: {e}")
        raise HTTPException(status_code=500, detail=str(e))
