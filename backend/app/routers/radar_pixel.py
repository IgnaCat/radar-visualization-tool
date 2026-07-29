from fastapi import APIRouter, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool

from ..models import RadarPixelRequest, RadarPixelResponse
from ..services.orchestrators import PixelOrchestrator
from ..dependencies.auth import get_current_user
from ..models.db.user import User

router = APIRouter(prefix="/stats", tags=["radar-pixel"])


@router.post("/pixel", response_model=RadarPixelResponse)
async def pixel_stat(
    payload: RadarPixelRequest,
    _user: User = Depends(get_current_user),
):
    """
    Obtiene el valor del radar en un píxel específico (coordenadas lat/lon).
    Usa interpolación bilinear cuando es posible, nearest neighbor en bordes.
    """
    try:
        # Ejecutar en threadpool (bloqueante pero seguro)
        response = await run_in_threadpool(
            PixelOrchestrator.process_pixel_request,
            payload,
            str(_user.id),
        )
        return response
    except ValueError as e:
        # Errores de validación o datos no disponibles
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Errores inesperados
        print(f"Error consultando píxel: {e}")
        raise HTTPException(status_code=500, detail=str(e))
