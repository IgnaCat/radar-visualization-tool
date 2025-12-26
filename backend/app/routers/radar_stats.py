from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..models import RadarStatsRequest, RadarStatsResponse
from ..services.orchestrators import StatsOrchestrator

router = APIRouter(prefix="/stats", tags=["radar-stats"])


@router.post("/area", response_model=RadarStatsResponse)
async def radar_stats(payload: RadarStatsRequest):
    """
    Calcula estadísticas del radar sobre un polígono (área seleccionada en el mapa),
    utilizando la grilla 2D cacheada (sin tocar disco).
    """
    try:
        # Ejecutar en threadpool (bloqueante pero seguro)
        response = await run_in_threadpool(
            StatsOrchestrator.process_stats_request,
            payload
        )
        return response
    except ValueError as e:
        # Errores de validación o no cobertura
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Errores inesperados
        print(f"Error calculando estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))
