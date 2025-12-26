from fastapi import APIRouter, HTTPException, status

from ..models import ProcessRequest, ProcessResponse
from ..services.orchestrators import ProcessingOrchestrator

router = APIRouter(prefix="/process", tags=["process"])


@router.post("", response_model=ProcessResponse)
async def process_file(payload: ProcessRequest):
    """
    Endpoint para procesar archivos de radar previamente subidos.
    """
    try:
        return ProcessingOrchestrator.process_radar_files(payload)
    except ValueError as e:
        # Errores de validación se convierten en 400 Bad Request
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        # Archivos no encontrados se convierten en 404 Not Found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        # Re-emitir las HTTPException tal cual
        raise
    except Exception as e:
        # 500 Internal Server Error para cualquier otro error
        print(f"Error procesando archivos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando archivos: {e}"
        )
