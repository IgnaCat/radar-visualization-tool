import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.concurrency import run_in_threadpool

from ..core.config import settings
from ..models import (
    ProcessRequest,
    ProcessResponse,
    GifAnimationRequest,
    GifAnimationResponse,
)
from ..services.animation import create_animation_from_layer_urls
from ..services.orchestrators import ProcessingOrchestrator
from ..dependencies.auth import get_current_user
from ..models.db.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/process", tags=["process"])


@router.post("", response_model=ProcessResponse)
async def process_file(
    payload: ProcessRequest,
    _user: User = Depends(get_current_user),
):
    """
    Endpoint para procesar archivos de radar previamente subidos.

    IMPORTANT: The heavy computation is offloaded to a thread via
    run_in_threadpool so that the asyncio event-loop is never blocked.
    Without this, a single long /process request (e.g. building the W
    operator) would freeze the entire server for ALL clients.
    """
    try:
        return await run_in_threadpool(
            ProcessingOrchestrator.process_radar_files, payload, str(_user.id)
        )
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
        logger.exception("Error procesando archivos")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando archivos: {e}"
        )


@router.post("/animation/gif", response_model=GifAnimationResponse)
async def create_gif_animation(
    payload: GifAnimationRequest,
    _user: User = Depends(get_current_user),
):
    """Genera un GIF animado a partir de rasters procesados ya existentes."""
    try:
        images_dir = Path(settings.IMAGES_DIR)
        output_dir = (
            images_dir / payload.session_id
            if payload.session_id
            else images_dir
        )

        gif_name = await run_in_threadpool(
            create_animation_from_layer_urls,
            payload.frames,
            str(output_dir),
            payload.fps,
            payload.session_id,
            payload.basemap_id,
            payload.show_logo,
            payload.show_colorbar,
            payload.colorbar_config,
            payload.show_metadata,
            payload.frame_labels,
        )

        gif_url = (
            f"/static/tmp/{payload.session_id}/{gif_name}"
            if payload.session_id
            else f"/static/tmp/{gif_name}"
        )
        return GifAnimationResponse(gif_url=gif_url)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Error generando GIF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando GIF: {e}"
        )
