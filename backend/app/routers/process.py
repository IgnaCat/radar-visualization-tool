import asyncio
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, status, Depends
from fastapi.concurrency import run_in_threadpool

from ..core.config import settings
from ..core.concurrency import acquire_processing_slot, release_processing_slot
from ..core.cancellation import CancellationToken, CancelledException, SESSION_ACTIVE_TOKENS
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
    request: Request,
    payload: ProcessRequest,
    _user: User = Depends(get_current_user),
):
    """
    Endpoint para procesar archivos de radar previamente subidos.

    Concurrency control:
    - acquire_processing_slot() limits how many heavy pipelines run at once
      (default: 2 simultaneous). Extra requests wait in queue up to 5 min,
      then get 503 Service Unavailable.
    - run_in_threadpool offloads CPU work to a thread so the asyncio
      event-loop stays responsive for lightweight requests (tiles, health).
    - numpy/scipy/rasterio release the GIL, so threads DO run in parallel
      for the heavy math. build_W_operator also uses multiprocessing.Pool
      internally for per-level parallelism.

    Cancellation:
    - Mientras el pipeline corre, el event loop sondea request.is_disconnected()
      cada 500ms. Si el cliente se fue (F5, cierre de pestaña, crash), activa
      el CancellationToken de la sesión. El pipeline chequea el token en puntos
      estratégicos (antes del W operator, entre archivos, entre campos) y lanza
      CancelledException para salir limpiamente y liberar el semáforo.
    - El multiprocessing.Pool de build_W_operator no se puede interrumpir
      mid-flight, pero termina solo sin leak y el semáforo se libera igual.
    """
    session_id = payload.session_id

    # Crear y registrar token de cancelación para esta sesión
    token = CancellationToken()
    if session_id:
        SESSION_ACTIVE_TOKENS[session_id] = token

    await acquire_processing_slot()
    try:
        # Lanzar el pipeline en threadpool y simultáneamente sondear desconexión
        task = asyncio.create_task(
            run_in_threadpool(
                ProcessingOrchestrator.process_radar_files, payload, str(_user.id)
            )
        )

        while not task.done():
            if await request.is_disconnected():
                token.cancel()
                logger.info(
                    "process_file: cliente desconectado, cancelando sesión '%s'",
                    session_id,
                )
                # Dar 3s para que el thread salga limpiamente por CancelledException
                # antes de abandonar la tarea (el Pool del W seguirá pero termina solo)
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                # La respuesta de todas formas no llega al cliente
                raise HTTPException(status_code=499, detail="Client disconnected")
            await asyncio.sleep(0.5)

        return task.result()

    except CancelledException:
        # Pipeline salió limpiamente por cancelación cooperativa
        raise HTTPException(status_code=499, detail="Client disconnected")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error procesando archivos")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando archivos: {e}",
        )
    finally:
        # Siempre limpiar el token y liberar el semáforo, incluso si hubo excepción
        if session_id:
            SESSION_ACTIVE_TOKENS.pop(session_id, None)
        release_processing_slot()


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
