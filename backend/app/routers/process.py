from fileinput import filename
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from typing import List
import os

from ..schemas import ProcessRequest, ProcessResponse, ProcessOutput
from ..core.config import settings

from ..services import radar_processor
from ..utils import helpers

router = APIRouter(prefix="/process", tags=["process"])

@router.post("", response_model=ProcessResponse)
async def process_file(payload: ProcessRequest):
    """
    Endpoint para procesar archivos de radar previamente subidos.
    """
    filepaths: List[str] = payload.filepaths
    product: str = payload.product
    height: int = payload.height
    elevation: int = payload.elevation

    # Validar inputs
    if product.upper() not in settings.ALLOWED_PRODUCTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Producto '{product}' no permitido. Debe ser uno de {settings.ALLOWED_PRODUCTS}"
        )
    if height < 0 or height > 12000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La altura debe estar entre 0 y 12000 metros."
        )
    if elevation < 0 or elevation > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El ángulo de elevación debe estar entre 0 y 12."
        )

    # Verificar que se proporcionen filepaths
    if not filepaths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debe proporcionar una lista de 'filepaths'"
        )
    
    UPLOAD_DIR = settings.UPLOAD_DIR
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in filepaths:
        filepath = os.path.join(UPLOAD_DIR, file)
        if not Path(filepath).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Archivo no encontrado: {file}"
            )

    try:

        # Limpieza de temporales (en threadpool para no bloquear)
        await run_in_threadpool(helpers.cleanup_tmp)

        processed: List[ProcessOutput] = []

        for file in filepaths:
            filepath = Path(UPLOAD_DIR) / file

            # Extraer metadata del nombre del archivo
            _, _, _, timestamp = await run_in_threadpool(
                helpers.extract_metadata_from_filename, filepath
            )

            # Ejecutar el procesamiento bloqueante para generar PNG
            #result_dict = await run_in_threadpool(radar_processor.process_radar, filepath)


            # Generar COG
            result_dict = await run_in_threadpool(radar_processor.process_radar_to_cog, filepath, product, height, elevation)

            result_dict["timestamp"] = timestamp
            processed.append(ProcessOutput(**result_dict))

        # Decidir si animación
        animate = await run_in_threadpool(helpers.should_animate, [r.dict() for r in processed])

        # Ordenar los resultados por timestamp
        processed.sort(key=lambda x: x.timestamp)

        return ProcessResponse(animation=bool(animate), outputs=processed, product=product)

    except HTTPException:
        # Re-emitir las HTTPException tal cual
        raise
    except Exception as e:
        # 500 Internal Server Error
        print(f"Error procesando archivos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando archivos: {e}"
        )
