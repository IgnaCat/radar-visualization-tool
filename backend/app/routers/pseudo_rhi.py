import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from ..schemas import PseudoRHIRequest, PseudoRHIResponse, RangeFilter
from ..services.pseudo_rhi import generate_pseudo_rhi_png
from ..core.config import settings
from ..utils import helpers

router = APIRouter(prefix="/process", tags=["process"])

@router.post("/pseudo_rhi", response_model=List[PseudoRHIResponse])
async def pseudo_rhi(payload: PseudoRHIRequest):
    """
    Generate a pseudo RHI images.
    """

    filepaths: List[str] = payload.filepaths
    field: str = payload.field
    end_lon: float = payload.end_lon
    end_lat: float = payload.end_lat
    start_lon: float = payload.start_lon
    start_lat: float = payload.start_lat
    max_length_km: float = payload.max_length_km
    max_height_km: float = payload.max_height_km
    elevation: int = payload.elevation
    filters: List[RangeFilter] = payload.filters

    # Validar inputs
    if not filepaths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debe proporcionar una lista de 'filepaths'"
        )
    if max_length_km <= 0 or max_length_km > 240:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La longitud máxima debe estar entre 0 y 240 km."
        )
    if max_height_km < 0.5 or max_height_km > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La altura máxima debe estar entre 0.5 y 30 km."
        )
    
    if elevation < 0 or elevation > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El ángulo de elevación debe estar entre 0 y 12."
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
    
    # Procesar y generar PNG
    try:
        # Limpieza de temporales (en threadpool para no bloquear)
        await run_in_threadpool(helpers.cleanup_tmp)

        processed: List[PseudoRHIResponse] = []

        for file in filepaths:
            filepath = Path(UPLOAD_DIR) / file

            # Extraer metadata del nombre del archivo
            _, _, _, timestamp = await run_in_threadpool(
                helpers.extract_metadata_from_filename, filepath
            )

            # Ejecutar el procesamiento bloqueante para generar PNG
            result_dict = await run_in_threadpool(
                generate_pseudo_rhi_png,
                filepath=filepath,
                field=field,
                end_lon=end_lon,
                end_lat=end_lat,
                max_length_km=max_length_km,
                max_height_km=max_height_km,
                elevation=elevation,
                filters=filters,
                start_lon=start_lon,
                start_lat=start_lat,
                colormap_overrides=payload.colormap_overrides,
                session_id=payload.session_id,
            )
            result_dict["timestamp"] = timestamp
            processed.append(PseudoRHIResponse(**result_dict))

        return processed
    
    except HTTPException:
        # Re-emitir las HTTPException tal cual
        raise
    except Exception as e:
        # 500 Internal Server Error
        print(f"Error pseudo RHI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error pseudo RHI: {e}"
        )