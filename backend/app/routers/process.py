from fileinput import filename
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
import os

from ..schemas import ProcessRequest, ProcessResponse, LayerResult, RangeFilter
from ..core.config import settings

from ..services import radar_processor
from ..utils import helpers

router = APIRouter(prefix="/process", tags=["process"])

def _timestamp_of(filepath: str):
    try:
        _,_,_, timestamp = helpers.extract_metadata_from_filename(filepath)
        return timestamp  # datetime | None
    except Exception:
        return None

@router.post("", response_model=ProcessResponse)
async def process_file(payload: ProcessRequest):
    """
    Endpoint para procesar archivos de radar previamente subidos.
    """
    filepaths: List[str] = payload.filepaths
    product: str = payload.product
    fields: List[str] = payload.fields
    height: int = payload.height
    elevation: int = payload.elevation
    filters: List[RangeFilter] = payload.filters
    selected_volumes: List[str] = getattr(payload, "selectedVolumes", None) or []
    warnings: List[str] = []

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
    if elevation < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El ángulo de elevación debe ser positivo."
        )

    # Verificar que se proporcionen filepaths
    if not filepaths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debe proporcionar una lista de 'filepaths'"
        )   
        
    UPLOAD_DIR = settings.UPLOAD_DIR
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Verificar que los archivos existan
    for file in filepaths:
        filepath = os.path.join(UPLOAD_DIR, file)
        if not Path(filepath).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Archivo no encontrado: {file}"
            )
        
    # Filtrar archivos por los volúmenes seleccionados
    if len(selected_volumes) > 0:
        def extract_volume_from_filename(filename):
            _,_,volume,_ = helpers.extract_metadata_from_filename(filename)
            return str(volume) if volume else None
        filtered_filepaths = []
        for f in filepaths:
            vol = extract_volume_from_filename(f)
            filename = Path(f).name
            if vol == '03' and product.upper() == 'PPI':
                warnings.append(f"{filename}: El volumen '03' no es válido para el producto PPI.")
                continue
            if vol in selected_volumes:
                filtered_filepaths.append(f)
            else:
                warnings.append(f"{filename}: Volumen '{vol}' no seleccionado, se omite.")
    else:
        print("No se seleccionaron volúmenes, procesando todos los archivos.")
        warnings.append("No se seleccionaron volúmenes, procesando todo.")

    filepaths = filtered_filepaths
    files_with_ts = [(f, _timestamp_of(f)) for f in filepaths]
    files_sorted = sorted(files_with_ts, key=lambda x: (x[1] is None, x[1] or 0))
    try:
        # Limpieza de temporales (en threadpool para no bloquear)
        await run_in_threadpool(helpers.cleanup_tmp)

        frames: List[List[LayerResult]] = []
        # Iteramos por file → procesar todas las capas (posible paralelismo)
        for file, timestamp in files_sorted:
            filepath = Path(UPLOAD_DIR) / file

            future_to_meta = {} # mapear cada future a su metadata (idx, field)
            results: List[LayerResult] = [] # imagenes procesadas de este archivo
            with ThreadPoolExecutor(max_workers=min(8, len(fields))) as ex:
                for idx, field in enumerate(fields):

                    fut = ex.submit(
                        radar_processor.process_radar_to_cog,
                        filepath=filepath,
                        product=product,
                        field_requested=field,
                        cappi_height=height,
                        elevation=elevation,
                        filters=filters,
                        volume=extract_volume_from_filename(filepath.name)
                    )
                    future_to_meta[fut] = (idx, field)

                for future in as_completed(future_to_meta):
                    idx, field = future_to_meta[future]
                    try:
                        result_dict = future.result()
                        result_dict["timestamp"] = timestamp
                        result_dict["order"] = idx
                        results.append(LayerResult(**result_dict))
                    except Exception as e:
                        print(f"Error procesando {field}: {e}")
                        warnings.append(f"{filepath.name}: {e}")

            if results:
                # ordenar capas por order antes de agregar el frame
                results.sort(key=lambda r: r.order)
                frames.append(results)

        if not frames:
            warnings.append("No se generaron imágenes de salida.")
            
        # Decidir si animación
        # animate = await run_in_threadpool(helpers.should_animate, [r.dict() for r in frames])

        return ProcessResponse(
            animation=(len(frames) > 1),
            outputs=frames,
            product=product,
            warnings=warnings
        )

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
