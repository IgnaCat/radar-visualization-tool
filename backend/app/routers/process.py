from fileinput import filename
from fastapi import APIRouter, HTTPException, status
<<<<<<< HEAD
from fastapi.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor, as_completed
=======
>>>>>>> 14ecc66fede379e1713e30b02a21c905ba0baad7
from pathlib import Path
from typing import List
from math import ceil
import os

from ..schemas import ProcessRequest, ProcessResponse, LayerResult, RangeFilter, RadarProcessResult
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
    selected_radars: List[str] = getattr(payload, "selectedRadars", None) or []
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
        
    # Determinar directorio de uploads según session_id
    UPLOAD_DIR = Path(settings.UPLOAD_DIR)
    if payload.session_id:
        UPLOAD_DIR = UPLOAD_DIR / payload.session_id
    UPLOAD_DIR = str(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filtered_filepaths = []

    # Límite duro: máximo 3 radares en simultáneo
    if selected_radars and len(selected_radars) > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pueden seleccionar más de 3 radares a la vez."
        )

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
        for f in filepaths:
            vol = helpers.extract_volume_from_filename(f)
            filename = Path(f).name
            if vol == '03' and product.upper() == 'PPI':
                warnings.append(f"{filename}: El volumen '03' no es válido para el producto PPI.")
                continue
            if vol in selected_volumes:
                filtered_filepaths.append(f)
            else:
                warnings.append(f"{filename}: Volumen '{vol}' no seleccionado, se omite.")
        filepaths = filtered_filepaths
    else:
        print("No se seleccionaron volúmenes, procesando todos los archivos.")
        warnings.append("No se seleccionaron volúmenes, procesando todo.")

    # Filtrar archivos por radares seleccionados (site)
    if selected_radars:
        fp2 = []
        for f in filepaths:
            try:
                radar, _, _, _ = helpers.extract_metadata_from_filename(Path(f).name)
            except Exception:
                radar = None
            if radar and radar in selected_radars:
                fp2.append(f)
            else:
                warnings.append(f"{Path(f).name}: Radar '{radar}' no seleccionado, se omite.")
        filepaths = fp2

    # Preparamos (filepath_abs, ts, vol, radar) por archivo
    items = []
    for f in filepaths:
        fp_abs = str(Path(UPLOAD_DIR) / f)
        ts = _timestamp_of(f)  # datetime|None
        radar, _, vol, _ = helpers.extract_metadata_from_filename(Path(f).name)
        items.append((f, fp_abs, ts, vol, radar))

<<<<<<< HEAD
    # Lanzamos todo en paralelo: (archivo × campo)
    future_to_meta = {}
    max_workers = min(max(4, (os.cpu_count() or 4) * 2), len(items) * max(1, len(fields)))
    # Agrupación: radar -> { timestamp -> [LayerResult, ...] }
    results_by_radar = {}
    warnings_by_radar = {}


=======
    # Crear directorio de salida temporal para la sesión ANTES del procesamiento paralelo
    # Esto evita race conditions cuando múltiples threads intentan crear el directorio
    if payload.session_id:
        session_tmp_dir = Path(settings.IMAGES_DIR) / payload.session_id
        os.makedirs(session_tmp_dir, exist_ok=True)

    # Agrupación: radar -> { timestamp -> [LayerResult, ...] }
    results_by_radar = {}
    warnings_by_radar = {}
>>>>>>> 14ecc66fede379e1713e30b02a21c905ba0baad7
    # Para trackear campos y volúmenes por radar
    fields_by_radar = {}
    volumes_by_radar = {}

<<<<<<< HEAD
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for (f_rel, f_abs, ts, vol, radar) in items:
                for idx, field in enumerate(fields):
                    fut = ex.submit(
                        radar_processor.process_radar_to_cog,
=======
    # Procesamiento secuencial - PyART/GDAL/NetCDF4 no son thread-safe
    # ThreadPoolExecutor causa corrupción de memoria (malloc errors) incluso sin volúmenes montados
    try:
        for (f_rel, f_abs, ts, vol, radar) in items:
            for idx, field in enumerate(fields):
                try:
                    result_dict = radar_processor.process_radar_to_cog(
>>>>>>> 14ecc66fede379e1713e30b02a21c905ba0baad7
                        filepath=f_abs,
                        product=product,
                        field_requested=field,
                        cappi_height=height,
                        elevation=elevation,
                        filters=filters,
                        volume=vol,
                        colormap_overrides=payload.colormap_overrides,
                        session_id=payload.session_id
                    )
<<<<<<< HEAD
                    future_to_meta[fut] = (f_rel, ts, idx, field, radar, vol)

            for fut in as_completed(future_to_meta):
                f_rel, ts, idx, field, radar, vol = future_to_meta[fut]
                try:
                    result_dict = fut.result()
=======
>>>>>>> 14ecc66fede379e1713e30b02a21c905ba0baad7
                    result_dict["timestamp"] = ts
                    result_dict["order"] = idx
                    # Agrupar por radar y timestamp
                    if radar not in results_by_radar:
                        results_by_radar[radar] = {}
                    if ts not in results_by_radar[radar]:
                        results_by_radar[radar][ts] = []
                    results_by_radar[radar][ts].append(LayerResult(**result_dict))
                    # Track campos y volúmenes
                    fields_by_radar.setdefault(radar, set()).add(field)
                    if vol:
                        volumes_by_radar.setdefault(radar, set()).add(vol)
                except Exception as e:
<<<<<<< HEAD
                    print(f"Error procesando {field}: {e}")
=======
>>>>>>> 14ecc66fede379e1713e30b02a21c905ba0baad7
                    if radar not in warnings_by_radar:
                        warnings_by_radar[radar] = []
                    warnings_by_radar[radar].append(f"{Path(f_rel).name}: {e}")

<<<<<<< HEAD

=======
>>>>>>> 14ecc66fede379e1713e30b02a21c905ba0baad7
        # Calcular warnings por campos/volúmenes faltantes
        all_fields = set()
        all_volumes = set()
        for s in fields_by_radar.values():
            all_fields.update(s)
        for s in volumes_by_radar.values():
            all_volumes.update(s)

        for radar in results_by_radar:
            missing_fields = all_fields - fields_by_radar.get(radar, set())
            missing_vols = all_volumes - volumes_by_radar.get(radar, set())
            if missing_fields:
                warnings_by_radar.setdefault(radar, []).append(
                    f"El radar {radar} no tiene los siguientes campos: {', '.join(sorted(missing_fields))}"
                )
            if missing_vols:
                warnings_by_radar.setdefault(radar, []).append(
                    f"El radar {radar} no tiene los siguientes volúmenes: {', '.join(sorted(missing_vols))}"
                )

        # Unificar todos los warnings (globales y por radar)
        all_warnings = list(warnings)
        for radar_warns in warnings_by_radar.values():
            all_warnings.extend(radar_warns)

        # Para cada radar, ordenar frames por timestamp y capas por 'order'
        radar_results = []
        for radar, ts_dict in results_by_radar.items():
            # Ordenar timestamps (None al final)
            sorted_ts = sorted(ts_dict.keys(), key=lambda t: (t is None, t or 0))
            frames = []
            for ts in sorted_ts:
                layers = ts_dict[ts]
                layers.sort(key=lambda r: r.order)
                frames.append(layers)
            # Decidir si animación
            animation = len(frames) > 1
            radar_results.append(RadarProcessResult(
                radar=radar,
                animation=animation,
                outputs=frames,
            ))

        if not radar_results:
            all_warnings.append("No se generaron imágenes de salida.")

        return ProcessResponse(
            results=radar_results,
            product=product,
            warnings=all_warnings
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
