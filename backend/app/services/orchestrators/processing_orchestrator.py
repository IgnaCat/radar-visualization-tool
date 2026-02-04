"""
Orchestrator para coordinar el procesamiento de múltiples archivos de radar.
Contiene la lógica de negocio previamente en el router process.py.
"""
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import traceback

from ...models import ProcessRequest, ProcessResponse, LayerResult, RangeFilter, RadarProcessResult
from ...core.config import settings
from .. import radar_processor
from ...utils import helpers


class ProcessingOrchestrator:
    """
    Coordina el procesamiento de múltiples archivos de radar.
    Maneja validación, filtrado, agrupación y procesamiento paralelo.
    """

    @staticmethod
    def validate_request(payload: ProcessRequest) -> List[str]:
        """
        Valida los parámetros de la solicitud.
        Returns: Lista de warnings (vacía si todo está correcto)
        Raises: ValueError con mensaje de error si hay problemas críticos
        """
        warnings = []
        
        # Validar producto
        if payload.product.upper() not in settings.ALLOWED_PRODUCTS:
            raise ValueError(
                f"Producto '{payload.product}' no permitido. Debe ser uno de {settings.ALLOWED_PRODUCTS}"
            )
        
        # Validar altura
        if payload.height < 0 or payload.height > 12000:
            raise ValueError("La altura debe estar entre 0 y 12000 metros.")
        
        # Validar elevación
        if payload.elevation < 0:
            raise ValueError("El ángulo de elevación debe ser positivo.")
        
        # Validar filepaths
        if not payload.filepaths:
            raise ValueError("Debe proporcionar una lista de 'filepaths'")
        
        # Validar límite de radares
        selected_radars = getattr(payload, "selectedRadars", None) or []
        if selected_radars and len(selected_radars) > 3:
            raise ValueError("No se pueden seleccionar más de 3 radares a la vez.")
        
        return warnings

    @staticmethod
    def get_upload_directory(session_id: Optional[str] = None) -> Path:
        """
        Determina el directorio de uploads según session_id.
        Crea el directorio si no existe.
        """
        upload_dir = Path(settings.UPLOAD_DIR)
        if session_id:
            upload_dir = upload_dir / session_id
        os.makedirs(upload_dir, exist_ok=True)
        return upload_dir

    @staticmethod
    def verify_files_exist(filepaths: List[str], upload_dir: Path) -> None:
        """
        Verifica que todos los archivos existan.
        Raises: ValueError si algún archivo no existe
        """
        for file in filepaths:
            filepath = upload_dir / file
            if not filepath.exists():
                raise ValueError(f"Archivo no encontrado: {file}")

    @staticmethod
    def filter_by_volumes(
        filepaths: List[str],
        selected_volumes: List[str],
        product: str
    ) -> Tuple[List[str], List[str]]:
        """
        Filtra archivos por los volúmenes seleccionados.
        Returns: (filtered_filepaths, warnings)
        """
        warnings = []
        
        if not selected_volumes:
            msg = "No se seleccionaron volúmenes, procesando todo."
            warnings.append(msg)
            print(f"[WARNING] {msg}")
            return filepaths, warnings
        
        filtered_filepaths = []
        for f in filepaths:
            vol = helpers.extract_volume_from_filename(f)
            filename = Path(f).name
            
            # Validar volumen 03 con producto PPI
            if vol == '03' and product.upper() == 'PPI':
                msg = f"{filename}: El volumen '03' no es válido para el producto PPI."
                warnings.append(msg)
                print(f"[WARNING] {msg}")
                continue
            
            if vol in selected_volumes:
                filtered_filepaths.append(f)
            else:
                msg = f"{filename}: Volumen '{vol}' no seleccionado, se omite."
                warnings.append(msg)
                print(f"[WARNING] {msg}")
        
        return filtered_filepaths, warnings

    @staticmethod
    def filter_by_radars(
        filepaths: List[str],
        selected_radars: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Filtra archivos por radares seleccionados.
        Returns: (filtered_filepaths, warnings)
        """
        warnings = []
        
        if not selected_radars:
            return filepaths, warnings
        
        filtered_filepaths = []
        for f in filepaths:
            try:
                radar, _, _, _ = helpers.extract_metadata_from_filename(Path(f).name)
            except Exception:
                radar = None
            
            if radar and radar in selected_radars:
                filtered_filepaths.append(f)
            else:
                msg = f"{Path(f).name}: Radar '{radar}' no seleccionado, se omite."
                warnings.append(msg)
                print(f"[WARNING] {msg}")
        
        return filtered_filepaths, warnings

    @staticmethod
    def extract_timestamp(filepath: str) -> Optional[datetime]:
        """
        Extrae el timestamp de un archivo.
        Returns: datetime o None si no se puede extraer
        """
        try:
            _, _, _, timestamp = helpers.extract_metadata_from_filename(filepath)
            return timestamp
        except Exception:
            return None

    @staticmethod
    def prepare_items(
        filepaths: List[str],
        upload_dir: Path
    ) -> List[Tuple[str, str, Optional[datetime], str, str, str]]:
        """
        Prepara la lista de items para procesar.
        Returns: List of (filepath_rel, filepath_abs, timestamp, volume, radar, estrategia)
        """
        items = []
        for f in filepaths:
            fp_abs = str(upload_dir / f)
            ts = ProcessingOrchestrator.extract_timestamp(f)
            radar, estrategia, vol, _ = helpers.extract_metadata_from_filename(Path(f).name)
            items.append((f, fp_abs, ts, vol, radar, estrategia))
        return items

    @staticmethod
    def create_session_tmp_dir(session_id: Optional[str]) -> None:
        """
        Crea directorio temporal para la sesión si existe session_id.
        """
        if session_id:
            session_tmp_dir = Path(settings.IMAGES_DIR) / session_id
            os.makedirs(session_tmp_dir, exist_ok=True)

    @staticmethod
    def process_files(
        items: List[Tuple[str, str, Optional[datetime], str, str]],
        product: str,
        fields: List[str],
        height: int,
        elevation: int,
        filters: List[RangeFilter],
        colormap_overrides: Optional[Dict] = None,
        session_id: Optional[str] = None,
        max_workers: int = 4
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Procesa archivos de radar en paralelo por archivo.
        
        Cada archivo NetCDF es independiente y se procesa en un thread separado.
        Internamente cada archivo puede paralelizar por niveles Z si el operador W
        no está cacheado. El lock por cache_key evita construcción duplicada.
        
        Args:
            items: Lista de (filepath_rel, filepath_abs, timestamp, volume, radar, estrategia)
            max_workers: Número máximo de threads (default 4, conservador porque cada
                        archivo puede usar threads internos para niveles Z)
        
        Returns:
            (results_by_radar, warnings_by_radar, fields_by_radar, volumes_by_radar)
        """
        results_by_radar = {}
        warnings_by_radar = {}
        fields_by_radar = {}
        volumes_by_radar = {}

        def process_single_item(item_data):
            """Procesa un archivo con todos sus campos. Retorna lista de resultados."""
            item_idx, (f_rel, f_abs, ts, vol, radar, estrategia) = item_data
            item_results = []
            item_warnings = []
            item_fields = set()
            
            for field_idx, field in enumerate(fields):
                try:
                    result_dict = radar_processor.process_radar_to_cog(
                        filepath=f_abs,
                        product=product,
                        field_requested=field,
                        cappi_height=height,
                        elevation=elevation,
                        filters=filters,
                        radar_name=radar,
                        estrategia=estrategia,
                        volume=vol,
                        colormap_overrides=colormap_overrides,
                        session_id=session_id
                    )
                    result_dict["timestamp"] = ts
                    result_dict["order"] = field_idx
                    item_results.append((radar, ts, vol, field, LayerResult(**result_dict)))
                    item_fields.add(field)
                    
                except Exception as e:
                    msg = f"{Path(f_rel).name}: {e}"
                    item_warnings.append((radar, msg))
                    print(f"[ERROR] {radar}: {msg}")
                    traceback.print_exc()
            
            return item_results, item_warnings, item_fields, vol, radar

        # Preparar items con índice para tracking
        indexed_items = list(enumerate(items))
        
        # Procesar en paralelo (max_workers conservador por paralelismo interno de niveles Z)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_item, item): item for item in indexed_items}
            
            for future in as_completed(futures):
                try:
                    item_results, item_warnings, item_fields, vol, radar = future.result()
                    
                    # Agregar resultados
                    for (r_radar, r_ts, r_vol, r_field, layer_result) in item_results:
                        if r_radar not in results_by_radar:
                            results_by_radar[r_radar] = {}
                        if r_ts not in results_by_radar[r_radar]:
                            results_by_radar[r_radar][r_ts] = []
                        results_by_radar[r_radar][r_ts].append(layer_result)
                        
                        fields_by_radar.setdefault(r_radar, set()).add(r_field)
                        if r_vol:
                            volumes_by_radar.setdefault(r_radar, set()).add(r_vol)
                    
                    # Agregar warnings
                    for (w_radar, msg) in item_warnings:
                        warnings_by_radar.setdefault(w_radar, []).append(msg)
                        
                except Exception as e:
                    print(f"[ERROR] Error procesando future: {e}")
                    traceback.print_exc()

        return results_by_radar, warnings_by_radar, fields_by_radar, volumes_by_radar

    @staticmethod
    def calculate_missing_fields_warnings(
        results_by_radar: Dict,
        fields_by_radar: Dict,
        volumes_by_radar: Dict,
        warnings_by_radar: Dict
    ) -> None:
        """
        Calcula warnings por campos/volúmenes faltantes.
        Modifica warnings_by_radar in-place.
        """
        # Calcular todos los campos y volúmenes disponibles
        all_fields = set()
        all_volumes = set()
        for s in fields_by_radar.values():
            all_fields.update(s)
        for s in volumes_by_radar.values():
            all_volumes.update(s)

        # Generar warnings para cada radar
        for radar in results_by_radar:
            missing_fields = all_fields - fields_by_radar.get(radar, set())
            missing_vols = all_volumes - volumes_by_radar.get(radar, set())
            
            if missing_fields:
                msg = f"El radar {radar} no tiene los siguientes campos: {', '.join(sorted(missing_fields))}"
                warnings_by_radar.setdefault(radar, []).append(msg)
                print(f"[WARNING] {msg}")
            if missing_vols:
                msg = f"El radar {radar} no tiene los siguientes volúmenes: {', '.join(sorted(missing_vols))}"
                warnings_by_radar.setdefault(radar, []).append(msg)
                print(f"[WARNING] {msg}")

    @staticmethod
    def build_radar_results(
        results_by_radar: Dict,
        warnings_by_radar: Dict,
        initial_warnings: List[str]
    ) -> Tuple[List[RadarProcessResult], List[str]]:
        """
        Construye la respuesta final con resultados por radar.
        Returns: (radar_results, all_warnings)
        """
        # Unificar todos los warnings (globales y por radar)
        all_warnings = list(initial_warnings)
        for radar_warns in warnings_by_radar.values():
            all_warnings.extend(radar_warns)

        # Para cada radar, ordenar frames por timestamp y capas por 'order'
        radar_results = []
        for radar, ts_dict in results_by_radar.items():
            # Ordenar timestamps (None al final usando tupla: is_none, timestamp)
            sorted_ts = sorted(ts_dict.keys(), key=lambda t: (t is None, t or datetime.min))
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
            msg = "No se generaron imágenes de salida."
            all_warnings.append(msg)
            print(f"[WARNING] {msg}")

        return radar_results, all_warnings

    @staticmethod
    def process_radar_files(payload: ProcessRequest) -> ProcessResponse:
        """
        Método principal que orquesta todo el procesamiento.
        """
        # 1. Validar request
        warnings = ProcessingOrchestrator.validate_request(payload)

        # 2. Obtener directorio de uploads
        upload_dir = ProcessingOrchestrator.get_upload_directory(payload.session_id)

        # 3. Verificar que archivos existan
        ProcessingOrchestrator.verify_files_exist(payload.filepaths, upload_dir)

        # 4. Filtrar por volúmenes
        selected_volumes = getattr(payload, "selectedVolumes", None) or []
        filepaths, volume_warnings = ProcessingOrchestrator.filter_by_volumes(
            payload.filepaths,
            selected_volumes,
            payload.product
        )
        warnings.extend(volume_warnings)

        # 5. Filtrar por radares
        selected_radars = getattr(payload, "selectedRadars", None) or []
        filepaths, radar_warnings = ProcessingOrchestrator.filter_by_radars(
            filepaths,
            selected_radars
        )
        warnings.extend(radar_warnings)

        # 6. Preparar items para procesar
        items = ProcessingOrchestrator.prepare_items(filepaths, upload_dir)

        # 7. Crear directorio temporal de sesión
        ProcessingOrchestrator.create_session_tmp_dir(payload.session_id)

        # 8. Procesar archivos
        results_by_radar, warnings_by_radar, fields_by_radar, volumes_by_radar = \
            ProcessingOrchestrator.process_files(
                items,
                payload.product,
                payload.fields,
                payload.height,
                payload.elevation,
                payload.filters,
                payload.colormap_overrides,
                payload.session_id
            )

        # 9. Calcular warnings por campos/volúmenes faltantes
        ProcessingOrchestrator.calculate_missing_fields_warnings(
            results_by_radar,
            fields_by_radar,
            volumes_by_radar,
            warnings_by_radar
        )

        # 10. Construir respuesta final
        radar_results, all_warnings = ProcessingOrchestrator.build_radar_results(
            results_by_radar,
            warnings_by_radar,
            warnings
        )

        return ProcessResponse(
            results=radar_results,
            product=payload.product,
            warnings=all_warnings
        )
