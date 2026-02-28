"""
Orchestrator para cálculo de estadísticas sobre áreas de radar.
Contiene la lógica de negocio previamente en el router radar_stats.py.
"""
import pyart
import pyproj
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as shp_transform
from rasterio.features import geometry_mask

from ...models import RadarStatsRequest, RadarStatsResponse
from ...core.cache import GRID2D_CACHE
from ...core.config import settings
from ...utils.helpers import extract_metadata_from_filename
from ..radar_common import (
    grid2d_cache_key,
    md5_file,
)
from ..radar_processing import (
    separate_filters,
    apply_visual_filters,
    apply_qc_filters,
)
from ..grid_generator import generate_grid2d_on_demand


ALLOWED_GEOMS = {"Polygon", "MultiPolygon"}


class StatsOrchestrator:
    """
    Coordina el cálculo de estadísticas sobre polígonos de radar.
    Usa la grilla 2D cacheada sin tocar disco.
    """

    @staticmethod
    def validate_request(payload: RadarStatsRequest) -> None:
        """
        Valida los parámetros de la solicitud.
        Raises: ValueError si hay problemas críticos
        """
        if getattr(payload, "filepath", None) in (None, "", "undefined"):
            raise ValueError("El campo 'filepath' es obligatorio.")

    @staticmethod
    def get_filepath(payload: RadarStatsRequest) -> str:
        """
        Construye el path completo del archivo desde el request.
        
        Returns:
            Path absoluto al archivo de radar
        """
        UPLOAD_DIR = Path(settings.UPLOAD_DIR)
        if payload.session_id:
            UPLOAD_DIR = UPLOAD_DIR / payload.session_id
        return str(UPLOAD_DIR / payload.filepath)

    @staticmethod
    def resolve_field_name(product: str, field: str) -> str:
        """
        Resuelve el nombre del campo según el producto.
        
        Args:
            product: Tipo de producto (PPI, CAPPI, COLMAX)
            field: Campo solicitado
        
        Returns:
            Nombre del campo resuelto
        """
        if product.upper() == "CAPPI":
            return "cappi"
        if product.upper() == "COLMAX" and field.upper() == "DBZH":
            return "composite_reflectivity"
        return field

    @staticmethod
    def generate_cache_key(
        filepath: str,
        product: str,
        field: str,
        elevation: Optional[int] = 0,
        cappi_height: Optional[int] = 4000,
        volume: Optional[str] = None,
        filters: Optional[List] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Genera cache key incluyendo filtros QC (afectan interpolación).
        
        Returns:
            Cache key para GRID2D_CACHE
        """
        product_upper = product.upper()
        field_to_use = field.upper()
        interp = "Barnes2"

        # Hash del archivo
        file_hash = md5_file(filepath)[:12]
        
        # Generar signature de qc_filters para cache key
        qc_filters, _ = separate_filters(filters or [], field_to_use)
        qc_sig = tuple(sorted([
            (f.field, f.min, f.max) for f in qc_filters
        ])) if qc_filters else tuple()

        cache_key = grid2d_cache_key(
            file_hash=file_hash,
            product_upper=product_upper,
            field_to_use=field_to_use,
            elevation=elevation if product_upper == "PPI" else None,
            cappi_height=cappi_height if product_upper == "CAPPI" else None,
            volume=volume,
            interp=interp,
            qc_sig=qc_sig,  # Incluir filtros QC en cache key
            session_id=session_id,
        )

        return cache_key

    @staticmethod
    def apply_filters_to_cached_data(
        arr: np.ma.MaskedArray,
        pkg: Dict,
        filters: List,
        field: str
    ) -> np.ma.MaskedArray:
        """
        Aplica filtros dinámicamente sobre el array cacheado.
        
        Args:
            arr: Array cacheado (posiblemente warped)
            pkg: Package del cache con metadata
            filters: Lista de filtros a aplicar
            field: Nombre del campo
        
        Returns:
            Array con filtros aplicados
        """
        field_to_use = field.upper()
        _, visual_filters = separate_filters(filters, field_to_use)
        
        # Solo aplicar filtros visuales - los QC ya están aplicados en el cache
        # (fueron aplicados durante la interpolación al generar la grilla)
        arr = apply_visual_filters(arr, visual_filters, field_to_use)
        
        return arr

    @staticmethod
    def extract_geometry(obj: dict) -> dict:
        """
        Acepta Geometry, Feature o FeatureCollection y devuelve un Geometry.
        
        Args:
            obj: Objeto GeoJSON
        
        Returns:
            Geometría extraída
        """
        t = obj.get("type", "")
        if t in ALLOWED_GEOMS:
            return obj
        if t == "Feature":
            return obj.get("geometry", {})
        if t == "FeatureCollection":
            features = obj.get("features", [])
            if not features:
                raise ValueError("FeatureCollection vacía")
            return features[0].get("geometry", {})
        raise ValueError(f"Tipo de geometría no soportado: {t}")

    @staticmethod
    def reproject_polygon(
        polygon_gj_4326: dict,
        target_crs: pyproj.CRS
    ) -> dict:
        """
        Reproyecta un polígono de WGS84 al CRS del grid.
        
        Args:
            polygon_gj_4326: Polígono en EPSG:4326
            target_crs: CRS destino
        
        Returns:
            Geometría reproyectada (mapping dict)
        """
        geom4326 = StatsOrchestrator.extract_geometry(polygon_gj_4326)
        g_src = shape(geom4326)
        tf = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        g_dst = shp_transform(lambda x, y, z=None: tf.transform(x, y), g_src)
        return mapping(g_dst)

    @staticmethod
    def calculate_stats(
        arr: np.ma.MaskedArray,
        polygon_mask: np.ndarray
    ) -> Optional[Dict]:
        """
        Calcula estadísticas básicas dentro del polígono.
        
        Args:
            arr: Array de datos con máscara
            polygon_mask: Máscara del polígono (True = dentro)
        
        Returns:
            Dict con estadísticas o None si no hay datos válidos
        """
        base_mask = np.ma.getmaskarray(arr)
        valid = polygon_mask & (~base_mask)

        vals = np.asarray(arr)[valid].astype("float32", copy=False)
        if vals.size == 0:
            return None

        return {
            "min": round(float(np.nanmin(vals)), 2),
            "max": round(float(np.nanmax(vals)), 2),
            "mean": round(float(np.nanmean(vals)), 2),
            "median": round(float(np.nanmedian(vals)), 2),
            "std": round(float(np.nanstd(vals)), 2),
            "count": int(vals.size),
            "valid_pct": round(100.0 * vals.size / arr.size, 2)
        }

    @staticmethod
    def compute_stats_from_cache(
        cache_key: str,
        polygon_gj_4326: dict,
        filters: List = [],
        field: str = ""
    ) -> Dict:
        """
        Calcula estadísticas desde la grilla cacheada.
        
        Args:
            cache_key: Key del cache para recuperar datos
            polygon_gj_4326: Polígono GeoJSON en EPSG:4326
            filters: Filtros a aplicar dinámicamente
            field: Nombre del campo
        
        Returns:
            Dict con stats, noCoverage y reason
        """
        pkg = GRID2D_CACHE.get(cache_key)
        if pkg is None:
            return {"noCoverage": True, "reason": "No cacheado"}

        # Usar versión warped si está disponible (optimizado para stats desde WGS84)
        arr = pkg["arr_warped"] if pkg.get("arr_warped") is not None else pkg["arr"]
        crs_wkt = pkg["crs_warped"] if pkg.get("crs_warped") is not None else pkg["crs"]
        transform = pkg["transform_warped"] if pkg.get("transform_warped") is not None else pkg["transform"]
        crs = pyproj.CRS.from_user_input(crs_wkt)
        
        # Aplicar filtros dinámicamente
        arr = StatsOrchestrator.apply_filters_to_cached_data(arr, pkg, filters, field)

        # Reproyectar polígono: 4326 → crs del grid
        gj_dst = StatsOrchestrator.reproject_polygon(polygon_gj_4326, crs)
        g_dst = shape(gj_dst)

        # Límites del raster
        ny, nx = arr.shape
        xmin, ymax = transform * (0, 0)
        xmax, ymin = transform * (nx, ny)

        # Verificar intersección
        if not g_dst.intersects(box(xmin, ymin, xmax, ymax)):
            return {"noCoverage": True, "reason": "Afuera de limites"}

        # Máscara del polígono sobre la grilla
        poly_mask = geometry_mask(
            [gj_dst],
            invert=True,
            out_shape=arr.shape,
            transform=transform
        )

        # Calcular estadísticas
        stats = StatsOrchestrator.calculate_stats(arr, poly_mask)
        
        if stats is None:
            return {"noCoverage": True, "reason": "Selección vacia"}

        return {
            "stats": stats,
            "noCoverage": False,
            "reason": None,
        }

    @staticmethod
    def compute_stats_from_generated_grid(
        filepath: str,
        field_requested: str,
        product: str,
        elevation: Optional[int],
        cappi_height: Optional[int],
        radar_name: str,
        estrategia: Optional[str],
        volume: Optional[str],
        polygon_gj_4326: dict,
        filters: List,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Genera grilla 2D bajo demanda cuando no está en cache y calcula estadísticas.
        No cachea el resultado - solo para uso temporal en stats.
        
        Args:
            filepath: Path al archivo NetCDF
            field_requested: Campo solicitado
            product: Tipo de producto (PPI, CAPPI, COLMAX)
            elevation: Índice de elevación
            cappi_height: Altura CAPPI
            radar_name: Nombre del radar
            estrategia: Estrategia de procesamiento
            volume: Volumen del radar
            polygon_gj_4326: Polígono GeoJSON en EPSG:4326
            filters: Filtros a aplicar
            session_id: ID de sesión
        
        Returns:
            Dict con stats, noCoverage y reason
        """
        # Leer radar desde disco
        radar = pyart.io.read(filepath)
        
        # Generar hash del archivo
        file_hash = md5_file(filepath)[:12]
        
        # Generar grilla 2D bajo demanda (reutiliza grilla 3D cacheada con todos los campos QC)
        pkg = generate_grid2d_on_demand(
            radar=radar,
            field_requested=field_requested,
            product=product,
            file_hash=file_hash,
            radar_name=radar_name,
            estrategia=estrategia,
            volume=volume,
            elevation=elevation,
            cappi_height=cappi_height,
            filters=filters,
        )
        
        # Extraer datos de la grilla generada
        arr = pkg["arr"]
        transform = pkg["transform"]
        crs_wkt = pkg["crs"]
        
        # Aplicar solo filtros visuales (QC ya aplicados durante interpolación)
        field_upper = field_requested.upper()
        _, visual_filters = separate_filters(filters, field_upper)
        arr = apply_visual_filters(arr, visual_filters, field_upper)
        
        # Reproyectar polígono al CRS de la grilla
        crs = pyproj.CRS.from_user_input(crs_wkt)
        gj_dst = StatsOrchestrator.reproject_polygon(polygon_gj_4326, crs)
        g_dst = shape(gj_dst)
        
        # Límites del raster
        ny, nx = arr.shape
        xmin, ymax = transform * (0, 0)
        xmax, ymin = transform * (nx, ny)
        
        # Verificar intersección
        if not g_dst.intersects(box(xmin, ymin, xmax, ymax)):
            return {"noCoverage": True, "reason": "Afuera de limites"}
        
        # Máscara del polígono sobre la grilla
        poly_mask = geometry_mask(
            [gj_dst],
            invert=True,
            out_shape=arr.shape,
            transform=transform
        )
        
        # Calcular estadísticas
        stats = StatsOrchestrator.calculate_stats(arr, poly_mask)
        
        if stats is None:
            return {"noCoverage": True, "reason": "Selección vacia"}
        
        return {
            "stats": stats,
            "noCoverage": False,
            "reason": None,
        }

    @staticmethod
    def process_stats_request(payload: RadarStatsRequest) -> RadarStatsResponse:
        """
        Método principal que orquesta el cálculo de estadísticas.
        
        Args:
            payload: Request de estadísticas
        
        Returns:
            Response con estadísticas calculadas
        
        Raises:
            ValueError: Si hay errores de validación o procesamiento
        """
        # 1. Validar request
        StatsOrchestrator.validate_request(payload)

        # 2. Obtener filepath completo
        filepath = StatsOrchestrator.get_filepath(payload)

        # 3. Resolver nombre del campo
        field = StatsOrchestrator.resolve_field_name(payload.product, payload.field)

        # 4. Generar cache key (incluyendo filtros QC)
        radar_name, estrategia, volume, _ = extract_metadata_from_filename(payload.filepath)
        cache_key = StatsOrchestrator.generate_cache_key(
            filepath=filepath,
            product=payload.product,
            field=field,
            elevation=payload.elevation,
            cappi_height=payload.height,
            volume=volume,
            filters=payload.filters,
            session_id=payload.session_id,
        )

        # 5. Intentar calcular estadísticas desde cache
        stats_result = StatsOrchestrator.compute_stats_from_cache(
            cache_key,
            payload.polygon_geojson,
            payload.filters or [],
            field
        )

        # 6. Si no está cacheado, generar grilla bajo demanda
        if stats_result.get("noCoverage") and stats_result.get("reason") == "No cacheado":
            stats_result = StatsOrchestrator.compute_stats_from_generated_grid(
                filepath=filepath,
                field_requested=payload.field,
                product=payload.product,
                elevation=payload.elevation,
                cappi_height=payload.height,
                radar_name=radar_name,
                estrategia=estrategia,
                volume=volume,
                polygon_gj_4326=payload.polygon_geojson,
                filters=payload.filters or [],
                session_id=payload.session_id,
            )

        # 7. Verificar cobertura final
        if stats_result.get("noCoverage"):
            raise ValueError(stats_result.get("reason", "No hay cobertura"))

        # 8. Retornar respuesta
        return RadarStatsResponse(**stats_result)
