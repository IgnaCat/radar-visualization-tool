"""
Orchestrator para consulta de valores en píxeles individuales de radar.
Contiene la lógica de negocio previamente en el router radar_pixel.py.
"""
import numpy as np
import pyproj
from pathlib import Path
from typing import Optional, List, Dict
from pyproj import Transformer
from rasterio.transform import xy

from ...models import RadarPixelRequest, RadarPixelResponse
from ...core.cache import GRID2D_CACHE
from ...core.config import settings
from ...utils.helpers import extract_volume_from_filename
from ..radar_common import (
    grid2d_cache_key,
    md5_file,
)
from ..filter_application import (
    separate_filters,
    apply_visual_filters,
    apply_qc_filters,
)


class PixelOrchestrator:
    """
    Coordina la consulta de valores en píxeles individuales de radar.
    Usa interpolación bilinear sobre la grilla 2D cacheada.
    """

    @staticmethod
    def validate_request(payload: RadarPixelRequest) -> None:
        """
        Valida los parámetros de la solicitud.
        Raises: ValueError si hay problemas críticos
        """
        if getattr(payload, "filepath", None) in (None, "", "undefined"):
            raise ValueError("El campo 'filepath' es obligatorio.")
        
        if not (-90 <= float(payload.lat) <= 90 and -180 <= float(payload.lon) <= 180):
            raise ValueError("Coordenadas no WGS84 (use lat∈[-90,90], lon∈[-180,180])")

    @staticmethod
    def get_filepath(payload: RadarPixelRequest) -> str:
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
        session_id: Optional[str] = None,
    ) -> str:
        """
        Genera cache key sin incluir filtros (se aplican dinámicamente).
        
        Returns:
            Cache key para GRID2D_CACHE
        """
        product_upper = product.upper()
        field_to_use = field.upper()
        interp = "Barnes2"

        # Hash del archivo
        file_hash = md5_file(filepath)[:12]

        cache_key = grid2d_cache_key(
            file_hash=file_hash,
            product_upper=product_upper,
            field_to_use=field_to_use,
            elevation=elevation if product_upper == "PPI" else None,
            cappi_height=cappi_height if product_upper == "CAPPI" else None,
            volume=volume,
            interp=interp,
            qc_sig=tuple(),  # Filtros se aplican dinámicamente
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
        qc_filters, visual_filters = separate_filters(filters, field_to_use)
        
        # Aplicar filtros visuales
        arr = apply_visual_filters(arr, visual_filters, field_to_use)
        
        # Aplicar filtros QC
        if qc_filters:
            # Usar qc_warped si arr_warped está disponible para coincidir dimensiones
            if pkg.get("arr_warped") is not None:
                qc_dict = pkg.get("qc_warped", {}) or {}
            else:
                qc_dict = pkg.get("qc", {}) or {}
            arr = apply_qc_filters(arr, qc_filters, qc_dict)
        
        return arr

    @staticmethod
    def transform_coordinates_to_grid(
        lon: float,
        lat: float,
        crs: pyproj.CRS,
        transform
    ) -> tuple:
        """
        Transforma coordenadas WGS84 a coordenadas del grid.
        
        Args:
            lon: Longitud en WGS84
            lat: Latitud en WGS84
            crs: CRS del grid
            transform: Affine transform del grid
        
        Returns:
            Tupla (col_f, row_f) coordenadas fraccionarias en pixel space
        """
        # 4326 -> CRS del grid (always_xy=True porque pasamos (lon,lat))
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xg, yg = tf.transform(lon, lat)

        # Coordenadas continuas (fraccionarias) en pixel space
        col_f, row_f = ~transform * (xg, yg)  # inverso de transform
        
        return col_f, row_f

    @staticmethod
    def get_pixel_value_nearest(
        arr: np.ma.MaskedArray,
        row_f: float,
        col_f: float,
        transform,
        crs: pyproj.CRS
    ) -> RadarPixelResponse:
        """
        Obtiene valor del píxel más cercano (sin interpolación).
        
        Args:
            arr: Array de datos
            row_f: Fila fraccionaria
            col_f: Columna fraccionaria
            transform: Affine transform
            crs: CRS del grid
        
        Returns:
            RadarPixelResponse con el valor o masked
        """
        row_int = int(round(row_f))
        col_int = int(round(col_f))
        ny, nx = arr.shape
        
        if row_int < 0 or row_int >= ny or col_int < 0 or col_int >= nx:
            return RadarPixelResponse(
                value=None, 
                masked=True, 
                row=row_int, 
                col=col_int, 
                message="Fuera de limites"
            )
        
        m = np.ma.getmaskarray(arr)
        if m[row_int, col_int]:
            return RadarPixelResponse(
                value=None, 
                masked=True, 
                row=row_int, 
                col=col_int, 
                message="masked"
            )
        
        val = float(arr[row_int, col_int])
        
        # Coordenada del centro del pixel para respuesta
        xc, yc = xy(transform, row_int, col_int, offset="center")
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lonc, latc = to_wgs84.transform(xc, yc)
        
        return RadarPixelResponse(
            value=round(val, 2), 
            masked=False, 
            row=row_int, 
            col=col_int, 
            lat=latc, 
            lon=lonc
        )

    @staticmethod
    def get_pixel_value_bilinear(
        arr: np.ma.MaskedArray,
        row_f: float,
        col_f: float,
        user_lat: float,
        user_lon: float
    ) -> RadarPixelResponse:
        """
        Obtiene valor interpolado bilinearmente entre 4 píxeles vecinos.
        
        Args:
            arr: Array de datos
            row_f: Fila fraccionaria
            col_f: Columna fraccionaria
            user_lat: Latitud original del usuario
            user_lon: Longitud original del usuario
        
        Returns:
            RadarPixelResponse con el valor interpolado
        """
        # Encontrar los 4 píxeles vecinos
        r0 = int(np.floor(row_f))
        c0 = int(np.floor(col_f))
        r1 = r0 + 1
        c1 = c0 + 1
        
        # Pesos
        dr = row_f - r0
        dc = col_f - c0
        
        m = np.ma.getmaskarray(arr)
        
        # Extraer valores y máscaras de los 4 vecinos
        v00 = arr[r0, c0] if not m[r0, c0] else np.nan
        v01 = arr[r0, c1] if not m[r0, c1] else np.nan
        v10 = arr[r1, c0] if not m[r1, c0] else np.nan
        v11 = arr[r1, c1] if not m[r1, c1] else np.nan
        
        # Si todos masked -> retornar masked
        if np.isnan([v00, v01, v10, v11]).all():
            row_int = int(round(row_f))
            col_int = int(round(col_f))
            return RadarPixelResponse(
                value=None, 
                masked=True, 
                row=row_int, 
                col=col_int, 
                message="masked (todos vecinos)"
            )
        
        # Interpolación bilinear (ignora NaN promediando los válidos con sus pesos)
        w00 = (1 - dr) * (1 - dc)
        w01 = (1 - dr) * dc
        w10 = dr * (1 - dc)
        w11 = dr * dc
        
        total_weight = 0.0
        val_interp = 0.0
        
        if not np.isnan(v00):
            val_interp += w00 * v00
            total_weight += w00
        if not np.isnan(v01):
            val_interp += w01 * v01
            total_weight += w01
        if not np.isnan(v10):
            val_interp += w10 * v10
            total_weight += w10
        if not np.isnan(v11):
            val_interp += w11 * v11
            total_weight += w11
        
        if total_weight > 0:
            val_interp /= total_weight
        else:
            row_int = int(round(row_f))
            col_int = int(round(col_f))
            return RadarPixelResponse(
                value=None, 
                masked=True, 
                row=row_int, 
                col=col_int, 
                message="masked"
            )
        
        row_int = int(round(row_f))
        col_int = int(round(col_f))
        
        return RadarPixelResponse(
            value=round(val_interp, 2), 
            masked=False, 
            row=row_int, 
            col=col_int, 
            lat=user_lat, 
            lon=user_lon
        )

    @staticmethod
    def process_pixel_request(payload: RadarPixelRequest) -> RadarPixelResponse:
        """
        Método principal que orquesta la consulta de píxel.
        
        Args:
            payload: Request de consulta de píxel
        
        Returns:
            Response con valor del píxel (interpolado o nearest)
        
        Raises:
            ValueError: Si hay errores de validación o datos no disponibles
        """
        # 1. Validar request
        PixelOrchestrator.validate_request(payload)

        # 2. Obtener filepath completo
        filepath = PixelOrchestrator.get_filepath(payload)

        # 3. Resolver nombre del campo
        field = PixelOrchestrator.resolve_field_name(payload.product, payload.field)

        # 4. Generar cache key
        volume = extract_volume_from_filename(payload.filepath)
        cache_key = PixelOrchestrator.generate_cache_key(
            filepath=filepath,
            product=payload.product,
            field=field,
            elevation=payload.elevation,
            cappi_height=payload.height,
            volume=volume,
            session_id=payload.session_id,
        )

        # 5. Obtener datos del cache
        pkg = GRID2D_CACHE.get(cache_key)
        if pkg is None:
            raise ValueError("No cacheado")

        # 6. Usar versión warped si está disponible (optimizado desde WGS84)
        arr = pkg["arr_warped"] if pkg.get("arr_warped") is not None else pkg["arr"]
        crs_wkt = pkg["crs_warped"] if pkg.get("crs_warped") is not None else pkg["crs"]
        transform = pkg["transform_warped"] if pkg.get("transform_warped") is not None else pkg["transform"]
        crs = pyproj.CRS.from_wkt(crs_wkt)

        # 7. Aplicar filtros dinámicamente
        arr = PixelOrchestrator.apply_filters_to_cached_data(
            arr, 
            pkg, 
            payload.filters or [], 
            field
        )

        # 8. Transformar coordenadas a grid
        col_f, row_f = PixelOrchestrator.transform_coordinates_to_grid(
            payload.lon, 
            payload.lat, 
            crs, 
            transform
        )

        ny, nx = arr.shape

        # 9. Verificar si está en bordes (no se puede interpolar)
        if row_f < 0 or row_f >= ny - 1 or col_f < 0 or col_f >= nx - 1:
            # Fuera de límites o en borde -> usar nearest
            return PixelOrchestrator.get_pixel_value_nearest(
                arr, row_f, col_f, transform, crs
            )

        # 10. Interpolación bilinear (dentro de la grilla)
        return PixelOrchestrator.get_pixel_value_bilinear(
            arr, 
            row_f, 
            col_f, 
            payload.lat, 
            payload.lon
        )
