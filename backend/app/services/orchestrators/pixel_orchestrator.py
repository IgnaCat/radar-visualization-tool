"""
Orchestrator para consulta de valores en píxeles individuales de radar.
Contiene la lógica de negocio previamente en el router radar_pixel.py.
"""

import numpy as np
import pyproj
from pathlib import Path
from typing import Optional, List, Dict
from affine import Affine
from pyproj import Transformer
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling

from ...models import RadarPixelRequest, RadarPixelResponse
from ...core.cache import GRID2D_CACHE, GRID2D_LOCK
from ...core.config import settings
from ...core.constants import DEFAULT_WEIGHT_FUNC, DEFAULT_MAX_NEIGHBORS
from ...utils.helpers import extract_volume_from_filename
from ..radar_common import (
    grid2d_cache_key,
    md5_file,
)
from ..radar_processing import (
    separate_filters,
    apply_visual_filters,
    apply_qc_filters,
)


class PixelOrchestrator:
    """
    Coordina la consulta de valores en píxeles individuales de radar.
    Usa interpolación bilinear sobre la grilla 2D cacheada.
    """

    WEB_MERCATOR_ORIGIN_SHIFT = 20037508.342789244
    WEB_MERCATOR_TILE_SIZE = 256

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
    def get_filepath(payload: RadarPixelRequest, user_id: str) -> str:
        """
        Construye el path completo del archivo desde el request.
        Estructura: UPLOAD_DIR/{user_id}/{session_id}/{filepath}
        Si no hay session_id cae en UPLOAD_DIR/{user_id}/{filepath}.

        Returns:
            Path absoluto al archivo de radar
        """
        base = Path(settings.UPLOAD_DIR) / str(user_id)
        if payload.session_id:
            base = base / payload.session_id
        return str(base / payload.filepath)

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
        weight_func: str = DEFAULT_WEIGHT_FUNC,
        max_neighbors: Optional[int] = DEFAULT_MAX_NEIGHBORS,
    ) -> str:
        """
        Genera cache key incluyendo filtros QC (afectan interpolación).

        Returns:
            Cache key para GRID2D_CACHE
        """
        product_upper = product.upper()
        field_to_use = field.upper()
        interp = weight_func

        # Hash del archivo
        file_hash = md5_file(filepath)[:12]

        # Generar signature de filtros para cache keys
        # Incluir QC + visuales, igual que radar_processor, porque ambos
        # se aplican durante la interpolación y afectan la grilla cacheada.
        qc_filters, visual_filters = separate_filters(filters or [], field_to_use)
        qc_sig = (
            tuple(sorted([(f.field, f.min, f.max) for f in qc_filters]))
            if qc_filters
            else tuple()
        )
        visual_sig = (
            tuple(sorted([(f.field, f.min, f.max) for f in visual_filters]))
            if visual_filters
            else tuple()
        )
        filter_sig = qc_sig + visual_sig

        cache_key = grid2d_cache_key(
            file_hash=file_hash,
            product_upper=product_upper,
            field_to_use=field_to_use,
            elevation=elevation if product_upper == "PPI" else None,
            cappi_height=cappi_height if product_upper == "CAPPI" else None,
            volume=volume,
            interp=interp,
            qc_sig=filter_sig,  # QC + visuales, misma firma que radar_processor
            max_neighbors=max_neighbors,
            session_id=session_id,
        )

        return cache_key

    @staticmethod
    def apply_filters_to_cached_data(
        arr: np.ma.MaskedArray, pkg: Dict, filters: List, field: str
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
    def transform_coordinates_to_grid(
        lon: float, lat: float, crs: pyproj.CRS, transform
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
    def estimate_web_mercator_native_zoom(
        transform,
        crs: Optional[pyproj.CRS],
    ) -> Optional[int]:
        """
        Estima el zoom nativo entero de un raster warped en EPSG:3857.

        El objetivo es no hacer snap a una grilla WebMercatorQuad más fina
        que la resolución real del raster servido en tiles. Si el usuario hace
        overzoom en Leaflet, el mismo tile visible se escala en pantalla y
        el click debe alinearse al zoom nativo del raster, no al zoom visual.

        Args:
            transform: Affine transform del raster consultado
            crs: CRS del raster consultado

        Returns:
            Zoom nativo entero estimado. Si no puede estimarse, devuelve None.
        """
        if transform is None or crs is None:
            return None

        try:
            crs_obj = pyproj.CRS.from_user_input(crs)
        except Exception:
            return None

        if crs_obj.to_epsg() != 3857:
            return None

        pixel_size_x = abs(float(getattr(transform, "a", 0.0)))
        pixel_size_y = abs(float(getattr(transform, "e", 0.0)))
        raster_resolution = max(pixel_size_x, pixel_size_y)

        if not np.isfinite(raster_resolution) or raster_resolution <= 0.0:
            return None

        world_span = 2.0 * PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT
        native_zoom_float = np.log2(
            world_span
            / (PixelOrchestrator.WEB_MERCATOR_TILE_SIZE * raster_resolution)
        )

        if not np.isfinite(native_zoom_float):
            return None

        # floor evita snapear a una resolución más fina que la del raster.
        return max(0, int(np.floor(native_zoom_float)))

    @staticmethod
    def resolve_effective_tile_zoom(
        render_zoom: Optional[int],
        transform=None,
        crs: Optional[pyproj.CRS] = None,
        render_native_zoom: Optional[int] = None,
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Resuelve el zoom efectivo de la malla visible WebMercatorQuad.

        Returns:
            Tupla (zoom_efectivo, zoom_nativo). Si no hay render_zoom, el
            zoom efectivo queda en None.
        """
        native_zoom = (
            int(render_native_zoom)
            if render_native_zoom is not None
            else PixelOrchestrator.estimate_web_mercator_native_zoom(
                transform=transform,
                crs=crs,
            )
        )

        if render_zoom is None:
            return None, native_zoom

        zoom = int(render_zoom)
        effective_zoom = min(zoom, native_zoom) if native_zoom is not None else zoom
        return effective_zoom, native_zoom

    @staticmethod
    def build_display_aligned_grid(
        arr: np.ma.MaskedArray,
        transform,
        effective_zoom: int,
    ) -> Dict:
        """
        Remuestrea un raster ya warped en EPSG:3857 a la malla exacta de
        pixeles WebMercatorQuad del zoom solicitado.
        """
        ny, nx = arr.shape
        left, bottom, right, top = array_bounds(ny, nx, transform)

        world_span = 2.0 * PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT
        resolution = world_span / (
            PixelOrchestrator.WEB_MERCATOR_TILE_SIZE * (2**int(effective_zoom))
        )

        left_px = int(
            np.floor(
                (left + PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT) / resolution
            )
        )
        right_px = int(
            np.ceil(
                (right + PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT) / resolution
            )
        )
        top_px = int(
            np.floor(
                (PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT - top) / resolution
            )
        )
        bottom_px = int(
            np.ceil(
                (PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT - bottom) / resolution
            )
        )

        dst_width = max(1, right_px - left_px)
        dst_height = max(1, bottom_px - top_px)
        dst_left = (left_px * resolution) - PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT
        dst_top = PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT - (
            top_px * resolution
        )
        dst_transform = Affine.translation(dst_left, dst_top) * Affine.scale(
            resolution,
            -resolution,
        )

        src_data = np.ma.filled(arr, fill_value=np.nan).astype(np.float32)
        dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=transform,
            src_crs="EPSG:3857",
            dst_transform=dst_transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        dst_arr = np.ma.masked_invalid(dst_data)
        return {
            "arr": dst_arr,
            "transform": dst_transform,
            "crs": "EPSG:3857",
            "zoom": int(effective_zoom),
            "resolution": float(resolution),
        }

    @staticmethod
    def get_or_build_display_aligned_grid(
        pkg: Dict,
        cache_key: str,
        arr: np.ma.MaskedArray,
        transform,
        crs: pyproj.CRS,
        effective_zoom: Optional[int],
    ) -> Optional[Dict]:
        """
        Obtiene o construye una grilla numérica alineada a la malla visible
        WebMercatorQuad del zoom efectivo.
        """
        if effective_zoom is None or transform is None or crs is None:
            return None

        try:
            crs_obj = pyproj.CRS.from_user_input(crs)
        except Exception:
            return None

        if crs_obj.to_epsg() != 3857:
            return None

        display_grids = pkg.setdefault("display_grids", {})
        zoom_key = int(effective_zoom)
        cached_display_grid = display_grids.get(zoom_key)
        if cached_display_grid is not None:
            return cached_display_grid

        display_grid = PixelOrchestrator.build_display_aligned_grid(
            arr=arr,
            transform=transform,
            effective_zoom=zoom_key,
        )
        display_grids[zoom_key] = display_grid
        with GRID2D_LOCK:
            GRID2D_CACHE[cache_key] = pkg
        return display_grid

    @staticmethod
    def snap_coordinates_to_tile_pixel(
        lon: float,
        lat: float,
        render_zoom: Optional[int],
        transform=None,
        crs: Optional[pyproj.CRS] = None,
        render_native_zoom: Optional[int] = None,
    ) -> tuple[float, float]:
        """
        Ajusta un click WGS84 al centro del pixel visible de la grilla
        WebMercatorQuad para el zoom actual del mapa.

        Esto ayuda a que múltiples clicks dentro del mismo bloque visible
        del tile consulten el mismo pixel científico subyacente.

        Args:
            lon: Longitud original del click
            lat: Latitud original del click
            render_zoom: Zoom entero del mapa al momento del click
            transform: Affine transform del raster consultado
            crs: CRS del raster consultado
            render_native_zoom: Max native zoom real de la capa visible

        Returns:
            Tupla (lon_snapped, lat_snapped) ajustada al centro del pixel visible.
            Si no hay zoom válido, devuelve las coordenadas originales.
        """
        if render_zoom is None:
            return lon, lat

        zoom = int(render_zoom)
        effective_zoom, _native_zoom = PixelOrchestrator.resolve_effective_tile_zoom(
            render_zoom=render_zoom,
            transform=transform,
            crs=crs,
            render_native_zoom=render_native_zoom,
        )

        # 4326 -> 3857 para trabajar en la misma grilla global de tiles.
        tf_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_merc, y_merc = tf_to_merc.transform(lon, lat)

        world_span = 2.0 * PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT
        resolution = world_span / (
            PixelOrchestrator.WEB_MERCATOR_TILE_SIZE * (2**effective_zoom)
        )

        # Convertir a pixel global de la pirámide WebMercatorQuad.
        pixel_x = (
            x_merc + PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT
        ) / resolution
        pixel_y = (
            PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT - y_merc
        ) / resolution

        # Snap al centro del pixel visible del tile.
        pixel_x_center = np.floor(pixel_x) + 0.5
        pixel_y_center = np.floor(pixel_y) + 0.5

        x_snapped = (
            pixel_x_center * resolution
        ) - PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT
        y_snapped = PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT - (
            pixel_y_center * resolution
        )

        # 3857 -> 4326 para reutilizar luego el flujo normal de consulta.
        tf_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon_snapped, lat_snapped = tf_to_wgs84.transform(x_snapped, y_snapped)

        return lon_snapped, lat_snapped

    @staticmethod
    def get_pixel_value_nearest(
        arr: np.ma.MaskedArray,
        row_f: float,
        col_f: float,
        transform,
        crs: pyproj.CRS,
        user_lat: float,
        user_lon: float,
    ) -> RadarPixelResponse:
        """
        Obtiene valor del píxel más cercano (sin interpolación).

        Args:
            arr: Array de datos
            row_f: Fila fraccionaria
            col_f: Columna fraccionaria
            transform: Affine transform
            crs: CRS del grid
            user_lat: Latitud original del usuario (para respuesta)
            user_lon: Longitud original del usuario (para respuesta)

        Returns:
            RadarPixelResponse con el valor y coordenadas originales del usuario
        """
        # Usar floor en lugar de round porque el transform mapea índices a esquinas de píxeles:
        # col_f ∈ [N, N+1) corresponde al pixel N
        row_int = int(np.floor(row_f))
        col_int = int(np.floor(col_f))
        ny, nx = arr.shape

        if row_int < 0 or row_int >= ny or col_int < 0 or col_int >= nx:
            return RadarPixelResponse(
                value=None,
                masked=True,
                row=row_int,
                col=col_int,
                message="Fuera de limites",
            )

        m = np.ma.getmaskarray(arr)
        if m[row_int, col_int]:
            return RadarPixelResponse(
                value=None, masked=True, row=row_int, col=col_int, message="masked"
            )

        val = float(arr[row_int, col_int])

        # Devolver coordenadas originales del usuario (no del centro del pixel)
        return RadarPixelResponse(
            value=round(val, 2),
            masked=False,
            row=row_int,
            col=col_int,
            lat=user_lat,
            lon=user_lon,
        )

    @staticmethod
    def get_pixel_value_bilinear(
        arr: np.ma.MaskedArray,
        row_f: float,
        col_f: float,
        transform,
        crs: pyproj.CRS,
        user_lat: float,
        user_lon: float,
    ) -> RadarPixelResponse:
        """
        Obtiene valor interpolado bilinearmente entre 4 píxeles vecinos.

        Args:
            arr: Array de datos
            row_f: Fila fraccionaria
            col_f: Columna fraccionaria
            transform: Affine transform del grid
            crs: CRS del grid
            user_lat: Latitud original del usuario (para respuesta)
            user_lon: Longitud original del usuario (para respuesta)

        Returns:
            RadarPixelResponse con el valor interpolado y coordenadas originales del usuario
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
            row_int = int(np.floor(row_f))
            col_int = int(np.floor(col_f))
            return RadarPixelResponse(
                value=None,
                masked=True,
                row=row_int,
                col=col_int,
                message="masked (todos vecinos)",
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
            row_int = int(np.floor(row_f))
            col_int = int(np.floor(col_f))
            return RadarPixelResponse(
                value=None, masked=True, row=row_int, col=col_int, message="masked"
            )

        row_int = int(np.floor(row_f))
        col_int = int(np.floor(col_f))

        # Devolver coordenadas originales del usuario (no del centro del pixel)
        return RadarPixelResponse(
            value=round(val_interp, 2),
            masked=False,
            row=row_int,
            col=col_int,
            lat=user_lat,
            lon=user_lon,
        )

    @staticmethod
    def process_pixel_request(payload: RadarPixelRequest, user_id: str) -> RadarPixelResponse:
        """
        Método principal que orquesta la consulta de píxel.

        Args:
            payload: Request de consulta de píxel
            user_id: ID del usuario autenticado (para localizar sus uploads)

        Returns:
            Response con valor del píxel (interpolado o nearest)

        Raises:
            ValueError: Si hay errores de validación o datos no disponibles
        """
        # 1. Validar request
        PixelOrchestrator.validate_request(payload)
        # 2. Obtener filepath completo
        filepath = PixelOrchestrator.get_filepath(payload, user_id)

        # 3. Resolver nombre del campo
        field = PixelOrchestrator.resolve_field_name(payload.product, payload.field)

        # 4. Generar cache key
        volume = extract_volume_from_filename(payload.filepath)
        weight_func = payload.weight_func or DEFAULT_WEIGHT_FUNC
        max_neighbors = payload.max_neighbors or DEFAULT_MAX_NEIGHBORS
        cache_key = PixelOrchestrator.generate_cache_key(
            filepath=filepath,
            product=payload.product,
            field=field,
            elevation=payload.elevation,
            cappi_height=payload.height,
            volume=volume,
            filters=payload.filters,
            session_id=payload.session_id,
            weight_func=weight_func,
            max_neighbors=max_neighbors,
        )

        # 5. Obtener datos del cache
        with GRID2D_LOCK:
            pkg = GRID2D_CACHE.get(cache_key)
        if pkg is None:
            raise ValueError("No cacheado")

        # 6. Usar versión warped si está disponible (optimizado desde WGS84)
        arr_base = (
            pkg["arr_warped"] if pkg.get("arr_warped") is not None else pkg["arr"]
        )
        crs_wkt_base = (
            pkg["crs_warped"] if pkg.get("crs_warped") is not None else pkg["crs"]
        )
        transform_base = (
            pkg["transform_warped"]
            if pkg.get("transform_warped") is not None
            else pkg["transform"]
        )
        crs_base = pyproj.CRS.from_user_input(crs_wkt_base)
        # 7. Resolver la grilla numérica sobre la que se va a consultar.
        effective_zoom, _native_zoom = PixelOrchestrator.resolve_effective_tile_zoom(
            render_zoom=payload.render_zoom,
            transform=transform_base,
            crs=crs_base,
            render_native_zoom=payload.render_native_zoom,
        )
        display_grid = PixelOrchestrator.get_or_build_display_aligned_grid(
            pkg=pkg,
            cache_key=cache_key,
            arr=arr_base,
            transform=transform_base,
            crs=crs_base,
            effective_zoom=effective_zoom,
        )
        if display_grid is not None:
            arr_query_base = display_grid["arr"]
            transform_query = display_grid["transform"]
            crs_query = pyproj.CRS.from_user_input(display_grid["crs"])
        else:
            arr_query_base = arr_base
            transform_query = transform_base
            crs_query = crs_base

        # 8. Aplicar filtros dinámicamente
        arr = PixelOrchestrator.apply_filters_to_cached_data(
            arr_query_base, pkg, payload.filters or [], field
        )

        # 9. Alinear el click a la grilla visible de tiles si llega el zoom.
        lon_query, lat_query = PixelOrchestrator.snap_coordinates_to_tile_pixel(
            payload.lon,
            payload.lat,
            payload.render_zoom,
            transform=transform_query,
            crs=crs_query,
            render_native_zoom=payload.render_native_zoom,
        )
        # 10. Transformar coordenadas a grid
        col_f, row_f = PixelOrchestrator.transform_coordinates_to_grid(
            lon_query, lat_query, crs_query, transform_query
        )

        # Usar valor directo del pixel (sin interpolación bilinear)
        # Esto asegura que cualquier click dentro del mismo pixel devuelva el mismo valor
        return PixelOrchestrator.get_pixel_value_nearest(
            arr,
            row_f,
            col_f,
            transform_query,
            crs_query,
            payload.lat,
            payload.lon,
        )

        # NOTA: Interpolación bilinear comentada - producía valores diferentes
        # para clicks dentro del mismo pixel debido a los pesos fraccionarios.
        # Si se desea reactivar, descomentar el bloque siguiente:

        # # 9. Verificar si está en bordes (no se puede interpolar)
        # if row_f < 0 or row_f >= ny - 1 or col_f < 0 or col_f >= nx - 1:
        #     # Fuera de límites o en borde -> usar nearest
        #     return PixelOrchestrator.get_pixel_value_nearest(
        #         arr, row_f, col_f, transform, crs, payload.lat, payload.lon
        #     )
        #
        # # 10. Interpolación bilinear (dentro de la grilla)
        # return PixelOrchestrator.get_pixel_value_bilinear(
        #     arr,
        #     row_f,
        #     col_f,
        #     transform,
        #     crs,
        #     payload.lat,
        #     payload.lon
        # )
