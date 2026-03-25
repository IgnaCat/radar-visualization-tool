"""
Orchestrator para consulta de valores en píxeles individuales de radar.
Contiene la lógica de negocio previamente en el router radar_pixel.py.
"""

import numpy as np
import pyproj
from pathlib import Path
from typing import Optional, List, Dict
from pyproj import Transformer

from ...models import RadarPixelRequest, RadarPixelResponse
from ...core.cache import GRID2D_CACHE
from ...core.config import settings
from ...core.constants import DEFAULT_WEIGHT_FUNC, DEFAULT_MAX_NEIGHBORS
from ...utils.helpers import extract_volume_from_filename
from ..radar_common import (
    grid2d_cache_key,
    md5_file,
    colormap_for,
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
    def debug_print(message: str, **kwargs) -> None:
        """
        Imprime trazas simples para debugging del flujo de consulta de pixel.

        Args:
            message: Mensaje corto del evento
            **kwargs: Pares clave/valor para inspeccionar el estado interno
        """
        if kwargs:
            details = ", ".join(f"{key}={value}" for key, value in kwargs.items())
            print(f"[pixel_debug] {message} | {details}")
        else:
            print(f"[pixel_debug] {message}")

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

        # Generar signature de qc_filters para cache keys
        qc_filters, _ = separate_filters(filters or [], field_to_use)
        qc_sig = (
            tuple(sorted([(f.field, f.min, f.max) for f in qc_filters]))
            if qc_filters
            else tuple()
        )

        cache_key = grid2d_cache_key(
            file_hash=file_hash,
            product_upper=product_upper,
            field_to_use=field_to_use,
            elevation=elevation if product_upper == "PPI" else None,
            cappi_height=cappi_height if product_upper == "CAPPI" else None,
            volume=volume,
            interp=interp,
            qc_sig=qc_sig,  # Incluir filtros QC en cache key
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
            PixelOrchestrator.debug_print(
                "snap_tile_omitido",
                motivo="sin_zoom",
                lon=round(lon, 8),
                lat=round(lat, 8),
            )
            return lon, lat

        zoom = int(render_zoom)
        native_zoom = (
            int(render_native_zoom)
            if render_native_zoom is not None
            else PixelOrchestrator.estimate_web_mercator_native_zoom(
                transform=transform,
                crs=crs,
            )
        )
        effective_zoom = min(zoom, native_zoom) if native_zoom is not None else zoom

        if effective_zoom != zoom:
            PixelOrchestrator.debug_print(
                "snap_zoom_clamp",
                zoom_solicitado=zoom,
                zoom_nativo=native_zoom,
                zoom_efectivo=effective_zoom,
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

        PixelOrchestrator.debug_print(
            "snap_tile_aplicado",
            zoom_solicitado=zoom,
            zoom_nativo=native_zoom,
            zoom_efectivo=effective_zoom,
            lon_original=round(lon, 8),
            lat_original=round(lat, 8),
            lon_snapped=round(lon_snapped, 8),
            lat_snapped=round(lat_snapped, 8),
            pixel_global_x=round(float(pixel_x), 3),
            pixel_global_y=round(float(pixel_y), 3),
            pixel_global_x_center=round(float(pixel_x_center), 3),
            pixel_global_y_center=round(float(pixel_y_center), 3),
            resolucion_mpp=round(float(resolution), 6),
        )

        return lon_snapped, lat_snapped

    @staticmethod
    def _debug_format_pixel_sample(
        arr: np.ma.MaskedArray,
        row: int,
        col: int,
        field: str,
    ) -> str:
        """
        Formatea un pixel para logs de diagnóstico incluyendo RGB esperado.

        Returns:
            String corto con formato: r{row}c{col}=valor rgb=(r,g,b)
        """
        ny, nx = arr.shape
        if row < 0 or row >= ny or col < 0 or col >= nx:
            return f"r{row}c{col}=OOB"

        mask = np.ma.getmaskarray(arr)
        if mask[row, col]:
            return f"r{row}c{col}=masked"

        value = float(arr[row, col])

        try:
            cmap, vmin, vmax, _ = colormap_for(field)
            if vmax == vmin:
                normalized = 0.0
            else:
                normalized = (value - float(vmin)) / (float(vmax) - float(vmin))
            normalized = float(np.clip(normalized, 0.0, 1.0))
            rgba = cmap(normalized)
            rgb = tuple(int(round(float(channel) * 255.0)) for channel in rgba[:3])
            return f"r{row}c{col}={value:.2f} rgb={rgb}"
        except Exception as exc:
            return f"r{row}c{col}={value:.2f} rgb=ERR({type(exc).__name__})"

    @staticmethod
    def debug_log_pixel_diagnostics(
        arr: np.ma.MaskedArray,
        row_f: float,
        col_f: float,
        transform,
        field: str,
        render_zoom: Optional[int],
        render_native_zoom: Optional[int],
    ) -> None:
        """
        Emite trazas para investigar offsets de 1 pixel entre render y consulta.

        Incluye:
        - Pixel seleccionado por convención actual (floor)
        - Pixel alternativo si se interpretara la coordenada como centro
        - Vecindad 3x3 alrededor del pixel actual
        - Relación entre resolución del tile visible y resolución del raster
        """
        row_floor = int(np.floor(row_f))
        col_floor = int(np.floor(col_f))
        frac_row = float(row_f - row_floor)
        frac_col = float(col_f - col_floor)

        # Candidato alternativo si existiera un corrimiento de +1 por convención
        # center-based vs corner-based.
        row_center_based = int(np.floor(row_f + 0.5))
        col_center_based = int(np.floor(col_f + 0.5))

        raster_res_x = abs(float(getattr(transform, "a", np.nan)))
        raster_res_y = abs(float(getattr(transform, "e", np.nan)))

        effective_zoom = None
        if render_zoom is not None:
            if render_native_zoom is not None:
                effective_zoom = min(int(render_zoom), int(render_native_zoom))
            else:
                effective_zoom = int(render_zoom)

        tile_resolution = None
        ratio_x = None
        ratio_y = None
        if effective_zoom is not None:
            world_span = 2.0 * PixelOrchestrator.WEB_MERCATOR_ORIGIN_SHIFT
            tile_resolution = world_span / (
                PixelOrchestrator.WEB_MERCATOR_TILE_SIZE * (2**effective_zoom)
            )
            if np.isfinite(raster_res_x) and raster_res_x > 0:
                ratio_x = tile_resolution / raster_res_x
            if np.isfinite(raster_res_y) and raster_res_y > 0:
                ratio_y = tile_resolution / raster_res_y

        PixelOrchestrator.debug_print(
            "pixel_diagnostico",
            frac_row=round(frac_row, 6),
            frac_col=round(frac_col, 6),
            pixel_actual=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor, col_floor, field
            ),
            pixel_alt_center_based=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_center_based, col_center_based, field
            ),
            raster_res_x_mpp=round(raster_res_x, 6) if np.isfinite(raster_res_x) else None,
            raster_res_y_mpp=round(raster_res_y, 6) if np.isfinite(raster_res_y) else None,
            tile_res_mpp=round(tile_resolution, 6) if tile_resolution is not None else None,
            tile_vs_raster_x=round(ratio_x, 6) if ratio_x is not None else None,
            tile_vs_raster_y=round(ratio_y, 6) if ratio_y is not None else None,
        )

        boundary_ranking = [
            ("down", 1.0 - frac_row),
            ("right", 1.0 - frac_col),
            ("left", frac_col),
            ("up", frac_row),
        ]
        boundary_ranking.sort(key=lambda item: item[1])
        boundary_ranking_str = " | ".join(
            f"{direction}:{distance:.6f}px" for direction, distance in boundary_ranking
        )

        PixelOrchestrator.debug_print(
            "pixel_candidatos_offset",
            actual=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor, col_floor, field
            ),
            down=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor + 1, col_floor, field
            ),
            up=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor - 1, col_floor, field
            ),
            right=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor, col_floor + 1, field
            ),
            left=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor, col_floor - 1, field
            ),
            diag_down_right=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor + 1, col_floor + 1, field
            ),
            diag_down_left=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor + 1, col_floor - 1, field
            ),
            diag_up_right=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor - 1, col_floor + 1, field
            ),
            diag_up_left=PixelOrchestrator._debug_format_pixel_sample(
                arr, row_floor - 1, col_floor - 1, field
            ),
            ranking_borde=boundary_ranking_str,
        )

        neighborhood = []
        for dr in (-1, 0, 1):
            row_samples = []
            for dc in (-1, 0, 1):
                row_samples.append(
                    f"{dr:+d},{dc:+d}:"
                    f"{PixelOrchestrator._debug_format_pixel_sample(arr, row_floor + dr, col_floor + dc, field)}"
                )
            neighborhood.append(" | ".join(row_samples))

        PixelOrchestrator.debug_print(
            "pixel_vecindad_3x3",
            fila_superior=neighborhood[0],
            fila_central=neighborhood[1],
            fila_inferior=neighborhood[2],
        )

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
            PixelOrchestrator.debug_print(
                "pixel_fuera_de_limites",
                row_f=round(float(row_f), 4),
                col_f=round(float(col_f), 4),
                row=row_int,
                col=col_int,
                ny=ny,
                nx=nx,
            )
            return RadarPixelResponse(
                value=None,
                masked=True,
                row=row_int,
                col=col_int,
                message="Fuera de limites",
            )

        m = np.ma.getmaskarray(arr)
        if m[row_int, col_int]:
            PixelOrchestrator.debug_print(
                "pixel_masked",
                row_f=round(float(row_f), 4),
                col_f=round(float(col_f), 4),
                row=row_int,
                col=col_int,
            )
            return RadarPixelResponse(
                value=None, masked=True, row=row_int, col=col_int, message="masked"
            )

        val = float(arr[row_int, col_int])
        PixelOrchestrator.debug_print(
            "pixel_valido",
            row_f=round(float(row_f), 4),
            col_f=round(float(col_f), 4),
            row=row_int,
            col=col_int,
            valor=round(val, 4),
            user_lat=round(float(user_lat), 8),
            user_lon=round(float(user_lon), 8),
        )

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
        PixelOrchestrator.debug_print(
            "request_recibido",
            filepath=payload.filepath,
            product=payload.product,
            field=payload.field,
            lat=round(float(payload.lat), 8),
            lon=round(float(payload.lon), 8),
            render_zoom=payload.render_zoom,
            render_native_zoom=payload.render_native_zoom,
            weight_func=payload.weight_func,
            max_neighbors=payload.max_neighbors,
        )

        # 2. Obtener filepath completo
        filepath = PixelOrchestrator.get_filepath(payload)

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
        pkg = GRID2D_CACHE.get(cache_key)
        if pkg is None:
            raise ValueError("No cacheado")

        # 6. Usar versión warped si está disponible (optimizado desde WGS84)
        arr = pkg["arr_warped"] if pkg.get("arr_warped") is not None else pkg["arr"]
        crs_wkt = pkg["crs_warped"] if pkg.get("crs_warped") is not None else pkg["crs"]
        transform = (
            pkg["transform_warped"]
            if pkg.get("transform_warped") is not None
            else pkg["transform"]
        )
        crs = pyproj.CRS.from_user_input(crs_wkt)
        PixelOrchestrator.debug_print(
            "cache_y_raster_resueltos",
            cache_key=cache_key,
            usa_warped=bool(pkg.get("arr_warped") is not None),
            shape=getattr(arr, "shape", None),
            crs=str(crs),
        )

        # 7. Aplicar filtros dinámicamente
        arr = PixelOrchestrator.apply_filters_to_cached_data(
            arr, pkg, payload.filters or [], field
        )

        # 8. Alinear el click a la grilla visible de tiles si llega el zoom.
        lon_query, lat_query = PixelOrchestrator.snap_coordinates_to_tile_pixel(
            payload.lon,
            payload.lat,
            payload.render_zoom,
            transform=transform,
            crs=crs,
            render_native_zoom=payload.render_native_zoom,
        )
        PixelOrchestrator.debug_print(
            "coordenadas_consulta",
            lon_original=round(float(payload.lon), 8),
            lat_original=round(float(payload.lat), 8),
            lon_query=round(float(lon_query), 8),
            lat_query=round(float(lat_query), 8),
        )

        # 9. Transformar coordenadas a grid
        col_f, row_f = PixelOrchestrator.transform_coordinates_to_grid(
            lon_query, lat_query, crs, transform
        )
        PixelOrchestrator.debug_print(
            "grid_coords_calculadas",
            row_f=round(float(row_f), 6),
            col_f=round(float(col_f), 6),
        )
        PixelOrchestrator.debug_log_pixel_diagnostics(
            arr=arr,
            row_f=row_f,
            col_f=col_f,
            transform=transform,
            field=field,
            render_zoom=payload.render_zoom,
            render_native_zoom=payload.render_native_zoom,
        )

        ny, nx = arr.shape

        # Usar valor directo del pixel (sin interpolación bilinear)
        # Esto asegura que cualquier click dentro del mismo pixel devuelva el mismo valor
        return PixelOrchestrator.get_pixel_value_nearest(
            arr, row_f, col_f, transform, crs, payload.lat, payload.lon
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
