"""
Registro y renderizado de mapas base permitidos para exportaciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import ceil, floor, isfinite, log2

import requests
from PIL import Image


WEB_MERCATOR_HALF_WORLD = 20037508.342789244
WEB_MERCATOR_WORLD_WIDTH = WEB_MERCATOR_HALF_WORLD * 2.0
TILE_SIZE = 256
INITIAL_RESOLUTION = WEB_MERCATOR_WORLD_WIDTH / TILE_SIZE


@dataclass(frozen=True)
class BaseMapSpec:
    id: str
    name: str
    url_template: str
    min_zoom: int = 0
    max_zoom: int = 18


BASEMAP_SPECS: dict[str, BaseMapSpec] = {
    "osm": BaseMapSpec(
        id="osm",
        name="Mapa Base",
        url_template="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        max_zoom=19,
    ),
    "argenmap": BaseMapSpec(
        id="argenmap",
        name="Argenmap",
        url_template="https://wms.ign.gob.ar/geoserver/gwc/service/tms/1.0.0/capabaseargenmap@EPSG%3A3857@png/{z}/{x}/{-y}.png",
        max_zoom=19,
    ),
    "argenmap-gris": BaseMapSpec(
        id="argenmap-gris",
        name="Argenmap gris",
        url_template="https://wms.ign.gob.ar/geoserver/gwc/service/tms/1.0.0/mapabase_gris@EPSG%3A3857@png/{z}/{x}/{-y}.png",
        max_zoom=19,
    ),
    "argenmap-oscuro": BaseMapSpec(
        id="argenmap-oscuro",
        name="Argenmap oscuro",
        url_template="https://wms.ign.gob.ar/geoserver/gwc/service/tms/1.0.0/argenmap_oscuro@EPSG%3A3857@png/{z}/{x}/{-y}.png",
        max_zoom=19,
    ),
    "topografico": BaseMapSpec(
        id="topografico",
        name="Argenmap topográfico",
        url_template="https://wms.ign.gob.ar/geoserver/gwc/service/tms/1.0.0/mapabase_topo@EPSG%3A3857@png/{z}/{x}/{-y}.png",
        max_zoom=19,
    ),
    "satellite": BaseMapSpec(
        id="satellite",
        name="Imágenes satelitales Esri",
        url_template="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        max_zoom=18,
    ),
    "topo-esri": BaseMapSpec(
        id="topo-esri",
        name="Mapa topográfico Esri",
        url_template="https://server.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}",
        max_zoom=18,
    ),
    "ocean": BaseMapSpec(
        id="ocean",
        name="Mapa Esri Fondo Oceánico",
        url_template="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        max_zoom=16,
    ),
}


def get_basemap_spec(basemap_id: str | None) -> BaseMapSpec | None:
    if not basemap_id:
        return None
    return BASEMAP_SPECS.get(str(basemap_id).strip())


def choose_basemap_zoom(target_resolution: float, spec: BaseMapSpec) -> int:
    """Elige un zoom cercano a la resolución objetivo del GIF."""
    if not isfinite(target_resolution) or target_resolution <= 0:
        return spec.min_zoom

    # En Web Mercator cada zoom divide la resolución por 2.
    raw_zoom = log2(INITIAL_RESOLUTION / target_resolution)
    # Redondeamos hacia arriba para no pedir un fondo más borroso que el GIF.
    zoom = int(ceil(raw_zoom))
    return max(spec.min_zoom, min(spec.max_zoom, zoom))


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _mercator_to_world_pixel(x: float, y: float, zoom: int) -> tuple[float, float]:
    # Convierte coordenadas EPSG:3857 a píxeles globales del mundo tileado.
    map_size = TILE_SIZE * (2**zoom)
    pixel_x = ((x + WEB_MERCATOR_HALF_WORLD) / WEB_MERCATOR_WORLD_WIDTH) * map_size
    pixel_y = ((WEB_MERCATOR_HALF_WORLD - y) / WEB_MERCATOR_WORLD_WIDTH) * map_size
    return pixel_x, pixel_y


def _format_tile_url(template: str, zoom: int, tile_x: int, tile_y: int) -> str:
    max_tile = (1 << zoom) - 1
    return (
        template.replace("{z}", str(zoom))
        .replace("{x}", str(tile_x))
        # Algunos proveedores usan TMS y necesitan Y invertida.
        .replace("{-y}", str(max_tile - tile_y))
        .replace("{y}", str(tile_y))
        .replace("{s}", "a")
    )


def render_basemap_for_canvas(
    spec: BaseMapSpec,
    left: float,
    bottom: float,
    right: float,
    top: float,
    width: int,
    height: int,
    target_resolution: float,
    timeout: float = 10.0,
) -> Image.Image | None:
    """
    Renderiza el mapa base exactamente sobre el canvas del GIF.
    """
    zoom = choose_basemap_zoom(target_resolution, spec)
    map_size = TILE_SIZE * (2**zoom)

    # Pasamos de bbox en metros a bbox en píxeles del mundo para este zoom.
    pixel_left, pixel_top = _mercator_to_world_pixel(left, top, zoom)
    pixel_right, pixel_bottom = _mercator_to_world_pixel(right, bottom, zoom)

    pixel_left = _clamp(pixel_left, 0.0, float(map_size))
    pixel_right = _clamp(pixel_right, 0.0, float(map_size))
    pixel_top = _clamp(pixel_top, 0.0, float(map_size))
    pixel_bottom = _clamp(pixel_bottom, 0.0, float(map_size))

    if pixel_right <= pixel_left or pixel_bottom <= pixel_top:
        return None

    # Calculamos qué tiles cubren por completo la bbox pedida.
    tile_x_min = max(0, int(floor(pixel_left / TILE_SIZE)))
    tile_x_max = min((1 << zoom) - 1, int(ceil(pixel_right / TILE_SIZE) - 1))
    tile_y_min = max(0, int(floor(pixel_top / TILE_SIZE)))
    tile_y_max = min((1 << zoom) - 1, int(ceil(pixel_bottom / TILE_SIZE) - 1))

    mosaic_width = (tile_x_max - tile_x_min + 1) * TILE_SIZE
    mosaic_height = (tile_y_max - tile_y_min + 1) * TILE_SIZE
    mosaic = Image.new("RGBA", (mosaic_width, mosaic_height), (255, 255, 255, 0))

    fetched_tiles = 0
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "tesis-radar-gif-export/1.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
    )

    try:
        # Armamos un mosaico pegando todos los tiles necesarios.
        for tile_x in range(tile_x_min, tile_x_max + 1):
            for tile_y in range(tile_y_min, tile_y_max + 1):
                url = _format_tile_url(spec.url_template, zoom, tile_x, tile_y)
                try:
                    response = session.get(url, timeout=timeout)
                    response.raise_for_status()
                    tile = Image.open(BytesIO(response.content)).convert("RGBA")
                except Exception:
                    continue

                mosaic.paste(
                    tile,
                    ((tile_x - tile_x_min) * TILE_SIZE, (tile_y - tile_y_min) * TILE_SIZE),
                )
                fetched_tiles += 1
    finally:
        session.close()

    if fetched_tiles == 0:
        return None

    # Recortamos la región exacta de la bbox dentro del mosaico.
    crop_left = int(round(pixel_left - tile_x_min * TILE_SIZE))
    crop_top = int(round(pixel_top - tile_y_min * TILE_SIZE))
    crop_right = int(round(pixel_right - tile_x_min * TILE_SIZE))
    crop_bottom = int(round(pixel_bottom - tile_y_min * TILE_SIZE))

    crop_right = max(crop_left + 1, min(crop_right, mosaic.width))
    crop_bottom = max(crop_top + 1, min(crop_bottom, mosaic.height))

    cropped = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))
    # Ajustamos el recorte al tamaño exacto del canvas del GIF.
    return cropped.resize((width, height), resample=Image.Resampling.BILINEAR)
