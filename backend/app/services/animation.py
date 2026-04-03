"""
Generación de animaciones GIF a partir de rasters procesados.
"""

import os
import uuid
import logging
from math import ceil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from ..core.config import settings
from .basemaps import get_basemap_spec, render_basemap_for_canvas


logger = logging.getLogger(__name__)


def create_animation(image_paths):
    """
    Mantiene compatibilidad con el helper legacy de animaciones simples.
    """
    resolved = [Path(p) for p in image_paths if Path(p).exists()]
    if not resolved:
        raise ValueError("No se pudieron abrir las imágenes para la animación")

    frames = [Image.open(path).convert("RGB") for path in resolved]
    gif_name = f"anim_{uuid.uuid4().hex[:8]}.gif"
    gif_path = resolved[0].parent / gif_name
    duration_ms = 1000

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    return str(gif_path)


def _resolve_animation_image_path(image_ref: str, session_id: str | None = None) -> Path:
    """
    Resuelve una URL/ruta lógica de imagen a un path local seguro dentro de IMAGES_DIR.
    """
    raw = str(image_ref or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("Se recibió una ruta de imagen vacía")

    if "://" in raw:
        raw = raw.split("://", 1)[1]
        raw = raw.split("/", 1)[1] if "/" in raw else ""

    raw = raw.lstrip("/")
    images_dir = Path(settings.IMAGES_DIR).resolve()

    candidates: list[Path] = []

    if raw.startswith("static/tmp/"):
        candidates.append(images_dir / raw.split("static/tmp/", 1)[1])
    elif raw.startswith("tmp/"):
        candidates.append(images_dir / raw.split("tmp/", 1)[1])
    else:
        if session_id:
            candidates.append(images_dir / session_id / raw)
        candidates.append(images_dir / raw)

    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        try:
            if resolved.is_relative_to(images_dir) and resolved.exists():  # type: ignore[attr-defined]
                return resolved
        except AttributeError:
            from os.path import commonpath

            if (
                commonpath([str(resolved), str(images_dir)]) == str(images_dir)
                and resolved.exists()
            ):
                return resolved

    raise ValueError(f"No se encontró la imagen para animación: {image_ref}")


def _alpha_composite_rgba(base_rgba: np.ndarray, over_rgba: np.ndarray) -> np.ndarray:
    """
    Compone dos arrays RGBA uint8 con alpha straight.
    """
    base = base_rgba.astype(np.float32) / 255.0
    over = over_rgba.astype(np.float32) / 255.0

    base_rgb = base[:3]
    base_a = base[3:4]
    over_rgb = over[:3]
    over_a = over[3:4]

    out_a = over_a + base_a * (1.0 - over_a)
    safe_out_a = np.where(out_a == 0, 1.0, out_a)
    out_rgb = (
        over_rgb * over_a + base_rgb * base_a * (1.0 - over_a)
    ) / safe_out_a

    out = np.concatenate([out_rgb, out_a], axis=0)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _rgba_to_transparent_gif_frame(rgba_frame: np.ndarray) -> Image.Image:
    """
    Convierte un frame RGBA a un frame GIF paletizado preservando transparencia.
    """
    rgba_image = Image.fromarray(rgba_frame, mode="RGBA")
    alpha = rgba_frame[..., 3]
    transparent_mask = Image.fromarray(
        np.where(alpha == 0, 255, 0).astype(np.uint8),
        mode="L",
    )

    paletted = rgba_image.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT)
    paletted.paste(255, mask=transparent_mask)
    paletted.info["transparency"] = 255

    return paletted


def _render_frame_rgba(
    frame_paths: list[Path],
    width: int,
    height: int,
    dst_transform,
    dst_crs,
    background_rgba: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reproyecta y compone todas las capas visibles de un frame en un solo RGBA.
    """
    if background_rgba is not None:
        composed = np.moveaxis(background_rgba.copy(), -1, 0)
    else:
        composed = np.zeros((4, height, width), dtype=np.uint8)

    # La primera capa del array tiene mayor prioridad visual en el mapa,
    # así que componemos en orden inverso: fondo -> frente.
    for path in reversed(frame_paths):
        with rasterio.open(path) as src:
            layer = np.zeros((4, height, width), dtype=np.uint8)

            for band_idx in range(1, min(src.count, 4) + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=layer[band_idx - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0,
                )

            if src.count < 4:
                layer[3] = np.where(layer[:3].any(axis=0), 255, 0).astype(np.uint8)

            composed = _alpha_composite_rgba(composed, layer)

    return np.moveaxis(composed, 0, -1)


def create_animation_from_layer_urls(
    frame_layers,
    output_dir,
    fps=1,
    session_id=None,
    basemap_id=None,
):
    """
    Crea un GIF animado a partir de capas raster ya procesadas.

    Cada frame puede contener una o varias capas visibles. Las capas se reproyectan
    a un canvas común en EPSG:3857 y se componen en un frame completo.
    """
    resolved_frames = []
    all_paths = []

    for frame in frame_layers:
        paths = [
            _resolve_animation_image_path(image_ref, session_id=session_id)
            for image_ref in (frame or [])
            if image_ref
        ]
        if paths:
            resolved_frames.append(paths)
            all_paths.extend(paths)

    if not resolved_frames:
        raise ValueError("No hay capas visibles válidas para generar la animación")

    datasets_meta = []
    for path in all_paths:
        with rasterio.open(path) as src:
            datasets_meta.append(
                {
                    "bounds": src.bounds,
                    "res": src.res,
                    "crs": src.crs,
                }
            )

    crs = datasets_meta[0]["crs"]
    left = min(meta["bounds"].left for meta in datasets_meta)
    bottom = min(meta["bounds"].bottom for meta in datasets_meta)
    right = max(meta["bounds"].right for meta in datasets_meta)
    top = max(meta["bounds"].top for meta in datasets_meta)
    res_x = min(abs(meta["res"][0]) for meta in datasets_meta)
    res_y = min(abs(meta["res"][1]) for meta in datasets_meta)

    width = max(1, int(ceil((right - left) / res_x)))
    height = max(1, int(ceil((top - bottom) / res_y)))
    dst_transform = from_origin(left, top, res_x, res_y)
    # Usamos los bounds reales del canvas para alinear bien el fondo.
    canvas_right = left + width * res_x
    canvas_bottom = top - height * res_y

    background_rgba = None
    if basemap_id:
        spec = get_basemap_spec(basemap_id)
        if spec is None:
            raise ValueError(f"Mapa base no soportado para GIF: {basemap_id}")

        background_image = render_basemap_for_canvas(
            spec=spec,
            left=left,
            bottom=canvas_bottom,
            right=canvas_right,
            top=top,
            width=width,
            height=height,
            target_resolution=min(res_x, res_y),
        )
        if background_image is None:
            logger.warning(
                "No se pudo renderizar el mapa base '%s' para el GIF; se mantiene fondo transparente",
                basemap_id,
            )
        else:
            background_rgba = np.array(background_image, dtype=np.uint8)

    rendered_frames = []
    for frame_paths in resolved_frames:
        rendered_frames.append(
            _render_frame_rgba(
                frame_paths=frame_paths,
                width=width,
                height=height,
                dst_transform=dst_transform,
                dst_crs=crs,
                background_rgba=background_rgba,
            )
        )

    if not rendered_frames:
        raise ValueError("No se pudieron renderizar frames para la animación")

    os.makedirs(output_dir, exist_ok=True)

    gif_name = f"anim_{uuid.uuid4().hex[:8]}.gif"
    gif_path = Path(output_dir) / gif_name
    duration_ms = max(40, int(1000 / max(1, fps)))

    pil_frames = [
        _rgba_to_transparent_gif_frame(frame)
        for frame in rendered_frames
    ]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
        transparency=255,
    )

    return gif_name
