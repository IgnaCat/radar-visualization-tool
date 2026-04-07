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
from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from ..core.config import settings
from .basemaps import get_basemap_spec, render_basemap_for_canvas


logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_LOGO_PATH = _ASSETS_DIR / "lrsr_logo.png"

# Colores que coinciden exactamente con el frontend
_COLORBAR_BG = (0, 0, 0, 178)         # rgba(0,0,0,0.7) - Paper de ColorLegendField
_COLORBAR_RADIUS = 12                  # borderRadius: 3 en MUI = 12px
_HEADER_BG = (74, 144, 226, 242)       # rgba(74,144,226,0.95) - HeaderCard
_HEADER_RADIUS = 8                     # borderRadius: "8px" - HeaderCard
_META_BG = (255, 255, 255, 230)        # bgcolor: white semi-opaco - AnimationControls metadata
_META_RADIUS = 12                      # borderRadius: 3 = 12px

# Altura del strip inferior dedicado a la metadata (en px del canvas)
_BOTTOM_STRIP = 38


def _get_font(size: int = 11):
    """Intenta cargar una fuente TrueType; fallback a la fuente por defecto de Pillow."""
    # Rutas comunes en Linux (Docker), macOS y Windows
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _get_bold_font(size: int = 11):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return _get_font(size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Devuelve (width, height) del texto dado."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple, radius: int, fill):
    """Dibuja un rectángulo con esquinas redondeadas (Pillow 8.2+)."""
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def _text_with_shadow(
    draw: ImageDraw.ImageDraw,
    pos: tuple,
    text: str,
    font,
    fill=(255, 255, 255, 255),
    shadow=(0, 0, 0, 180),
    anchor: str = "mm",
):
    """Dibuja texto con sombra (imita textShadow: 1px 1px 1px rgba(0,0,0,0.9))."""
    x, y = pos
    draw.text((x + 1, y + 1), text, fill=shadow, font=font, anchor=anchor)
    draw.text((x, y), text, fill=fill, font=font, anchor=anchor)


# ── ColorLegendField (coincide con ColorLegendField.jsx) ─────────────────

def _generate_colorbar_pil(field: str, colormap_name: str | None = None) -> Image.Image | None:
    """
    Genera la barra de colores replicando el estilo de ColorLegendField.jsx.

    Medidas tomadas del componente React (1 MUI unit = 8px):
      paddingBlock:  1.5 → 12px   paddingLeft: 2 → 16px   paddingRight: 3.5 → 28px
      barHeight: 200px   barWidth: 14px
      labelsGap: 8px     labelsWidth: 36px
      borderRadius: 3 → 12px   background: rgba(0,0,0,0.7)
    """
    from .radar_common import colormap_for
    from ..core.constants import FIELD_LEGEND_VALUES, VARIABLE_UNITS

    try:
        field_key = field.upper()
        cmap, vmin, vmax, _ = colormap_for(field_key, override_cmap=colormap_name)
        values = FIELD_LEGEND_VALUES.get(field_key, [])
        unit = VARIABLE_UNITS.get(field_key, "").strip()
    except Exception as e:
        logger.warning("No se pudo generar colorbar para '%s': %s", field, e)
        return None

    if not values:
        return None

    # Dimensiones exactas del frontend
    pad_top = 12
    pad_bottom = 12
    pad_left = 10
    pad_right = 10
    bar_h = 200
    bar_w = 14
    label_gap = 8
    label_w = 32
    title_mb = 12     # marginBottom debajo del título
    unit_mt = 12      # marginTop encima de la unidad

    font_title = _get_bold_font(13)   # subtitle2 bold 0.8rem≈12.8px
    font_label = _get_font(11)        # 0.7rem≈11.2px
    font_unit = _get_font(11)         # caption 0.68rem≈10.9px

    # Medir alturas reales de texto
    tmp_img = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)
    _, title_h = _text_size(tmp_draw, field_key, font_title)
    _, unit_h = _text_size(tmp_draw, unit, font_unit) if unit else (0, 0)

    total_w = pad_left + bar_w + label_gap + label_w + pad_right
    total_h = (pad_top + title_h + title_mb
               + bar_h
               + (unit_mt + unit_h if unit else 0)
               + pad_bottom)

    # Canvas transparente + rounded rect de fondo
    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _draw_rounded_rect(draw, [(0, 0), (total_w - 1, total_h - 1)], _COLORBAR_RADIUS, _COLORBAR_BG)

    # Título centrado (con sombra)
    title_y = pad_top + title_h // 2
    _text_with_shadow(draw, (total_w // 2, title_y), field_key, font_title, anchor="mm")

    # Barra de gradiente (top=vmax, bottom=vmin — igual que el CSS reversed)
    bar_x = pad_left + 10
    bar_y = pad_top + title_h + title_mb
    
    # Dibujamos el gradiente en una imagen temporal
    grad_img = Image.new("RGBA", (bar_w, bar_h))
    grad_draw = ImageDraw.Draw(grad_img)
    for i in range(bar_h):
        norm = 1.0 - i / max(bar_h - 1, 1)
        rgba = cmap(norm)
        color = tuple(int(c * 255) for c in rgba[:3]) + (255,)
        grad_draw.line([(0, i), (bar_w - 1, i)], fill=color)
        
    # Creamos una máscara con los bordes redondeados
    bar_radius = 4  # Radio de redondeo de la varita
    mask_img = Image.new("L", (bar_w, bar_h), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    _draw_rounded_rect(mask_draw, [(0, 0), (bar_w - 1, bar_h - 1)], radius=bar_radius, fill=255)
    
    # Pegamos la barra temporal redondeada sobre la imagen principal
    img.paste(grad_img, (bar_x, bar_y), mask=mask_img)

    # Tick labels con sombra (posición: (1 - norm) * bar_h desde el top del bar)
    label_x_start = bar_x + bar_w + label_gap
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if vmax == vmin:
            continue
        norm = (v - vmin) / (vmax - vmin)
        norm = max(0.0, min(1.0, norm))
        y = bar_y + int((1.0 - norm) * (bar_h - 1))
        # Tick horizontal
        draw.line([(bar_x + bar_w, y), (bar_x + bar_w + label_gap - 1, y)],
                  fill=(200, 200, 200, 200))
        # Valor con sombra
        label_str = str(value)
        _text_with_shadow(draw, (label_x_start, y), label_str, font_label, anchor="lm")

    # Unidad centrada debajo del bar (fontWeight 500, ligeramente más tenue)
    if unit:
        unit_y = bar_y + bar_h + unit_mt + unit_h // 2
        _text_with_shadow(draw, (total_w // 2, unit_y), unit, font_unit,
                          fill=(200, 200, 200, 255), anchor="mm")

    return img


# ── Helpers de composición ────────────────────────────────────────────────

def _paste_overlay(frame_rgba: np.ndarray, overlay: Image.Image, x: int, y: int) -> np.ndarray:
    """Pega un overlay RGBA (PIL) sobre un frame RGBA (numpy) en la posición (x, y)."""
    frame_pil = Image.fromarray(frame_rgba, mode="RGBA")
    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")
    frame_pil.paste(overlay, (x, y), mask=overlay)
    return np.array(frame_pil)


# ── Logo (coincide con HeaderCard.jsx) ───────────────────────────────────

def _overlay_logo(frame_rgba: np.ndarray, target_h: int = 36, padding: int = 10) -> np.ndarray:
    """
    Superpone el logo LSRS replicando el estilo de HeaderCard.jsx:
      background: rgba(74, 144, 226, 0.95)   borderRadius: 8px
      padding: 8px 14px   logo height: 36px
    """
    if not _LOGO_PATH.exists():
        logger.warning("Logo no encontrado en %s", _LOGO_PATH)
        return frame_rgba
    try:
        logo = Image.open(_LOGO_PATH).convert("RGBA")
        ratio = target_h / logo.height
        new_w = max(1, int(logo.width * ratio))
        logo_resized = logo.resize((new_w, target_h), Image.LANCZOS)

        pad_v, pad_h = 8, 14
        card_w = new_w + pad_h * 2
        card_h = target_h + pad_v * 2

        card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
        card_draw = ImageDraw.Draw(card)
        _draw_rounded_rect(card_draw, [(0, 0), (card_w - 1, card_h - 1)], _HEADER_RADIUS, _HEADER_BG)
        card.paste(logo_resized, (pad_h, pad_v), mask=logo_resized)

        return _paste_overlay(frame_rgba, card, padding, padding)
    except Exception as e:
        logger.warning("Error superponiendo logo: %s", e)
        return frame_rgba


# ── Margen extra del canvas ───────────────────────────────────────────────

def _expand_canvas(
    frame_rgba: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    fill: tuple = (255, 255, 255, 255),
) -> np.ndarray:
    """
    Añade margen alrededor del frame con el color de relleno indicado.
    Úsese antes de aplicar overlays para que no se superpongan al radar.
    """
    h, w = frame_rgba.shape[:2]
    new_h = h + pad_top + pad_bottom
    new_w = w + pad_left + pad_right
    canvas = np.empty((new_h, new_w, 4), dtype=np.uint8)
    canvas[:] = fill
    canvas[pad_top:pad_top + h, pad_left:pad_left + w] = frame_rgba
    return canvas


# ── Colorbar posicionado (bottom-left, encima del strip de metadata) ─────

def _overlay_colorbar(
    frame_rgba: np.ndarray,
    colorbar_img: Image.Image,
    padding: int = 10,
    bottom_offset: int = 0,
) -> np.ndarray:
    """
    Superpone la colorbar en la esquina inferior izquierda.
    bottom_offset: espacio extra a respetar desde el borde inferior (strip de metadata).
    """
    h = frame_rgba.shape[0]
    cb_h = colorbar_img.height
    y = max(0, h - cb_h - bottom_offset - padding)
    return _paste_overlay(frame_rgba, colorbar_img, padding, y)


# ── Metadata (coincide con AnimationControls.jsx metadata box) ───────────

def _overlay_metadata(frame_rgba: np.ndarray, text: str) -> np.ndarray:
    """
    Superpone texto de metadata centrado en el strip inferior del canvas.
    La fuente es más chica que en el frontend para no ocupar demasiado ancho.
    """
    if not text:
        return frame_rgba

    h, w = frame_rgba.shape[:2]
    font = _get_font(11)  # reducido para que entre en el strip

    pad_v, pad_h = 5, 12

    # Medir texto
    tmp_img = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)
    text_w, text_h = _text_size(tmp_draw, text, font)

    card_w = min(text_w + pad_h * 2, w - 20)
    card_h = text_h + pad_v * 2

    card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card)
    _draw_rounded_rect(card_draw, [(0, 0), (card_w - 1, card_h - 1)], _META_RADIUS, _META_BG)
    card_draw.text(
        (card_w // 2, card_h // 2),
        text,
        fill=(0, 0, 0, 255),
        font=font,
        anchor="mm",
    )

    # Centrar horizontalmente; centrar verticalmente dentro del _BOTTOM_STRIP
    x = max(0, (w - card_w) // 2)
    strip_start = h - _BOTTOM_STRIP
    y = strip_start + (_BOTTOM_STRIP - card_h) // 2
    y = max(strip_start, y)
    return _paste_overlay(frame_rgba, card, x, y)


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


def _rgba_to_transparent_gif_frame(rgba_frame: np.ndarray, global_palette: Image.Image | None = None) -> Image.Image:
    """
    Convierte un frame RGBA a un frame GIF paletizado preservando transparencia.
    Si se provee global_palette, fuerza que este frame utilice exactamente esa paleta.
    """
    rgba_image = Image.fromarray(rgba_frame, mode="RGBA")
    alpha = rgba_frame[..., 3]
    transparent_mask = Image.fromarray(
        np.where(alpha == 0, 255, 0).astype(np.uint8),
        mode="L",
    )

    rgb_image = rgba_image.convert("RGB")
    if global_palette is not None:
        paletted = rgb_image.quantize(palette=global_palette, dither=0)
    else:
        paletted = rgb_image.quantize(colors=255, method=Image.MEDIANCUT)

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
    show_logo=False,
    show_colorbar=False,
    colorbar_config=None,
    show_metadata=False,
    frame_labels=None,
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
    
    # Factor de escala: Aumenta este valor (ej: 1.5, 2.0) para hacer el mapa más grande
    scale_factor = 1.5
    res_x = min(abs(meta["res"][0]) for meta in datasets_meta) / scale_factor
    res_y = min(abs(meta["res"][1]) for meta in datasets_meta) / scale_factor

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

    # Preparar overlays reutilizables
    colorbar_img = None
    if show_colorbar and colorbar_config:
        field = getattr(colorbar_config, "field", None) or colorbar_config.get("field", "") if isinstance(colorbar_config, dict) else colorbar_config.field
        colormap_name = getattr(colorbar_config, "colormap", None) or (colorbar_config.get("colormap") if isinstance(colorbar_config, dict) else None)
        colorbar_img = _generate_colorbar_pil(field, colormap_name)

    # Agregar strip inferior solo para la metadata (logo y colorbar se superponen al mapa)
    if show_metadata:
        strip_color = (255, 255, 255, 255) if background_rgba is not None else (20, 20, 20, 255)
        rendered_frames = [
            _expand_canvas(f, 0, _BOTTOM_STRIP, 0, 0, strip_color)
            for f in rendered_frames
        ]

    # Aplicar overlays a cada frame
    cb_bottom_offset = _BOTTOM_STRIP if show_metadata else 0
    for i, frame in enumerate(rendered_frames):
        if show_metadata and frame_labels and i < len(frame_labels) and frame_labels[i]:
            frame = _overlay_metadata(frame, frame_labels[i])
        if colorbar_img is not None:
            frame = _overlay_colorbar(frame, colorbar_img, bottom_offset=cb_bottom_offset)
        if show_logo:
            frame = _overlay_logo(frame)
        rendered_frames[i] = frame

    os.makedirs(output_dir, exist_ok=True)

    gif_name = f"anim_{uuid.uuid4().hex[:8]}.gif"
    gif_path = Path(output_dir) / gif_name
    duration_ms = max(40, int(1000 / max(1, fps)))

    # Generar una paleta global combinando todos los frames en una sola imagen temporal (evita parpadeos)
    global_palette = None
    if rendered_frames:
        # Apilar verticalmente los arrays RGBA y calcular la mejor paleta común de 255 colores
        combined_rgba = np.vstack(rendered_frames)
        combined_pil = Image.fromarray(combined_rgba, mode="RGBA").convert("RGB")
        global_palette = combined_pil.quantize(colors=255, method=Image.MEDIANCUT)

    pil_frames = [
        _rgba_to_transparent_gif_frame(frame, global_palette)
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
