from fastapi import APIRouter, HTTPException
from typing import Dict, List
from ..core.constants import (
    FIELD_COLORMAP_OPTIONS,
    FIELD_RENDER,
    FIELD_LEGEND_VALUES,
    VARIABLE_UNITS,
)
from ..services.radar_common import colormap_for
import numpy as np

router = APIRouter(prefix="/colormap", tags=["colormap"])


def _rgba_to_hex_colors(rgba_colors):
    return [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _a in rgba_colors
    ]


def _sample_colormap_hex(cmap, normalized_values):
    rgba_colors = cmap(normalized_values)
    return _rgba_to_hex_colors(rgba_colors)

@router.get("/options", response_model=Dict[str, List[str]])
async def get_colormap_options():
    """
    Devuelve las opciones de colormaps disponibles para cada campo.
    """
    return FIELD_COLORMAP_OPTIONS

@router.get("/defaults", response_model=Dict[str, str])
async def get_colormap_defaults():
    """
    Devuelve el colormap por defecto para cada campo.
    """
    return {field: config["cmap"] for field, config in FIELD_RENDER.items()}


@router.get("/colors/{cmap_name}")
async def get_colormap_colors(cmap_name: str, steps: int = 256):
    """
    Devuelve una lista de colores RGB hexadecimales para el colormap especificado.
    
    Args:
        cmap_name: Nombre del colormap (ej: 'grc_th', 'pyart_NWSRef')
        steps: Número de pasos de color a generar (default 256)
    
    Returns:
        Lista de colores en formato hex: ['#RRGGBB', ...]
    """
    try:
        # Obtener el colormap usando la función existente
        # Usamos un campo dummy ya que solo nos interesa el colormap
        cmap, _, _, _ = colormap_for("DBZH", override_cmap=cmap_name)

        # Generar valores normalizados uniformemente espaciados
        normalized_values = np.linspace(0, 1, steps)

        hex_colors = _sample_colormap_hex(cmap, normalized_values)

        return {"colors": hex_colors, "steps": steps}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error obteniendo colormap '{cmap_name}': {str(e)}")


@router.get("/legend/{field_key}")
async def get_colormap_legend(field_key: str, cmap_name: str | None = None):
    """
    Devuelve una leyenda completa para un campo: valores, rango y colores alineados.

    - values: puntos de referencia de la leyenda para el campo
    - colors: color de cada value, muestreado en el cmap dentro de [vmin, vmax]
    """
    key = field_key.upper()
    if key not in FIELD_RENDER:
        raise HTTPException(status_code=404, detail=f"Campo no soportado: '{field_key}'")

    values = FIELD_LEGEND_VALUES.get(key)
    if not values:
        raise HTTPException(status_code=404, detail=f"No hay valores de leyenda para: '{field_key}'")

    try:
        cmap, vmin, vmax, cmap_key = colormap_for(key, override_cmap=cmap_name)

        values_arr = np.asarray(values, dtype=float)
        if vmax == vmin:
            normalized_values = np.zeros_like(values_arr)
        else:
            normalized_values = (values_arr - float(vmin)) / (float(vmax) - float(vmin))
            normalized_values = np.clip(normalized_values, 0.0, 1.0)

        hex_colors = _sample_colormap_hex(cmap, normalized_values)

        return {
            "field": key,
            "colormap": cmap_key,
            "vmin": float(vmin),
            "vmax": float(vmax),
            "values": values,
            "colors": hex_colors,
            "unit": VARIABLE_UNITS.get(key, ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error generando leyenda para campo '{key}': {str(e)}",
        )
