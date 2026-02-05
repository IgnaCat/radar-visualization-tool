from fastapi import APIRouter, HTTPException
from typing import Dict, List
from ..core.constants import FIELD_COLORMAP_OPTIONS, FIELD_RENDER
from ..services.radar_common import colormap_for
import numpy as np

router = APIRouter(prefix="/colormap", tags=["colormap"])

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
        
        # Obtener colores RGBA del colormap
        rgba_colors = cmap(normalized_values)
        
        # Convertir RGBA (0-1) a hex
        hex_colors = [
            "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            for r, g, b, a in rgba_colors
        ]
        
        return {"colors": hex_colors, "steps": steps}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error obteniendo colormap '{cmap_name}': {str(e)}")
