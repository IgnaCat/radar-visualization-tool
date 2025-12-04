from fastapi import APIRouter
from typing import Dict, List
from ..core.constants import FIELD_COLORMAP_OPTIONS, FIELD_RENDER

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
