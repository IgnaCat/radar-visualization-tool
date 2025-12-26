"""
Modelos comunes compartidos entre diferentes dominios.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class RangeFilter(BaseModel):
    """Filtro de rango para campos de radar."""
    field: str = Field(..., description="Nombre del campo")
    type: Literal["range"] = "range"
    min: Optional[float] = Field(default=0, description="Límite inferior (inclusivo)")
    max: Optional[float] = Field(default=1, description="Límite superior (inclusivo)")
