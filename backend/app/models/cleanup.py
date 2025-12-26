"""
Modelos para limpieza de archivos temporales.
"""
from pydantic import BaseModel, Field
from typing import Optional


class CleanupRequest(BaseModel):
    """Request para limpieza de archivos temporales."""
    uploads: list[str] = []
    cogs: list[str] = []
    delete_cache: bool = False
    session_id: Optional[str] = Field(
        default=None,
        description="ID de sesión para cleanup selectivo"
    )
