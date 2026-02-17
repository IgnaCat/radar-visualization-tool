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


class FileCleanupRequest(BaseModel):
    """Request para eliminar archivos específicos subidos por el usuario."""
    filepaths: list[str] = Field(..., min_length=1, description="Lista de rutas de archivos a eliminar")
    session_id: Optional[str] = Field(
        default=None,
        description="ID de sesión para resolución de rutas y cleanup de cache"
    )
