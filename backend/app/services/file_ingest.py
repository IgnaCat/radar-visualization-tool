"""
Descripción de un NetCDF ya presente en disco, para /admin/demo/load.

El endpoint demo, después de copiar el archivo a la sesión, debe describirlo con el
mismo shape que /upload (metadata + radar + volumen) para que el frontend no distinga
el origen. Esa lógica vive acá, aislada y testeable.

Nota: /upload mantiene su propia construcción inline por decisión de alcance (no se
toca upload.py). Si en el futuro se unifica, extender este helper (p. ej. size_bytes /
extra override) y hacer que upload lo use.
"""
from pathlib import Path
from typing import Optional

from .metadata import extract_radar_metadata
from ..utils import helpers


def build_file_entry(
    path: Path,
    *,
    filepath: Optional[str] = None,
    filename: Optional[str] = None,
) -> tuple[dict, Optional[str], Optional[str]]:
    """
    Describe un NetCDF ya presente en disco.

    Args:
        path: Ruta al NetCDF en disco (se lee su metadata y su tamaño).
        filepath: Nombre relativo con el que el frontend lo referencia. Default: path.name.
        filename: Nombre "humano" del archivo. Default: path.name.

    Returns:
        (entry, volume, radar). entry es el dict que consume el frontend;
        volume y radar salen del nombre del archivo (pueden ser None).
    """
    path = Path(path)
    meta = extract_radar_metadata(str(path))
    radar, _, volume, _ = helpers.extract_metadata_from_filename(str(path))

    entry = {
        "filepath": filepath if filepath is not None else path.name,
        "filename": filename if filename is not None else path.name,
        "size_bytes": path.stat().st_size,
        "metadata": meta,
    }
    return entry, volume, radar
