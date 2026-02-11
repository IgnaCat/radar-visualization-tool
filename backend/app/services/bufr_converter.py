"""
Servicio para convertir archivos BUFR de radar a formato NetCDF (CFRadial).

Utiliza el módulo local app.services.bufr para decodificar archivos BUFR y
escribirlos como NetCDF, permitiendo que el resto del pipeline (procesamiento
PyART, generación COG, etc.) trabaje con un formato único.

Los archivos BUFR de un volumen de radar vienen separados por campo
(ej. DBZH, VRAD, ZDR cada uno en un archivo .BUFR distinto).
Este servicio los agrupa por volumen y fusiona todos los campos en
un NetCDF por volumen — coincidiendo con lo que el SMN provee como .nc.
"""

from __future__ import annotations

import logging
import re
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Patrón:  RADAR_ESTRATEGIA_VOL_CAMPO_TIMESTAMPZ.BUFR
_BUFR_NAME_RE = re.compile(
    r"^(?P<radar>\w+)_(?P<strategy>\d+)_(?P<vol>\d+)_(?P<field>\w+)_(?P<timestamp>\d{8}T\d{6}Z)\.BUFR$",
    re.IGNORECASE,
)


def _volume_key(path: Path) -> str | None:
    """
    Extrae una clave de agrupación desde un nombre de archivo BUFR.

    Los archivos que comparten el mismo (radar, estrategia, vol, timestamp) pertenecen al
    mismo volumen de radar y deben fusionarse en un único NetCDF.
    Retorna None si el nombre no coincide con el patrón esperado.
    """
    m = _BUFR_NAME_RE.match(path.name)
    if not m:
        return None
    return f"{m['radar']}_{m['strategy']}_{m['vol']}_{m['timestamp']}"


def group_bufr_by_volume(bufr_paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Agrupa paths de archivos BUFR por clave de volumen.

    Retorna un dict mapeando volume_key -> lista de paths BUFR.
    Los archivos que no coinciden con el patrón de nombre se agrupan bajo
    una clave sintética basada en su stem.
    """
    groups: Dict[str, List[Path]] = {}
    for p in bufr_paths:
        key = _volume_key(p) or p.stem
        groups.setdefault(key, []).append(p)
    return groups


def _netcdf_name_from_volume_key(volume_key: str) -> str:
    """
    Deriva el nombre del archivo NetCDF desde una clave de volumen.

    Formato de clave de volumen: RADAR_ESTRATEGIA_VOL_TIMESTAMPZ
    Nombre NetCDF:               RADAR_ESTRATEGIA_VOL_TIMESTAMPZ.nc
    (Elimina el componente por campo que llevan los archivos BUFR.)
    """
    return f"{volume_key}.nc"


def convert_bufr_to_netcdf(
    bufr_paths: List[Path],
    output_dir: Path,
) -> List[Tuple[Path, str]]:
    """
    Convierte una lista de archivos BUFR a NetCDF (CFRadial).

    Agrupa archivos por volumen, decodifica + fusiona campos,
    y escribe un .nc por volumen en *output_dir*.

    Returns:
        Lista de (netcdf_path, volume_key) para cada volumen convertido exitosamente.
        Los archivos que fallan en la conversión se registran en log y se omiten.
    """
    from app.services.bufr import (
        bufr_to_dict,
        bufr_fields_to_pyart_radar,
        save_radar_to_cfradial,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    volumes = group_bufr_by_volume(bufr_paths)
    results: List[Tuple[Path, str]] = []

    for vol_key, paths in volumes.items():
        nc_name = _netcdf_name_from_volume_key(vol_key)
        nc_path = output_dir / nc_name

        # Saltar si ya fue convertido (idempotente)
        if nc_path.exists():
            logger.info("BUFR→NetCDF ya existe, se omite conversión: %s", nc_path.name)
            results.append((nc_path, vol_key))
            continue

        logger.info(
            "Convirtiendo volumen BUFR '%s' (%d archivos) → %s",
            vol_key,
            len(paths),
            nc_name,
        )

        # Paso 1: decodificar cada archivo BUFR a un diccionario de campo
        fields = []
        for p in paths:
            try:
                vol = bufr_to_dict(str(p), root_resources=None, legacy=False)
                if vol is not None:
                    fields.append(vol)
                else:
                    logger.warning("bufr_to_dict retornó None para %s", p.name)
            except Exception:
                logger.exception("Error decodificando BUFR %s", p.name)

        if not fields:
            logger.error(
                "No se pudo decodificar ningún BUFR del volumen '%s' — se omite.",
                vol_key,
            )
            continue

        # Paso 2: fusionar campos en un único Radar PyART
        try:
            radar = bufr_fields_to_pyart_radar(fields)
        except Exception:
            logger.exception("Error creando Radar PyART para volumen '%s'", vol_key)
            continue

        # Paso 3: escribir NetCDF CFRadial
        try:
            save_radar_to_cfradial(radar, nc_path, format="NETCDF4")
            logger.info("✓ Volumen '%s' guardado como %s", vol_key, nc_path.name)
            results.append((nc_path, vol_key))
        except Exception:
            logger.exception("Error escribiendo NetCDF para volumen '%s'", vol_key)

    return results
