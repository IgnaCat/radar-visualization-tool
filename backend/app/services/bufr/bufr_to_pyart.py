"""
Conversión de campos BUFR decodificados a un objeto PyART Radar.

Adaptado de radarlib.io.bufr.bufr_to_pyart — se eliminaron dependencias a
radarlib, reemplazadas por imports locales.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


PRODUCT_UNITS = {
    "DBZH": "dBZ",
    "DBZV": "dBZ",
    "ZDR": "dB",
    "KDP": "deg/km",
    "VRAD": "m/s",
}


def _find_reference_field(fields: List[dict]) -> int:
    """Retorna el índice del campo que tiene el rango más lejano."""
    if not fields:
        raise ValueError("no fields provided")

    max_last = -1
    max_idx = 0
    for i, f in enumerate(fields):
        sweeps = f.get("info", {}).get("sweeps")
        if sweeps is None or sweeps.empty:
            continue
        last_gate = (
            sweeps["gate_offset"] + sweeps["gate_size"] * sweeps["ngates"]
        ).max()
        if last_gate > max_last:
            max_last = last_gate
            max_idx = i
    return max_idx


def _create_empty_radar(n_gates: int, n_rays: int, n_sweeps: int):
    """Crea un objeto Radar PPI vacío usando make_empty_ppi_radar de pyart."""
    import pyart

    return pyart.testing.make_empty_ppi_radar(n_gates, n_rays, n_sweeps)


def _align_field_to_reference(
    field: dict,
    ref_gate_offset: int,
    ref_gate_size: int,
    ref_ngates: int,
):
    """Alinea el array de datos de un campo al grillado de referencia."""
    out = field.copy()
    data = np.array(field["data"], copy=True)
    nrays, ngates = data.shape
    out_data = np.ma.masked_all((nrays, ref_ngates), dtype=np.float32)

    field_offset = int(field["info"]["sweeps"]["gate_offset"].iloc[0])
    field_gate_size = int(field["info"]["sweeps"]["gate_size"].iloc[0])

    if field_gate_size != ref_gate_size:
        raise ValueError("gate_size mismatch not supported in align routine")

    if field_offset == ref_gate_offset:
        out_data[:, :ngates] = data
    else:
        init = int((field_offset - ref_gate_offset) // ref_gate_size)
        if init < 0 or init + ngates > ref_ngates:
            raise ValueError("field cannot be aligned to reference grid")
        out_data[:, init : init + ngates] = data

    # Asegurar que NaN queden enmascarados (crítico para BUFR donde
    # los valores faltantes son NaN en lugar de _FillValue)
    out_data = np.ma.masked_invalid(out_data)

    out["data"] = out_data
    return out


def bufr_fields_to_pyart_radar(
    fields: List[dict],
    *,
    include_scan_metadata: bool = False,
    root_scan_config_files: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convierte una lista de campo-dicts BUFR decodificados en un Radar PyART.

    Cada elemento de `fields` proviene de `bufr_to_dict()` y contiene:
      - 'data': ndarray 2-D
      - 'info': dict con 'sweeps' DataFrame
    """
    if not fields:
        raise ValueError("fields is empty")

    ref_idx = _find_reference_field(fields)
    ref_field = fields[ref_idx]

    ref_ngates = int(ref_field["info"]["sweeps"]["ngates"].iloc[0])
    ref_rays_per_sweep = int(ref_field["info"]["sweeps"]["nrayos"].iloc[0])
    ref_nsweeps = int(ref_field["info"].get("nsweeps", 1))

    radar = _create_empty_radar(ref_ngates, ref_rays_per_sweep, ref_nsweeps)

    # Eje de rango
    gate_size = int(ref_field["info"]["sweeps"]["gate_size"].iloc[0])
    gate_offset = int(ref_field["info"]["sweeps"]["gate_offset"].iloc[0])
    range_data = gate_offset + gate_size * np.arange(radar.ngates)
    radar.range["data"] = range_data
    radar.range["meters_between_gates"] = gate_size
    radar.range["meters_to_center_of_first_gate"] = gate_offset

    # Elevación / azimut / fixed_angle
    rays_per_sweep = ref_field["info"]["sweeps"]["nrayos"].to_numpy()
    elevs = np.array(
        ref_field["info"]["sweeps"]["elevaciones"], dtype=np.float32
    )
    radar.elevation["data"] = np.repeat(elevs, rays_per_sweep)
    radar.azimuth["data"] = np.concatenate(
        [np.arange(n, dtype=np.float32) for n in rays_per_sweep]
    )
    radar.fixed_angle["data"] = elevs

    # Metadatos
    radar.metadata.update(ref_field["info"]["metadata"])

    # Coordenadas geográficas
    radar.latitude["data"] = np.array(
        [ref_field["info"].get("lat", 0)], dtype=np.float64
    )
    radar.latitude["units"] = "degrees"
    radar.latitude["long_name"] = "latitude"
    radar.latitude["_fillValue"] = -9999.0

    radar.longitude["data"] = np.array(
        [ref_field["info"].get("lon", 0)], dtype=np.float64
    )
    radar.longitude["units"] = "degrees"
    radar.longitude["long_name"] = "longitude"
    radar.longitude["_fillValue"] = -9999.0

    radar.altitude["data"] = np.array(
        [ref_field["info"].get("altura", 0)], dtype=np.float64
    )
    radar.altitude["units"] = "meters"
    radar.altitude["long_name"] = "altitude"
    radar.altitude["_fillValue"] = -9999.0

    # Agregar campos alineados a la referencia
    for field in fields:
        aligned = _align_field_to_reference(
            field, gate_offset, gate_size, ref_ngates
        )
        name = field["info"].get("tipo_producto", "UNKNOWN")
        if "info" in aligned:
            del aligned["info"]
        units = PRODUCT_UNITS.get(name)
        if units:
            aligned["units"] = units
        radar.add_field(name, aligned, replace_existing=True)

    return radar


def bufr_to_pyart(
    fields: List[dict],
    *,
    include_scan_metadata: bool = False,
    root_scan_config_files: Optional[Path] = None,
) -> Any:
    """
    Wrapper de conveniencia: convierte lista de campos BUFR decodificados
    a un Radar PyART.
    """
    if not fields:
        raise ValueError("fields is empty")
    return bufr_fields_to_pyart_radar(
        fields,
        include_scan_metadata=include_scan_metadata,
        root_scan_config_files=root_scan_config_files,
    )


def save_radar_to_cfradial(
    radar: Any, out_file: Path, format: str = "NETCDF4"
) -> Path:
    """Guarda un Radar PyART como NetCDF CFRadial."""
    import pyart

    try:
        pyart.io.cfradial.write_cfradial(str(out_file), radar, format=format)
    except Exception as exc:
        logger.error("Failed to write CFRadial file %s: %s", out_file, exc)
        raise
    return out_file


def get_netcdf_filename_from_bufr(ref_filename: str) -> str:
    """
    Genera nombre de archivo NetCDF a partir de un nombre BUFR.

    Formato BUFR: RADAR_ESTRATEGIA_VOL_CAMPO_TIMESTAMP.BUFR
    Formato NC:   RADAR_ESTRATEGIA_VOL_TIMESTAMP.nc
    """
    fichero = ref_filename.split(".")[0]
    parts = fichero.split("_")
    if len(parts) >= 5:
        # Quitar el componente de campo (parts[3])
        fichero = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[4]}"
    return fichero + ".nc"
