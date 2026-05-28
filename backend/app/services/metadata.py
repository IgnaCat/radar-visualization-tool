from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)

EXCLUDED_FIELDS_PRESENT = {"COLMAX", "DBZHF"}

# Variables de CFRadial que NO son campos de radar — se excluyen del listado.
# Incluye coordenadas, tiempo, geometría y variables de sweep.
_NC_NON_FIELD_VARS = {
    "time", "range", "azimuth", "elevation", "latitude", "longitude",
    "altitude", "altitude_agl", "sweep_number", "sweep_mode",
    "fixed_angle", "sweep_start_ray_index", "sweep_end_ray_index",
    "target_scan_rate", "rays_are_indexed", "ray_angle_res",
    "scan_rate", "antenna_transition", "instrument_parameters",
    "radar_parameters", "radar_calibration", "georefs_applied",
    "northward_wind", "eastward_wind", "vertical_wind",
    "heading", "roll", "pitch", "drift", "rotation", "tilt",
    "side_slip", "georefs_applied",
}


def _safe_scalar(var: Any, default=None) -> Optional[float]:
    """Extrae el primer valor numérico de una variable netCDF4."""
    try:
        arr = np.asarray(var[:])
        return float(arr.flat[0])
    except Exception:
        return default


def _extract_via_netcdf4(path: str) -> Dict[str, Any]:
    """
    Lee metadata del archivo usando netCDF4 directamente (sin PyART).

    Más rápido y más estable que pyart.io.read() para extraer solo headers:
    no construye el objeto Radar de PyART (ni sus destructores problemáticos).
    Asume formato CFRadial — el estándar de los archivos SMN/SiNaRaMe.
    """
    with nc.Dataset(path, "r") as ds:
        # ── Campos presentes ──────────────────────────────────────────────
        fields_present = [
            v for v in ds.variables
            if v.lower() not in _NC_NON_FIELD_VARS
            and v.strip().upper() not in EXCLUDED_FIELDS_PRESENT
            and ds.variables[v].ndim >= 2  # los campos tienen (ray, range) o similar
        ]

        # ── Elevaciones ───────────────────────────────────────────────────
        elevations: list[float] = []
        if "fixed_angle" in ds.variables:
            try:
                raw = np.asarray(ds.variables["fixed_angle"][:]).flatten()
                elevations = [round(float(x), 2) for x in raw.tolist()]
            except Exception:
                pass

        # ── Número de sweeps ──────────────────────────────────────────────
        try:
            nsweeps = int(getattr(ds, "nsweeps", len(elevations)))
        except Exception:
            nsweeps = len(elevations)

        # ── Rango máximo ──────────────────────────────────────────────────
        range_max_m: Optional[float] = None
        if "range" in ds.variables:
            try:
                rarr = np.asarray(ds.variables["range"][:]).flatten()
                range_max_m = float(rarr[-1])
            except Exception:
                pass

        # ── Coordenadas del sitio ─────────────────────────────────────────
        lat = _safe_scalar(ds.variables.get("latitude"))
        lon = _safe_scalar(ds.variables.get("longitude"))
        alt = _safe_scalar(ds.variables.get("altitude"))

        # ── Metadatos globales ────────────────────────────────────────────
        instrument = getattr(ds, "instrument_name", None)
        time_units: Optional[str] = None
        if "time" in ds.variables:
            try:
                time_units = getattr(ds.variables["time"], "units", None)
            except Exception:
                pass

    return {
        "fields_present": fields_present,
        "nsweeps": nsweeps,
        "elevations": elevations,
        "range_max_m": range_max_m,
        "last_gate_range_m": range_max_m,
        "radar_site": {"lat": lat, "lon": lon, "alt_m": alt},
        "site": {"lat": lat, "lon": lon, "alt_m": alt},
        "instrument": instrument,
        "time_units": time_units,
    }


def _extract_via_pyart(path: str) -> Dict[str, Any]:
    """
    Fallback: usa PyART cuando netCDF4 directo falla (formatos no-CFRadial).
    Hace del + gc.collect explícito para evitar que el destructor del objeto
    Radar corra en el GC del event loop y colisione con threads de procesamiento.
    """
    import pyart
    from ..core.cache import NETCDF_READ_LOCK

    radar = None
    try:
        with NETCDF_READ_LOCK:
            radar = pyart.io.read(str(path), delay_field_loading=True)

        fields_present = [
            f for f in radar.fields.keys()
            if str(f).strip().upper() not in EXCLUDED_FIELDS_PRESENT
        ]

        try:
            elevs = radar.fixed_angle["data"]
            if hasattr(elevs, "filled"):
                elevs = elevs.filled(np.nan)
            elevations = [round(float(x), 2) for x in np.asarray(elevs).tolist()]
        except Exception:
            elevations = []

        try:
            nsweeps = int(radar.nsweeps)
        except Exception:
            nsweeps = len(elevations)

        try:
            rarr = radar.range["data"]
            range_max_m: Optional[float] = float(np.asarray(rarr)[-1])
        except Exception:
            range_max_m = None

        def _first(v, default=None):
            try:
                return float(np.asarray(v["data"])[0])
            except Exception:
                return default

        lat = _first(getattr(radar, "latitude", {}))
        lon = _first(getattr(radar, "longitude", {}))
        alt = _first(getattr(radar, "altitude", {}))

        try:
            instrument = radar.metadata.get("instrument_name")
        except Exception:
            instrument = None
        try:
            time_units = radar.time.get("units")
        except Exception:
            time_units = None

        return {
            "fields_present": fields_present,
            "nsweeps": nsweeps,
            "elevations": elevations,
            "range_max_m": range_max_m,
            "last_gate_range_m": range_max_m,
            "radar_site": {"lat": lat, "lon": lon, "alt_m": alt},
            "site": {"lat": lat, "lon": lon, "alt_m": alt},
            "instrument": instrument,
            "time_units": time_units,
        }
    except Exception as e:
        return {"error": f"pyart_read_failed: {e.__class__.__name__}: {e}"}
    finally:
        # Destrucción explícita del objeto Radar ANTES de que el GC lo haga
        # de manera impredecible en el context de otro thread.
        if radar is not None:
            del radar
            gc.collect()


def extract_radar_metadata(path: str) -> Dict[str, Any]:
    """
    Extrae metadata básica de un archivo NetCDF de radar.

    Intenta primero con netCDF4 directo (más rápido y estable).
    Si falla (formato no-CFRadial, archivo corrupto, etc.), usa PyART como fallback.

    Retorna:
        - fields_present: lista de campos del radar
        - nsweeps: cantidad de sweeps
        - elevations: ángulos de elevación en grados
        - range_max_m: alcance máximo en metros
        - site / radar_site: lat/lon/alt del radar
        - instrument, time_units: metadatos adicionales
    """
    p = Path(path)
    if not p.exists():
        return {"error": f"file_not_found: {path}"}

    # Intento 1: netCDF4 directo (sin PyART, sin destructores problemáticos)
    try:
        result = _extract_via_netcdf4(path)
        # Sanity check mínimo: si no encontramos ningún campo, el archivo
        # probablemente no es CFRadial → probamos PyART
        if result.get("fields_present") or result.get("elevations"):
            return result
        logger.debug(
            "netCDF4 directo no encontró campos en %s, reintentando con PyART", path
        )
    except Exception as e:
        logger.debug("netCDF4 directo falló para %s (%s), usando PyART", path, e)

    # Intento 2: PyART (más robusto para formatos no-CFRadial)
    return _extract_via_pyart(path)
