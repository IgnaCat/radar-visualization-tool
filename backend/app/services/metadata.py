from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..core.constants import FIELD_ALIASES

import numpy as np
import pyart


def extract_radar_metadata(path: str) -> Dict[str, Any]:
    """
    Lee un NetCDF de radar con Py-ART y devuelve metadata básica:
      - fields_present: lista de campos presentes en el archivo
      - nsweeps: cantidad de sweeps
      - elevations: lista de ángulos de elevación (si está disponible)
      - range_max_m: alcance máximo (m)
      - site: lat/lon/alt del radar
      - instrument, time_units: metadatos útiles
    """
    p = Path(path)
    if not p.exists():
        return {"error": f"file_not_found: {path}"}

    try:
        radar = pyart.io.read(str(p), delay_field_loading=True)  # más rápido
    except Exception as e:
        return {"error": f"pyart_read_failed: {e.__class__.__name__}: {e}"}

    # Campos presentes
    try:
        fields_present = list(radar.fields.keys())
    except Exception:
        fields_present = []

    # Elevaciones
    try:
        elevs = radar.fixed_angle["data"]
        if hasattr(elevs, "filled"):
            elevs = elevs.filled(np.nan)
        elevations = [round(float(x), 2) for x in np.asarray(elevs).tolist()]
    except Exception:
        elevations = []

    # Sweeps
    try:
        nsweeps = int(radar.nsweeps)
    except Exception:
        nsweeps = len(elevations) if elevations else 0

    # Rango máximo
    try:
        rarr = radar.range["data"]
        range_max_m: Optional[float] = float(np.asarray(rarr)[-1])
    except Exception:
        range_max_m = None

    # Sitio
    def _first(v, default=None):
        try:
            arr = v["data"]
            return float(np.asarray(arr)[0])
        except Exception:
            return default

    lat = _first(getattr(radar, "latitude", {}), None)
    lon = _first(getattr(radar, "longitude", {}), None)
    alt = _first(getattr(radar, "altitude", {}), None)

    # Otros metadatos
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
        "elevations": elevations,               # lista de grados (puede traer NaN)
        "range_max_m": range_max_m,
        "site": {"lat": lat, "lon": lon, "alt_m": alt},
        "instrument": instrument,
        "time_units": time_units,
    }
