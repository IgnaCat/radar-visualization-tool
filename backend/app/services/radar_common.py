from __future__ import annotations
from typing import Iterable, Optional, Tuple

import numpy as np
import pyart
from pyproj import Geod
import hashlib

from ..utils import colores
from ..core.constants import FIELD_ALIASES, FIELD_RENDER
from ..schemas import RangeFilter


# ------------------------------
# Hash de archivo
# ------------------------------

def md5_file(path, chunk=1024*1024):
    """
    Devuelve el hash MD5 (hexadecimal) de un archivo.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------
# Campos / colormaps
# ------------------------------

def resolve_field(radar: pyart.core.Radar, requested: str) -> Tuple[str, str]:
    """
    Devuelve (field_name_en_archivo, field_key_canon) a partir de un 'requested'.
    Usa FIELD_ALIASES. Lanza KeyError si no encuentra.
    """
    key = requested.upper()
    if key not in FIELD_ALIASES:
        raise KeyError(f"Campo no soportado: {requested}")
    for cand in FIELD_ALIASES[key]:
        if cand in radar.fields:
            return cand, key
    raise KeyError(f"No se encontró alias disponible para '{requested}' en el archivo.")

def colormap_for(field_key: str):
    """
    Devuelve defaults (cmap, vmin, vmax, cmap_key) según FIELD_RENDER.
    """
    spec = FIELD_RENDER.get(field_key.upper(), {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"})
    vmin, vmax, cmap_key = spec["vmin"], spec["vmax"], spec["cmap"]
    if field_key not in ["VRAD", "WRAD", "PHIDP"]:
        cmap = getattr(colores, f"get_cmap_{cmap_key}")()
    else: # Usamos directamente cmap de pyart
        cmap = cmap_key
    return cmap, vmin, vmax, cmap_key


# ------------------------------
# Radar metadata segura
# ------------------------------

def get_radar_site(radar: pyart.core.Radar) -> Tuple[float, float, float]:
    """
    Devuelve (lon, lat, alt_m) del sitio del radar.
    """
    lat = float(np.asarray(radar.latitude["data"]).ravel()[0])
    lon = float(np.asarray(radar.longitude["data"]).ravel()[0])
    alt = 0.0
    try:
        alt = float(np.asarray(radar.altitude["data"]).ravel()[0])
    except Exception:
        pass
    return lon, lat, alt

def safe_range_max_m(radar: pyart.core.Radar, default: float = 240e3) -> float:
    """
    Devuelve el alcance máximo (último gate) en metros, con fallback.
    """
    r = radar.range["data"]
    arr = np.asarray(getattr(r, "filled", lambda v: r)(np.nan), dtype=float)
    if arr.size == 0:
        return float(default)
    last = float(arr[-1])
    if np.isfinite(last):
        return last
    # fallback al máximo finito
    finite = arr[np.isfinite(arr)]
    return float(finite.max()) if finite.size else float(default)


# ------------------------------
# GateFilter común
# ------------------------------

def build_gatefilter(
    radar: pyart.core.Radar,
    field: Optional[str],
    filters: Optional[Iterable[RangeFilter]] = []
) -> pyart.filters.GateFilter:
    """
    Construye un GateFilter consistente:
      - exclude_transition()
      - exclude_invalid/masked para el campo base (si existe)
      - aplica filtros por rango (min/max) por campo
    """
    gf = pyart.filters.GateFilter(radar)
    try:
        gf.exclude_transition()
    except Exception:
        pass

    if field in radar.fields:
        try:
            gf.exclude_invalid(field)
            gf.exclude_masked(field)
        except Exception:
            pass

    for f in filters:
        fld = f.field
        if fld in radar.fields:
            if f.min is not None and fld != "RHOHV" and f.min <= 0.3:
                    gf.exclude_below(fld, float(f.min))
            if f.max is not None:
                gf.exclude_above(fld, float(f.max))
    return gf


# ------------------------------
# Geodesia utilitaria (pseudo-RHI)
# ------------------------------

_GEOD = Geod(ellps="WGS84")

def limit_line_to_range(
    lon0: float, lat0: float, lon1: float, lat1: float, max_len_km: float
) -> Tuple[float, float, float]:
    """
    Limita el punto final (lon1,lat1) a una distancia máxima desde (lon0,lat0).
    Devuelve (lon_final, lat_final, length_km_efectiva).
    """
    az12, az21, dist_m = _GEOD.inv(lon0, lat0, lon1, lat1)
    max_m = max_len_km * 1000.0
    if dist_m <= max_m:
        return lon1, lat1, dist_m / 1000.0
    lon2, lat2, _ = _GEOD.fwd(lon0, lat0, az12, max_m)
    return lon2, lat2, max_len_km
