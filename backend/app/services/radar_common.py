from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any

import numpy as np
import pyart
import hashlib
import json
from pyproj import Geod

from ..utils import colores
from ..core.constants import FIELD_ALIASES, FIELD_RENDER, AFFECTS_INTERP_FIELDS
from ..schemas import RangeFilter


# ------------------------------
# Hashes utilitarios
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

def stable_hash(obj):
    """Hash estable de un objeto JSON-serializable."""
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _roundf(x: float, nd=6) -> float:
    try:
        return float(round(float(x), nd))
    except Exception:
        return float(x)

def _stable(obj: Any):
    """
    Convierte a una estructura JSON-estable (tuplas/listas → listas, floats redondeados).
    """
    if isinstance(obj, dict):
        return {k: _stable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_stable(v) for v in obj]
    if isinstance(obj, float):
        return _roundf(obj, 6)
    return obj

# Usado para generar las cache keys
def _hash_of(payload: Any) -> str:
    s = json.dumps(_stable(payload), separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


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
    filters: Optional[Iterable[RangeFilter]] = [],
    is_rhi: Optional[bool] = False
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

    for f in (filters or []):
        fld = getattr(f, "field", None)
        if not fld:
            continue
        # solo aplicamos por ahora si es RHOHV (QC) si es otro campo se hace post-grid el filtro
        # si es RHI, los aplicamos todos (porque no hay grilla ni cacheo)
        if fld in radar.fields and (fld in AFFECTS_INTERP_FIELDS or (is_rhi and fld == field)):
            fmin = getattr(f, "min", None)
            fmax = getattr(f, "max", None)
            if fmin is not None:
                    if fmin <= 0.3:
                        continue
                    else:
                        gf.exclude_below(fld, float(fmin))
            if fmax is not None:
                gf.exclude_above(fld, float(fmax))
    return gf


# ------------------------------
# Grilla 2D cacheada
# ------------------------------

def filters_affect_interpolation(filters, field_to_use):
    """
    Regla: regridear si hay filtros sobre campos QC (RHOHV/NCP/SNR)
    o sobre un campo distinto al visualizado (porque cambia qué gates aportan).
    """
    ft = field_to_use.upper()
    for f in (filters or []):
        ffield = getattr(f, "field", None)
        if not ffield:
            continue
        up = str(ffield).upper()
        if up in AFFECTS_INTERP_FIELDS:
            return True
        if up != ft:
            # Cualquier filtro sobre OTRA variable (e.g., filtrar por RHOHV mientras muestro DBZH)
            return True
    return False

def qc_signature(filters):
    """
    Solo los filtros que afectan interpolación entran a la firma QC (para cache).
    QC(Quality Control): filtros que sí cambian qué datos se usan para construir la grilla
    """
    sig = []
    for f in (filters or []):
        ffield = getattr(f, "field", None)
        if not ffield:
            continue
        up = str(ffield).upper()
        if up in AFFECTS_INTERP_FIELDS:
            sig.append((up, getattr(f, "min", None), getattr(f, "max", None)))
        # también contamos filtros sobre otras variables (distintas al campo visualizado)
        # la distinción campo mostrado la hacemos arriba; acá metemos todo “potencialmente QC”
        elif up not in AFFECTS_INTERP_FIELDS:
            # los no-QC no suman aquí; si querés ser más estricto, podés incluirlos
            pass
    return tuple(sig)

def grid2d_cache_key(*, file_hash, product_upper, field_to_use,
                     elevation, cappi_height, volume,
                     interp, qc_sig) -> str:
    payload = {
        "v": 1,  # versión del formato de clave
        "file": file_hash,
        "prod": product_upper,
        "field": str(field_to_use).upper(),
        "elev": float(elevation) if elevation is not None else None,
        "h": int(cappi_height) if cappi_height is not None else None,
        "vol": str(volume) if volume is not None else None,
        "interp": str(interp),
        "qc": list(qc_sig) if isinstance(qc_sig, (list, tuple)) else qc_sig,
    }
    return "g2d_" + _hash_of(payload)

def normalize_proj_dict(grid, grid_origin):
    """
    Convierte el dict de proyección de Py-ART a algo que pyproj entienda.
    """
    proj = dict(getattr(grid, "projection", {}) or {})
    if not proj:
        proj = dict(grid.get_projparams() or {})

    # Fallback a origen del radar si faltan lat/lon
    lat0 = float(proj.get("lat_0", grid_origin[0]))
    lon0 = float(proj.get("lon_0", grid_origin[1]))

    # Py-ART usa "pyart_aeqd" como alias interno; PROJ quiere "aeqd"
    if proj.get("proj") in ("pyart_aeqd", None):
        proj = {
            "proj": "aeqd",
            "lat_0": lat0,
            "lon_0": lon0,
            "datum": "WGS84",
            "units": "m",
            # "no_defs": True,   # opcional
        }
    else:
        # Asegurar unidades/datum razonables
        proj.setdefault("datum", "WGS84")
        proj.setdefault("units", "m")

    # A veces viene "type":"crs" que a ciertos builds les molesta
    proj.pop("type", None)
    return proj



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
