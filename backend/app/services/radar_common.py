from __future__ import annotations
from typing import Iterable, Optional, Tuple
from dataclasses import dataclass, asdict

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

def nbytes(arr):
    if isinstance(arr, np.ma.MaskedArray):
        base = arr.data.nbytes
        mask = 0 if arr.mask is np.ma.nomask else np.asarray(arr.mask, dtype=np.bool_).nbytes
        return base + mask
    return arr.nbytes

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

def grid2d_cache_key(*, file_hash, product_upper, field_to_use, elevation, cappi_height,
                      grid_shape, grid_limits, interp, qc_sig):
    return (
        file_hash, product_upper, field_to_use,
        float(elevation) if elevation is not None else None,
        int(cappi_height) if cappi_height is not None else None,
        tuple(grid_shape),
        tuple((tuple(x) for x in grid_limits)),
        str(interp),
        qc_sig  # firma de QC: si cambia, es otra entrada
    )


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
