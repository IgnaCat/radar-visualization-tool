from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any

import numpy as np
import pyart
import hashlib
import json
from pyproj import Geod

from ..utils import colores
from ..core.constants import FIELD_ALIASES, FIELD_RENDER, AFFECTS_INTERP_FIELDS
from ..models import RangeFilter


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

def colormap_for(field_key: str, override_cmap: Optional[str] = None):
    """
    Devuelve defaults (cmap, vmin, vmax, cmap_key) según FIELD_RENDER.
    Si override_cmap se provee, lo usa en lugar del default.
    """
    import matplotlib.pyplot as plt
    
    spec = FIELD_RENDER.get(field_key.upper(), {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"})
    vmin, vmax = spec["vmin"], spec["vmax"]
    cmap_key = override_cmap if override_cmap else spec["cmap"]
    
    # Determinar si es un cmap personalizado (grc_*) o uno de pyart/matplotlib
    if cmap_key.startswith("grc_"):
        # Colormap personalizado del módulo colores
        cmap = getattr(colores, f"get_cmap_{cmap_key}")()
    elif cmap_key.startswith("pyart_"):
        # Colormap de pyart (obtener objeto colormap real)
        cmap_name = cmap_key.replace("pyart_", "")
        try:
            cmap = pyart.graph.cm.get_colormap(cmap_name)
        except (AttributeError, KeyError):
            # Fallback: intentar como colormap estándar de matplotlib
            cmap = plt.get_cmap(cmap_name)
    else:
        # Colormap estándar de matplotlib/pyart (NWSVel, Theodore16, etc)
        # Intentar primero de PyART, luego matplotlib
        try:
            cmap = pyart.graph.cm.get_colormap(cmap_key)
        except (AttributeError, KeyError):
            cmap = plt.get_cmap(cmap_key)
    
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
                    if fmin <= 0.3 and fld in AFFECTS_INTERP_FIELDS:
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
            if ft in AFFECTS_INTERP_FIELDS:
                return False
            else:
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
                     interp, qc_sig, session_id=None) -> str:
    """
    Genera cache key para grilla 2D con soporte para aislamiento por sesión.
    
    Args:
        session_id: Identificador único de sesión (None = compartido globalmente)
    """
    payload = {
        "v": 2,  # versión del formato de clave (incrementada para session support)
        "file": file_hash,
        "prod": product_upper,
        "field": str(field_to_use).upper(),
        "elev": float(elevation) if elevation is not None else None,
        "h": int(cappi_height) if cappi_height is not None else None,
        "vol": str(volume) if volume is not None else None,
        "interp": str(interp),
        "qc": list(qc_sig) if isinstance(qc_sig, (list, tuple)) else qc_sig,
        "sess": str(session_id) if session_id else None,  # Aislar por sesión
    }
    return "g2d_" + _hash_of(payload)


def grid3d_cache_key(*, file_hash: str,
                     volume: str | None, qc_sig, grid_res_xy: float,
                     grid_res_z: float, z_top_m: float, session_id=None) -> str:
    """
    Genera cache key para grilla 3D multi-campo con soporte para aislamiento por sesión.
    CAMBIO: Ya no depende de field_to_use - una grilla sirve para todos los campos.
    
    Args:
        session_id: Identificador único de sesión (None = compartido globalmente)
    """
    payload = {
        "v": 3,  # versión incrementada para multi-campo (incompatible con v2)
        "file": file_hash,
        # "field": ELIMINADO - grideamos todos los campos
        "vol": str(volume) if volume is not None else None,
        "qc": list(qc_sig) if isinstance(qc_sig, (list, tuple)) else qc_sig,
        "gxy": float(grid_res_xy),
        "gz": float(grid_res_z),
        "ztop": float(z_top_m),
        "sess": str(session_id) if session_id else None,  # Aislar por sesión
    }
    return "g3d_" + _hash_of(payload)

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

def collapse_field_3d_to_2d(data3d, product, *,
                            x_coords=None, y_coords=None, z_levels=None,
                            elevation_deg=None, target_height_m=None):
    """Versión no destructiva para colapsar un solo campo 3D a 2D.
    No modifica el objeto Grid, sólo recibe los arrays necesarios.
    """
    # PyART Grid puede tener dimensión temporal (time, z, y, x)
    # Eliminar dimensión temporal si existe
    if data3d.ndim == 4:
        data3d = data3d[0, :, :, :]  # Tomar primer (y único) timestep
    
    if data3d.ndim == 2:
        arr2d = data3d
    else:
        if product == "ppi":
            assert elevation_deg is not None and x_coords is not None and y_coords is not None and z_levels is not None
            X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
            r = np.sqrt(X**2 + Y**2)
            Re = 8.49e6
            z_target = r * np.sin(np.deg2rad(elevation_deg)) + (r**2) / (2.0 * Re)
            iz = np.abs(z_target[..., None] - z_levels[None, None, :]).argmin(axis=2)
            yy = np.arange(len(y_coords))[:, None]
            xx = np.arange(len(x_coords))[None, :]
            arr2d = data3d[iz, yy, xx]
        elif product == "cappi":
            assert target_height_m is not None and z_levels is not None
            iz = np.abs(z_levels - float(target_height_m)).argmin()
            arr2d = data3d[iz, :, :]
        elif product == "colmax":
            arr2d = data3d.max(axis=0)
        else:
            raise ValueError("Producto inválido")
    return np.ma.array(arr2d.astype(np.float32), mask=np.ma.getmaskarray(arr2d))



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
