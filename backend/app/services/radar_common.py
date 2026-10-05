from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any

import numpy as np
import pyart
import pyproj
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from affine import Affine
from pyproj import Geod

from ..utils import colores
from ..core.config import settings
from ..core.constants import (
    FIELD_ALIASES,
    FIELD_RENDER,
    AFFECTS_INTERP_FIELDS,
    ROI_PARAMS_BY_VOLUME,
    ROI_PARAMS_VOL01,
    DEFAULT_WEIGHT_FUNC,
    DEFAULT_MAX_NEIGHBORS,
    GRID_RANGE_ROUND_TO_KM,
)
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

def safe_range_max_m(radar: pyart.core.Radar, default: float = 240e3, round_to_km: int = GRID_RANGE_ROUND_TO_KM) -> float:
    """
    Devuelve el alcance máximo (último gate) en metros, con fallback.
    Redondea hacia arriba al múltiplo de round_to_km km para alinear grids.
    
    Args:
        radar: Objeto radar de PyART
        default: Valor por defecto si no se puede determinar
        round_to_km: Redondear hacia arriba a múltiplos de este valor en km (default: 20km)
                     Ejemplos: 116580m → 120000m, 236460m → 240000m
    
    Returns:
        Rango máximo redondeado en metros
    """
    import math
    
    r = radar.range["data"]
    arr = np.asarray(getattr(r, "filled", lambda v: r)(np.nan), dtype=float)
    if arr.size == 0:
        range_m = float(default)
    else:
        last = float(arr[-1])
        if np.isfinite(last):
            range_m = last
        else:
            # fallback al máximo finito
            finite = arr[np.isfinite(arr)]
            range_m = float(finite.max()) if finite.size else float(default)
    
    # Redondear hacia arriba al múltiplo más cercano
    if round_to_km > 0:
        step_m = round_to_km * 1000.0
        range_m = math.ceil(range_m / step_m) * step_m
    
    return float(range_m)


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
                     interp, qc_sig, max_neighbors=DEFAULT_MAX_NEIGHBORS, session_id=None) -> str:
    """
    Genera cache key para grilla 2D con soporte para aislamiento por sesión.
    
    Args:
        max_neighbors: Máximo número de vecinos usados en interpolación (afecta W operator)
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
        "maxn": int(max_neighbors) if max_neighbors is not None else None,
        "qc": list(qc_sig) if isinstance(qc_sig, (list, tuple)) else qc_sig,
        "sess": str(session_id) if session_id else None,  # Aislar por sesión
    }
    return "g2d_" + _hash_of(payload)

def w_operator_cache_key(
    *,
    radar: str,
    estrategia: str,
    volumen: str,
    grid_shape: tuple,
    grid_limits: tuple,
    weight_func: str = DEFAULT_WEIGHT_FUNC,
    max_neighbors: int | None = DEFAULT_MAX_NEIGHBORS,
    blind_range_m: float | None = None,
    last_gate_range_m: float | None = None,
) -> str:
    """
    Genera cache key para operador W basado en:
    - Identificación del radar (radar_estrategia_volumen)
    - Geometría de grilla (shape, limits)
    - Parámetros ROI específicos del volumen (ROI_PARAMS_BY_VOLUME)
    - Función de ponderación y max_neighbors
    
    NOTA: Cache W es COMPARTIDO entre sesiones - el operador depende solo
    de la geometría del radar, no de datos específicos del archivo.
    
    Args:
        radar: Código del radar (ej: RMA1)
        estrategia: Estrategia de escaneo (ej: 0315)
        volumen: Número de volumen (ej: 01)
        grid_shape: (nz, ny, nx)
        grid_limits: ((z_min, z_max), (y_min, y_max), (x_min, x_max))
        weight_func: Función de ponderación ('Barnes', 'Barnes2', 'Cressman', 'nearest')
        max_neighbors: Máximo número de vecinos (None = todos)
        blind_range_m: Radio ciego cercano al radar incluido en W
        last_gate_range_m: Alcance máximo del último gate incluido en W
    """
    # Seleccionar parámetros ROI específicos del volumen (constantes, no adaptativos)
    roi_params = ROI_PARAMS_BY_VOLUME.get(volumen, ROI_PARAMS_VOL01)
    
    payload = {
        "v": 7,  # versión 7: máscaras radial interna y externa integradas en W
        "radar": str(radar),
        "strat": str(estrategia),
        "vol": str(volumen),
        "shape": list(grid_shape),
        "limits": [
            [float(grid_limits[0][0]), float(grid_limits[0][1])],
            [float(grid_limits[1][0]), float(grid_limits[1][1])],
            [float(grid_limits[2][0]), float(grid_limits[2][1])],
        ],
        "roi": list(roi_params),  # (h_factor, nb, bsp, min_radius) - constante por volumen
        "wfunc": str(weight_func),
        "maxn": int(max_neighbors) if max_neighbors is not None else None,
    }
    
    return f"W_{radar}_{estrategia}_{volumen}_{_hash_of(payload)}"


# ------------------------------
# Nombre de archivo COG / resumen de salida
# ------------------------------

def effective_smoothing_enabled(
    smoothing_enabled: bool,
    smoothing_sigma: float,
    weight_func: str,
    smoothing_only_when_nearest: bool = True,
) -> bool:
    """
    Decide si el suavizado configurado realmente se va a aplicar.

    Se usa en dos lugares que deben coincidir siempre: generate_cog_filename
    (para el sufijo del nombre del COG) y process_radar_to_cog (para decidir
    si ejecuta apply_smoothing_masked). Antes de esta extracción, la misma
    fórmula estaba escrita por separado en los dos lugares — un riesgo real
    de que se desincronicen si alguien cambia una y no la otra (ej: el
    nombre del archivo diría "_nosmooth" pero el píxel vendría suavizado).

    Args:
        smoothing_enabled: Si el usuario pidió suavizado
        smoothing_sigma: Intensidad configurada (debe ser > 0 para aplicar)
        weight_func: Función de ponderación usada en la interpolación
        smoothing_only_when_nearest: Si True, el suavizado solo se aplica
            cuando weight_func es 'nearest' — ver generate_cog_filename

    Returns:
        True si el suavizado debe aplicarse de verdad.
    """
    return (
        bool(smoothing_enabled)
        and float(smoothing_sigma) > 0.0
        and (weight_func == "nearest" or not smoothing_only_when_nearest)
    )


def generate_cog_filename(
    field_requested: str,
    product: str,
    elevation: int,
    cappi_height: float,
    filters,
    file_hash: str,
    colormap_overrides: dict | None,
    weight_func: str = DEFAULT_WEIGHT_FUNC,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
    smoothing_enabled: bool = False,
    smoothing_method: str = "median",
    smoothing_sigma: float = 0.8,
    smoothing_median_size: int = 3,
    smoothing_only_when_nearest: bool = True,
) -> str:
    """
    Genera nombre único pero estable para el archivo COG.

    Args:
        field_requested: Campo solicitado (ej: 'DBZH')
        product: Tipo de producto ('PPI', 'CAPPI', 'COLMAX')
        elevation: Índice de elevación (para PPI)
        cappi_height: Altura CAPPI en metros
        filters: Lista de filtros aplicados
        file_hash: Hash del archivo radar
        colormap_overrides: Dict opcional con overrides de colormap
        weight_func: Función de ponderación usada en interpolación
        max_neighbors: Máximo número de vecinos
        smoothing_enabled: Si se aplica suavizado opcional
        smoothing_method: Método de suavizado ('gaussian' o 'median')
        smoothing_sigma: Intensidad de suavizado gaussiano
        smoothing_median_size: Tamaño de ventana para mediana
        smoothing_only_when_nearest: Si el suavizado aplica solo para nearest

    Returns:
        Nombre del archivo COG (sin path)
    """
    filters_str = (
        "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters])
        if filters
        else "nofilter"
    )
    aux = (
        elevation
        if product.upper() == "PPI"
        else (cappi_height if product.upper() == "CAPPI" else "")
    )

    # Incluir cmap en el nombre si hay override
    cmap_override_key = (colormap_overrides or {}).get(field_requested, None)
    cmap_suffix = f"_{cmap_override_key}" if cmap_override_key else ""

    interp_suffix = (
        f"_{weight_func}_n{max_neighbors if max_neighbors is not None else 'all'}"
    )

    effective_smoothing = effective_smoothing_enabled(
        smoothing_enabled, smoothing_sigma, weight_func, smoothing_only_when_nearest
    )
    if effective_smoothing:
        method = (smoothing_method or "median").lower()
        mode_tag = "nearestonly" if smoothing_only_when_nearest else "allinterp"
        if method == "median":
            smooth_suffix = f"_smooth_m{int(smoothing_median_size)}_{mode_tag}"
        else:
            sigma_tag = f"{float(smoothing_sigma):.2f}".rstrip("0").rstrip(".")
            sigma_tag = sigma_tag.replace(".", "p")
            smooth_suffix = f"_smooth_g{sigma_tag}_{mode_tag}"
    else:
        smooth_suffix = "_nosmooth"

    return f"radar_{field_requested}_{product}_{filters_str}_{aux}_{file_hash}{cmap_suffix}{interp_suffix}{smooth_suffix}.tif"


def build_output_summary(
    unique_cog_name: str,
    field_requested: str,
    filepath: str,
    cog_path: Path,
    cmap_key: str,
    metadata: dict | None = None,
    version_tag: str | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Construye el diccionario de resumen para la respuesta API.

    Args:
        unique_cog_name: Nombre del archivo COG
        field_requested: Campo procesado
        filepath: Path del archivo radar original
        cog_path: Path completo al archivo COG
        cmap_key: Colormap usado (ej: 'grc_th')
        session_id: Identificador de sesión para aislar archivos

    Returns:
        Dict con image_url, field, source_file, tilejson_url, colormap
    """
    file_uri = cog_path.resolve().as_posix()
    style = "&resampling=nearest&warp_resampling=nearest"
    relative_url = (
        f"static/tmp/{session_id}/{unique_cog_name}"
        if session_id
        else f"static/tmp/{unique_cog_name}"
    )

    tilejson_url = f"{settings.BASE_URL}/cog/WebMercatorQuad/tilejson.json?url={quote(file_uri, safe=':/')}{style}"
    if version_tag:
        tilejson_url += f"&v={quote(str(version_tag), safe='')}"

    return {
        "image_url": relative_url,
        "field": field_requested,
        "source_file": Path(filepath).name,
        "tilejson_url": tilejson_url,
        "colormap": cmap_key,
        "metadata": metadata or None,
    }


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


def build_local_affine_and_crs(grid, radar, arr2d_shape, x_grid_limits, y_grid_limits):
    """
    Construye el Affine transform (con offset de medio píxel) y el CRS WKT
    en la proyección local (Azimuthal Equidistant centrada en el radar) para
    una grilla 2D ya colapsada.

    Movida desde process_radar_to_cog (radar_processor.py): compone
    directamente con normalize_proj_dict, que vive en este mismo módulo.

    Args:
        grid: objeto Grid de PyART (usa grid.x/grid.y y su proyección)
        radar: objeto Radar de PyART, para el origen (lat, lon) como fallback
        arr2d_shape: (ny, nx) del array 2D colapsado — solo se usa para el
            caso borde de una grilla con un único punto por eje
        x_grid_limits, y_grid_limits: (min, max) en metros, usados solo
            como fallback cuando grid.x/grid.y tienen un único punto

    Returns:
        (transform, crs_wkt)
    """
    grid_origin = (
        float(radar.latitude["data"][0]),
        float(radar.longitude["data"][0]),
    )

    x = grid.x["data"].astype(float)
    y = grid.y["data"].astype(float)
    ny, nx = arr2d_shape
    dx = (
        float(np.mean(np.diff(x)))
        if x.size > 1
        else (x_grid_limits[1] - x_grid_limits[0]) / max(nx - 1, 1)
    )
    dy = (
        float(np.mean(np.diff(y)))
        if y.size > 1
        else (y_grid_limits[1] - y_grid_limits[0]) / max(ny - 1, 1)
    )
    xmin = float(x.min()) if x.size else x_grid_limits[0]
    ymax = float(y.max()) if y.size else y_grid_limits[1]
    # Los valores de linspace(-R, R, N) representan CENTROS de píxeles.
    # El dominio va desde (xmin - dx/2) hasta (xmax + dx/2).
    # Después del flip [::-1], fila 0 = pixel con centro en ymax.
    # Transform debe mapear (col=0, row=0) a la ESQUINA superior izquierda del dominio.
    transform = Affine.translation(xmin - dx / 2, ymax + dy / 2) * Affine.scale(dx, -dy)
    proj_dict_norm = normalize_proj_dict(grid, grid_origin)
    crs_wkt = pyproj.CRS.from_dict(proj_dict_norm).to_wkt()
    return transform, crs_wkt


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
