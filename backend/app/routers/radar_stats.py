import pyproj
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as shp_transform
from rasterio.features import geometry_mask
from typing import Optional, List
from ..core.cache import GRID2D_CACHE
from ..schemas import RadarStatsRequest, RadarStatsResponse
from ..utils.helpers import extract_volume_from_filename
from ..services.radar_common import (
    grid2d_cache_key,
    qc_signature,
    filters_affect_interpolation,
    md5_file,
)
from ..services.filter_application import (
    separate_filters,
    apply_visual_filters,
    apply_qc_filters,
)

ALLOWED_GEOMS = {"Polygon", "MultiPolygon"}

router = APIRouter(prefix="/stats", tags=["radar-stats"])

@router.post("/area", response_model=RadarStatsResponse)
async def radar_stats(payload: RadarStatsRequest):
    """
    Calcula estadísticas del radar sobre un polígono (área seleccionada en el mapa),
    utilizando la grilla 2D cacheada (sin tocar disco).
    """
    if getattr(payload, "filepath", None) in (None, "", "undefined"):
        raise HTTPException(
            status_code=400,
            detail="El campo 'filepath' es obligatorio."
        )

    polygon_gj_4326: dict = payload.polygon_geojson
    filepath = payload.filepath
    product = payload.product
    field = payload.field

    if (product.upper() == "CAPPI"): field = "cappi"
    if (product.upper() == "COLMAX" and field.upper() == "DBZH"): field = "composite_reflectivity"
    
    volume = extract_volume_from_filename(filepath)
    cache_key = get_cache_key_for_radar_stats(
        filepath=filepath,
        product=product,
        field=field,
        elevation=payload.elevation,
        cappi_height=payload.height,
        volume=volume,
        session_id=payload.session_id,
    )

    try:
        # Ejecutar en threadpool (bloqueante pero seguro)
        stats_result = await run_in_threadpool(
            stats_from_cache,
            cache_key,
            polygon_gj_4326,
            payload.filters or [],
            field
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    if stats_result.get("noCoverage"):
        raise HTTPException(status_code=404, detail=stats_result.get("reason", "No hay cobertura"))

    return RadarStatsResponse(**stats_result)


def get_cache_key_for_radar_stats(
    filepath: str,
    product: str,
    field: str,
    elevation: Optional[int] = 0,
    cappi_height: Optional[int] = 4000,
    volume: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Genera cache key sin incluir filtros (se aplican dinámicamente).
    """
    product_upper = product.upper()
    field_to_use = field.upper()

    interp = "nearest"  # método de interpolación

    # clave del archivo (hash del contenido)
    file_hash = md5_file(filepath)[:12]

    cache_key = grid2d_cache_key(
        file_hash=file_hash,
        product_upper=product_upper,
        field_to_use=field_to_use,
        elevation=elevation if product_upper == "PPI" else None,
        cappi_height=cappi_height if product_upper == "CAPPI" else None,
        volume=volume,
        interp=interp,
        qc_sig=tuple(),  # Filtros se aplican dinámicamente, no en cache
        session_id=session_id,
    )

    return cache_key


def stats_from_cache(cache_key: str, polygon_gj_4326: dict, filters: List = [], field: str = ""):
    """
    Calcula estadísticas básicas (min, max, mean, etc.) dentro del polígono
    usando la grilla 2D cacheada en GRID2D_CACHE.
    Aplica filtros dinámicamente sobre el array cacheado.
    """
    pkg = GRID2D_CACHE.get(cache_key)
    if pkg is None:
        return {"noCoverage": True, "reason": "No cacheado"}

    # Usar versión warped si está disponible (optimizado para stats desde WGS84)
    arr = pkg["arr_warped"] if pkg.get("arr_warped") is not None else pkg["arr"]
    crs_wkt = pkg["crs_warped"] if pkg.get("crs_warped") is not None else pkg["crs"]
    transform = pkg["transform_warped"] if pkg.get("transform_warped") is not None else pkg["transform"]
    crs = pyproj.CRS.from_wkt(crs_wkt)
    
    # Aplicar filtros dinámicamente (cache tiene array sin filtrar)
    field_to_use = field.upper()
    qc_filters, visual_filters = separate_filters(filters, field_to_use)
    
    arr = apply_visual_filters(arr, visual_filters, field_to_use)
    
    # Aplicar filtros QC si existen en el cache
    # Usar qc_warped si arr_warped está disponible para coincidir dimensiones
    if qc_filters:
        if pkg.get("arr_warped") is not None:
            qc_dict = pkg.get("qc_warped", {}) or {}
        else:
            qc_dict = pkg.get("qc", {}) or {}
        arr = apply_qc_filters(arr, qc_filters, qc_dict)

    # reproyectar polígono: 4326 → crs del grid
    geom4326 = _extract_geometry(polygon_gj_4326)
    g_src = shape(geom4326)
    tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    g_dst = shp_transform(lambda x, y, z=None: tf.transform(x, y), g_src)
    gj_dst = mapping(g_dst)

    # límites del raster (en coordenadas del grid)
    ny, nx = arr.shape
    xmin, ymax = transform * (0, 0)
    xmax, ymin = transform * (nx, ny)

    # si el polígono no interseca el raster → sin cobertura
    if not g_dst.intersects(box(xmin, ymin, xmax, ymax)):
        return {"noCoverage": True, "reason": "Afuera de limites"}

    # máscara del polígono sobre la grilla
    poly_mask = geometry_mask(
        [gj_dst],
        invert=True,
        out_shape=arr.shape,
        transform=transform
    )

    base_mask = np.ma.getmaskarray(arr)
    valid = poly_mask & (~base_mask)

    vals = np.asarray(arr)[valid].astype("float32", copy=False)
    if vals.size == 0:
        return {"noCoverage": True, "reason": "Selección vacia"}

    return {
        "stats": {
            "min": round(float(np.nanmin(vals)), 2),
            "max": round(float(np.nanmax(vals)), 2),
            "mean": round(float(np.nanmean(vals)), 2),
            "median": round(float(np.nanmedian(vals)), 2),
            "std": round(float(np.nanstd(vals)), 2),
            "count": int(vals.size),
            "valid_pct": round(100.0 * vals.size / arr.size, 2)
        },
        "noCoverage": False,
        "reason": None,
    } 


def _extract_geometry(obj: dict) -> dict:
    """Acepta Geometry, Feature o FeatureCollection y devuelve un Geometry."""
    if not isinstance(obj, dict):
        raise ValueError("GeoJSON inválido")

    t = obj.get("type")
    if t == "Feature":
        geom = obj.get("geometry")
        if not geom:
            raise ValueError("Feature sin 'geometry'")
        return _extract_geometry(geom)

    if t == "FeatureCollection":
        feats = obj.get("features") or []
        if not feats:
            raise ValueError("FeatureCollection vacía")
        # Tomamos la primera; si querés, podés unirlas luego con shapely.ops.unary_union
        return _extract_geometry(feats[0])

    # Geometry
    if t not in ALLOWED_GEOMS:
        raise ValueError(f"Geometry no soportada: {t}. Usá Polygon o MultiPolygon.")
    return obj