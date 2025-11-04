import pyproj
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from shapely.geometry import shape, box
from rasterio.warp import transform_geom
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
    
    volume = extract_volume_from_filename(filepath)
    cache_key = get_cache_key_for_radar_stats(
        filepath=filepath,
        product=payload.product,
        field=payload.field,
        elevation=payload.elevation,
        cappi_height=payload.height,
        volume=volume,
        filters=payload.filters,
    )

    try:
        # Ejecutar en threadpool (bloqueante pero seguro)
        stats_result = await run_in_threadpool(
            stats_from_cache,
            cache_key,
            polygon_gj_4326
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
    filters: Optional[list] = [],
) -> str:

    product_upper = product.upper()
    field_to_use = field.upper()

    interp = "nearest"  # método de interpolación (podría ser otro)
    qc_sig = qc_signature(filters)
    needs_regrid = filters_affect_interpolation(filters, field_to_use)

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
        qc_sig=qc_sig if needs_regrid else tuple()
    )

    return cache_key


def stats_from_cache(cache_key: str, polygon_gj_4326: dict):
    """
    Calcula estadísticas básicas (min, max, mean, etc.) dentro del polígono
    usando la grilla 2D cacheada en GRID2D_CACHE.
    """
    pkg = GRID2D_CACHE.get(cache_key)
    if pkg is None:
        return {"noCoverage": True, "reason": "No cacheado"}

    arr = pkg["arr"]  # np.ma.MaskedArray (ny, nx)
    crs = pyproj.CRS.from_wkt(pkg["crs"])
    transform = pkg["transform"]

    # reproyectar polígono: 4326 → CRS del grid
    geom4326 = _extract_geometry(polygon_gj_4326)
    gj_dst = transform_geom("EPSG:4326", crs.to_string(), geom4326, precision=6)
    geom = shape(gj_dst)

    # límites del raster (en coordenadas del grid)
    ny, nx = arr.shape
    xmin, ymax = transform * (0, 0)
    xmax, ymin = transform * (nx, ny)

    # si el polígono no interseca el raster → sin cobertura
    if not geom.intersects(box(xmin, ymin, xmax, ymax)):
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
    
    print("Values inside polygon:", vals)

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