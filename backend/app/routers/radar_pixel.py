from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import numpy as np
import pyproj
from pyproj import Transformer
from affine import Affine
from rasterio.transform import rowcol, xy

from ..core.cache import GRID2D_CACHE
from ..schemas import RadarPixelRequest, RadarPixelResponse
from ..utils.helpers import extract_volume_from_filename
from ..services.radar_common import (
    grid2d_cache_key,
    qc_signature,
    filters_affect_interpolation,
    md5_file,
)

router = APIRouter(prefix="/stats", tags=["radar-pixel"])

@router.post("/pixel", response_model=RadarPixelResponse)
async def probe_pixel(p: RadarPixelRequest):
    try:
        return await run_in_threadpool(_probe_pixel_impl, p)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    

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


def _probe_pixel_impl(p: RadarPixelRequest) -> RadarPixelResponse:
    
    if getattr(p, "filepath", None) in (None, "", "undefined"):
        raise HTTPException(
            status_code=400,
            detail="El campo 'filepath' es obligatorio."
        )

    filepath = p.filepath
    product = p.product
    field = p.field

    if (product.upper() == "CAPPI"): field = "cappi"
    if (product.upper() == "COLMAX" and field.upper() == "DBZH"): field = "composite_reflectivity"
    
    volume = extract_volume_from_filename(filepath)
    cache_key = get_cache_key_for_radar_stats(
        filepath=filepath,
        product=product,
        field=field,
        elevation=p.elevation,
        cappi_height=p.height,
        volume=volume,
        filters=p.filters,
    )

    pkg = GRID2D_CACHE.get(cache_key)
    if pkg is None:
        raise HTTPException(status_code=404, detail="No cacheado")
    
    if not (-90 <= float(p.lat) <= 90 and -180 <= float(p.lon) <= 180):
        raise HTTPException(status_code=400, detail="Coordenadas no WGS84 (use lat∈[-90,90], lon∈[-180,180])")

    arr = pkg["arr"]                 # np.ma.MaskedArray (ny, nx)
    crs = pyproj.CRS.from_wkt(pkg["crs"])
    transform: Affine = pkg["transform"]

    # 4326 -> CRS del grid (siempre_xy=True porque pasamos (lon,lat))
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xg, yg = tf.transform(p.lon, p.lat)

    # coords -> (col,row)
    row, col = rowcol(transform, xg, yg, op=round)

    # transformar a WGS84 (lon/lat)
    xc, yc = xy(transform, row, col, offset="center")
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lonc, latc = to_wgs84.transform(xc, yc)

    ny, nx = arr.shape
    if row < 0 or row >= ny or col < 0 or col >= nx:
        return RadarPixelResponse(value=None, masked=True, row=row, col=col, message="Fuera de limites")

    m = np.ma.getmaskarray(arr)
    if m[row, col]:
        return RadarPixelResponse(value=None, masked=True, row=row, col=col, message="masked")

    val = float(arr[row, col])
    return RadarPixelResponse(value=round(val, 2), masked=False, row=row, col=col, lat=latc, lon=lonc)
