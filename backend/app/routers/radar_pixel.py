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
async def pixel_stat(p: RadarPixelRequest):
    try:
        return await run_in_threadpool(_pixel_stat_impl, p)
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


def _pixel_stat_impl(p: RadarPixelRequest) -> RadarPixelResponse:
    
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

    # coords continuas (fraccionarias) en pixel space
    col_f, row_f = ~transform * (xg, yg)  # inverso de transform: (xg, yg) -> (col, row) float
    
    ny, nx = arr.shape
    
    # Verificar límites básicos
    if row_f < 0 or row_f >= ny - 1 or col_f < 0 or col_f >= nx - 1:
        # Fuera de límites o en borde (no hay 4 vecinos para bilinear)
        row_int = int(round(row_f))
        col_int = int(round(col_f))
        if row_int < 0 or row_int >= ny or col_int < 0 or col_int >= nx:
            return RadarPixelResponse(value=None, masked=True, row=row_int, col=col_int, message="Fuera de limites")
        m = np.ma.getmaskarray(arr)
        if m[row_int, col_int]:
            return RadarPixelResponse(value=None, masked=True, row=row_int, col=col_int, message="masked")
        val = float(arr[row_int, col_int])
        # Coordenada del centro del pixel para respuesta
        xc, yc = xy(transform, row_int, col_int, offset="center")
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lonc, latc = to_wgs84.transform(xc, yc)
        return RadarPixelResponse(value=round(val, 2), masked=False, row=row_int, col=col_int, lat=latc, lon=lonc)
    
    # Interpolación bilinear: encontrar los 4 píxeles vecinos
    r0 = int(np.floor(row_f))
    c0 = int(np.floor(col_f))
    r1 = r0 + 1
    c1 = c0 + 1
    
    # Pesos
    dr = row_f - r0
    dc = col_f - c0
    
    m = np.ma.getmaskarray(arr)
    
    # Extraer valores y máscaras de los 4 vecinos
    v00 = arr[r0, c0] if not m[r0, c0] else np.nan
    v01 = arr[r0, c1] if not m[r0, c1] else np.nan
    v10 = arr[r1, c0] if not m[r1, c0] else np.nan
    v11 = arr[r1, c1] if not m[r1, c1] else np.nan
    
    # Si todos masked -> retornar masked
    if np.isnan([v00, v01, v10, v11]).all():
        row_int = int(round(row_f))
        col_int = int(round(col_f))
        return RadarPixelResponse(value=None, masked=True, row=row_int, col=col_int, message="masked (todos vecinos)")
    
    # Interpolación bilinear (ignora NaN promediando los válidos con sus pesos)
    # Normalizar pesos solo con celdas válidas
    w00 = (1 - dr) * (1 - dc)
    w01 = (1 - dr) * dc
    w10 = dr * (1 - dc)
    w11 = dr * dc
    
    total_weight = 0.0
    val_interp = 0.0
    
    if not np.isnan(v00):
        val_interp += w00 * v00
        total_weight += w00
    if not np.isnan(v01):
        val_interp += w01 * v01
        total_weight += w01
    if not np.isnan(v10):
        val_interp += w10 * v10
        total_weight += w10
    if not np.isnan(v11):
        val_interp += w11 * v11
        total_weight += w11
    
    if total_weight > 0:
        val_interp /= total_weight
    else:
        # Todos NaN (ya chequeado arriba, pero por si acaso)
        row_int = int(round(row_f))
        col_int = int(round(col_f))
        return RadarPixelResponse(value=None, masked=True, row=row_int, col=col_int, message="masked")
    
    # Devolver coordenadas del punto interpolado (podemos devolver las del pixel más cercano o las exactas)
    # Usaremos las exactas (lat/lon del usuario)
    row_int = int(round(row_f))
    col_int = int(round(col_f))
    
    return RadarPixelResponse(value=round(val_interp, 2), masked=False, row=row_int, col=col_int, lat=p.lat, lon=p.lon)
