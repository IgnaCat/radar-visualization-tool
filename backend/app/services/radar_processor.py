import os
import pyart
import uuid
import pyproj
import numpy as np
from pathlib import Path
import rasterio
from urllib.parse import quote
from affine import Affine

from ..core.config import settings
from ..core.cache import GRID2D_CACHE, SESSION_CACHE_INDEX
from ..core.constants import AFFECTS_INTERP_FIELDS

from .radar_common import (
    md5_file,
    resolve_field,
    colormap_for,
    grid2d_cache_key,
    normalize_proj_dict,
    safe_range_max_m,
)
from .radar_processing import (
    get_or_build_grid3d_with_operator,
    collapse_grid_to_2d,
    create_cog_from_warped_array,
    calculate_z_limits,
    calculate_grid_resolution,
    calculate_grid_points,
    prepare_radar_for_product,
    fill_dbzh_if_needed,
    separate_filters,
    apply_visual_filters,
    apply_qc_filters
)


def _generate_cog_filename(
    field_requested: str,
    product: str,
    elevation: int,
    cappi_height: float,
    filters,
    file_hash: str,
    colormap_overrides: dict = None
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
    
    Returns:
        Nombre del archivo COG (sin path)
    """
    filters_str = "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters]) if filters else "nofilter"
    aux = elevation if product.upper() == "PPI" else (cappi_height if product.upper() == "CAPPI" else "")
    
    # Incluir cmap en el nombre si hay override
    cmap_override_key = (colormap_overrides or {}).get(field_requested, None)
    cmap_suffix = f"_{cmap_override_key}" if cmap_override_key else ""
    
    return f"radar_{field_requested}_{product}_{filters_str}_{aux}_{file_hash}{cmap_suffix}.tif"


def _build_output_summary(
    unique_cog_name: str,
    field_requested: str,
    filepath: str,
    cog_path: Path,
    session_id: str | None = None
) -> dict:
    """
    Construye el diccionario de resumen para la respuesta API.
    
    Args:
        unique_cog_name: Nombre del archivo COG
        field_requested: Campo procesado
        filepath: Path del archivo radar original
        cog_path: Path completo al archivo COG
        session_id: Identificador de sesión para aislar archivos
    
    Returns:
        Dict con image_url, field, source_file, tilejson_url
    """
    file_uri = cog_path.resolve().as_posix()
    style = "&resampling=nearest&warp_resampling=nearest"
    relative_url = f"static/tmp/{session_id}/{unique_cog_name}" if session_id else f"static/tmp/{unique_cog_name}"
    
    return {
        "image_url": relative_url,
        "field": field_requested,
        "source_file": filepath,
        "tilejson_url": f"{settings.BASE_URL}/cog/WebMercatorQuad/tilejson.json?url={quote(file_uri, safe=':/')}{style}",
    }


def process_radar_to_cog(
        filepath, 
        product="PPI", 
        field_requested="DBZH", 
        cappi_height=4000, 
        elevation=0, 
        filters=[], 
        output_dir=None,  # Will use settings.IMAGES_DIR if None
        radar_name=None,
        estrategia=None,
        volume=None,
        colormap_overrides=None,
        session_id=None
    ):
    """
    Procesa un archivo NetCDF de radar y genera una COG (Cloud Optimized GeoTIFF).
    Devuelve un resumen de los datos procesados.
    Si ya existe un COG generado para este archivo, devuelve directamente la info.
    
    Args:
        colormap_overrides: dict opcional {field: cmap_key} para personalizar paletas
        session_id: Identificador de sesión para aislar archivos y cache
    """
    # Use settings.IMAGES_DIR if output_dir not specified
    if output_dir is None:
        output_dir = settings.IMAGES_DIR
    
    # Crear subdirectorio por sesión si se provee session_id
    if session_id:
        output_dir = str(Path(output_dir) / session_id)
        os.makedirs(output_dir, exist_ok=True)

    # Crear nombre único pero estable a partir del NetCDF
    file_hash = md5_file(filepath)[:12]
    unique_cog_name = _generate_cog_filename(
        field_requested, product, elevation, cappi_height,
        filters, file_hash, colormap_overrides
    )
    cog_path = Path(output_dir) / unique_cog_name

    # Generamos el resumen de salida
    summary = _build_output_summary(unique_cog_name, field_requested, filepath, cog_path, session_id=session_id)

    # Si ya existe el COG, devolvemos directo
    if cog_path.exists():
        return summary

    # Si no existe, lo procesamos...
    if not Path(filepath).exists():
        raise ValueError(f"Archivo no encontrado: {filepath}")
    
    # Leer archivo NetCDF con PyART
    radar = pyart.io.read(filepath)

    try:
        field_name, field_key = resolve_field(radar, field_requested)
    except KeyError as e:
        raise ValueError(e)
    
    if elevation > radar.nsweeps - 1:
        raise ValueError(f"El ángulo de elevación {elevation} no existe en el archivo.")
    
    # defaults de render por variable con posible override
    cmap_override = (colormap_overrides or {}).get(field_requested, None)
    cmap, vmin, vmax, cmap_key = colormap_for(field_key, override_cmap=cmap_override)

    # Calcular límites Z según producto ANTES de prepare_radar_for_product
    range_max_m = safe_range_max_m(radar)
    z_min, z_max, elev_deg = calculate_z_limits(
        range_max_m, elevation, cappi_height, radar.fixed_angle['data']
    )

    # TOA (Top of Atmosphere)
    toa = 15000.0

    # Preparar radar según producto (PPI/CAPPI/COLMAX)
    field_name = fill_dbzh_if_needed(radar, field_name, product)
    radar_to_use, field_to_use = prepare_radar_for_product(
        radar, product, field_name, elevation, cappi_height
    )

    # Generamos la imagen PNG para previsualización y referencia
    # png.create_png(
    #     radar_to_use, 
    #     product, 
    #     output_dir, 
    #     field_to_use, 
    #     filters=filters, 
    #     elevation=elevation, 
    #     height=cappi_height, 
    #     vmin=vmin, 
    #     vmax=vmax, 
    #     cmap_key=cmap_key
    # )

    # Creamos la grilla
    # Definimos los limites de nuestra grilla en las 3 dimensiones (x,y,z)

    # Método de interpolación para grilla
    interp = 'Barnes2'

    # Calculamos la cantidad de puntos en cada dimensión
    # XY depende del volumen, pero Z siempre usa resolución fina para transectos suaves
    grid_resolution_xy, grid_resolution_z = calculate_grid_resolution(volume)
    z_grid_limits = (0.0, toa)
    y_grid_limits = (-range_max_m, range_max_m)
    x_grid_limits = (-range_max_m, range_max_m)
    grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits)

    # Calcular puntos de grilla
    z_points, y_points, x_points = calculate_grid_points(
        grid_limits[0], grid_limits[1], grid_limits[2],
        grid_resolution_z, grid_resolution_xy
    )
    grid_shape = (z_points, y_points, x_points)

    # Separar filtros QC (RHOHV) vs otros
    # QC filters se aplican durante interpolación (afectan grilla 3D y cache key)
    # Visual filters se aplican post-grid como máscaras 2D
    qc_filters, visual_filters = separate_filters(filters, field_to_use)

    # Generar signature de qc_filters para cache key
    qc_sig = tuple(sorted([
        (f.field, f.min, f.max) for f in qc_filters
    ])) if qc_filters else tuple()

    # Intentamos cachear la 2D colapsada
    product_upper = product.upper()
    cache_key = grid2d_cache_key(
        file_hash=file_hash,
        product_upper=product_upper,
        field_to_use=field_to_use,
        elevation=elevation if product_upper == "PPI" else None,
        cappi_height=cappi_height if product_upper == "CAPPI" else None,
        volume=volume,
        interp=interp,
        qc_sig=qc_sig,  # Cache depende de filtros QC aplicados durante interpolación
        session_id=session_id,
    )

    pkg_cached = GRID2D_CACHE.get(cache_key)

    if pkg_cached is None:
        # Construir o recuperar grilla 3D multi-campo con el operador W
        grid = get_or_build_grid3d_with_operator(
            radar_to_use=radar_to_use,
            file_hash=file_hash,
            radar=radar_name,
            estrategia=estrategia,
            volume=volume,
            range_max_m=range_max_m,
            toa=toa,
            grid_limits=grid_limits,
            grid_shape=grid_shape,
            grid_resolution_xy=grid_resolution_xy,
            grid_resolution_z=grid_resolution_z,
            weight_func=interp,
            qc_filters=qc_filters,
            session_id=session_id
        )

        # Verificar que el campo solicitado existe en la grilla
        if field_to_use not in grid.fields:
            available = list(grid.fields.keys())
            raise ValueError(f"Campo '{field_to_use}' no encontrado en grilla. Disponibles: {available}")

        collapse_grid_to_2d(
            grid,
            field=field_to_use,
            product=product.lower(),
            elevation_deg=elev_deg,
            target_height_m=cappi_height,
            vmin=vmin,
        )
        arr2d = grid.fields[field_to_use]['data'][0, :, :]
        arr2d = np.ma.array(arr2d.astype(np.float32), mask=np.ma.getmaskarray(arr2d))

        # Obtener grid_origin para normalize_proj_dict
        grid_origin = (
            float(radar_to_use.latitude['data'][0]),
            float(radar_to_use.longitude['data'][0]),
        )

        x = grid.x['data'].astype(float)
        y = grid.y['data'].astype(float)
        ny, nx = arr2d.shape
        dx = float(np.mean(np.diff(x))) if x.size > 1 else (x_grid_limits[1]-x_grid_limits[0]) / max(nx-1, 1)
        dy = float(np.mean(np.diff(y))) if y.size > 1 else (y_grid_limits[1]-y_grid_limits[0]) / max(ny-1, 1)
        xmin = float(x.min()) if x.size else x_grid_limits[0]
        ymax = float(y.max()) if y.size else y_grid_limits[1]
        transform = Affine.translation(xmin - dx/2, ymax + dy/2) * Affine.scale(dx, -dy)
        proj_dict_norm = normalize_proj_dict(grid, grid_origin)
        crs_wkt = pyproj.CRS.from_dict(proj_dict_norm).to_wkt()
        
        # Guardar en CRS local (se agregará versión warped después del primer warp de PyART)
        pkg_cached = {
            "arr": arr2d,
            "crs": crs_wkt,
            "transform": transform,
            "arr_warped": None,  # Se llenará después del primer warp
            "crs_warped": None,
            "transform_warped": None,
        }
        GRID2D_CACHE[cache_key] = pkg_cached
        
        # Registrar en índice de sesión si existe session_id
        if session_id:
            if session_id not in SESSION_CACHE_INDEX:
                SESSION_CACHE_INDEX[session_id] = set()
            SESSION_CACHE_INDEX[session_id].add(cache_key)


    os.makedirs(output_dir, exist_ok=True)

    # WARPING: Warpear si es la primera vez (sin filtros visuales, se aplican después)
    if pkg_cached.get("arr_warped") is None:
        # Crear grid PyART temporal con arr2d sin filtros visuales
        arr2d = pkg_cached["arr"]
        ny, nx = arr2d.shape
        grid_temp = pyart.core.Grid(
            time={'data': np.array([0])},
            fields={field_to_use: {'data': arr2d[np.newaxis, :, :], '_FillValue': -9999.0}},
            metadata={'instrument_name': 'RADAR'},
            origin_latitude={'data': radar_to_use.latitude['data']},
            origin_longitude={'data': radar_to_use.longitude['data']},
            origin_altitude={'data': radar_to_use.altitude['data']},
            x={'data': np.linspace(x_grid_limits[0], x_grid_limits[1], nx).astype(np.float32)},
            y={'data': np.linspace(y_grid_limits[0], y_grid_limits[1], ny).astype(np.float32)},
            z={'data': np.array([0.0], dtype=np.float32)}
        )
        
        # Generar GeoTIFF numérico warped con PyART
        temp_numeric_tif = Path(output_dir) / f"numeric_{uuid.uuid4().hex}.tif"
        
        pyart.io.write_grid_geotiff(
            grid=grid_temp,
            filename=str(temp_numeric_tif),
            field=field_to_use,
            level=0,
            rgb=False,
            warp_to_mercator=True
        )

        # Leer el GeoTIFF numérico warped
        with rasterio.open(temp_numeric_tif) as src_numeric:
            arr_warped = src_numeric.read(1, masked=True)
            transform_warped = src_numeric.transform
            crs_warped = src_numeric.crs.to_wkt()
            
            # Enmascarar valores extremadamente bajos (ruido de interpolación en bordes)
            if not np.ma.is_masked(arr_warped):
                arr_warped = np.ma.array(arr_warped, mask=np.zeros_like(arr_warped, dtype=bool))
            arr_warped = np.ma.masked_less(arr_warped, vmin)
            
            pkg_cached["arr_warped"] = arr_warped.astype(np.float32)
            pkg_cached["transform_warped"] = transform_warped
            pkg_cached["crs_warped"] = crs_warped
            GRID2D_CACHE[cache_key] = pkg_cached
                
            # Registrar en índice de sesión si existe session_id
            if session_id:
                if session_id not in SESSION_CACHE_INDEX:
                    SESSION_CACHE_INDEX[session_id] = set()
                SESSION_CACHE_INDEX[session_id].add(cache_key)
        
        # Limpiar GeoTIFF numérico temporal
        try:
            temp_numeric_tif.unlink()
        except OSError:
            pass
    else:
        # Sino usar el warp cachea
        arr_warped = pkg_cached["arr_warped"]
        transform_warped = pkg_cached["transform_warped"]
        crs_warped = pkg_cached["crs_warped"]
    
    # Aplicar filtros visuales post-cache sobre el array warped
    # Los filtros QC ya fueron aplicados durante interpolación y están en el cache
    arr_warped_filtered = apply_visual_filters(arr_warped, visual_filters, field_to_use)

    # Crear COG RGB desde el array warped cacheado usando la función optimizada
    create_cog_from_warped_array(
        data_warped=arr_warped_filtered,
        output_path=cog_path,
        transform=transform_warped,
        crs=crs_warped,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    return summary
