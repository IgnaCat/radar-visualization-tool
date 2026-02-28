import os
import pyart
import pyproj
import numpy as np
from pathlib import Path
import rasterio
import rasterio.transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
from urllib.parse import quote
from affine import Affine

from ..core.config import settings
from ..core.cache import GRID2D_CACHE, SESSION_CACHE_INDEX, NETCDF_READ_LOCK
from ..core.constants import AFFECTS_INTERP_FIELDS, FIELD_RENDER

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
    fill_dbzh_if_needed,
    separate_filters,
    apply_visual_filters,
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
    cmap_key: str = None,
    session_id: str | None = None
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
    relative_url = f"static/tmp/{session_id}/{unique_cog_name}" if session_id else f"static/tmp/{unique_cog_name}"
    
    return {
        "image_url": relative_url,
        "field": field_requested,
        "source_file": filepath,
        "tilejson_url": f"{settings.BASE_URL}/cog/WebMercatorQuad/tilejson.json?url={quote(file_uri, safe=':/')}{style}",
        "colormap": cmap_key,
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

    # Si ya existe el COG, necesitamos calcular cmap_key para el summary
    # Usar field_requested en lugar de field_key ya que no hemos leído el radar
    cmap_override_key = (colormap_overrides or {}).get(field_requested, None)
    spec = FIELD_RENDER.get(field_requested.upper(), {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"})
    cmap_key_for_summary = cmap_override_key if cmap_override_key else spec.get("cmap", "grc_th")

    # Generamos el resumen de salida
    summary = _build_output_summary(unique_cog_name, field_requested, filepath, cog_path, cmap_key=cmap_key_for_summary, session_id=session_id)

    # Si ya existe el COG, devolvemos directo
    if cog_path.exists():
        return summary

    # Si no existe, lo procesamos...
    if not Path(filepath).exists():
        raise ValueError(f"Archivo no encontrado: {filepath}")
    
    # Leer archivo NetCDF con PyART (protegido con lock - NetCDF/HDF5 no es thread-safe)
    with NETCDF_READ_LOCK:
        radar = pyart.io.read(filepath)

    try:
        field_to_use, field_key = resolve_field(radar, field_requested)
    except KeyError as e:
        raise ValueError(e)
    
    if elevation > radar.nsweeps - 1:
        raise ValueError(f"El ángulo de elevación {elevation} no existe en el archivo.")
    
    # defaults de render por variable con posible override
    cmap, vmin, vmax, cmap_key = colormap_for(field_key, override_cmap=cmap_override_key)

    # Calcular límites Z según producto ANTES de prepare_radar_for_product
    range_max_m = safe_range_max_m(radar)
    z_min, z_max, elev_deg = calculate_z_limits(
        range_max_m, elevation, cappi_height, radar.fixed_angle['data']
    )

    # TOA (Top of Atmosphere)
    toa = 12000.0

    # Generamos la imagen PNG para previsualización y referencia
    # png.create_png(
    #     radar, 
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
    
    # Volumen 03 (bird bath) necesita grid XY más grande para COLMAX/CAPPI
    # El scan vertical con 360 azimuts crea un patrón circular amplio
    if volume == '03' and product.upper() in ['COLMAX', 'CAPPI']:
        # Usar grid más grande para capturar el patrón circular completo
        # Ajustado a 50km radio (gates alcanzan ~35km, con ROI ~9km total ~44km)
        grid_extent_m = 50000.0  # 50 km de radio
        y_grid_limits = (-grid_extent_m, grid_extent_m)
        x_grid_limits = (-grid_extent_m, grid_extent_m)
    else:
        y_grid_limits = (-range_max_m, range_max_m)
        x_grid_limits = (-range_max_m, range_max_m)
    
    grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits)

    # Calcular puntos de grilla
    z_points, y_points, x_points = calculate_grid_points(
        grid_limits[0], grid_limits[1], grid_limits[2],
        grid_resolution_xy, grid_resolution_z
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
            radar_to_use=radar,
            file_hash=file_hash,
            radar=radar_name,
            estrategia=estrategia,
            volume=volume,
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

        # PyART grid: y[0]=ymin (sur), y[-1]=ymax (norte).
        # GeoTIFF north-up: fila 0 = norte.  Flip para que coincidan.
        arr2d = arr2d[::-1, :]

        # Obtener grid_origin para normalize_proj_dict
        grid_origin = (
            float(radar.latitude['data'][0]),
            float(radar.longitude['data'][0]),
        )

        x = grid.x['data'].astype(float)
        y = grid.y['data'].astype(float)
        ny, nx = arr2d.shape
        dx = float(np.mean(np.diff(x))) if x.size > 1 else (x_grid_limits[1]-x_grid_limits[0]) / max(nx-1, 1)
        dy = float(np.mean(np.diff(y))) if y.size > 1 else (y_grid_limits[1]-y_grid_limits[0]) / max(ny-1, 1)
        xmin = float(x.min()) if x.size else x_grid_limits[0]
        ymax = float(y.max()) if y.size else y_grid_limits[1]
        # Los valores de linspace(-R, R, N) representan CENTROS de píxeles.
        # El dominio va desde (xmin - dx/2) hasta (xmax + dx/2).
        # Después del flip [::-1], fila 0 = pixel con centro en ymax.
        # Transform debe mapear (col=0, row=0) a la ESQUINA superior izquierda del dominio.
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

    # WARPING: Reproyectar de AzEq a Web Mercator usando rasterio directamente.
    # Se usa el Affine transform correcto (con offset de medio pixel) calculado
    # al construir pkg_cached, evitando PyART write_grid_geotiff que tiene
    # errores de GeoTransform (sin half-pixel offset + asume grilla cuadrada).
    if pkg_cached.get("arr_warped") is None:
        arr2d = pkg_cached["arr"]
        src_transform = pkg_cached["transform"]
        src_crs = pkg_cached["crs"]
        ny, nx = arr2d.shape

        # CRS destino: Web Mercator
        dst_crs = 'EPSG:3857'

        # Bounds del raster fuente (edge-to-edge, calculados desde el Affine transform)
        src_bounds = rasterio.transform.array_bounds(ny, nx, src_transform)

        # Calcular transform y dimensiones en Web Mercator
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, nx, ny,
            left=src_bounds[0], bottom=src_bounds[1],
            right=src_bounds[2], top=src_bounds[3]
        )

        # Preparar arrays: NaN para datos enmascarados
        src_data = np.ma.filled(arr2d, fill_value=np.nan).astype(np.float32)
        dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        # Reproyectar de Azimuthal Equidistant a Web Mercator
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

        # Crear MaskedArray y enmascarar ruido de bordes
        arr_warped = np.ma.masked_invalid(dst_data)
        arr_warped = np.ma.masked_less(arr_warped, vmin)

        transform_warped = dst_transform
        crs_warped = dst_crs

        pkg_cached["arr_warped"] = arr_warped.astype(np.float32)
        pkg_cached["transform_warped"] = transform_warped
        pkg_cached["crs_warped"] = crs_warped
        GRID2D_CACHE[cache_key] = pkg_cached

        # Registrar en índice de sesión si existe session_id
        if session_id:
            if session_id not in SESSION_CACHE_INDEX:
                SESSION_CACHE_INDEX[session_id] = set()
            SESSION_CACHE_INDEX[session_id].add(cache_key)
    else:
        # Usar el warp cacheado
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
