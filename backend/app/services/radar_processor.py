import logging
import os
import pyart
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

from ..core.config import settings
import time
from ..core.cache import GRID2D_CACHE, GRID2D_LOCK, SESSION_CACHE_INDEX, SESSION_LAST_ACTIVITY, NETCDF_READ_LOCK
from ..core.constants import (
    AFFECTS_INTERP_FIELDS,
    FIELD_RENDER,
    DEFAULT_WEIGHT_FUNC,
    DEFAULT_MAX_NEIGHBORS,
)

from .radar_common import (
    md5_file,
    resolve_field,
    colormap_for,
    grid2d_cache_key,
    normalize_proj_dict,
    safe_range_max_m,
    generate_cog_filename,
    build_output_summary,
    build_local_affine_and_crs,
    effective_smoothing_enabled,
)
from .radar_processing import (
    get_or_build_grid3d_with_operator,
    collapse_grid_to_2d,
    create_cog_from_warped_array,
    calculate_z_limits,
    compute_grid_limits,
    calculate_grid_resolution,
    calculate_grid_points,
    fill_dbzh_if_needed,
    separate_filters,
    apply_smoothing_masked,
    warp_local_to_mercator,
)


def process_radar_to_cog(
    filepath,
    product="PPI",
    field_requested="DBZH",
    cappi_height=4000,
    elevation=0,
    filters=[],
    output_dir="",  # Will use settings.IMAGES_DIR if None
    radar_name="",
    estrategia="",
    volume="",
    colormap_overrides=None,
    session_id=None,
    weight_func=DEFAULT_WEIGHT_FUNC,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
    smoothing_enabled=False,
    smoothing_method="median",
    smoothing_sigma=0.8,
    smoothing_median_size=3,
    smoothing_only_when_nearest=True,
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
    if output_dir == "":
        output_dir = settings.IMAGES_DIR

    # Crear subdirectorio por sesión si se provee session_id
    if session_id:
        output_dir = str(Path(output_dir) / session_id)
        os.makedirs(output_dir, exist_ok=True)
        # Marcar actividad para el cleanup de inactividad.
        # Si la sesión no estaba en el dict (fue evictada tras inactividad),
        # reactivarla en DB — el usuario volvió después de 2 h.
        was_evicted = session_id not in SESSION_LAST_ACTIVITY
        SESSION_LAST_ACTIVITY[session_id] = time.time()
        if was_evicted:
            from .inactivity_cleanup import reactivate_session_in_db
            reactivate_session_in_db(session_id)

    # Crear nombre único pero estable a partir del NetCDF
    file_hash = md5_file(filepath)[:12]
    unique_cog_name = generate_cog_filename(
        field_requested,
        product,
        elevation,
        cappi_height,
        filters,
        file_hash,
        colormap_overrides,
        weight_func=weight_func,
        max_neighbors=max_neighbors,
        smoothing_enabled=smoothing_enabled,
        smoothing_method=smoothing_method,
        smoothing_sigma=smoothing_sigma,
        smoothing_median_size=smoothing_median_size,
        smoothing_only_when_nearest=smoothing_only_when_nearest,
    )
    cog_path = Path(output_dir) / unique_cog_name

    # Si ya existe el COG, necesitamos calcular cmap_key para el summary
    # Usar field_requested en lugar de field_key ya que no hemos leído el radar
    cmap_override_key = (colormap_overrides or {}).get(field_requested, None)
    spec = FIELD_RENDER.get(
        field_requested.upper(), {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"}
    )
    cmap_key_for_summary = (
        cmap_override_key if cmap_override_key else spec.get("cmap", "grc_th")
    )
    layer_metadata: dict = {}

    # Si ya existe el COG, devolvemos directo
    if cog_path.exists():
        version_tag = str(cog_path.stat().st_mtime_ns)
        return build_output_summary(
            unique_cog_name,
            field_requested,
            filepath,
            cog_path,
            cmap_key=cmap_key_for_summary,
            metadata=layer_metadata,
            version_tag=version_tag,
            session_id=session_id,
        )

    # Si no existe, lo procesamos...
    if not Path(filepath).exists():
        raise ValueError(f"Archivo no encontrado: {filepath}")

    # Leer archivo NetCDF con PyART (protegido con lock - NetCDF/HDF5 no es thread-safe)
    with NETCDF_READ_LOCK:
        radar = pyart.io.read(filepath, delay_field_loading=False)

    try:
        field_to_use, field_key = resolve_field(radar, field_requested)
    except KeyError as e:
        raise ValueError(e)

    if elevation > radar.nsweeps - 1:
        raise ValueError(f"El ángulo de elevación {elevation} no existe en el archivo.")

    # defaults de render por variable con posible override
    cmap, vmin, vmax, cmap_key = colormap_for(
        field_key, override_cmap=cmap_override_key
    )

    # Calcular límites Z según producto ANTES de prepare_radar_for_product
    range_max_m = safe_range_max_m(radar, round_to_km=20)
    z_min, z_max, elev_deg = calculate_z_limits(
        range_max_m, elevation, cappi_height, radar.fixed_angle["data"]
    )

    # TOA (Top of Athmosphere) dinámico: z_max ya viene redondeado a múltiplos de 20 km por calculate_z_limits.
    toa = z_max

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
    interp = weight_func

    # Calculamos la cantidad de puntos en cada dimensión
    # XY depende del volumen, pero Z siempre usa resolución fina para transectos suaves
    grid_resolution_xy, grid_resolution_z = calculate_grid_resolution(volume)
    z_grid_limits, y_grid_limits, x_grid_limits = compute_grid_limits(
        range_max_m, toa, volume
    )
    grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits)

    # Calcular puntos de grilla
    z_points, y_points, x_points = calculate_grid_points(
        grid_limits[0],
        grid_limits[1],
        grid_limits[2],
        grid_resolution_xy,
        grid_resolution_z,
    )
    grid_shape = (z_points, y_points, x_points)

    # Separar filtros QC (RHOHV) vs otros
    # QC filters se aplican durante interpolación (afectan grilla 3D y cache key)
    # Visual filters se aplican post-grid como máscaras 2D
    qc_filters, visual_filters = separate_filters(filters, field_to_use)

    # Generar signature de filtros para cache key
    qc_sig = (
        tuple(sorted([(f.field, f.min, f.max) for f in qc_filters]))
        if qc_filters
        else tuple()
    )
    visual_sig = (
        tuple(sorted([(f.field, f.min, f.max) for f in visual_filters]))
        if visual_filters
        else tuple()
    )
    # Combinar ambas signatures (ambos tipos de filtro afectan interpolación)
    filter_sig = qc_sig + visual_sig

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
        qc_sig=filter_sig,  # Cache depende de filtros QC + visuales aplicados durante interpolación
        max_neighbors=max_neighbors,
        session_id=session_id,
    )

    with GRID2D_LOCK:
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
            max_neighbors=max_neighbors,
            qc_filters=qc_filters,
            visual_filters=visual_filters,
            field_to_use=field_to_use,
            session_id=session_id,
        )

        grid_metadata = dict(getattr(grid, "metadata", {}) or {})
        if grid_metadata:
            for key in ("last_gate_range_m", "blind_range_m"):
                if key in grid_metadata:
                    layer_metadata[key] = grid_metadata[key]

        # Verificar que el campo solicitado existe en la grilla
        if field_to_use not in grid.fields:
            available = list(grid.fields.keys())
            raise ValueError(
                f"Campo '{field_to_use}' no encontrado en grilla. Disponibles: {available}"
            )

        collapse_grid_to_2d(
            grid,
            field=field_to_use,
            product=product.lower(),
            elevation_deg=elev_deg,
            target_height_m=cappi_height,
            vmin=vmin,
        )
        arr2d = grid.fields[field_to_use]["data"][0, :, :]
        arr2d = np.ma.array(arr2d.astype(np.float32), mask=np.ma.getmaskarray(arr2d))

        # PyART grid: y[0]=ymin (sur), y[-1]=ymax (norte).
        # GeoTIFF north-up: fila 0 = norte.  Flip para que coincidan.
        arr2d = arr2d[::-1, :]

        transform, crs_wkt = build_local_affine_and_crs(
            grid, radar, arr2d.shape, x_grid_limits, y_grid_limits
        )

        # Guardar en CRS local (se agregará versión warped después del primer warp de PyART)
        pkg_cached = {
            "arr": arr2d,
            "crs": crs_wkt,
            "transform": transform,
            "arr_warped": None,  # Se llenará después del primer warp
            "crs_warped": None,
            "transform_warped": None,
        }
        # Registrar en índice ANTES de guardar en caché.
        # Así un cleanup concurrente que llegue justo después del setdefault
        # encontrará la key en el índice y la buscará en GRID2D_CACHE (donde aún
        # no está → la saltea), que es mejor que el caso inverso donde la key
        # existe en caché pero no en el índice y queda huérfana para siempre.
        with GRID2D_LOCK:
            if session_id:
                SESSION_CACHE_INDEX.setdefault(session_id, set()).add(cache_key)
            GRID2D_CACHE[cache_key] = pkg_cached

    os.makedirs(output_dir, exist_ok=True)

    # WARPING: Reproyectar de AzEq a Web Mercator usando rasterio directamente.
    # Se usa el Affine transform correcto (con offset de medio pixel) calculado
    # al construir pkg_cached, evitando PyART write_grid_geotiff que tiene
    # errores de GeoTransform (sin half-pixel offset + asume grilla cuadrada).
    if pkg_cached.get("arr_warped") is None:
        arr2d = pkg_cached["arr"]
        src_transform = pkg_cached["transform"]
        src_crs = pkg_cached["crs"]

        arr_warped, transform_warped, crs_warped = warp_local_to_mercator(
            arr2d, src_transform, src_crs, vmin
        )

        pkg_cached["arr_warped"] = arr_warped.astype(np.float32)
        pkg_cached["transform_warped"] = transform_warped
        pkg_cached["crs_warped"] = crs_warped
        # Mismo orden que arriba: índice primero, caché después.
        with GRID2D_LOCK:
            if session_id:
                SESSION_CACHE_INDEX.setdefault(session_id, set()).add(cache_key)
            GRID2D_CACHE[cache_key] = pkg_cached
    else:
        # Usar el warp cacheado
        arr_warped = pkg_cached["arr_warped"]
        transform_warped = pkg_cached["transform_warped"]
        crs_warped = pkg_cached["crs_warped"]

    should_apply_smoothing = effective_smoothing_enabled(
        smoothing_enabled, smoothing_sigma, weight_func, smoothing_only_when_nearest
    )

    # Mantener cache base intacto: aplicar suavizado sobre copia para salida final.
    arr_to_render = arr_warped
    if should_apply_smoothing:
        arr_to_render = apply_smoothing_masked(
            arr=arr_warped,
            method=smoothing_method,
            sigma=float(smoothing_sigma),
            median_size=int(smoothing_median_size),
        )

    # Crear COG RGB desde el array warped (suavizado opcional) usando la función optimizada
    create_cog_from_warped_array(
        data_warped=arr_to_render,
        output_path=cog_path,
        transform=transform_warped,
        crs=crs_warped,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    version_tag = str(cog_path.stat().st_mtime_ns)
    return build_output_summary(
        unique_cog_name,
        field_requested,
        filepath,
        cog_path,
        cmap_key=cmap_key_for_summary,
        metadata=layer_metadata,
        version_tag=version_tag,
        session_id=session_id,
    )
