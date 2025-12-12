import math
import os
import pyart
import uuid
import pyproj
import shutil
import numpy as np
from ..utils import cappi as cappi_utils
from ..utils import png
from pathlib import Path
import rasterio
from rasterio.shutil import copy
from rasterio.enums import ColorInterp
from rasterio.warp import Resampling
from urllib.parse import quote
from ..core.config import settings
from ..core.cache import GRID2D_CACHE, GRID3D_CACHE
from ..core.constants import AFFECTS_INTERP_FIELDS
from affine import Affine

from .radar_common import (
    md5_file,
    resolve_field,
    build_gatefilter,
    colormap_for,
    filters_affect_interpolation,
    qc_signature,
    grid2d_cache_key,
    grid3d_cache_key,
    normalize_proj_dict,
    safe_range_max_m,
    collapse_field_3d_to_2d
)
from .grid_geometry import (
    calculate_z_limits,
    calculate_grid_resolution,
    calculate_grid_points
)
from .product_preparation import (
    prepare_radar_for_product,
    fill_dbzh_if_needed
)
from .filter_application import (
    separate_filters,
    apply_visual_filters,
    apply_qc_filters
)


def _get_or_build_grid3d(
    radar_to_use: pyart.core.Radar,
    field_to_use: str,
    file_hash: str,
    volume: str | None,
    qc_filters,
    z_grid_limits: tuple,
    y_grid_limits: tuple,
    x_grid_limits: tuple,
    grid_resolution_xy: float,
    grid_resolution_z: float,
    session_id: str | None = None,
) -> pyart.core.Grid:
    """
    Función interna para obtener o construir una grilla 3D cacheada.
    
    Args:
        session_id: Identificador de sesión para aislar cache
    """
    # Generar cache key con session_id
    qc_sig = qc_signature(qc_filters)
    cache_key = grid3d_cache_key(
        file_hash=file_hash,
        field_to_use=field_to_use,
        volume=volume,
        qc_sig=qc_sig,
        grid_res_xy=grid_resolution_xy,
        grid_res_z=grid_resolution_z,
        z_top_m=z_grid_limits[1],
        session_id=session_id,  # Incluir session_id
    )
    
    # Verificar cache 3D
    pkg_cached = GRID3D_CACHE.get(cache_key)
    
    if pkg_cached is not None:
        # Reconstruir Grid desde cache con todos los metadatos del campo
        cached_field_name = pkg_cached.get("field_name", field_to_use)
        field_metadata = pkg_cached.get("field_metadata", {})
        
        # Restaurar el campo completo con todos sus metadatos
        field_dict = field_metadata.copy()
        field_dict['data'] = pkg_cached["arr3d"]
        
        # Asegurar metadatos mínimos si no existen en cache
        if 'units' not in field_dict:
            field_dict['units'] = 'unknown'
        if '_FillValue' not in field_dict:
            field_dict['_FillValue'] = -9999.0
        if 'long_name' not in field_dict:
            field_dict['long_name'] = cached_field_name
        
        # Crear Grid con metadatos completos a partir del array cacheado
        grid = pyart.core.Grid(
            time={
                'data': np.array([0]),
                'units': 'seconds since 2000-01-01T00:00:00Z',
                'calendar': 'gregorian',
                'standard_name': 'time'
            },
            fields={cached_field_name: field_dict},
            metadata={'instrument_name': 'RADAR'},
            origin_latitude={'data': radar_to_use.latitude['data']},
            origin_longitude={'data': radar_to_use.longitude['data']},
            origin_altitude={'data': radar_to_use.altitude['data']},
            x={'data': pkg_cached["x"]},
            y={'data': pkg_cached["y"]},
            z={'data': pkg_cached["z"]},
        )
        grid.projection = pkg_cached["projection"]
        return grid
    
    # Construir grilla 3D desde radar
    gf = build_gatefilter(radar_to_use, field_to_use, qc_filters, is_rhi=False)
    
    grid_origin = (
        float(radar_to_use.latitude['data'][0]),
        float(radar_to_use.longitude['data'][0]),
    )
    
    range_max_m = (y_grid_limits[1] - y_grid_limits[0]) / 2
    constant_roi = max(
        grid_resolution_xy * 1.5,
        800 + (range_max_m / 100000) * 400
    )
    
    # Calcular puntos de grilla usando la función del módulo grid_geometry
    z_points, y_points, x_points = calculate_grid_points(
        z_grid_limits, y_grid_limits, x_grid_limits,
        grid_resolution_z, grid_resolution_xy
    )
    
    # Campos a incluir en la grilla: principal + todos los campos de filtrado QC (ej. RHOHV)
    fields_for_grid = {field_to_use}
    for qc_name in AFFECTS_INTERP_FIELDS:
        if qc_name in radar_to_use.fields:
            fields_for_grid.add(qc_name)
    fields_for_grid = list(fields_for_grid)
    
    grid = pyart.map.grid_from_radars(
        radar_to_use,
        grid_shape=(z_points, y_points, x_points),
        grid_limits=(z_grid_limits, y_grid_limits, x_grid_limits),
        gridding_algo="map_gates_to_grid",
        grid_origin=grid_origin,
        fields=fields_for_grid,
        weighting_function='nearest',
        gatefilters=gf,
        roi_func="constant",
        constant_roi=constant_roi,
    )
    grid.to_xarray()
    
    # Guardamos en caché el 3D grid completo, antes de colapsar.
    # Guardar todos los metadatos del campo excepto 'data'
    field_metadata = {k: v for k, v in grid.fields[field_to_use].items() if k != 'data'}
    
    pkg_to_cache = {
        "arr3d": grid.fields[field_to_use]['data'].copy(),
        "x": grid.x['data'].copy(),
        "y": grid.y['data'].copy(),
        "z": grid.z['data'].copy(),
        "projection": dict(getattr(grid, "projection", {}) or {}),
        "field_name": field_to_use,
        "field_metadata": field_metadata,  # Incluir todos los metadatos (units, long_name, etc.)
    }
    GRID3D_CACHE[cache_key] = pkg_to_cache
    
    return grid


def convert_to_cog(src_path, cog_path):
    """
    Convierte un GeoTIFF existente a un COG (Cloud Optimized GeoTIFF) optimizado para tiling rápido.
    Re-escribe completamente el archivo con estructura tiled y overviews.
    """
    from rasterio.shutil import copy
    from rasterio.enums import Resampling
    
    # Si el COG ya existe y es válido, no regenerar
    if cog_path.exists():
        try:
            with rasterio.open(cog_path) as test:
                if (test.profile.get('tiled') and 
                    test.overviews(1) and 
                    test.profile.get('blockxsize', 0) >= 256):
                    return cog_path
        except:
            pass    
    try:
        # Abrir con IGNORE_COG_LAYOUT_BREAK para permitir modificar archivos COG
        with rasterio.open(src_path, IGNORE_COG_LAYOUT_BREAK='YES') as src:
            # Configurar perfil COG optimizado con tiles grandes
            # SIN compresión para máxima velocidad de lectura en tiles
            profile = src.profile.copy()
            profile.update(
                driver='COG',
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress='DEFLATE',
                BIGTIFF='IF_NEEDED',
                NUM_THREADS='ALL_CPUS',
                COPY_SRC_OVERVIEWS='YES',
            )
            
            # Escribir archivo intermedio tiled
            temp_tiled = Path(cog_path.parent) / f"temp_{cog_path.name}"
            
            with rasterio.open(temp_tiled, 'w', **profile) as dst:
                # Copiar todas las bandas
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
                
                # Copiar colorinterp si existe
                try:
                    dst.colorinterp = src.colorinterp
                except:
                    pass
            
            # Ahora agregar overviews al archivo tiled
            # También necesita IGNORE_COG_LAYOUT_BREAK porque el driver COG ya optimizó el archivo
            with rasterio.open(temp_tiled, 'r+', IGNORE_COG_LAYOUT_BREAK='YES') as dst:
                factors = [2, 4, 8, 16, 32]
                dst.build_overviews(factors, Resampling.nearest)
                dst.update_tags(ns='rio_overview', resampling='nearest')
            
            # Mover archivo temporal al destino final
            # shutil.move elimina el temp_tiled automáticamente
            shutil.move(str(temp_tiled), str(cog_path))
            
    except Exception as e:
        print(f"Error generando COG optimizado: {e}")
        
        # Limpiar archivo temporal si quedó creado
        try:
            if temp_tiled.exists():
                temp_tiled.unlink()
        except:
            pass
        
        # Fallback: copiar el original
        if not cog_path.exists():
            shutil.copy2(src_path, cog_path)
    
    return cog_path


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


def collapse_grid_to_2d(grid, field, product, *,
                        elevation_deg=None,       # para PPI
                        target_height_m=None,     # para CAPPI
                        vmin=-30.0):
    """
    Convierte la grilla 3D a 2D según el producto:
      - "ppi": sigue el haz del sweep con elevación `elevation_deg`
      - "cappi": toma el nivel z más cercano a `target_height_m`
    """
    data3d = grid.fields[field]['data']
    z = grid.z['data']
    x = grid.x['data']
    y = grid.y['data']
    ny, nx = len(y), len(x)

    if data3d.ndim == 2:  # ya llegó 2D (raro)
        arr2d = data3d
    else:
        if product == "ppi":
            # Buscamos recortar la superficie que sigue el haz del sweep seleccionado
            assert elevation_deg is not None
            # Calculamos distancia horizontal r de cada píxel al radar
            X, Y = np.meshgrid(x, y, indexing='xy')
            r = np.sqrt(X**2 + Y**2)
            Re = 8.49e6  # m, 4/3 R_tierra

            # Altura donde debería estar el haz en cada píxel
            z_target = r * np.sin(np.deg2rad(elevation_deg)) + (r**2) / (2.0 * Re)

            # Para cada pixel (y,x), buscamos el índice z cuyo nivel esté más cerca de z_target
            iz = np.abs(z_target[..., None] - z[None, None, :]).argmin(axis=2)

            # Tomamos el valor en ese z
            yy = np.arange(ny)[:, None]
            xx = np.arange(nx)[None, :]
            arr2d = data3d[iz, yy, xx]

        elif product == "cappi":
            assert target_height_m is not None
            iz = np.abs(z - float(target_height_m)).argmin()
            arr2d = data3d[iz, :, :]
        elif product == "colmax":
            arr2d = data3d.max(axis=0)
        else:
            raise ValueError("Producto inválido")

    # Re-máscarar
    arr2d = np.ma.masked_invalid(arr2d)
    if field in ["filled_DBZH", "DBZH", "DBZV", "DBZHF", "composite_reflectivity", "cappi"]:
        arr2d = np.ma.masked_less_equal(arr2d, vmin)
    elif field in ["KDP", "ZDR"]:
        arr2d = np.ma.masked_less(arr2d, vmin)

    # Lo escribimos como un único nivel
    grid.fields[field]['data'] = arr2d[np.newaxis, ...]   # (1,ny,nx)
    grid.fields[field]['_FillValue'] = -9999.0
    grid.z['data'] = np.array([0.0], dtype=float)


def process_radar_to_cog(
        filepath, 
        product="PPI", 
        field_requested="DBZH", 
        cappi_height=4000, 
        elevation=0, 
        filters=[], 
        output_dir="app/storage/tmp",
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
    range_max_m = safe_range_max_m(radar)

    # Calcular límites Z según producto
    z_min, z_max, elev_deg = calculate_z_limits(
        product, range_max_m, elevation, cappi_height, radar.fixed_angle['data']
    )
    z_grid_limits = (z_min, z_max)
    y_grid_limits = (-range_max_m, range_max_m)
    x_grid_limits = (-range_max_m, range_max_m)

    # Calculamos la cantidad de puntos en cada dimensión
    # XY depende del volumen, pero Z siempre usa resolución fina para transectos suaves
    grid_resolution_xy, grid_resolution_z = calculate_grid_resolution(volume)

    interp = 'nearest'

    # Separar filtros QC (RHOHV) vs otros (todos se aplicarán post-grid como máscaras 2D)
    # Ya no forzamos regridding por filtros: la grilla se genera UNA sola vez
    qc_filters, visual_filters = separate_filters(filters, field_to_use)

    # No usamos needs_regrid ni qc_sig (el cache ignora filtros para regridding)

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
        qc_sig=tuple(),  # no dependemos de filtros para cache
        session_id=session_id,
    )

    pkg_cached = GRID2D_CACHE.get(cache_key)

    if pkg_cached is None:
        # Construir o recuperar grilla 3D cacheada
        grid = _get_or_build_grid3d(
            radar_to_use=radar_to_use,
            field_to_use=field_to_use,
            file_hash=file_hash,
            volume=volume,
            qc_filters=qc_filters,
            z_grid_limits=z_grid_limits,
            y_grid_limits=y_grid_limits,
            x_grid_limits=x_grid_limits,
            grid_resolution_xy=grid_resolution_xy,
            grid_resolution_z=grid_resolution_z,
            session_id=session_id,
        )

        # Guardamos niveles z completos antes de colapsar el campo principal
        z_levels_full = grid.z['data'].copy()

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

        # Colapsar grilla campo QC (RHOHV) a 2D usando la versión no destructiva y los niveles originales
        # (refactor pendiente: esto podría hacerse dentro de collapse_grid_to_2d)
        qc_2d = {}
        # Obtener lista de campos QC que están en la grilla
        for qc_name in AFFECTS_INTERP_FIELDS:
            if qc_name == field_to_use:
                continue
            if qc_name not in grid.fields:
                continue
            data3d_q = grid.fields[qc_name]['data']
            # data3d_q puede seguir en 3D aunque ya colapsamos el principal (collapse_grid_to_2d sólo afecta ese campo)
            q2d = collapse_field_3d_to_2d(
                data3d_q,
                product=product.lower(),
                x_coords=grid.x['data'],
                y_coords=grid.y['data'],
                z_levels=z_levels_full,
                elevation_deg=elev_deg,
                target_height_m=cappi_height,
            )
            qc_2d[qc_name] = q2d

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
            "qc": qc_2d,
            "crs": crs_wkt,
            "transform": transform,
            "arr_warped": None,  # Se llenará después del primer warp
            "crs_warped": None,
            "transform_warped": None,
        }
        GRID2D_CACHE[cache_key] = pkg_cached

    # Si hay filtros "visuales" sobre el MISMO campo, aplicar máscara post-grid
    # Regla: cualquier filtro cuyo .field == field_to_use (mismo campo) lo aplicamos como máscara 2D
    masked = apply_visual_filters(pkg_cached["arr"], visual_filters, field_to_use)

    # Aplicar filtros QC post-grid (sin regridding)
    if qc_filters:
        qc_dict = pkg_cached.get("qc", {}) or {}
        masked = apply_qc_filters(masked, qc_filters, qc_dict)

    # Crear path único para el GeoTIFF temporal (antes de convertir a Cloud Optimized GeoTIFF)
    os.makedirs(output_dir, exist_ok=True)
    unique_tif_name = f"radar_{uuid.uuid4().hex}.tif"
    tiff_path = Path(output_dir) / unique_tif_name

    # Creamos un grid (ahora 2D) de "bolsillo” (sin reinterpolar) (usando el arr2d)
    # Reusamos la malla x/y de la cache: como no guardamos grid anterior completo, derivamos dims del array
    # Armamos un grid pyart mínimo para write_grid_geotiff, escribiendo el 2D como nivel 0
    ny, nx = masked.shape
    grid_fake = pyart.core.Grid(
        time={'data': np.array([0])},
        fields={field_to_use: {'data': masked[np.newaxis, :, :], '_FillValue': -9999.0}},
        metadata={'instrument_name': 'RADAR'},
        origin_latitude={'data': radar_to_use.latitude['data']},
        origin_longitude={'data': radar_to_use.longitude['data']},
        origin_altitude={'data': radar_to_use.altitude['data']},
        x={'data': np.linspace(x_grid_limits[0], x_grid_limits[1], nx).astype(np.float32)},
        y={'data': np.linspace(y_grid_limits[0], y_grid_limits[1], ny).astype(np.float32)},
        z={'data': np.array([0.0], dtype=np.float32)}
    )

    # Exportar a GeoTIFF
    pyart.io.write_grid_geotiff(
        grid=grid_fake,
        filename=str(tiff_path),
        field=field_to_use,
        level=0,
        rgb=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        warp_to_mercator=True
    )
    
    # ACTUALIZAR CACHE: 
    # Si es la primera vez que guardamos versión warped del array, la generamos ahora
    # (solo se hace una vez por cache_key, luego se reutiliza en futuros procesos)
    # Hacemos este paso para cachear el array warped en mercator y usarlo en stats futuros
    if pkg_cached.get("arr_warped") is None:
        temp_numeric_tif = Path(output_dir) / f"numeric_{uuid.uuid4().hex}.tif"
        pyart.io.write_grid_geotiff(
            grid=grid_fake,
            filename=str(temp_numeric_tif),
            field=field_to_use,
            level=0,
            rgb=False,  # Sin colormap, valores numéricos
            warp_to_mercator=True
        )
        
        # Leer el GeoTIFF numérico warped para stats
        with rasterio.open(temp_numeric_tif) as src_numeric:
            arr_warped = src_numeric.read(1, masked=True)
            transform_warped = src_numeric.transform
            crs_warped = src_numeric.crs.to_wkt()
            
            # Agregar versión warped al cache (mantiene arr local para GeoTIFFs)
            pkg_cached["arr_warped"] = arr_warped.astype(np.float32)
            pkg_cached["transform_warped"] = transform_warped
            pkg_cached["crs_warped"] = crs_warped
        
        # Limpiar GeoTIFF numérico temporal
        try:
            temp_numeric_tif.unlink()
        except OSError:
            pass
        
        # Generar versiones warped de campos QC para que coincidan con arr_warped
        qc_dict = pkg_cached.get("qc", {}) or {}
        qc_warped = {}
        
        for qc_field_name, qc_arr2d in qc_dict.items():
            # Crear grid temporal con este campo QC
            grid_qc = pyart.core.Grid(
                time={'data': np.array([0])},
                fields={qc_field_name: {'data': qc_arr2d[np.newaxis, :, :], '_FillValue': -9999.0}},
                metadata={'instrument_name': 'RADAR'},
                origin_latitude={'data': radar_to_use.latitude['data']},
                origin_longitude={'data': radar_to_use.longitude['data']},
                origin_altitude={'data': radar_to_use.altitude['data']},
                x={'data': grid_fake.x['data']},
                y={'data': grid_fake.y['data']},
                z={'data': np.array([0.0], dtype=np.float32)}
            )
            
            temp_qc_tif = Path(output_dir) / f"qc_{qc_field_name}_{uuid.uuid4().hex}.tif"
            pyart.io.write_grid_geotiff(
                grid=grid_qc,
                filename=str(temp_qc_tif),
                field=qc_field_name,
                level=0,
                rgb=False,
                warp_to_mercator=True
            )
            
            # Leer versión warped
            with rasterio.open(temp_qc_tif) as src_qc:
                qc_warped[qc_field_name] = src_qc.read(1, masked=True).astype(np.float32)
            
            # Limpiar temporal
            try:
                temp_qc_tif.unlink()
            except OSError:
                pass
        
        pkg_cached["qc_warped"] = qc_warped
        GRID2D_CACHE[cache_key] = pkg_cached

    # Convertir a COG (Cloud Optimized GeoTIFF)
    _ = convert_to_cog(tiff_path, cog_path)

     # Limpiar el GeoTIFF temporal (queda SOLO el COG)
    try:
        tiff_path.unlink()
    except OSError:
        pass

    return summary
