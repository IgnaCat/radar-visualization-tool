import math
import os
import pyart
import uuid
import hashlib
import numpy as np
import cartopy.crs as ccrs
import pyproj
from ..utils import colores
from ..utils import cappi as cappi_utils
from ..utils import png
from pathlib import Path
import rasterio
from rasterio.shutil import copy
from rasterio.enums import ColorInterp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from urllib.parse import quote
from ..core.config import settings

FIELD_ALIASES = { 
    "DBZH": ["DBZH", "reflectivity", "corrected_reflectivity_horizontal"], 
    "ZDR": ["ZDR", "zdr"], 
    "RHOHV": ["RHOHV", "rhohv"], 
    "KDP": ["KDP", "kdp"] 
}

FIELD_RENDER = { 
    "DBZH": {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"}, 
    "ZDR": {"vmin": -5.0, "vmax": 10.5, "cmap": "grc_zdr"}, 
    "RHOHV": {"vmin": 0.5, "vmax": 1.0, "cmap": "grc_rho"}, 
    "KDP": {"vmin": 0.0, "vmax": 8.0, "cmap": "grc_rain"} 
}

def resolve_field(radar, field_requested: str):
    """Devuelve el nombre real del campo en el radar a partir del 'field_requested'."""
    key = field_requested.upper()
    if key not in FIELD_ALIASES:
        raise KeyError(f"Campo no soportado: {field_requested}")
    for candidate in FIELD_ALIASES[key]:
        if candidate in radar.fields:
            return candidate, key
    raise KeyError(f"No se encontró ningún alias disponible para {field_requested} en el radar.")


def reproject_to_cog(src_path, cog_path, dst_crs="EPSG:3857"):
    """
    Reproyecta un archivo Geotiff a un nuevo CRS y lo guarda como COG.
    """

    with rasterio.open(src_path) as src:
        # Calcular transform, width y height para el nuevo CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Definir el perfil base
        profile = src.profile.copy()
        profile.update(
            driver="COG",
            compress="DEFLATE",
            predictor=2,
            BIGTIFF="IF_NEEDED",
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            photometric="RGB"
        )
        profile["band_descriptions"] = ["Red", "Green", "Blue", "Alpha"]

        # Crear el archivo COG directamente
        with rasterio.open(cog_path, "w+", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha
            )

            # Generar overviews dentro del COG
            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")

    return cog_path


def create_colmax(radar):
    """
    Crea un campo de reflectividad compuesto (COLMAX) a partir de todas las
    elevaciones disponibles en el radar.
    """

    compz = pyart.retrieve.composite_reflectivity(radar, field="filled_DBZH")

    # Cambiamos el long_name para que en el titulo de la figura salga COLMAX
    compz.fields['composite_reflectivity']['long_name'] = 'COLMAX'

    # Volver a la máscara antes de exportar
    data = compz.fields['composite_reflectivity']['data']
    mask = np.isnan(data) | np.isclose(data, -30) | (data < -40)
    compz.fields['composite_reflectivity']['data'] = np.ma.array(data, mask=mask)
    compz.fields['composite_reflectivity']['_FillValue'] = -9999.0

    return compz


def beam_height_max_km(range_max_m, elev_deg, antenna_alt_m=0.0):
    """
    Calcula la altura máxima del haz en km para un rango y elevación dados.
    """
    Re = 8.49e6  # m
    r = float(range_max_m)
    th = math.radians(float(elev_deg))
    h = r*math.sin(th) + (r*r)/(2.0*Re) + antenna_alt_m
    return h/1000.0  # km


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
    if field in ["filled_DBZH", "DBZH", "composite_reflectivity", "cappi"]:
        arr2d = np.ma.masked_less_equal(arr2d, vmin)
    elif field in ["KDP", "ZDR", "RHOHV"]:
        arr2d = np.ma.masked_less(arr2d, vmin)

    # Lo escribimos como un único nivel
    grid.fields[field]['data'] = arr2d[np.newaxis, ...]   # (1,ny,nx)
    grid.z['data'] = np.array([0.0], dtype=float)


def md5_file(path, chunk=1024*1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def process_radar_to_cog(filepath, product="PPI", field_requested="DBZH", cappi_height=4000, elevation=0, filters=[], output_dir="app/storage/tmp"):
    """
    Procesa un archivo NetCDF de radar y genera una COG (Cloud Optimized GeoTIFF).
    Devuelve un resumen de los datos procesados.
    Si ya existe un COG generado para este archivo, devuelve directamente la info.
    """

    # Crear nombre único pero estable a partir del NetCDF
    file_hash = md5_file(filepath)[:12]
    filters_str = "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters]) if filters else "nofilter"
    aux = elevation if product.upper() == "PPI" else (cappi_height if product.upper() == "CAPPI" else "")
    unique_cog_name = f"radar_{field_requested}_{product}_{filters_str}_{aux}_{file_hash}.tif"
    cog_path = Path(output_dir) / unique_cog_name
    file_uri = Path(cog_path).resolve().as_posix()

    style = "&resampling=nearest&warp_resampling=nearest"
    summary = {
        "method": "pyart",
        "image_url": f"static/tmp/{unique_cog_name}",
        "source_file": filepath,
        "tilejson_url": f"{settings.BASE_URL}/cog/WebMercatorQuad/tilejson.json?url={quote(file_uri, safe=':/')}{style}",
    }

    # Si ya existe el COG, devolvemos directo
    if cog_path.exists():
        return summary

    # Si no existe, lo procesamos...
    radar = pyart.io.read(filepath)

    try:
        field_name, field_key = resolve_field(radar, field_requested)
    except KeyError as e:
        return {"Error": str(e)}
    
    # defaults de render por variable
    render = FIELD_RENDER.get(field_key, {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"})
    vmin = render["vmin"]; vmax = render["vmax"]; cmap_key = render["cmap"] 
    cmap = getattr(colores, f"get_cmap_{cmap_key}")()

    if field_name == "DBZH" and product.upper() in ["CAPPI", "COLMAX"]:
        # Relleno el campo DBZH sino los -- no dejan interpolar
        filled_DBZH = radar.fields[field_name]['data'].filled(fill_value=-30)
        radar.add_field_like(field_name, 'filled_DBZH', filled_DBZH, replace_existing=True)
        field_name = 'filled_DBZH'

    # Definimos qué radar y campo usar según el producto
    product_upper = product.upper()
    if product_upper == "PPI":
        radar_to_use = radar.extract_sweeps([elevation])
        field_to_use = field_name
    elif product_upper == "CAPPI":
        cappi = cappi_utils.create_cappi(radar, fields=["filled_DBZH"], height=cappi_height)
        # Creamos un campo de 5400x523 y lo rellenamos con el cappi
        # Hacemos esto por problemas con el interpolador de pyart
        template = cappi.fields['filled_DBZH']['data']   # (360, 523)
        zeros_array = np.tile(template, (15, 1))   # (5400, 523)
        radar.add_field_like('DBZH', 'cappi', zeros_array, replace_existing=True)

        radar_to_use = radar
        field_to_use = "cappi"
    elif product_upper == "COLMAX" and field_key.upper() == "DBZH":
        radar_to_use = create_colmax(radar)
        field_to_use = 'composite_reflectivity'
    else:
        raise ValueError(f"Producto inválido: {product_upper}")

    # Aplicar filtros si se proporcionan
    gf = pyart.filters.GateFilter(radar_to_use)
    if field_to_use in radar_to_use.fields:
        gf.exclude_invalid(field_to_use)
        gf.exclude_masked(field_to_use)
    gf.exclude_transition()

    for f in filters:
        if f.field in radar_to_use.fields:
            gf.exclude_below(f.field, f.min)
            gf.exclude_above(f.field, f.max)

    # Generamos la imagen PNG para previsualización y referencia
    png.create_png(
        radar_to_use, 
        product, 
        output_dir, 
        field_to_use, 
        filters=filters, 
        elevation=elevation, 
        height=cappi_height, 
        vmin=vmin, 
        vmax=vmax, 
        cmap_key=cmap_key
    )

    # Creamos la grilla
    # Definimos los limites de nuestra grilla en las 3 dimensiones (x,y,z)
    if product_upper == "CAPPI":
        z_top_m = cappi_height + 2000  # +2 km de margen
        elev_deg = None
    else:
        range_max_m = 240e3
        elev_deg = float(radar.fixed_angle['data'][elevation])
        hmax_km = beam_height_max_km(range_max_m, elev_deg)
        z_top_m = int((hmax_km + 3) * 1000)  # +3 km de margen
    
    z_grid_limits = (0.0, z_top_m)
    y_grid_limits = (-240e3, 240e3)
    x_grid_limits = (-240e3, 240e3)

    # Calculamos la cantidad de puntos en cada dimensión
    grid_resolution = 1000
    z_points = int(np.ceil((z_grid_limits[1] - z_grid_limits[0]) / grid_resolution)) + 1
    z_points = max(z_points, 2)
    y_points = int((y_grid_limits[1] - y_grid_limits[0]) / grid_resolution)
    x_points = int((x_grid_limits[1] - x_grid_limits[0]) / grid_resolution)


    # Esta proyeccion en el grid no funciona, lo deja en Azimutal Equidistance
    # projection = ccrs.Mercator()
    # merc = pyproj.CRS.from_epsg(3857)

    grid = pyart.map.grid_from_radars(
        radar_to_use,
        grid_shape=(z_points, y_points, x_points),
        grid_limits=(z_grid_limits, y_grid_limits, x_grid_limits),
        # projection=merc,
        weighting_function='nearest',
        gatefilters=gf
    )
    grid.to_xarray()

    # Pasamos de grilla 3D a 2D
    collapse_grid_to_2d(
        grid,
        field=field_to_use,
        product=product.lower(),
        elevation_deg=elev_deg,
        target_height_m=cappi_height,
        vmin=vmin
    )

    # Crear path único para el GeoTIFF temporal (antes de convertir a Cloud Optimized GeoTIFF)
    os.makedirs(output_dir, exist_ok=True)
    unique_tif_name = f"radar_{uuid.uuid4().hex}.tif"
    tiff_path = Path(output_dir) / unique_tif_name

    # Exportar a GeoTIFF
    pyart.io.write_grid_geotiff(
        grid=grid,
        filename=str(tiff_path),
        field=field_to_use,
        level=0,
        rgb=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    # Convertir a COG y reproyectar a EPSG:3857
    _ = reproject_to_cog(tiff_path, cog_path, dst_crs="EPSG:3857")

     # Limpiar el GeoTIFF temporal (queda SOLO el COG)
    try:
        tiff_path.unlink()
    except OSError:
        pass

    return summary
