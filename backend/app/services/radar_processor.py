import os
import pyart
import uuid
import hashlib
import numpy as np
import cartopy.crs as ccrs
import pyproj
from ..utils import colores
from ..utils import cappi as cappi_utils
from pathlib import Path
import rasterio
from rasterio.shutil import copy
from rasterio.enums import ColorInterp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from urllib.parse import quote
from ..core.config import settings


def get_reflectivity_field(radar):
    """
    Devuelve el campo de reflectividad disponible en el radar.
    Lanza KeyError si ninguno existe.
    """
    fields = {'DBZH', 'reflectivity', 'corrected_reflectivity_horizontal'}

    for field in fields:
        if field in radar.fields:
            return radar.fields[field]['data'], field

    raise KeyError("No se encontró ningún campo de reflectividad en el radar.")


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


def create_colmax(radar, gatefilter):
    """
    Crea un campo de reflectividad compuesto (COLMAX) a partir de todas las
    elevaciones disponibles en el radar.
    """

    compz = pyart.retrieve.composite_reflectivity(
        radar, field="filled_DBZH", gatefilter=gatefilter
    )
    # Cambiamos el long_name para que en el titulo de la figura salga COLMAX
    compz.fields['composite_reflectivity']['long_name'] = 'COLMAX'

    # volver a máscara antes de exportar
    data = compz.fields['composite_reflectivity']['data']
    mask = np.isnan(data) | np.isclose(data, -30) | (data < -40)
    compz.fields['composite_reflectivity']['data'] = np.ma.array(data, mask=mask)
    compz.fields['composite_reflectivity']['_FillValue'] = -9999.0

    return compz


def process_radar_to_cog(filepath, product="PPI", cappi_height=4000, elevation=0, output_dir="app/storage/tmp"):
    """
    Procesa un archivo NetCDF de radar y genera una COG (Cloud Optimized GeoTIFF).
    Devuelve un resumen de los datos procesados.
    Si ya existe un COG generado para este archivo, devuelve directamente la info.
    """

    # Crear nombre único pero estable a partir del NetCDF
    file_hash = hashlib.md5(open(filepath, "rb").read()).hexdigest()[:12]
    aux = elevation if product.upper() == "PPI" else (cappi_height if product.upper() == "CAPPI" else "")
    unique_cog_name = f"radar_{product}_{aux}_{file_hash}.tif"
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

    # print(radar.range['data'])    # distancias en metros de cada gate
    # print(radar.azimuth['data'])  # ángulos de cada rayo
    # print(radar.elevation['data'])

    try:
        _, reflectivity_field = get_reflectivity_field(radar)
    except KeyError as e:
        return {"Error": str(e)}
    

    gf = None
    if "RHOHV" in radar.fields:
        gf = pyart.filters.GateFilter(radar)
        # if product.upper() == "COLMAX":
        #     gf.exclude_transition()
        #     gf.exclude_below("RHOHV", 0.80)
        # elif product.upper() == "CAPPI":
        #     gf.exclude_below("RHOHV", 0.5)
        # else:
        #     gf.exclude_below("RHOHV", 0.92)

    
    
    compz = None
    cappi = None
    # Relleno el campo DBZH sino los -- no dejan interpolar
    filled_DBZH = radar.fields[reflectivity_field]['data'].filled(fill_value=-30)
    radar.add_field_like(reflectivity_field, 'filled_DBZH', filled_DBZH, replace_existing=True)

    if product.upper() == "PPI":
        ppi = radar.extract_sweeps([elevation])
    elif product.upper() == "CAPPI":
        cappi = cappi_utils.create_cappi(radar, fields=["filled_DBZH"], height=cappi_height, gatefilter=gf)
    else:
        compz = create_colmax(radar, gf)

    # Definimos los limites de nuestra grilla en las 3 dimensiones (x,y,z)
    z_grid_limits = (0.0, 0.0)
    y_grid_limits = (-240e3, 240e3)
    x_grid_limits = (-240e3, 240e3)

    # Definimos una resolución. A mayor resolución más lento va a ser el procesamiento de la grilla.
    grid_resolution = 1000

    # Calculamos la cantidad de puntos en cada dimensión
    z_points = 1
    y_points = int((y_grid_limits[1] - y_grid_limits[0]) / grid_resolution)
    x_points = int((x_grid_limits[1] - x_grid_limits[0]) / grid_resolution)


    radar_to_use = ppi if product.upper() == "PPI" else (cappi if product.upper() == "CAPPI" else compz)

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

    # Crear path único para el GeoTIFF temporal
    os.makedirs(output_dir, exist_ok=True)
    unique_tif_name = f"radar_{uuid.uuid4().hex}.tif"
    tiff_path = Path(output_dir) / unique_tif_name

    field_to_use = reflectivity_field if product.upper() == "PPI" else ("filled_DBZH" if product.upper() == "CAPPI" else 'composite_reflectivity')

    # Exportar a GeoTIFF
    pyart.io.write_grid_geotiff(
        grid=grid,
        filename=str(tiff_path),
        field=field_to_use,
        level=0,
        rgb=True,
        cmap=colores.get_cmap_grc_th(),
        vmin=-30,
        vmax=70
    )

    # Convertir a COG y reproyectar a EPSG:3857
    _ = reproject_to_cog(tiff_path, cog_path, dst_crs="EPSG:3857")

     # Limpiar el GeoTIFF temporal (queda SOLO el COG)
    try:
        tiff_path.unlink()
    except OSError:
        pass

    return summary
