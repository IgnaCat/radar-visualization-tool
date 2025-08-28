import os
import pyart
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import uuid
from ..utils import colores
from pathlib import Path
import rasterio
from rasterio.shutil import copy
from rasterio.enums import ColorInterp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from urllib.parse import quote
from ..core.config import settings
import pyproj

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

def process_radar_to_png(filepath, output_dir="app/storage/tmp"):
    """
    Procesa un archivo NetCDF de radar y genera una imagen PPI, usando Py-ART.
    Devuelve un resumen de los datos procesados.
    """
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    radar = pyart.io.read(filepath)

    try:
        reflectivity, field_used = get_reflectivity_field(radar)
    except KeyError as e:
        return {"Error": str(e)}
    
    # Crear path único para la imagen
    os.makedirs(output_dir, exist_ok=True)
    unique_name = f"ppi_{uuid.uuid4().hex}.png"
    output_path = os.path.join(output_dir, unique_name)

    fig = plt.figure(figsize=[15, 10])
    ax = plt.axes(projection=ccrs.Mercator()) #PlateCarree
    display = pyart.graph.RadarMapDisplay(radar)

    # Filtro. Extraido de las notebooks de Ignacio Montamat (Grupo Radar Córdoba).
    gf = None
    if 'RHOHV' in radar.fields:
        gf = pyart.correct.GateFilter(radar)
        gf.exclude_below('RHOHV', 0.92)

    display.plot_ppi_map(field_used,
                    sweep=0,
                    vmin=-30,
                    vmax=70,
                    projection=ccrs.Mercator(),
                    ax=ax,
                    colorbar_flag=False,
                    cmap=colores.get_cmap_grc_th(),
                    gatefilter=gf)
    

    # Calcular bounds
    lat = radar.gate_latitude['data']
    lon = radar.gate_longitude['data']
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    # Eliminar límites del mapa
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)


    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True )
    plt.close()

    summary = {
        "method": "pyart",
        "image_url": f"static/tmp/{unique_name}",
        "bounds": bounds,
        "field_used": field_used,
        "source_file": filepath,
    }

    return summary


def reproject_to_cog(src_path, output_dir, dst_crs="EPSG:4326"):
    """
    Reproyecta un archivo Geotiff a un nuevo CRS y lo guarda como COG.
    """
    unique_cog_name = f"radar_cog_{uuid.uuid4().hex}.tif"
    cog_path = Path(output_dir) / unique_cog_name

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

    return cog_path, unique_cog_name


def process_radar_to_cog(filepath, output_dir="app/storage/tmp"):
    """
    Procesa un archivo NetCDF de radar y genera una COG (Cloud Optimized GeoTIFF).
    Devuelve un resumen de los datos procesados.
    """
    radar = pyart.io.read(filepath)

    # print(radar.range['data'])    # distancias en metros de cada gate
    # print(radar.azimuth['data'])  # ángulos de cada rayo
    # print(radar.elevation['data'])

    try:
        _, field_used = get_reflectivity_field(radar)
    except KeyError as e:
        return {"Error": str(e)}

     # Filtro. Extraido de las notebooks de Ignacio Montamat (Grupo Radar Córdoba).
    gf = None
    if 'RHOHV' in radar.fields:
        gf = pyart.correct.GateFilter(radar)
        gf.exclude_below('RHOHV', 0.92)

    # Definimos los limites de nuestra grilla en las 3 dimensiones (x,y,z)
    z_grid_limits = (1000, 10_000)
    y_grid_limits = (-240e3, 240e3)
    x_grid_limits = (-240e3, 240e3)

    # Definimos una resolución. A mayor resolución más lento va a ser el procesamiento de la grilla.
    grid_resolution = 1000

    # Calculamos la cantidad de puntos en cada dimensión
    z_points = int((z_grid_limits[1] - z_grid_limits[0]) / grid_resolution)
    y_points = int((y_grid_limits[1] - y_grid_limits[0]) / grid_resolution)
    x_points = int((x_grid_limits[1] - x_grid_limits[0]) / grid_resolution)

    merc = pyproj.CRS.from_epsg(4326)

    grid = pyart.map.grid_from_radars(
        radar,
        grid_shape=(z_points, y_points, x_points),
        grid_limits=(z_grid_limits, y_grid_limits, x_grid_limits),
        projection=merc,
        gatefilters=gf
    )
    grid.to_xarray()

    # Crear path único
    os.makedirs(output_dir, exist_ok=True)
    unique_tif_name = f"radar_{uuid.uuid4().hex}.tif"
    tiff_path = Path(output_dir) / unique_tif_name

    # Exportar a GeoTIFF
    pyart.io.write_grid_geotiff(
        grid=grid,
        filename=str(tiff_path),
        field=field_used,
        level=0,
        rgb=True,
        cmap=colores.get_cmap_grc_th(),
        vmin=-30,
        vmax=70
    )

    # Convertir a COG y reproyectar a EPSG:4326 (lat-lon)
    cog_path, unique_cog_name = reproject_to_cog(tiff_path, output_dir, dst_crs="EPSG:4326")

    # Generar overviews dentro del COG
    # with rasterio.open(cog_path, "r+", open_options={"IGNORE_COG_LAYOUT_BREAK": "YES"}) as dst:
    #     dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
    #     dst.update_tags(ns="rio_overview", resampling="nearest")
        

     # Limpiar el GeoTIFF temporal (queda SOLO el COG)
    try:
        tiff_path.unlink()
    except OSError:
        pass

    summary = {
        "method": "pyart",
        "image_url": f"static/tmp/{unique_cog_name}",
        "field_used": field_used,
        "source_file": filepath,
    }

    return summary

