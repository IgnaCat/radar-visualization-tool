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

def process_radar(filepath, output_dir="app/storage/tmp"):
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


def process_radar_to_cog(filepath, output_dir="app/storage/cogs"):
    """
    Procesa un archivo NetCDF de radar y genera una COG (Cloud Optimized GeoTIFF).
    Devuelve un resumen de los datos procesados.
    """
    radar = pyart.io.read(filepath)

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

    grid = pyart.map.grid_from_radars(
        radar,
        grid_shape=(z_points, y_points, x_points),
        grid_limits=(z_grid_limits, y_grid_limits, x_grid_limits),
        projection=ccrs.Mercator(),
        gatefilters=gf
    )
    grid.to_xarray()

    # Crear path único
    os.makedirs(output_dir, exist_ok=True)
    unique_tif_name = f"radar_{uuid.uuid4().hex}.tif"
    tiff_path = os.path.join(output_dir, unique_tif_name)

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

    # Convertir a COG
    unique_cog_name = f"radar_cog_{uuid.uuid4().hex}.tif"
    cog_path = Path(output_dir) / unique_cog_name
    file_uri = Path(cog_path).resolve().as_posix()
    style = "&resampling=nearest&warp_resampling=nearest"

    with rasterio.open(tiff_path) as src:
        profile = src.profile
        profile.update(
            driver="COG",
            compress="DEFLATE",
            predictor=2
        )
        copy(src, cog_path, **profile)

    summary = {
        "method": "pyart",
        "field_used": field_used,
        "source_file": filepath,
        "cog_url": f"static/cogs/{unique_cog_name}",
        "tilejson_url": f"{settings.BASE_URL}/cog/WebMercatorQuad/tilejson.json?url={quote(file_uri, safe=':/')}{style}",
    }

    return summary

