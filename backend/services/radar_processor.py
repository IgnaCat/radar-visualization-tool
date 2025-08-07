import os
import pyart
import xarray as xr
import dask.array as da
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import uuid
from utils import colores

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

def process_radar(filepath, output_dir="static/tmp"):
    """
    Procesa un archivo NetCDF de radar y genera una imagen PPI.
    Si el archivo es menor a 500MB, se procesa directamente con Py-ART.
    Si es mayor, se convierte a xarray + dask.
    Devuelve un resumen de los datos procesados.
    """
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    radar = pyart.io.read(filepath)

    try:
        reflectivity, field_used = get_reflectivity_field(radar)
    except KeyError as e:
        return {"error": str(e)}
    
    # Crear path único para la imagen
    os.makedirs(output_dir, exist_ok=True)
    unique_name = f"ppi_{uuid.uuid4().hex}.png"
    output_path = os.path.join(output_dir, unique_name)


    # Si el archivo es pequeño (menos de 500MB), procesar con Py-ART
    if file_size_mb < 500:

        # Crear figura, ejes con proyección Cartopy y el radar display
        fig = plt.figure(figsize=[15, 10])
        ax = plt.axes(projection=ccrs.PlateCarree())
        display = pyart.graph.RadarMapDisplay(radar)

        # Filtro. Extraido de las notebooks de Ignacio Montamat (Grupo Radar Córdoba).
        if 'RHOHV' in radar.fields:
            gf = pyart.correct.GateFilter(radar)
            gf.exclude_below('RHOHV', 0.92)

        display.plot_ppi_map(field_used,
                     sweep=0,
                     vmin=-30,
                     vmax=70,
                     projection=ccrs.PlateCarree(),
                     ax=ax,
                     colorbar_flag=False,
                     cmap='grc_th',
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
            "image_url": f"/{output_path.replace(os.sep, '/')}",
            "bounds": bounds,
            "field_used": field_used,
            "source_file": filepath
        }

    # Si es grande (mas 500MB), convertir a xarray + dask
    else:

        # Falta completar el generador de imagenes
        reflectivity_dask = da.from_array(reflectivity, chunks=(100, 100))
        ds = xr.DataArray(reflectivity_dask,
                          dims=["azimuth", "range"],
                          coords={"azimuth": radar.azimuth["data"],
                                  "range": radar.range["data"]},
                          name=field_used)
        summary = {
            "method": "pyart + xarray/dask",
        }

    return summary
