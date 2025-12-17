
import os
import pyart
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import uuid
import numpy as np
from . import colores
from ..services.radar_common import (
    build_gatefilter
)



def create_png(radar, product, output_dir, field_used, filters=[], elevation=0, height=None, vmin=-30, vmax=70, cmap_key="grc_th"):
    """
    Genera una imagen PNG a partir de un objeto Py-ART Radar.
    Retorna el path del archivo generado.
    """

    # Crear path único para la imagen
    os.makedirs(output_dir, exist_ok=True)
    filters_str = "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters]) if filters else "nofilter"
    aux = height if product.upper() == "CAPPI" else elevation
    unique_name = f"png_{field_used}_{product}_{aux}_{filters_str}_{uuid.uuid4().hex}.png"
    output_path = os.path.join(output_dir, unique_name)

    fig = plt.figure(figsize=[15, 10])
    ax = plt.axes(projection=ccrs.Mercator()) #PlateCarree
    display = pyart.graph.RadarMapDisplay(radar)

    # Aplicar filtros si se proporcionan
    gf = build_gatefilter(radar, field_used, filters) if filters else None

    if cmap_key == "grc_zdr2":
        cmap_key = "grc_zdr"

    # Determinar si es un cmap personalizado (grc_*) o uno de pyart/matplotlib
    if cmap_key.startswith("grc_"):
        # Colormap personalizado del módulo colores
        cmap = getattr(colores, f"get_cmap_{cmap_key}")()
    elif cmap_key.startswith("pyart_"):
        # Colormap de pyart (obtener objeto colormap real)
        cmap_name = cmap_key.replace("pyart_", "")
        try:
            cmap = pyart.graph.cm.get_colormap(cmap_name)
        except (AttributeError, KeyError):
            # Fallback: intentar como colormap estándar de matplotlib
            cmap = plt.get_cmap(cmap_name)
    else:
        # Colormap estándar de matplotlib/pyart (NWSVel, Theodore16, etc)
        # Intentar primero de PyART, luego matplotlib
        try:
            cmap = pyart.graph.cm.get_colormap(cmap_key)
        except (AttributeError, KeyError):
            cmap = plt.get_cmap(cmap_key)

    # Enmascarar datos inválidos
    # radar.fields[field_used]['data'] = np.ma.masked_invalid(radar.fields[field_used]['data'])
    # radar.fields[field_used]['data'] = np.ma.masked_less(radar.fields[field_used]['data'], vmin)

    if field_used == 'ppi':
        display.plot_ppi_map(field_used,
                        sweep=elevation,
                        vmin=vmin,
                        vmax=vmax,
                        projection=ccrs.Mercator(),
                        ax=ax,
                        colorbar_flag=False,
                        cmap=cmap,
                        gatefilter=gf)
    else:
        display.plot_ppi_map(field_used,
                        vmin=vmin,
                        vmax=vmax,
                        projection=ccrs.Mercator(),
                        ax=ax,
                        colorbar_flag=False,
                        cmap=cmap,
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

    return output_path