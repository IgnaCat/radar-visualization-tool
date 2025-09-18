
import os
import pyart
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import uuid
import numpy as np
from . import colores


def create_png(radar, product, output_dir, field_used, filters=[], elevation=0, height=None, vmin=-30, vmax=70, cmap_key="grc_th"):
    """
    Genera una imagen PNG a partir de un objeto Py-ART Radar.
    Retorna el path del archivo generado.
    """

    # Crear path único para la imagen
    os.makedirs(output_dir, exist_ok=True)
    aux = "_".join([f"{a[0]}{a[1]}" for a in filters]) if filters else "nofilter"
    aux2 = height if product.upper() == "CAPPI" else elevation
    unique_name = f"png_{field_used}_{product}_{aux2}_{aux}_{uuid.uuid4().hex}.png"
    output_path = os.path.join(output_dir, unique_name)

    fig = plt.figure(figsize=[15, 10])
    ax = plt.axes(projection=ccrs.Mercator()) #PlateCarree
    display = pyart.graph.RadarMapDisplay(radar)

    # Aplicar filtros si se proporcionan
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_transition()
    for filter in filters:
        if filter[0] in radar.fields:
            gf.exclude_below(filter[0], filter[1])

    cmap = getattr(colores, f"get_cmap_{cmap_key}")()

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