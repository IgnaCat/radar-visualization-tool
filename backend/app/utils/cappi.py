"""
Plot Max-CAPPI

This module provides a function to plot a Maximum Constant Altitude Plan Position Indicator (Max-CAPPI)
from radar data using an xarray dataset. The function includes options for adding map features, range rings,
color bars, and customized visual settings.

Author: Syed Hamid Ali (@syedhamidali)
"""

import os
import warnings

import cartopy.crs as ccrs
import cartopy.feature as feat
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.ticker import NullFormatter

from netCDF4 import num2date
from pandas import to_datetime
from scipy.interpolate import RectBivariateSpline

from pyart.core import Radar

warnings.filterwarnings("ignore")

def plot_maxcappi(
    grid,
    field,
    cmap=None,
    vmin=None,
    vmax=None,
    title=None,
    lat_lines=None,
    lon_lines=None,
    add_map=False,
    projection=None,
    colorbar=True,
    range_rings=False,
    dpi=100,
    savedir=None,
    show_figure=True,
    add_slogan=False,
    shapefile=None,
    shapefile_kwargs=None,
    facecolor='white',
    max_height=int(20),
    sideticks=[5,10,15,20],
    **kwargs,):
    """
    Plot a Maximum Constant Altitude Plan Position Indicator (Max-CAPPI) using an xarray Dataset.

    Parameters
    ----------
    grid : pyart.core.Grid
        The grid object containing the radar data to be plotted.
    field : str
        The radar field to be plotted (e.g., "REF", "VEL", "WIDTH").
    cmap : str or matplotlib colormap, optional
        Colormap to use for the plot. Default is "HomeyerRainbow".
    vmin : float, optional
        Minimum value for the color scaling. Default is set to the minimum value of the data if not provided.
    vmax : float, optional
        Maximum value for the color scaling. Default is set to the maximum value of the data if not provided.
    title : str, optional
        Title of the plot. If None, the title is set to "Max-{field}".
    lat_lines : array-like, optional
        Latitude lines to be included in the plot. Default is calculated based on dataset coordinates.
    lon_lines : array-like, optional
        Longitude lines to be included in the plot. Default is calculated based on dataset coordinates.
    add_map : bool, optional
        Whether to include a map background in the plot. Default is True.
    projection : cartopy.crs.Projection, optional
        The map projection for the plot. Default is automatically determined based on dataset coordinates.
    colorbar : bool, optional
        Whether to include a colorbar in the plot. Default is True.
    range_rings : bool, optional
        Whether to include range rings at 50 km intervals. Default is False.
    dpi : int, optional
        DPI (dots per inch) for the plot. Default is 100.
    savedir : str, optional
        Directory where the plot will be saved. If None, the plot is not saved.
    show_figure : bool, optional
        Whether to display the plot. Default is True.
    add_slogan : bool, optional
        Whether to add a slogan like "Powered by Py-ART" to the plot. Default is False.
    ###########################################################################################################
    @msuarez agregado para plotear los shapes
    shapefile : str, optional
        Path to the shapefile to be plotted.
    shapefile_kwargs : dict, optional
        Additional keyword arguments to pass to cartopy's `add_geometries` function.
    ###########################################################################################################
    **kwargs : dict, optional
        Additional keyword arguments to pass to matplotlib's `pcolormesh` function.

    Returns
    -------
    None
        This function does not return any value. It generates and optionally displays or saves a plot.

    Notes
    -----
    - The function extracts the maximum value across the altitude (z) dimension to create the Max-CAPPI.
    - It supports customizations such as map projections, color scales, and range rings.
    - If the radar_name attribute in the dataset is a byte string, it will be decoded and limited to 4 characters.
    - If add_map is True, map features and latitude/longitude lines are included.
    - The plot can be saved to a specified directory in PNG format.

    Author: Syed Hamid Ali (@syedhamidali)
    """

    ds = grid.to_xarray().squeeze()

    if lon_lines is None:
        lon_lines = np.arange(int(ds.lon.min().values), int(ds.lon.max().values) + 1)
    if lat_lines is None:
        lat_lines = np.arange(int(ds.lat.min().values), int(ds.lat.max().values) + 1)

    plt.rcParams.copy()
    plt.rcParams.update({
        "font.weight": "normal",         # Hace el peso de todas las fuentes (títulos, textos) en negrita
        "axes.labelweight": "normal",    # Marca en negrita las etiquetas de los ejes (x e y)
        "xtick.direction": "out",       # Hace que las marcas (ticks) del eje x apunten hacia adentro del gráfico (in u out)
        "ytick.direction": "out",       # Igual pero para el eje y, dirección hacia adentro
        "xtick.major.size": 5,        # Longitud de las marcas principales (ticks) en el eje x en puntos
        "ytick.major.size": 5,        # Longitud de las marcas principales en el eje y
        "xtick.minor.size": 0,         # Longitud de las marcas menores (ticks secundarios) en el eje x
        "ytick.minor.size": 0,         # Longitud de las marcas menores en el eje y
        "font.size": 12,               # Tamaño base de fuente para textos generales en el gráfico
        "axes.linewidth": 0.5,          # Grosor de la línea que dibuja los bordes de los ejes (axis spines)
        "ytick.labelsize": 10,         # Tamaño de letra para las etiquetas numéricas del eje y
        "xtick.labelsize": 10,         # Tamaño de letra para las etiquetas del eje x
    })

    max_c = ds[field].max(dim="z")
    max_x = ds[field].max(dim="y")
    max_y = ds[field].max(dim="x").T

    trgx = ds["x"].values
    trgy = ds["y"].values
    trgz = ds["z"].values
    

    if cmap is None:
        cmap = "HomeyerRainbow"
    if vmin is None:
        vmin = grid.fields[field]["data"].min()
    if vmax is None:
        vmax = grid.fields[field]["data"].max()
    if title is None:
        title = f"Max-{field.upper()[:3]}"

    def plot_range_rings(ax_xy, max_range):
        """
        Plots range rings at 50 km intervals.

        Parameters
        ----------
        ax_xy : matplotlib.axes.Axes
            The axis on which to plot the range rings.
        max_range : float
            The maximum range for the range rings.

        Returns
        -------
        None
        """
        background_color = ax_xy.get_facecolor()
        color = "k" if sum(background_color[:3]) / 3 > 0.5 else "w"

        for i, r in enumerate([240e3]):
            label = f"Ring Dist. {int(r/1e3)} km" if i == 0 else None
            ax_xy.plot(
                r * np.cos(np.arange(0, 360) * np.pi / 180),
                r * np.sin(np.arange(0, 360) * np.pi / 180),
                color=color,
                ls="-",
                linewidth=1,
                alpha=0.5,
                label=label,
            )

        # for i, r in enumerate(np.arange(5e4, np.floor(max_range) + 1, 5e4)):
        #     label = f"Ring Dist. {int(r/1e3)} km" if i == 0 else None
        #     ax_xy.plot(
        #         r * np.cos(np.arange(0, 360) * np.pi / 180),
        #         r * np.sin(np.arange(0, 360) * np.pi / 180),
        #         color=color,
        #         ls="--",
        #         linewidth=0.4,
        #         alpha=0.3,
        #         label=label,
        #     )

        # ax_xy.legend(loc="upper right", prop={"weight": "normal", "size": 8})

    def _get_projection(ds):
        """
        Determine the central latitude and longitude from a dataset
        and return the corresponding projection.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset from which to extract latitude and longitude
            information.

        Returns
        -------
        projection : cartopy.crs.Projection
            A Cartopy projection object centered on the extracted or
            calculated latitude and longitude.
        """

        def get_coord_or_attr(ds, coord_name, attr_name):
            """Helper function to get a coordinate or attribute, or
            calculate median if available.
            """
            if coord_name in ds:
                return (
                    ds[coord_name].values.item()
                    if ds[coord_name].values.ndim == 0
                    else ds[coord_name].values[0]
                )
            if f"origin_{coord_name}" in ds.coords:
                return ds.coords[f"origin_{coord_name}"].median().item()
            if f"radar_{coord_name}" in ds.coords:
                return ds.coords[f"radar_{coord_name}"].median().item()
            return ds.attrs.get(attr_name, None)

        lat_0 = get_coord_or_attr(
            ds, "latitude", "origin_latitude"
        ) or get_coord_or_attr(ds, "radar_latitude", "origin_latitude")
        lon_0 = get_coord_or_attr(
            ds, "longitude", "origin_longitude"
        ) or get_coord_or_attr(ds, "radar_longitude", "origin_longitude")

        if lat_0 is None or lon_0 is None:
            lat_0 = ds.lat.mean().item()
            lon_0 = ds.lon.mean().item()

        projection = ccrs.LambertAzimuthalEqualArea(lon_0, lat_0)
        return projection

    projection = _get_projection(ds)

    # FIG
    fig = plt.figure(figsize=[10.3, 10])
    # Agregado por @msuarez para separar las figuras un delta
    delta = 0
    left, bottom, width, height = 0.1, 0.1, 0.6, 0.1
    # este es el plot central (colmax)
    ax_xy = plt.axes((left, bottom, width, width), projection=projection)
    # este es el max del eje x
    ax_x = plt.axes((left, bottom + width + delta, width, height))
    # este es el max del eje y
    ax_y = plt.axes((left + width + delta, bottom, height, width))
    # este es el plot de arriba a la derecha
    # ax_cnr = plt.axes((left + width, bottom + width, left + left, height))
    if colorbar:
        ax_cb = plt.axes((left - 0.015 + width + height + 0.02, bottom, 0.02, width))

    # Set axis label formatters
    ax_x.xaxis.set_major_formatter(NullFormatter())
    ax_y.yaxis.set_major_formatter(NullFormatter())
    # ax_cnr.yaxis.set_major_formatter(NullFormatter())
    # ax_cnr.xaxis.set_major_formatter(NullFormatter())
    ax_x.set_ylabel("Height (km)", size=10)
    ax_y.set_xlabel("Height (km)", size=10)

    # Draw CAPPI
    plt.sca(ax_xy)
    xy = ax_xy.pcolormesh(trgx, trgy, max_c, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if shapefile is not None:
        from cartopy.io.shapereader import Reader

        if shapefile_kwargs is None:
            shapefile_kwargs = {}
        if "crs" not in shapefile_kwargs:
            shapefile_kwargs["crs"] = ccrs.PlateCarree()
        ax_xy.add_geometries(Reader(shapefile).geometries(), **shapefile_kwargs)

    # Add map features
    if add_map:
        map_features(ax_xy, lat_lines, lon_lines)

    ax_xy.minorticks_on()

    if range_rings:
        plot_range_rings(ax_xy, trgx.max())

    ax_xy.set_xlim(trgx.min(), trgx.max())
    ax_xy.set_ylim(trgx.min(), trgx.max())

    # Draw colorbar
    if colorbar:
        cb = plt.colorbar(xy, cax=ax_cb)
        cb.set_label(ds[field].attrs["units"], size=15)

    background_color = ax_xy.get_facecolor()
    color = "k" if sum(background_color[:3]) / 3 > 0.5 else "w"
    # color = "white"


    plt.sca(ax_x)
    plt.pcolormesh(trgx / 1e3, trgz / 1e3, max_x, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.yticks(sideticks)
    ax_x.set_xlim(trgx.min() / 1e3, trgx.max() / 1e3)
    ax_x.set_ylim(0, max_height)
    ax_x.grid(axis="y", lw=0.5, color=color, alpha=0.5, ls="-")
    # ax_x.minorticks_on()

    plt.sca(ax_y)
    plt.pcolormesh(trgz / 1e3, trgy / 1e3, max_y, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_y.set_xticks(sideticks)
    ax_y.set_ylim(trgx.min() / 1e3, trgx.max() / 1e3)
    ax_y.set_xlim(0, max_height)
    ax_y.grid(axis="x", lw=0.5, color=color, alpha=0.5, ls="-")
    # ax_y.minorticks_on()

    ax_xy.set_facecolor(facecolor)
    ax_x.set_facecolor(facecolor)
    ax_y.set_facecolor(facecolor)

    # plot del borde arriba a la derecha
    # plt.sca(ax_cnr)
    # plt.tick_params(
    #     axis="both",  # changes apply to both axes
    #     which="both",  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     left=False,
    #     right=False,
    #     labelbottom=False,
    # )

    # Initialize an empty list to store the processed radar names
    # full_title = []

    # # Check if radar_name is a list (or list-like) or a simple string
    # if isinstance(ds.attrs["radar_name"], list):
    #     # Iterate over each radar name in the list
    #     for name in ds.attrs["radar_name"]:
    #         # Decode if it's a byte string and take the first 4 characters
    #         if isinstance(name, bytes):
    #             site_title = name.decode("utf-8")[:4]
    #         else:
    #             site_title = name[:4]
    #         full_title.append(site_title)
    # else:
    #     # Handle the case where radar_name is a single string
    #     site_title = ds.attrs["radar_name"][:4]
    #     full_title.append(site_title)

    # Join the processed radar names into a single string with commas separating them
    # formatted_title = ", ".join(full_title)

    # # Center-align text in the corner box
    # plt.text(
    #     0.5,
    #     0.90,
    #     f"{formatted_title}",
    #     size=13,
    #     weight="bold",
    #     ha="center",
    #     va="center",
    # )
    # plt.text(0.5, 0.76, title, size=13, weight="bold", ha="center", va="center")
    # plt.text(
    #     0.5,
    #     0.63,
    #     f"Max Range: {np.floor(trgx.max() / 1e3)} km",
    #     size=11,
    #     ha="center",
    #     va="center",
    # )
    # plt.text(
    #     0.5,
    #     0.47,
    #     f"Max Height: {np.floor(trgz.max() / 1e3)} km",
    #     size=11,
    #     ha="center",
    #     va="center",
    # )
    # plt.text(
    #     0.5,
    #     0.28,
    #     ds["time"].dt.strftime("%H:%M:%S Z").values.item(),
    #     weight="bold",
    #     size=16,
    #     ha="center",
    #     va="center",
    # )
    # plt.text(
    #     0.5,
    #     0.13,
    #     ds["time"].dt.strftime("%d %b, %Y UTC").values.item(),
    #     size=13.5,
    #     ha="center",
    #     va="center",
    # )

    ax_xy.set_aspect("auto")

    # if add_slogan:
    #     fig.text(
    #         0.1,
    #         0.06,
    #         "Powered by Py-ART",  # Coordinates close to (0, 0) for lower-left corner
    #         fontsize=9,
    #         fontname="Courier New",
    #         # bbox=dict(facecolor='none', boxstyle='round,pad=0.5')
    #     )

    # if savedir is not None:
    #     radar_name = ds.attrs.get("instrument_name", "Radar")
    #     time_str = ds["time"].dt.strftime("%Y%m%d%H%M%S").values.item()
    #     figname = f"{savedir}{os.sep}{title}_{radar_name}_{time_str}.png"
    #     plt.savefig(fname=figname, dpi=dpi, bbox_inches="tight")
    #     print(f"Figure(s) saved as {figname}")

    # plt.rcParams.update(original_rc_params)
    # plt.rcdefaults()

    if show_figure:
        plt.show()
    else:
        plt.close()


def create_cappi(
    radar,
    fields=None,
    height=2000,
    gatefilter=None,
):
    """
    Create a Constant Altitude Plan Position Indicator (CAPPI) from radar data.

    Parameters
    ----------
    radar : Radar
        Py-ART Radar object containing the radar data.
    fields : list of str, optional
        List of radar fields to be used for creating the CAPPI.
        If None, all available fields will be used. Default is None.
    height : float, optional
        The altitude at which to create the CAPPI. Default is 2000 meters.
    gatefilter : GateFilter, optional
        A GateFilter object to apply masking/filtering to the radar data.
        Default is None.

    Returns
    -------
    Radar
        A Py-ART Radar object containing the CAPPI at the specified height.

    Notes
    -----
    CAPPI (Constant Altitude Plan Position Indicator) is a radar visualization
    technique that provides a horizontal view of meteorological data at a fixed altitude.
    Reference: https://glossary.ametsoc.org/wiki/Cappi

    Author
    ------
    Hamid Ali Syed (@syedhamidali)
    """

    if fields is None:
        fields = list(radar.fields.keys())

    # Initialize the first sweep as the reference
    first_sweep = 0

    # Initialize containers for the stacked data and nyquist velocities
    data_stack = []

    # Process each sweep individually
    for sweep in range(radar.nsweeps):
        sweep_slice = radar.get_slice(sweep)

        sweep_data = {}

        for field in fields:
            data = radar.get_field(sweep, field)

            # Apply gatefilter if provided
            if gatefilter is not None:
                data = np.ma.masked_array(
                    data, gatefilter.gate_excluded[sweep_slice, :]
                )
            time = radar.time["data"][sweep_slice]

            # Extract and sort azimuth angles
            azimuth = radar.azimuth["data"][sweep_slice]
            azimuth_sorted_idx = np.argsort(azimuth)
            azimuth = azimuth[azimuth_sorted_idx]
            data = data[azimuth_sorted_idx]

            # Store initial lat/lon for reordering
            if sweep == first_sweep:
                azimuth_final = azimuth
                time_final = time
            else:
                # Interpolate data for consistent azimuth ordering across sweeps
                interpolator = RectBivariateSpline(azimuth, radar.range["data"], data)
                data = interpolator(azimuth_final, radar.range["data"])

            sweep_data[field] = data[np.newaxis, :, :]

        data_stack.append(sweep_data)


    # Generate CAPPI for each field using data_stack
    fields_data = {}
    for field in fields:
        data_3d = np.concatenate(
            [sweep_data[field] for sweep_data in data_stack], axis=0
        )

        # Sort azimuth for all sweeps
        dim0 = data_3d.shape[1:]
        azimuths = np.linspace(0, 359, dim0[0])
        elevation_angles = radar.fixed_angle["data"][: data_3d.shape[0]]
        ranges = radar.range["data"]

        theta = (450 - azimuths) % 360
        THETA, PHI, R = np.meshgrid(theta, elevation_angles, ranges)
        Z = R * np.sin(PHI * np.pi / 180)

        # Extract the data slice corresponding to the requested height
        height_idx = np.argmin(np.abs(Z - height), axis=0)
        CAPPI = np.array(
            [
                data_3d[height_idx[j, i], j, i]
                for j in range(dim0[0])
                for i in range(dim0[1])
            ]
        ).reshape(dim0)

        # Retrieve units and handle case where units might be missing
        units = radar.fields[field].get("units", "").lower()

        # Determine valid_min and valid_max based on units
        if units == "dbz":
            valid_min, valid_max = -10, 80
        elif units in ["m/s", "meters per second"]:
            valid_min, valid_max = -100, 100
        elif units == "db":
            valid_min, valid_max = -7.9, 7.9
        else:
            # If units are not found or don't match known types, set default values or skip masking
            valid_min, valid_max = None, None

        # If valid_min or valid_max are still None, set them to conservative defaults or skip
        if valid_min is None:
            print(f"Warning: valid_min not set for {field}, using default of -1e10")
            valid_min = -1e10  # Conservative default
        if valid_max is None:
            print(f"Warning: valid_max not set for {field}, using default of 1e10")
            valid_max = 1e10  # Conservative default

        # Apply valid_min and valid_max masking
        if valid_min is not None:
            CAPPI = np.ma.masked_less(CAPPI, valid_min)
        if valid_max is not None:
            CAPPI = np.ma.masked_greater(CAPPI, valid_max)

        # Convert to masked array with the specified fill value
        CAPPI.set_fill_value(radar.fields[field].get("_FillValue", np.nan))
        CAPPI = np.ma.masked_invalid(CAPPI)
        CAPPI = np.ma.masked_outside(CAPPI, valid_min, valid_max)

        fields_data[field] = {
            "data": CAPPI,
            "units": radar.fields[field]["units"],
            "long_name": f"CAPPI {field} at {height} meters",
            "comment": f"CAPPI {field} calculated at a height of {height} meters",
            "_FillValue": radar.fields[field].get("_FillValue", np.nan),
        }

    # Set the elevation to zeros for CAPPI
    elevation_final = np.zeros(dim0[0], dtype="float32")

    # Since we are using the whole volume scan, report mean time
    try:
        dtime = to_datetime(
            num2date(radar.time["data"], radar.time["units"]).astype(str),
            format="ISO8601",
        )
    except ValueError:
        dtime = to_datetime(
            num2date(radar.time["data"], radar.time["units"]).astype(str)
        )
    dtime = dtime.mean()

    time = radar.time.copy()
    time["data"] = time_final
    time["mean"] = dtime

    # Create the Radar object with the new CAPPI data
    return Radar(
        time=radar.time.copy(),
        _range=radar.range.copy(),
        fields=fields_data,
        metadata=radar.metadata.copy(),
        scan_type=radar.scan_type,
        latitude=radar.latitude.copy(),
        longitude=radar.longitude.copy(),
        altitude=radar.altitude.copy(),
        sweep_number=radar.sweep_number.copy(),
        sweep_mode=radar.sweep_mode.copy(),
        fixed_angle=radar.fixed_angle.copy(),
        sweep_start_ray_index=radar.sweep_start_ray_index.copy(),
        sweep_end_ray_index=radar.sweep_end_ray_index.copy(),
        azimuth=radar.azimuth.copy(),
        elevation={"data": elevation_final},
        instrument_parameters=radar.instrument_parameters,
    )