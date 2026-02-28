import os
import pyart
import copy
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from fastapi import HTTPException
from pathlib import Path
from typing import List, Optional
from geopy.distance import geodesic
from rasterio.vrt import WarpedVRT
from scipy.interpolate import RegularGridInterpolator
from rasterio.enums import Resampling
from ..core.constants import VARIABLE_UNITS
from ..core.config import settings

from ..models import RangeFilter
from .radar_processing import (
    beam_height_max_km,
    calculate_grid_points,
    calculate_grid_resolution,
    calculate_z_limits,
    get_or_build_grid3d_with_operator,
    get_or_build_W_operator,
    apply_operator,
    separate_filters,
    apply_visual_filters,
)
from .radar_common import (
    resolve_field, colormap_for, build_gatefilter,
    safe_range_max_m, get_radar_site, md5_file, limit_line_to_range,
    normalize_proj_dict, qc_signature,
)

def calcule_radial_angle(radar_lat, radar_lon, punto_lat, punto_lon):
    """
    Calcula el ángulo en grados desde la latitud y longitud del radar hasta el punto de interés.
    Mide desde Norte en sentido horario.

    Parámetros:
    - radar_lat: Latitud del radar.
    - radar_lon: Longitud del radar.
    - punto_lat: Latitud del punto de interés.
    - punto_lon: Longitud del punto de interés.

    Retorna:
    - radial_angle: Ángulo radial en grados desde el radar al punto de interés.
    """
    # Convertir latitud y longitud a radianes
    radar_lat_rad = np.radians(radar_lat)
    radar_lon_rad = np.radians(radar_lon)
    punto_lat_rad = np.radians(punto_lat)
    punto_lon_rad = np.radians(punto_lon)

    # Calcula la diferencia de longitud
    d_lon = punto_lon_rad - radar_lon_rad

    # Calcula el ángulo usando la fórmula de azimut
    x = np.sin(d_lon) * np.cos(punto_lat_rad)
    y = np.cos(radar_lat_rad) * np.sin(punto_lat_rad) - (np.sin(radar_lat_rad) * np.cos(punto_lat_rad) * np.cos(d_lon))

    azimuth = np.arctan2(x, y)

    # Convierte el ángulo a grados y ajustar el rango [0, 360)
    radial_angle = np.degrees(azimuth)
    radial_angle = (radial_angle + 360) % 360

    return radial_angle


def variable_radar_cross_section(
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    volumen_radar_data,
    output_path,
    range_max: float,
    variable='DBZH',
    cmap='viridis',
    filters: List[RangeFilter] = [],
    plot_max_length_km: Optional[float] = None,
    plot_max_height_km: Optional[float] = None,
):
    """
    Función para graficar datos radiales del radar en un perfil a un ángulo dado.
    Esta función grafica en un ángulo dado que esta definido por una latitud y longitud.
    También grafica el perfil de elevación del terreno.

    Parámetros:
    - start_lat: Latitud del punto inicial del perfil.
    - start_lon: Longitud del punto inicial del perfil.
    - end_lat: Latitud del punto final del perfil.
    - end_lon: Longitud del punto final del perfil.
    - volumen_radar_data: Datos del radar en formato Py-ART.
    - field_name: Nombre del campo a graficar.
    - output_path: Ruta donde se guardará la imagen generada.
    - range_max: Distancia máxima a graficar (en km).
    - variable: Variable a graficar (por defecto 'DBZH').
    - cmap: Colormap a utilizar (por defecto 'viridis').
    - gf: GateFilter para enmascarar datos (opcional).
    """
    ######################################################################
    tif_path = Path(settings.DATA_DIR) / "mosaico_argentina_2.tif"

    radial_angle = calcule_radial_angle(start_lat, start_lon, end_lat, end_lon)

    # Distancia máxima según el radar (en km)
    radar_range_km = round(volumen_radar_data.ngates * volumen_radar_data.range['meters_between_gates'] / 1000)

    # Defino densidad de muestreo sobre la línea (cada N metros)
    # Podés ajustar este step si querés más/menos detalle
    step_m = 150.0
    num_points = int((radar_range_km * 1000) // step_m) + 1
    # Genero los puntos geodésicos desde el radar hacia el bearing
    lat_points = np.empty(num_points, dtype=np.float64)
    lon_points = np.empty(num_points, dtype=np.float64)
    for i in range(num_points):
        d_km = (i * step_m) / 1000.0
        p = geodesic(kilometers=d_km).destination((start_lat, start_lon), radial_angle)
        lat_points[i], lon_points[i] = p.latitude, p.longitude

    # Muestreo del DEM (Digital Elevation Model) solo en esos puntos y en float32 con NaN
    with rasterio.open(tif_path) as src:
        # WarpedVRT reprojecta/ajusta al CRS del raster y lee solo ventanas necesarias
        with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
            coords = list(zip(lon_points, lat_points))  # (x=lon, y=lat)
            # sample devuelve un generador de arrays; lo convertimos y casteamos a float32
            perfil_elevacion = np.fromiter((v[0] for v in vrt.sample(coords)),
                                           dtype=np.float32, count=len(coords))
            # Manejo de nodata -> NaN
            nodata = vrt.nodata
            if nodata is not None:
                mask = perfil_elevacion == nodata
                if mask.any():
                    perfil_elevacion[mask] = np.nan

    # Detectar último índice válido
    last_valid_index = len(perfil_elevacion) - 1
    valid_indices = np.where(~np.isnan(perfil_elevacion))[0]
    if valid_indices.size > 0:
        last_valid_index = valid_indices[-1]
        for i in range(1, last_valid_index + 1):
            if perfil_elevacion[i] < perfil_elevacion[i - 1] - 500:
                last_valid_index = i - 1
                break
        perfil_elevacion = perfil_elevacion[:last_valid_index + 1]
        lat_points = lat_points[:last_valid_index + 1]
        lon_points = lon_points[:last_valid_index + 1]

    # Restar offset y pasar a km
    offset = 439.0423493233697
    perfil_elevacion_km = (perfil_elevacion - offset) / 1000.0

    # Distancias al radar para eje X del perfil
    distances = np.array([
        geodesic((start_lat, start_lon), (lat_points[i], lon_points[i])).km for i in range(len(lat_points))
    ], dtype=np.float64)

    # Empezamos con la parte del pseudo corte
    # Hacemos una copia profunda del objeto volumen_radar_data para no modificar el original
    radar_data_copy_3 = copy.deepcopy(volumen_radar_data)

    # Obtener y validar datos de la variable seleccionada del volumen de radar
    data = radar_data_copy_3.fields[variable]['data']

    # Determinamos los valores mínimos y máximos dinámicamente
    vmin = data.min()
    vmax = data.max()

    # Obtenemos unidades de la variable
    units = VARIABLE_UNITS.get(variable, '')

    # Se realiza gráfico del cross section
    xsect = pyart.util.cross_section_ppi(radar_data_copy_3, [radial_angle])
    display = pyart.graph.RadarDisplay(xsect)  # Crear el display de Py-ART

    # GateFilter PARA EL XSECT
    gf_xsect = build_gatefilter(xsect, variable, filters, is_rhi=True)

    # Crear la figura y el subplot
    fig = plt.figure(figsize=[15, 5.5])
    ax2 = plt.subplot(1, 1, 1)

    # Graficar la variable especificada
    display.plot(variable, 0, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax2, mask_outside=True, gatefilter=gf_xsect)
    # Limites solicitados por el usuario (con fallback)
    x_max = min(range_max, plot_max_length_km) if plot_max_length_km else range_max
    y_max = plot_max_height_km if plot_max_height_km else 30
    y_max = max(0.1, min(y_max, 30))  # clamp razonable
    display.set_limits(xlim=[0, x_max], ylim=[0, y_max])
    
    # Ajustar tamaño de fuente del colorbar generado por PyART
    for ax_cbar in fig.axes:
        if ax_cbar != ax2:  # El colorbar es el otro axes
            ax_cbar.tick_params(labelsize=14)  # Tamaño de los valores numéricos del colorbar
            ax_cbar.yaxis.label.set_size(16)  # Tamaño de la etiqueta del colorbar

    # Grafico el perfil de elevación del terreno
    ax2.plot(distances, perfil_elevacion_km, label=None, color='black', linewidth=2)

    # Marcar el punto de interés con un asterisco (punto final)
    distancia_interes = geodesic((start_lat, start_lon), (end_lat, end_lon)).km

    # Interpolar la elevación del punto interés directamente del DEM
    with rasterio.open(tif_path) as src:
        with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
            val = next(vrt.sample([(end_lon, end_lat)]))[0]
            elevacion_interes = np.float32(np.nan if (vrt.nodata is not None and val == vrt.nodata) else val)
    if np.isfinite(elevacion_interes):
        elevacion_interes_value = (elevacion_interes - offset) / 1000.0
        ax2.plot(distancia_interes, elevacion_interes_value, 'r*', markersize=15, label='Punto de interés')
    else:
        elevacion_interes_value = None
        print("No se pudo interpolar la elevación del punto de interés.")

    # Texto con el ángulo radial
    ax2.text(0.98, 0.95, f'Ángulo Radial: {radial_angle:.2f}°',
             horizontalalignment='right', verticalalignment='top',
             transform=ax2.transAxes, fontsize=16, bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Distancia (km)', fontsize=18)
    plt.ylabel(f'Altura (km)', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=14)  # Tamaño de los valores numéricos de los ejes
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

from pyproj import Geod
_GEOD = Geod(ellps="WGS84")

def _km_between(lon0, lat0, lon1, lat1):
  _, _, d = _GEOD.inv(lon0, lat0, lon1, lat1)
  return d / 1000.0

def generate_pseudo_rhi_png(
    filepath: str,
    field: str,
    end_lon: float,
    end_lat: float,
    max_length_km: float,
    max_height_km: float = 20.0,
    elevation: int = 0,
    filters: List[RangeFilter] = [],
    output_dir: str = "app/storage/tmp",
    start_lon: Optional[float] = None,
    start_lat: Optional[float] = None,
    colormap_overrides: Optional[dict] = None,
    session_id: Optional[str] = None,
):
    
    # Crear subdirectorio de sesión si se proporciona session_id
    if session_id:
        output_dir = str(Path(output_dir) / session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    file_hash = md5_file(filepath)[:12]
    filters_str = "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters]) if filters else "nofilter"
    points = (
        f"{start_lon}_{start_lat}__{end_lon}_{end_lat}"
        if (start_lon is not None and start_lat is not None)
        else f"{end_lon}_{end_lat}"
    )
    unique_out_name = f"pseudo_rhi_{field}_{points}_{filters_str}_{elevation}_{int(max_length_km)}km_{int(max_height_km)}km_{file_hash}.png"
    out_path = Path(output_dir) / unique_out_name

    # Construir URL relativa incluyendo session_id si existe
    relative_url = f"static/tmp/{session_id}/{unique_out_name}" if session_id else f"static/tmp/{unique_out_name}"

    if out_path.exists():
        return {"image_url": f"{settings.BASE_URL}/{relative_url}", "metadata": None}

    os.makedirs(output_dir, exist_ok=True)
    radar = pyart.io.read(filepath)
    # radar = radar.extract_sweeps([elevation]) if elevation < radar.nsweeps else radar.extract_sweeps([0])

    field_name, field_key = resolve_field(radar, field)
    
    site_lon, site_lat, site_alt = get_radar_site(radar)
    range_max_km = safe_range_max_m(radar) / 1000.0
    # Ajustar límites a capacidades físicas
    max_length_km = max(0.5, min(max_length_km, range_max_km))
    max_height_km = max(0.5, min(max_height_km, 30.0))

    # Validar puntos dentro del rango máximo
    if _km_between(site_lon, site_lat, end_lon, end_lat) > range_max_km:
        print(f"El punto está fuera del alcance ({range_max_km:.1f} km).")
        raise HTTPException(status_code=400, detail=f"El punto está fuera del alcance ({range_max_km:.1f} km).")
    if start_lon is not None and start_lat is not None:
        dstart = _km_between(site_lon, site_lat, float(start_lon), float(start_lat))
        if dstart > range_max_km:
            raise HTTPException(status_code=400, detail=f"El punto de inicio está fuera del alcance ({range_max_km:.1f} km).")

    # Filtros (se aplican por GateFilter para enmascarar fuera de rango)
    gf = build_gatefilter(radar, field_name, filters, is_rhi=True)

    # Colormap + vmin/vmax con posible override
    cmap_override = (colormap_overrides or {}).get(field, None)
    cmap, vmin, vmax, _ = colormap_for(field, override_cmap=cmap_override)

    # Branch: si hay un punto inicial distinto del radar, generar transecto entre dos puntos
    if start_lon is not None and start_lat is not None:
        # Compare if origin point using geodesic distance (meters)
        # Allow a tolerance to account for front-end selection imprecision (100 m).
        try:
            start_lat_f = float(start_lat)
            start_lon_f = float(start_lon)
        except Exception:
            start_lat_f = float(start_lat)
            start_lon_f = float(start_lon)
        same_origin = geodesic((start_lat_f, start_lon_f), (site_lat, site_lon)).meters <= 1000.0
        if not same_origin:
            _generate_segment_transect_png(
                filepath=filepath,
                radar=radar,
                field_name=field_name,
                file_hash=file_hash,
                start_lon=float(start_lon),
                start_lat=float(start_lat),
                end_lon=float(end_lon),
                end_lat=float(end_lat),
                max_length_km=float(max_length_km),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                output_path=out_path,
                filters=filters,
                max_height_km=max_height_km,
                session_id=session_id,
            )
        else:
            # Caso clásico: pseudo-RHI radial desde el radar
            variable_radar_cross_section(
                site_lat,
                site_lon,
                end_lat, 
                end_lon,
                radar,
                out_path,
                range_max=range_max_km,
                variable=field_name,
                cmap=cmap,
                filters=filters,
                plot_max_length_km=max_length_km,
                plot_max_height_km=max_height_km,
            )
    else:
        # Caso clásico: pseudo-RHI radial desde el radar
        variable_radar_cross_section(
            site_lat,
            site_lon,
            end_lat, 
            end_lon,
            radar,
            out_path,
            range_max=range_max_km,
            variable=field_name,
            cmap=cmap,
            filters=filters,
            plot_max_length_km=max_length_km,
            plot_max_height_km=max_height_km,
        )


    return {
        "image_url": f"{settings.BASE_URL}/{relative_url}",
        "metadata": {
            "radar_site": {"lon": site_lon, "lat": site_lat, "alt_m": site_alt},
            "field": field.upper(),
            "start_point": (
                {"lon": start_lon, "lat": start_lat}
                if (start_lon is not None and start_lat is not None)
                else {"lon": site_lon, "lat": site_lat}
            ),
            "end_point": {"lon": end_lon, "lat": end_lat},
            "max_length_km": max_length_km,
            "max_height_km": max_height_km,
        }
    }


def _generate_segment_transect_png(
    *,
    filepath: str,
    radar: pyart.core.Radar,
    field_name: str,
    file_hash: str,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    max_length_km: Optional[float] = None,
    cmap,
    vmin: Optional[float],
    vmax: Optional[float],
    output_path: Path,
    filters: List[RangeFilter] = [],
    max_height_km: Optional[float] = None,
    session_id: Optional[str] = None,
):
    """
    Genera transecto vertical entre dos puntos arbitrarios (no necesariamente radiales).
    
    1. Construye grilla 3D usando get_or_build_grid3d_with_operator (igual que radar_processor.py).
    2. Samplea la grilla 3D a lo largo de la línea entre los dos puntos con RegularGridInterpolator.
    3. Genera un gráfico similar a variable_radar_cross_section con perfil de terreno.
    """
    from pyproj import Geod
    from ..utils.helpers import extract_metadata_from_filename
    
    _geod = Geod(ellps="WGS84")
    site_lon, site_lat, site_alt = get_radar_site(radar)
    
    # ── Extraer metadata del filename ──
    try:
        radar_name, estrategia, volume, _ = extract_metadata_from_filename(Path(filepath).name)
        if volume is None or radar_name is None or estrategia is None:
            raise ValueError("No se pudo extraer metadata completa del filename")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo extraer metadata del archivo: {e}")
    
    # ── Parámetros de grilla (idéntico a radar_processor.py) ──
    range_max_m = safe_range_max_m(radar)
    
    _, z_top_m, elev_deg = calculate_z_limits(
        range_max_m, elevation=0, cappi_height=4000,
        radar_fixed_angles=radar.fixed_angle['data']
    )
    toa = 12000
    
    grid_resolution_xy, grid_resolution_z = calculate_grid_resolution(volume)
    
    z_grid_limits = (0.0, toa)
    y_grid_limits = (-range_max_m, range_max_m)
    x_grid_limits = (-range_max_m, range_max_m)
    grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits)
    
    z_points, y_points, x_points = calculate_grid_points(
        z_grid_limits, y_grid_limits, x_grid_limits,
        grid_resolution_xy, grid_resolution_z
    )
    grid_shape = (z_points, y_points, x_points)
    
    # ── Separar filtros QC vs visuales ──
    qc_filters, visual_filters = separate_filters(filters, field_name)
    
    # ── Construir grilla 3D (con cache de operador W) ──
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
        weight_func='Barnes2',
        qc_filters=qc_filters,
        session_id=session_id,
    )
    
    # Verificar que el campo solicitado existe en la grilla
    if field_name not in grid.fields:
        available = list(grid.fields.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Campo '{field_name}' no encontrado en grilla. Disponibles: {available}"
        )
    
    # ── Coordenadas de la grilla ──
    x_coords = grid.x['data'].astype(np.float64)  # metros, centrado en radar
    y_coords = grid.y['data'].astype(np.float64)
    z_coords = grid.z['data'].astype(np.float64)
    
    # Datos 3D del campo: shape (1, nz, ny, nx) → (nz, ny, nx)
    field_data_3d = grid.fields[field_name]['data'][0, :, :, :]
    # Convertir masked array a float con NaN
    if hasattr(field_data_3d, 'filled'):
        field_data_3d = field_data_3d.filled(np.nan).astype(np.float64)
    else:
        field_data_3d = np.asarray(field_data_3d, dtype=np.float64)
    
    # ── Limitar línea a max_length_km ──
    end_lon_eff, end_lat_eff, length_km = limit_line_to_range(
        start_lon, start_lat, end_lon, end_lat, max_length_km
    )
    
    # ── Samplear puntos a lo largo de la línea (en lat/lon) ──
    step_m = 150.0 #m
    n_sample = max(2, int((length_km * 1000.0) // step_m) + 1)
    
    az12, _, _ = _geod.inv(start_lon, start_lat, end_lon_eff, end_lat_eff)
    sample_lons = np.empty(n_sample, dtype=np.float64)
    sample_lats = np.empty(n_sample, dtype=np.float64)
    sample_dists_km = np.empty(n_sample, dtype=np.float64)
    
    for i in range(n_sample):
        d_m = i * step_m
        lon_i, lat_i, _ = _geod.fwd(start_lon, start_lat, az12, d_m)
        sample_lons[i], sample_lats[i] = lon_i, lat_i
        sample_dists_km[i] = d_m / 1000.0
    
    # ── Convertir lat/lon a coordenadas de grilla (x, y en metros relativo al radar) ──
    # La grilla usa proyección AEQD centrada en el radar
    # Aproximación precisa: usar pyproj para convertir lat/lon → x,y del radar
    from pyproj import Transformer, CRS
    
    aeqd_crs = CRS.from_dict({
        'proj': 'aeqd',
        'lat_0': site_lat,
        'lon_0': site_lon,
        'datum': 'WGS84',
        'units': 'm',
    })
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84_crs, aeqd_crs, always_xy=True)
    
    sample_x, sample_y = transformer.transform(sample_lons, sample_lats)
    
    z_km = z_coords / 1000.0  # metros → km
    
    # ── Muestreo directo de la grilla 3D en todos los niveles Z ──
    interpolator = RegularGridInterpolator(
        (z_coords, y_coords, x_coords),
        field_data_3d,
        method='nearest',
        bounds_error=False,
        fill_value=np.nan,
    )
    
    # Para cada punto (x_i, y_i) a lo largo de la línea, samplear TODOS los
    # niveles Z de la grilla.  Resultado: imagen 2D (n_z, n_sample).
    n_z = len(z_coords)
    zz = np.repeat(z_coords, n_sample)
    yy = np.tile(sample_y, n_z)
    xx = np.tile(sample_x, n_z)
    pts = np.column_stack([zz, yy, xx])
    
    image = interpolator(pts).reshape(n_z, n_sample)
    
    # ── Aplicar filtros visuales (máscaras post-interpolación) ──
    # Convertir a masked array
    image_ma = np.ma.masked_invalid(image)
    # Aplicar filtros visuales sobre el mismo campo
    image_filtered = apply_visual_filters(image_ma, visual_filters, field_name)
    # Convertir de vuelta: masked values → NaN para matplotlib
    image = np.ma.filled(image_filtered, np.nan)
    
    z_fine_km = z_km  # eje vertical = niveles Z reales de la grilla
    
    # Límite vertical para el gráfico
    y_max_val = max_height_km if max_height_km else float(z_km[-1])
    y_max_val = max(0.1, min(y_max_val, 30.0))
    
    # ── Obtener unidades de la variable ──
    units = VARIABLE_UNITS.get(field_name, '')
    
    # ── Samplear perfil de terreno (DEM) ──
    tif_path = Path(settings.DATA_DIR) / "mosaico_argentina_2.tif"
    offset_dem = 439.0423493233697
    terrain_dists_km = None
    terrain_elev_km = None
    
    try:
        with rasterio.open(tif_path) as src:
            with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
                coords = list(zip(sample_lons, sample_lats))
                elev_raw = np.fromiter(
                    (v[0] for v in vrt.sample(coords)),
                    dtype=np.float32, count=n_sample,
                )
                nodata = vrt.nodata
                if nodata is not None:
                    elev_raw[elev_raw == nodata] = np.nan
                
                terrain_dists_km = sample_dists_km.copy()
                terrain_elev_km = (elev_raw - offset_dem) / 1000.0
    except Exception as e:
        print(f"Warning: No se pudo leer perfil de terreno: {e}")
    
    # ── Elevación del punto final (para marcador) ──
    end_elev_km = None
    try:
        with rasterio.open(tif_path) as src:
            with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
                val = next(vrt.sample([(end_lon_eff, end_lat_eff)]))[0]
                if vrt.nodata is not None and val == vrt.nodata:
                    val = np.nan
                if np.isfinite(val):
                    end_elev_km = (float(val) - offset_dem) / 1000.0
    except Exception:
        pass
    
    # ── Graficar con forma de haz ──
    fig = plt.figure(figsize=[15, 5.5])
    ax = plt.subplot(1, 1, 1)
    
    # Meshgrid sobre grilla fina
    D, Z = np.meshgrid(sample_dists_km, z_fine_km)
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(D, Z, image, cmap=cmap, norm=norm, shading='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f'{field_name} ({units})', fontsize=16)
    cbar.ax.tick_params(labelsize=14)  # Tamaño de los valores numéricos del colorbar
    
    # Perfil de terreno
    if terrain_dists_km is not None and terrain_elev_km is not None:
        ax.plot(terrain_dists_km, terrain_elev_km, color='black', linewidth=2)
        ax.fill_between(
            terrain_dists_km, -1, terrain_elev_km,
            color='saddlebrown', alpha=0.6,
        )
    
    # Marcador del punto final
    dist_end_km = length_km
    if end_elev_km is not None:
        ax.plot(dist_end_km, end_elev_km, 'r*', markersize=15, label='Punto final')
    
    # Texto informativo con azimut y distancia
    azimuth_deg = az12 % 360
    info_text = f'Azimut: {azimuth_deg:.1f}°  |  Dist: {length_km:.1f} km'
    ax.text(
        0.98, 0.95, info_text,
        horizontalalignment='right', verticalalignment='top',
        transform=ax.transAxes, fontsize=16,
        bbox=dict(facecolor='white', alpha=0.7),
    )
    
    # Límites
    x_max = length_km
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max_val)
    
    ax.set_xlabel('Distancia (km)', fontsize=18)
    ax.set_ylabel('Altura (km)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)  # Tamaño de los valores numéricos de los ejes
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
