import os
import pyart
import copy
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from fastapi import HTTPException
from pathlib import Path
from typing import List, Optional
from geopy.distance import geodesic
from rasterio.vrt import WarpedVRT
from scipy.interpolate import RegularGridInterpolator
from rasterio.enums import Resampling
from ..core.constants import VARIABLE_UNITS
from ..core.config import settings
from ..core.cache import GRID3D_CACHE

from ..models import RangeFilter
from ..services.grid_geometry import beam_height_max_km
from .radar_common import (
    resolve_field, colormap_for, build_gatefilter,
    safe_range_max_m, get_radar_site, md5_file, limit_line_to_range,
    normalize_proj_dict, grid3d_cache_key, qc_signature,
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
    tif_path = Path("app/storage/data/mosaico_argentina_2.tif")

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
    y_max = max(0.5, min(y_max, 30))  # clamp razonable
    display.set_limits(xlim=[0, x_max], ylim=[-0.5, y_max])

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
             transform=ax2.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Distancia (km)', fontsize=14)
    plt.ylabel(f'Altura (km)', fontsize=14)
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
    Genera transecto vertical entre dos puntos usando GridMapDisplay.plot_cross_section
    con grilla 3D cacheada para máxima calidad y performance.
    """
    from pyproj import Geod
    
    site_lon, site_lat, _ = get_radar_site(radar)
    _geod = Geod(ellps="WGS84")
    
    # 1) Buscar grilla 3D en cache (debe existir de process_radar_to_cog previo)
    from ..utils.helpers import extract_metadata_from_filename
    from ..core.constants import AFFECTS_INTERP_FIELDS
    
    # Extraer volumen del nombre del archivo (crítico para cache key)
    try:
        _,_,volume,_ = extract_metadata_from_filename(Path(filepath).name)
        if volume is None:
            raise ValueError("No se pudo extraer volume del filename")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo extraer metadata del archivo: {e}"
        )
    
    # Separar filtros QC (igual que en radar_processor)
    qc_filters = []
    for f in (filters or []):
        ffield = str(getattr(f, "field", "") or "").upper()
        if ffield in AFFECTS_INTERP_FIELDS:
            qc_filters.append(f)
    
    # Calcular range_max_m del radar (igual que en radar_processor)
    range_max_m = safe_range_max_m(radar)
    
    # Calcular z_top_m dinámicamente (DEBE coincidir con radar_processor)
    # Nota: Para pseudo-RHI asumimos producto tipo PPI con elevación 0
    # Si el usuario procesó otro producto, puede no encontrar cache
    elev_deg = 0.0  # Elevación default para pseudo-RHI
    if radar.nsweeps > 0:
        try:
            elev_deg = float(radar.fixed_angle['data'][0])
        except Exception:
            elev_deg = 0.0
    
    hmax_km = beam_height_max_km(range_max_m, elev_deg)
    z_top_m = int((hmax_km + 3) * 1000)  # +3 km de margen (igual que PPI en radar_processor)
    
    # Generar cache key para buscar la grilla 3D multi-campo
    qc_sig = qc_signature(qc_filters)
    grid_resolution_xy = 300 if volume == '03' else 1200
    grid_resolution_z = 300  # Siempre 300m en Z (debe coincidir con radar_processor)
    
    cache_key = grid3d_cache_key(
        file_hash=file_hash,
        volume=volume,
        qc_sig=qc_sig,
        grid_res_xy=grid_resolution_xy,
        grid_res_z=grid_resolution_z,
        z_top_m=z_top_m,
        session_id=session_id,
    )
    
    # Buscar en cache
    pkg_cached = GRID3D_CACHE.get(cache_key)
    
    if pkg_cached is None:
        raise HTTPException(
            status_code=400,
            detail="No se encontró grilla 3D en cache. Debe procesarse primero un producto 2D (PPI/CAPPI/COLMAX) para generar la grilla 3D."
        )
    
    # Reconstruir Grid multi-campo desde cache
    fields_dict = {}
    for fname, fdata in pkg_cached["fields"].items():
        field_dict = fdata["metadata"].copy()
        field_dict['data'] = fdata["data"]
        fields_dict[fname] = field_dict
    
    # Verificar que el campo solicitado existe en la cache
    if field_name not in fields_dict:
        available = list(fields_dict.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Campo '{field_name}' no encontrado en grilla cacheada. Disponibles: {available}"
        )
    
    # Crear Grid con TODOS los campos cacheados
    grid = pyart.core.Grid(
        time={
            'data': np.array([0]),
            'units': 'seconds since 2000-01-01T00:00:00Z',
            'calendar': 'gregorian',
            'standard_name': 'time'
        },
        fields=fields_dict,
        metadata={'instrument_name': 'RADAR'},
        origin_latitude={'data': radar.latitude['data']},
        origin_longitude={'data': radar.longitude['data']},
        origin_altitude={'data': radar.altitude['data']},
        x={'data': pkg_cached["x"]},
        y={'data': pkg_cached["y"]},
        z={'data': pkg_cached["z"]},
    )
    grid.projection = pkg_cached["projection"]

    # DEBUG: verificar que la grilla sea verdaderamente 3D
    try:
        arr3d = grid.fields[field_name]['data']
        zvals = grid.z['data']
        print(f"DEBUG grid field shape: {getattr(arr3d, 'shape', None)}, z shape: {getattr(zvals, 'shape', None)}")
        print(f"DEBUG z range (m): min={float(np.nanmin(zvals))}, max={float(np.nanmax(zvals))}")
        if getattr(arr3d, 'ndim', 0) != 3 or getattr(zvals, 'size', 0) <= 1:
            raise HTTPException(
                status_code=400,
                detail="La grilla 3D cacheada tiene un solo nivel z o no es 3D. Procesá PPI/COLMAX primero para cachear grilla 3D."
            )
    except Exception as e:
        print(f"WARNING: No se pudo verificar dimensiones de grilla: {e}")
    
    # 2) Limitar línea a max_length_km
    end_lon_eff, end_lat_eff, length_km = limit_line_to_range(
        start_lon, start_lat, end_lon, end_lat, max_length_km
    )
    
    # 3) Calcular distancia acumulada para eje X
    _, _, dist_total_m = _geod.inv(start_lon, start_lat, end_lon_eff, end_lat_eff)
    dkm = dist_total_m / 1000.0
    
    # 4) Usar GridMapDisplay.plot_cross_section
    fig = plt.figure(figsize=[15, 5.5])
    ax = plt.subplot(1, 1, 1)
    
    display = pyart.graph.GridMapDisplay(grid)
    
    # plot_cross_section espera (start_lat, start_lon) y (end_lat, end_lon)
    # IMPORTANTE: Usar cached_field_name (el que está en el grid) no field_name
    try:
        display.plot_cross_section(
            field_name,  # Usar el nombre del campo que realmente está en el grid
            (start_lat, start_lon),
            (end_lat_eff, end_lon_eff),
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            #mask_outside=False,  # No enmascarar fuera del haz para ver toda la interpolación
        )
    except Exception as e:
        # Si falla plot_cross_section, intentar fallback manual
        print(f"Warning: plot_cross_section falló ({e}), usando fallback")
        import traceback
        traceback.print_exc()
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", fontsize=12)

    # Ajustar límites verticales basados en niveles z reales (m -> km) y asegurar terreno visible
    try:
        zvals = grid.z['data']
        zmax_km = float(np.nanmax(zvals)) / 1000.0
        y_max_km = max_height_km if max_height_km else zmax_km
        # expand slightly to include terrain curve
        y_max_km = max(0.5, min(y_max_km, max(zmax_km, 30.0)))
        ax.set_ylim(0.0, y_max_km)
        ax.set_aspect('auto')
        print(f"DEBUG set ylim: 0.0 to {y_max_km} km (zmax_km={zmax_km})")
    except Exception as e:
        print(f"WARNING: No se pudo ajustar límites verticales: {e}")
    
    # 5) Agregar perfil de terreno
    tif_path = Path("app/storage/data/mosaico_argentina_2.tif")
    try:
        # Samplear terreno a lo largo de la línea
        step_m = 500.0  # puntos cada 500m
        n_pts = max(2, int((length_km * 1000.0) // step_m) + 1)
        lons = np.empty(n_pts, dtype=np.float64)
        lats = np.empty(n_pts, dtype=np.float64)
        
        az12, _, _ = _geod.inv(start_lon, start_lat, end_lon_eff, end_lat_eff)
        for i in range(n_pts):
            d = i * step_m
            lon_i, lat_i, _ = _geod.fwd(start_lon, start_lat, az12, d)
            lons[i], lats[i] = lon_i, lat_i
        
        with rasterio.open(tif_path) as src:
            with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
                coords = list(zip(lons, lats))
                elev = np.fromiter((v[0] for v in vrt.sample(coords)), dtype=np.float32, count=len(coords))
                nodata = vrt.nodata
                if nodata is not None:
                    elev[elev == nodata] = np.nan
                
                # Calcular distancias acumuladas
                dkm_terrain = np.zeros(n_pts, dtype=np.float64)
                for i in range(1, n_pts):
                    _, _, dd = _geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
                    dkm_terrain[i] = dkm_terrain[i-1] + dd / 1000.0
                
                offset = 439.0423493233697
                ground_km = (elev - offset) / 1000.0
                ax.plot(dkm_terrain, ground_km, color="black", linewidth=2, label="Terreno")
    except Exception as e:
        print(f"Warning: No se pudo graficar terreno: {e}")
    
    # 6) Configurar ejes y límites
    ax.set_xlabel("Distancia (km)", fontsize=14)
    ax.set_ylabel("Altura (km)", fontsize=14)
    
    if max_height_km:
        ax.set_ylim(0, max_height_km)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
