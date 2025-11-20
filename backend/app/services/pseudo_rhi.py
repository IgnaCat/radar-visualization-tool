import os
import pyart
import copy
import numpy as np
import rasterio
from matplotlib import pyplot as plt
# matplotlib.rcParams['pcolormesh.shading'] = 'auto' # evita el error de pcolormesh
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

from ..schemas import RangeFilter
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
    gf=None,
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
    # Datos de la variable seleccionada del volumen de radar
    data = radar_data_copy_3.fields[variable]['data']

    # Determinamos los valores mínimos y máximos dinámicamente
    vmin = data.min()
    vmax = data.max()

    # Obtenemos unidades de la variable
    units = VARIABLE_UNITS.get(variable, '')

    # Se realiza gráfico del cross section
    xsect = pyart.util.cross_section_ppi(radar_data_copy_3, [radial_angle])
    display = pyart.graph.RadarDisplay(xsect)  # Crear el display de Py-ART

    # Crear la figura y el subplot
    fig = plt.figure(figsize=[15, 5.5])
    ax2 = plt.subplot(1, 1, 1)

    # Graficar la variable especificada
    display.plot(variable, 0, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax2, mask_outside=True, gatefilter=gf)
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

    plt.xlabel('Distancia al radar (km)', fontsize=14)
    plt.ylabel(f'Altura (km) - {units}', fontsize=14)
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
):
    
    file_hash = md5_file(filepath)[:12]
    filters_str = "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters]) if filters else "nofilter"
    points = (
        f"{start_lon}_{start_lat}__{end_lon}_{end_lat}"
        if (start_lon is not None and start_lat is not None)
        else f"{end_lon}_{end_lat}"
    )
    unique_out_name = f"pseudo_rhi_{field}_{points}_{filters_str}_{elevation}_{int(max_length_km)}km_{int(max_height_km)}km_{file_hash}.png"
    out_path = Path(output_dir) / unique_out_name

    if out_path.exists():
        return {"image_url": f"{settings.BASE_URL}/static/tmp/{unique_out_name}", "metadata": None}

    os.makedirs(output_dir, exist_ok=True)
    radar = pyart.io.read(filepath)
    # radar = radar.extract_sweeps([elevation]) if elevation < radar.nsweeps else radar.extract_sweeps([0])

    field_name, _ = resolve_field(radar, field)
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

    # Colormap + vmin/vmax
    cmap, vmin, vmax, _ = colormap_for(field)

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
                gf=gf,
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
            gf=gf,
            plot_max_length_km=max_length_km,
            plot_max_height_km=max_height_km,
        )


    return {
        "image_url": f"{settings.BASE_URL}/static/tmp/{unique_out_name}",
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
    radar: pyart.core.Radar,
    field_name: str,
    file_hash: str,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    max_length_km: float,
    cmap,
    vmin: Optional[float],
    vmax: Optional[float],
    output_path: Path,
    filters: List[RangeFilter] = [],
    max_height_km: float = 20.0,
):
    """
    Genera una "cortina" vertical entre dos puntos arbitrarios dentro del alcance del radar:
    eje X = distancia a lo largo del segmento start→end, eje Y = altura, colores = campo del radar.

    Implementación: crea una grilla 3D con pyart.map.grid_from_radars y re-muestrea
    a lo largo del plano definido por el segmento, usando RegularGridInterpolator.
    """
    # 1) Preparar grilla 3D (reusar cache si existe)
    site_lon, site_lat, _ = get_radar_site(radar)
    rng_max_m = safe_range_max_m(radar)
    z_top_m = max(2000.0, float(max_height_km) * 1000.0)  # limitar grilla a altura solicitada (mín 2 km)

    # Resolución: balance entre detalle y performance
    grid_res_xy = 1000.0  # 1.0 km
    grid_res_z = 300.0    # 300 m

    nx = int(rng_max_m / grid_res_xy)
    ny = int(rng_max_m / grid_res_xy)
    nz = int(z_top_m / grid_res_z) + 1

    nx = max(nx, 100)
    ny = max(ny, 100)
    nz = max(nz, 40)

    # Cache key (3D)
    sig = qc_signature(filters)
    key3d = grid3d_cache_key(
        file_hash=file_hash,
        field_to_use=field_name,
        volume=None,
        qc_sig=sig,
        grid_res_xy=grid_res_xy,
        grid_res_z=grid_res_z,
        z_top_m=z_top_m,
    )

    pkg3d = GRID3D_CACHE.get(key3d)
    if pkg3d is None:
        gf = build_gatefilter(radar, field_name, filters, is_rhi=True)
        grid = pyart.map.grid_from_radars(
            radar,
            grid_shape=(nz, ny, nx),
            grid_limits=((0.0, z_top_m), (-rng_max_m, rng_max_m), (-rng_max_m, rng_max_m)),
            grid_origin=(site_lat, site_lon),
            weighting_function="nearest",
            gatefilters=gf,
            roi_func="dist",
            z_factor=0.0,
            xy_factor=0.02,
            min_radius=max(800.0, 1.2 * grid_res_xy),
        )
        arr3d = np.ma.array(grid.fields[field_name]["data"], copy=False)
        z = np.asarray(grid.z["data"], dtype=np.float32)
        y = np.asarray(grid.y["data"], dtype=np.float32)
        x = np.asarray(grid.x["data"], dtype=np.float32)
        import pyproj
        proj_dict = normalize_proj_dict(grid, (site_lat, site_lon))
        crs_wkt = pyproj.CRS.from_dict(proj_dict).to_wkt()
        pkg3d = {"arr3d": arr3d, "z": z, "y": y, "x": x, "crs": crs_wkt}
        GRID3D_CACHE[key3d] = pkg3d
    else:
        arr3d = pkg3d["arr3d"]
        z = pkg3d["z"]
        y = pkg3d["y"]
        x = pkg3d["x"]
        crs_wkt = pkg3d.get("crs")

    # 2) Construir la polilínea start→end limitada por max_length_km
    end_lon_eff, end_lat_eff, length_km = limit_line_to_range(
        start_lon, start_lat, end_lon, end_lat, max_length_km
    )
    # muestreo cada ~150 m
    step_m = 150.0
    n_pts = max(2, int((length_km * 1000.0) // step_m) + 1)
    lons = np.empty(n_pts, dtype=np.float64)
    lats = np.empty(n_pts, dtype=np.float64)
    from pyproj import Geod
    _geod = Geod(ellps="WGS84")
    az12, _, dist_m = _geod.inv(start_lon, start_lat, end_lon_eff, end_lat_eff)
    for i in range(n_pts):
        d = i * step_m
        lon_i, lat_i, _ = _geod.fwd(start_lon, start_lat, az12, d)
        lons[i], lats[i] = lon_i, lat_i

    # 3) Transformar a CRS de la grilla (x,y en metros)
    import pyproj
    crs = pyproj.CRS.from_wkt(crs_wkt) if 'crs_wkt' in locals() and crs_wkt else pyproj.CRS.from_epsg(4326)
    tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tf.transform(lons, lats)

    # 4) Interpolador 3D (z,y,x). Usamos nearest para respetar máscaras/mejores bordes
    data = np.asarray(arr3d.filled(np.nan), dtype=np.float32)
    rgi = RegularGridInterpolator(
        (z.astype(np.float32), y.astype(np.float32), x.astype(np.float32)),
        data,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Construimos consultas para toda la "cortina": ZxN
    Z = z.astype(np.float32)
    N = xs.size
    ZZ, NN = np.meshgrid(Z, np.arange(N), indexing="ij")  # (nz, N)
    pts = np.column_stack((ZZ.ravel(), ys[NN].ravel(), xs[NN].ravel()))
    vals = rgi(pts).reshape(ZZ.shape)

    # 5) Perfil de terreno a lo largo de la línea (opcional, como en radial)
    tif_path = Path("app/storage/data/mosaico_argentina_2.tif")
    ground_km = None
    try:
        with rasterio.open(tif_path) as src:
            with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
                coords = list(zip(lons, lats))
                elev = np.fromiter((v[0] for v in vrt.sample(coords)), dtype=np.float32, count=len(coords))
                nodata = vrt.nodata
                if nodata is not None:
                    elev[elev == nodata] = np.nan
                offset = 439.0423493233697
                ground_km = (elev - offset) / 1000.0
    except Exception:
        ground_km = None

    # Distancias acumuladas para eje X (km)
    dkm = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        _, _, dd = _geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
        dkm[i] = dkm[i-1] + dd / 1000.0

    # 6) Graficar
    fig = plt.figure(figsize=[15, 5.5])
    ax = plt.subplot(1, 1, 1)

    # imshow con extent (x: km, y: km)
    # Recortar según altura solicitada
    if Z[-1] / 1000.0 > max_height_km:
        clip_idx = np.searchsorted(Z, max_height_km * 1000.0, side="right")
        Z = Z[:clip_idx]
        vals = vals[:clip_idx, :]

    im = ax.imshow(
        vals,
        origin="lower",
        aspect="auto",
        extent=[0, float(dkm[-1]), float(Z[0]/1000.0), float(Z[-1]/1000.0)],
        cmap=cmap,
        vmin=float(vmin) if np.isfinite(vmin) else None,
        vmax=float(vmax) if np.isfinite(vmax) else None,
    )
    if ground_km is not None:
        ax.plot(dkm, ground_km, color="black", linewidth=2)

    ax.set_xlabel("Distancia (km)")
    ax.set_ylabel("Altura (km)")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(VARIABLE_UNITS.get(field_name, ""))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
