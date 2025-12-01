"""
Servicio para generar perfiles de elevación a partir de coordenadas geográficas.
Utiliza el DEM (Digital Elevation Model) de Argentina para extraer las elevaciones.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import List, Dict, Tuple
from math import radians, cos, sin, asin, sqrt
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calcula la distancia entre dos puntos en la superficie terrestre usando la fórmula de Haversine.
    
    Args:
        lon1, lat1: Coordenadas del primer punto (en grados)
        lon2, lat2: Coordenadas del segundo punto (en grados)
    
    Returns:
        Distancia en kilómetros
    """
    # Convertir grados a radianes
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Fórmula de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radio de la Tierra en kilómetros
    r = 6371
    
    return c * r


def interpolate_line_points(coordinates: List[Dict], points_per_km: int = 10) -> List[Dict]:
    """
    Interpola puntos adicionales entre los puntos de la línea para mayor resolución.
    
    Args:
        coordinates: Lista de diccionarios con 'lat' y 'lon'
        points_per_km: Número de puntos por kilómetro
    
    Returns:
        Lista de coordenadas interpoladas
    """
    if len(coordinates) < 2:
        return coordinates
    
    interpolated = []
    
    for i in range(len(coordinates) - 1):
        p1 = coordinates[i]
        p2 = coordinates[i + 1]
        
        # Calcular distancia entre puntos
        dist_km = haversine_distance(p1['lon'], p1['lat'], p2['lon'], p2['lat'])
        
        # Número de puntos a interpolar
        num_points = max(2, int(dist_km * points_per_km))
        
        # Interpolar
        for j in range(num_points):
            t = j / num_points
            lat = p1['lat'] + t * (p2['lat'] - p1['lat'])
            lon = p1['lon'] + t * (p2['lon'] - p1['lon'])
            interpolated.append({'lat': lat, 'lon': lon})
    
    # Agregar el último punto
    interpolated.append(coordinates[-1])
    
    return interpolated


def extract_elevation_profile(
    coordinates: List[Dict],
    dem_path: Path,
    interpolate: bool = True,
    points_per_km: int = 10
) -> Dict:
    """
    Extrae el perfil de elevación para una línea de coordenadas.
    Usa WarpedVRT para muestreo eficiente del DEM pesado (optimización como en pseudo_rhi).
    
    Args:
        coordinates: Lista de diccionarios con 'lat' y 'lon'
        dem_path: Ruta al archivo DEM (GeoTIFF)
        interpolate: Si True, interpola puntos adicionales
        points_per_km: Puntos por kilómetro al interpolar
    
    Returns:
        Diccionario con:
        - profile: Lista de puntos con distance (km), elevation (m), lat, lon
    """
    if len(coordinates) < 2:
        raise ValueError("Se requieren al menos 2 coordenadas para generar un perfil")
    
    # Interpolar puntos si es necesario
    if interpolate:
        coords = interpolate_line_points(coordinates, points_per_km)
    else:
        coords = coordinates
    
    # Usar WarpedVRT para muestreo eficiente (como en pseudo_rhi)
    with rasterio.open(dem_path) as src:
        with WarpedVRT(src, resampling=Resampling.nearest, add_alpha=False) as vrt:
            profile = []
            cumulative_distance = 0.0
            
            # Preparar lista de coordenadas para muestreo en batch
            coords_list = [(coord['lon'], coord['lat']) for coord in coords]
            
            # Muestrear todas las elevaciones de una vez (mucho más rápido)
            elevations = np.fromiter(
                (v[0] for v in vrt.sample(coords_list)),
                dtype=np.float32,
                count=len(coords_list)
            )
            
            # Manejar valores nodata -> NaN
            nodata = vrt.nodata
            if nodata is not None:
                mask = elevations == nodata
                if mask.any():
                    elevations[mask] = np.nan
            
            # Restar offset (mismo que en pseudo_rhi para consistencia)
            offset = 439.0423493233697
            elevations = elevations - offset
            
            # Construir perfil con distancias
            for i, (coord, elevation) in enumerate(zip(coords, elevations)):
                lat = coord['lat']
                lon = coord['lon']
                
                # Calcular distancia acumulada
                if i > 0:
                    prev_coord = coords[i - 1]
                    segment_dist = haversine_distance(
                        prev_coord['lon'], prev_coord['lat'],
                        lon, lat
                    )
                    cumulative_distance += segment_dist
                
                profile.append({
                    'distance': cumulative_distance,
                    'elevation': float(elevation) if np.isfinite(elevation) else None,
                    'lat': lat,
                    'lon': lon
                })
    
    # Filtrar puntos sin elevación
    valid_profile = [p for p in profile if p['elevation'] is not None]
    
    if not valid_profile:
        return {
            'profile': [],
        }
    
    return {
        'profile': valid_profile,
    }
