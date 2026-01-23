"""
Utilidades para cálculo de geometría de grillas 3D.
Incluye límites espaciales, resolución, roi y altura del haz del radar.
"""

import math
import numpy as np


def beam_height_max_km(range_max_m: float, elev_deg: float, antenna_alt_m: float = 0.0) -> float:
    """
    Calcula la altura máxima del haz en km para un rango y elevación dados.
    
    Args:
        range_max_m: Rango máximo del radar en metros
        elev_deg: Ángulo de elevación en grados
        antenna_alt_m: Altura de la antena sobre el suelo en metros (default: 0)
    
    Returns:
        Altura máxima del haz en kilómetros
    """
    Re = 8.49e6  # Radio efectivo de la Tierra en metros
    r = float(range_max_m)
    th = math.radians(float(elev_deg))
    h = r * math.sin(th) + (r * r) / (2.0 * Re) + antenna_alt_m
    return h / 1000.0  # Convertir a km


def calculate_z_limits(
    range_max_m: float,
    elevation: int = 0,
    cappi_height: float = 4000,
    radar_fixed_angles=None
) -> tuple[float, float, float | None]:
    """
    Calcula límites verticales (z_min, z_max).
    
    Args:
        range_max_m: Rango máximo del radar en metros
        elevation: Índice de elevación (para PPI)
        cappi_height: Altura CAPPI en metros (para CAPPI)
        radar_fixed_angles: Array con ángulos de elevación fijos del radar
    
    Returns:
        Tupla (z_min, z_max, elev_deg) donde:
            - z_min: Altura mínima en metros (siempre 0.0)
            - z_max: Altura máxima en metros
            - elev_deg: Ángulo de elevación usado (None para CAPPI)
    """
    # if product_upper == "CAPPI":
    #     z_top_m = cappi_height + 2000  # +2 km de margen
    #     elev_deg = None
    # No hago diferenciacion por vista (PPI, etc) para manejar siempre la misma grilla 3d
        

    if radar_fixed_angles is None:
        raise ValueError("radar_fixed_angles requerido para PPI/COLMAX")
    
    elev_deg = float(radar_fixed_angles[elevation])
    hmax_km = beam_height_max_km(range_max_m, elev_deg)
    z_top_m = int((hmax_km + 3) * 1000)  # +3 km de margen

    
    return (0.0, z_top_m, elev_deg)


def calculate_grid_resolution(volume: str | None) -> tuple[float, float]:
    """
    Calcula resolución XY y Z para la grilla según el volumen del radar.
    
    Args:
        volume: Identificador del volumen del radar ('03' tiene mayor resolución)
    
    Returns:
        Tupla (grid_resolution_xy, grid_resolution_z) en metros:
            - grid_resolution_xy: Resolución horizontal (depende del volumen)
            - grid_resolution_z: Resolución vertical (siempre 300m para cross-sections)
    """
    # XY depende del volumen, pero Z siempre usa resolución fina para transectos suaves
    grid_resolution_xy = 300 if volume == '03' else 1200
    grid_resolution_z = 1000  # Siempre usar 300m en Z para cross-sections de calidad
    
    return grid_resolution_xy, grid_resolution_z


def calculate_grid_points(
    z_limits: tuple[float, float],
    y_limits: tuple[float, float],
    x_limits: tuple[float, float],
    resolution_xy: float,
    resolution_z: float
) -> tuple[int, int, int]:
    """
    Calcula número de puntos de la grilla en cada dimensión.
    
    Args:
        z_limits: Tupla (z_min, z_max) en metros
        xy_limits: Tupla (min, max) para X e Y en metros (asume cuadrado)
        resolution_xy: Resolución horizontal en metros
        resolution_z: Resolución vertical en metros
    
    Returns:
        Tupla (z_points, y_points, x_points) con cantidad de puntos en cada eje
    """
    z_points = int(np.ceil(z_limits[1] / resolution_z)) + 1
    y_points = int((y_limits[1] - y_limits[0]) / resolution_xy)
    x_points = int((x_limits[1] - x_limits[0]) / resolution_xy)
    
    return z_points, y_points, x_points


def calculate_roi_dist_beam(
    z_coords,
    y_coords,
    x_coords,
    h_factor: float = 1.0,
    nb: float = 1.5,
    bsp: float = 1.0,
    min_radius: float = 300.0,
    radar_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
):
    """
    Calcula el Radio de Influencia (ROI) usando el método dist_beam que simula
    el ensanchamiento geométrico del haz del radar con la distancia.
    
    El ROI crece con la distancia del radar basándose en:
    - Altura del voxel (componente vertical)
    - Distancia horizontal del voxel (componente que simula apertura del haz)
    
    Fórmula dist_beam (PyART):
        ROI = max(h_factor*(z/20) + distancia_xy*tan(nb*bsp*π/180), min_radius)
    
    Args:
        z_coords: Coordenada(s) Z en metros (escalar o array)
        y_coords: Coordenada(s) Y en metros (escalar o array)
        x_coords: Coordenada(s) X en metros (escalar o array)
        h_factor: Factor de escalado de altura (default 1.0)
            - Controla cuánto contribuye la altura al ROI
            - Dividido por 20 para evitar que domine sobre componente horizontal
        nb: Ancho de haz virtual en grados (default 1.5)
            - 0.5-1.0°: Radares de investigación alta resolución
            - 1.5°: Radares meteorológicos operacionales (recomendado)
            - 2.0-3.0°: Radares antiguos o baja resolución
        bsp: Espaciado entre haces (default 1.0)
            - Multiplica el ancho del haz para simular solapamiento
        min_radius: Radio mínimo en metros (default 800.0)
            - Garantiza ROI mínimo cerca del radar donde z y xy son pequeños
            - Previene ROI demasiado pequeño que causaría voxels sin datos
        radar_offset: Offset (z, y, x) del centro del radar en metros (default origen)
    
    Returns:
        ROI en metros (mismo shape que inputs: escalar o array)
    """
    # Convertir a arrays para operaciones vectorizadas
    z = np.asarray(z_coords, dtype=np.float32)
    y = np.asarray(y_coords, dtype=np.float32)
    x = np.asarray(x_coords, dtype=np.float32)
    
    # Aplicar offset del radar (normalmente en el origen)
    z_rel = z - radar_offset[0]
    y_rel = y - radar_offset[1]
    x_rel = x - radar_offset[2]
    
    # Componente vertical: altura dividida por 20 para evitar que domine
    vertical_component = h_factor * (z_rel / 20.0)
    
    # Componente horizontal: distancia XY multiplicada por tangente del ángulo del haz
    # tan(nb * bsp * π/180) simula el ensanchamiento del haz con la distancia
    xy_distance = np.sqrt(x_rel**2 + y_rel**2)
    beam_angle_rad = nb * bsp * np.pi / 180.0
    horizontal_component = xy_distance * np.tan(beam_angle_rad)
    
    # ROI total: suma de componentes con límite mínimo
    roi = np.maximum(
        vertical_component + horizontal_component,
        min_radius
    )
    
    return roi
