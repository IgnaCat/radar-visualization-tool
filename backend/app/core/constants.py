
FIELD_ALIASES = {
    "DBZH": ["DBZH", "corrected_reflectivity_horizontal"],  # Reflectividad horizontal
    "DBZV": ["DBZV", "corrected_reflectivity_vertical"],    # Reflectividad vertical
    "DBZHF": ["DBZHF", "DBZHf"],
    "ZDR":  ["ZDR", "zdr"],                                 # Diferencia de reflectividad
    "RHOHV":["RHOHV","rhohv"],                              # Correlación cruzada
    "KDP":  ["KDP","kdp"],                                  # Diferencial de fase
    "VRAD": ["VRAD","velocity","corrected_velocity"],       # Velocidad radial
    "WRAD": ["WRAD","spectrum_width"],                      # Velocidad radial del viento
    "PHIDP":["PHIDP","differential_phase"],                 # Fase diferencial específica
}

FIELD_RENDER = { 
    "DBZH": {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"},
    "DBZHF": {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"},
    "DBZV": {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"},
    "ZDR": {"vmin": -5.0, "vmax": 10.5, "cmap": "grc_zdr2"}, 
    "RHOHV": {"vmin": 0.0, "vmax": 1.0, "cmap": "grc_rho"}, 
    "KDP": {"vmin": 0.0, "vmax": 8.0, "cmap": "grc_rain"},
    "VRAD": {"vmin": -35.0, "vmax": 35.0, "cmap": "NWSVel"},
    "WRAD": {"vmin": 0.0, "vmax": 10.0, "cmap": "Oranges"},
    "PHIDP": {"vmin": 0.0, "vmax": 360.0, "cmap": "Theodore16"},
}

VARIABLE_UNITS = {
    'WRAD': 'm/s',
    'KDP': 'deg/km',
    'DBZV': 'dBZ',
    'DBZH': 'dBZ',
    'ZDR': 'dBZ',
    'VRAD': 'm/s',
    'RHOHV': '',
    'PHIDP': 'deg'
}

# Opciones de colormaps disponibles por campo
FIELD_COLORMAP_OPTIONS = {
    "DBZH": ["grc_th", "grc_th2", "grc_rain", "pyart_NWSRef", "pyart_HomeyerRainbow"],
    "DBZHF": ["grc_th", "grc_th2", "grc_rain", "pyart_NWSRef", "pyart_HomeyerRainbow"],
    "DBZV": ["grc_th", "grc_th2", "grc_rain", "pyart_NWSRef", "pyart_HomeyerRainbow"],
    "ZDR": ["grc_zdr2", "grc_zdr", "pyart_RefDiff", "pyart_Theodore16"],
    "RHOHV": ["grc_rho", "pyart_RefDiff", "Greys", "viridis"],
    "KDP": ["grc_rain", "grc_th", "pyart_Theodore16", "plasma"],
    "VRAD": ["NWSVel", "pyart_BuDRd18", "seismic", "RdBu_r"],
    "WRAD": ["Oranges", "YlOrRd", "hot", "plasma"],
    "PHIDP": ["Theodore16", "hsv", "twilight", "twilight_shifted"],
}

AFFECTS_INTERP_FIELDS = {"RHOHV"}

# Parámetros adaptativos de ROI por altura Z para interpolación Barnes2
# Formato: (z_threshold_m, (h_factor, nb, bsp, min_radius))
# Niveles bajos usan parámetros más agresivos, niveles altos más conservadores
# IMPORTANTE: Cambiar estos valores invalida el caché de operadores W
ADAPTIVE_ROI_PARAMS = [
    (2000,  (0.9, 1.2, 1.0, 350.0)),
    (4000,  (0.8, 1.0, 0.9, 350.0)),
    (6000,  (0.6, 0.8, 0.7, 300.0)),
    (9000,  (0.4, 0.7, 0.6, 300.0)),
    (float('inf'), (0.3, 0.5, 0.5, 200.0)),
]