
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

# Parámetros de ROI por volumen para interpolación Barnes2 con max_neighbors
# Formato: (h_factor, nb, bsp, min_radius)
# Valores constantes por volumen (sin variación por altura Z)
# Usa valores anteriormente aplicados a <2000m, ahora con max_neighbors limitando vecinos
# IMPORTANTE: Cambiar estos valores invalida el caché de operadores W (hace que se recalculen)

# Volumen 01: Parámetros base
ROI_PARAMS_VOL01 = (0.9, 1.2, 1.0, 700.0)

# Volumen 02: Escaneo estándar, alcance medio
ROI_PARAMS_VOL02 = (1.1, 1.6, 1.3, 900.0)

# Volumen 03: Bird bath (scan vertical ~90° con 360 azimuts)
# ROI muy grande para permitir proyección horizontal de gates verticales
# h_factor alto para escalar con Z (altura ≈ distancia radial en bird bath)
# min_radius extremadamente alto para cubrir todo el patrón circular
# Con estos valores, gates a Z=30km tendrán ROI de ~15km, cubriendo hasta 45km de radio
ROI_PARAMS_VOL03 = (5.0, 3.0, 2.5, 15000.0)

# Volumen 04: Largo alcance, mayor cobertura horizontal
ROI_PARAMS_VOL04 = (1.1, 1.6, 1.3, 900.0)

# Mapeo de volumen a sus parámetros (constantes, no adaptativos por altura)
ROI_PARAMS_BY_VOLUME = {
    '01': ROI_PARAMS_VOL01,
    '02': ROI_PARAMS_VOL02,
    '03': ROI_PARAMS_VOL03,
    '04': ROI_PARAMS_VOL04,
}