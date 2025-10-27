
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

AFFECTS_INTERP_FIELDS = {"RHOHV"}