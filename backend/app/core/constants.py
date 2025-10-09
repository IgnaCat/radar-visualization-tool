
FIELD_ALIASES = {
    "DBZH": ["DBZH", "reflectivity", "corrected_reflectivity_horizontal"],
    "ZDR":  ["ZDR", "zdr"],
    "RHOHV":["RHOHV","rhohv"],
    "KDP":  ["KDP","kdp"],
    "VRAD": ["VRAD","velocity","corrected_velocity"],
    "WRAD": ["WRAD","spectrum_width","corrected_spectrum_width"],
}

FIELD_RENDER = { 
    "DBZH": {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"}, 
    "ZDR": {"vmin": -5.0, "vmax": 10.5, "cmap": "grc_zdr2"}, 
    "RHOHV": {"vmin": 0.5, "vmax": 1.0, "cmap": "grc_rho"}, 
    "KDP": {"vmin": 0.0, "vmax": 8.0, "cmap": "grc_rain"},
    "VRAD": {"vmin": -35.0, "vmax": 35.0, "cmap": "NWSVel"},
    "WRAD": {"vmin": 00, "vmax": 10.0, "cmap": "Oranges"},
}