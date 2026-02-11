"""
Módulo local de decodificación BUFR para radar.

Funciones extraídas de radarlib para evitar instalar el paquete completo
(cuya dependencia arm-pyart>=2.1.1 entra en conflicto con el fork
personalizado que usa el backend).

Solo se copian las funciones necesarias para el flujo BUFR → PyART Radar:
  - bufr_to_dict: decodifica un archivo BUFR a diccionario Python
  - bufr_fields_to_pyart_radar: convierte campos BUFR en objeto PyART Radar
  - bufr_to_pyart: wrapper de conveniencia
  - save_radar_to_cfradial: guarda Radar como NetCDF CFRadial

Requiere:
  - bufr_resources/ (tablas BUFR + librería dinámica libdecbufr.so)
    Se configura con BUFR_RESOURCES_PATH env var o por defecto en
    app/services/bufr/bufr_resources/
"""

from app.services.bufr.bufr_decoder import bufr_to_dict
from app.services.bufr.bufr_to_pyart import (
    bufr_fields_to_pyart_radar,
    bufr_to_pyart,
    save_radar_to_cfradial,
)

__all__ = [
    "bufr_to_dict",
    "bufr_fields_to_pyart_radar",
    "bufr_to_pyart",
    "save_radar_to_cfradial",
]
