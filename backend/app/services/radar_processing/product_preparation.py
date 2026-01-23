"""
Preparación de radares según tipo de producto (PPI, CAPPI, COLMAX).
Cada producto requiere transformaciones específicas antes del gridding.
"""

import numpy as np
import pyart
from ...utils import cappi as cappi_utils


def prepare_radar_for_product(
    radar: pyart.core.Radar,
    product: str,
    field_name: str,
    elevation: int = 0,
    cappi_height: float = 4000
) -> tuple[pyart.core.Radar, str]:
    """
    Prepara el objeto radar según el tipo de producto a generar.
    
    Args:
        radar: Objeto radar PyART original
        product: Tipo de producto ('PPI', 'CAPPI', 'COLMAX')
        field_name: Nombre del campo a procesar (puede estar ya filled)
        elevation: Índice de elevación para PPI
        cappi_height: Altura en metros para CAPPI
    
    Returns:
        Tupla (radar_to_use, field_to_use):
            - radar_to_use: Radar preparado (puede ser subset o modificado)
            - field_to_use: Nombre del campo a usar en gridding
    """
    product_upper = product.upper()
    
    if product_upper == "PPI":
        # PPI: Extraer solo el sweep de elevación especificado
        radar_to_use = radar.extract_sweeps([elevation])

        # Forzar la inicialización de las coordenadas de puerta después de extract_sweeps
        # extract_sweeps crea un nuevo objeto radar pero las coordenadas pueden no estar inicializadas
        _ = radar_to_use.init_gate_x_y_z()
            
        field_to_use = field_name
        
    elif product_upper == "CAPPI":
        # CAPPI: Pre-interpolar a altura fija y replicar verticalmente
        cappi = cappi_utils.create_cappi(radar, fields=[field_name], height=cappi_height)
        
        # Creamos un campo de 5400x523 y lo rellenamos con el cappi
        # Hacemos esto por problemas con el interpolador de pyart
        template = cappi.fields[field_name]['data']   # (360, 523)
        zeros_array = np.tile(template, (15, 1))   # (5400, 523)
        radar.add_field_like('DBZH', 'cappi', zeros_array, replace_existing=True)
        
        radar_to_use = radar
        field_to_use = "cappi"
        
    elif product_upper == "COLMAX":
        # COLMAX: Crear campo de reflectividad compuesta
        radar_to_use = _create_colmax_field(radar, field_name)
        field_to_use = 'composite_reflectivity'
        
    else:
        raise ValueError(f"Producto inválido: {product_upper}")
    
    return radar_to_use, field_to_use


def fill_dbzh_if_needed(radar: pyart.core.Radar, field_name: str, product: str) -> str:
    """
    Rellena valores enmascarados de DBZH si el producto lo requiere.
    
    Los productos CAPPI y COLMAX necesitan DBZH sin máscaras para interpolar correctamente.
    
    Args:
        radar: Objeto radar PyART
        field_name: Nombre del campo original
        product: Tipo de producto
    
    Returns:
        Nombre del campo a usar (puede ser 'filled_DBZH' si se rellenó)
    """
    if field_name == "DBZH" and product.upper() in ["CAPPI", "COLMAX"]:
        # Relleno el campo DBZH sino los -- no dejan interpolar
        filled_DBZH = radar.fields[field_name]['data'].filled(fill_value=-30)
        radar.add_field_like(field_name, 'filled_DBZH', filled_DBZH, replace_existing=True)
        return 'filled_DBZH'
    
    return field_name


def _create_colmax_field(radar: pyart.core.Radar, filled_field_name: str) -> pyart.core.Radar:
    """
    Crea un campo de reflectividad compuesto (COLMAX) a partir de todas las
    elevaciones disponibles en el radar.
    
    Args:
        radar: Objeto radar PyART
        filled_field_name: Nombre del campo filled ('filled_DBZH')
    
    Returns:
        Radar con campo 'composite_reflectivity' agregado
    """
    compz = pyart.retrieve.composite_reflectivity(radar, field=filled_field_name)
    
    # Cambiamos el long_name para que en el titulo de la figura salga COLMAX
    compz.fields['composite_reflectivity']['long_name'] = 'COLMAX'
    
    # Volver a la máscara antes de exportar
    data = compz.fields['composite_reflectivity']['data']
    mask = np.isnan(data) | np.isclose(data, -30) | (data < -40)
    compz.fields['composite_reflectivity']['data'] = np.ma.array(data, mask=mask)
    compz.fields['composite_reflectivity']['_FillValue'] = -9999.0
    
    return compz
