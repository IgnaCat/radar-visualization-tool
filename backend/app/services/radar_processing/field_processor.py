"""
Colapso de grillas 3D a 2D según producto (PPI, CAPPI, COLMAX).
"""
import numpy as np


def collapse_grid_to_2d(grid, field, product, *,
                        elevation_deg=None,       # para PPI
                        target_height_m=None,     # para CAPPI
                        vmin=-30.0):
    """
    Convierte la grilla 3D a 2D según el producto:
      - "ppi": sigue el haz del sweep con elevación `elevation_deg`
      - "cappi": toma el nivel z más cercano a `target_height_m`
      - "colmax": toma el máximo en columna
    
    Args:
        grid: pyart.core.Grid con datos 3D
        field: Nombre del campo a colapsar
        product: Tipo de producto ('ppi', 'cappi', 'colmax')
        elevation_deg: Ángulo de elevación para PPI (requerido si product='ppi')
        target_height_m: Altura objetivo para CAPPI (requerido si product='cappi')
        vmin: Valor mínimo para enmascarado
    
    Modifica grid in-place, colapsando el campo 3D a 2D.
    """
    data3d = grid.fields[field]['data']
    
    # PyART Grid puede tener dimensión temporal (time, z, y, x)
    # Eliminar dimensión temporal si existe
    if data3d.ndim == 4:
        data3d = data3d[0, :, :, :]  # Tomar primer (y único) timestep
    
    z = grid.z['data']
    x = grid.x['data']
    y = grid.y['data']
    ny, nx = len(y), len(x)

    if data3d.ndim == 2:  # ya llegó 2D (raro)
        arr2d = data3d
    else:
        if product == "ppi":
            # Buscamos recortar la superficie que sigue el haz del sweep seleccionado
            assert elevation_deg is not None
            # Calculamos distancia horizontal r de cada píxel al radar
            X, Y = np.meshgrid(x, y, indexing='xy')
            r = np.sqrt(X**2 + Y**2)
            Re = 8.49e6  # m, 4/3 R_tierra

            # Altura donde debería estar el haz en cada píxel
            z_target = r * np.sin(np.deg2rad(elevation_deg)) + (r**2) / (2.0 * Re)

            # Para cada pixel (y,x), buscamos el índice z cuyo nivel esté más cerca de z_target
            iz = np.abs(z_target[..., None] - z[None, None, :]).argmin(axis=2)

            # Tomamos el valor en ese z
            yy = np.arange(ny)[:, None]
            xx = np.arange(nx)[None, :]
            arr2d = data3d[iz, yy, xx]

        elif product == "cappi":
            assert target_height_m is not None
            iz = np.abs(z - float(target_height_m)).argmin()
            arr2d = data3d[iz, :, :]
        elif product == "colmax":
            arr2d = data3d.max(axis=0)
        else:
            raise ValueError("Producto inválido")

    # Re-máscarar
    arr2d = np.ma.masked_invalid(arr2d)
    if field in ["filled_DBZH", "DBZH", "DBZV", "DBZHF", "composite_reflectivity", "cappi"]:
        arr2d = np.ma.masked_less_equal(arr2d, vmin)
    elif field in ["KDP", "ZDR"]:
        arr2d = np.ma.masked_less(arr2d, vmin)

    # Lo escribimos como un único nivel
    grid.fields[field]['data'] = arr2d[np.newaxis, ...]   # (1,ny,nx)
    grid.fields[field]['_FillValue'] = -9999.0
    grid.z['data'] = np.array([0.0], dtype=float)
