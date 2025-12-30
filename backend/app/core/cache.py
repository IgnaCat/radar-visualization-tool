from cachetools import LRUCache
import numpy as np

def _nbytes_arr(a) -> int:
    """Calcula el tamaño en bytes de un array o MaskedArray."""
    if isinstance(a, np.ma.MaskedArray):
        base = a.data.nbytes
        m = np.ma.getmaskarray(a)
        return base + (m.nbytes if m is not np.ma.nomask else 0)
    return getattr(a, "nbytes", 0)

def _nbytes_pkg(pkg) -> int:
    # pkg = {"arr": MaskedArray, "crs": str, "transform": Affine}
    n = 0
    a = pkg.get("arr")
    if a is not None:
        n += _nbytes_arr(a)
    # crs/transform pesan poco, los ignoramos
    return n

GRID2D_CACHE = LRUCache(maxsize=200 * 1024 * 1024, getsizeof=_nbytes_pkg)

# ---- 3D Grid Cache (for vertical transects) ----
def _nbytes_pkg3d(pkg) -> int:
    """
    Calcula tamaño en bytes de package 3D.
    Soporta estructura multi-campo: pkg['fields'] = {field_name: {"data": arr, "metadata": dict}}
    O estructura legacy: pkg['arr3d'] para retrocompat.
    """
    n = 0
    
    # Nuevo formato multi-campo
    fields_dict = pkg.get("fields")
    if fields_dict:
        for fname, fdata in fields_dict.items():
            arr = fdata.get("data")
            if arr is not None:
                n += _nbytes_arr(arr)
    else:
        # Formato legacy (retrocompat)
        a3 = pkg.get("arr3d")
        if a3 is not None:
            n += _nbytes_arr(a3)
    
    # x, y, z son pequeños (~KB), ignoramos
    return n

GRID3D_CACHE = LRUCache(maxsize=600 * 1024 * 1024, getsizeof=_nbytes_pkg3d)