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