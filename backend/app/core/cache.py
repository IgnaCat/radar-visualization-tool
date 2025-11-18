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
    # pkg = {"arr3d": MaskedArray(nz,ny,nx), "z": arr, "y": arr, "x": arr, "crs": str}
    n = 0
    a3 = pkg.get("arr3d")
    if a3 is not None:
        n += _nbytes_arr(a3)
    # axes are small; ignore
    return n

GRID3D_CACHE = LRUCache(maxsize=600 * 1024 * 1024, getsizeof=_nbytes_pkg3d)