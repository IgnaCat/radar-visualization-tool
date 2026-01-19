import numpy as np
import pickle
import logging
from pathlib import Path
from cachetools import LRUCache
from scipy.sparse import csr_matrix, save_npz, load_npz
from ..core.config import settings

logger = logging.getLogger(__name__)

def _nbytes_arr(a) -> int:
    """Calcula el tamaño en bytes de un array o MaskedArray."""
    if isinstance(a, np.ma.MaskedArray):
        base = a.data.nbytes
        m = np.ma.getmaskarray(a)
        return base + (m.nbytes if m is not np.ma.nomask else 0)
    return getattr(a, "nbytes", 0)

def _nbytes_pkg(pkg) -> int:
    """
    Calcula tamaño en bytes del paquete de datos de una grilla 2D.
    pkg = {"arr": MaskedArray, "crs": str, "transform": Affine}
    """
    n = 0
    a = pkg.get("arr")
    if a is not None:
        n += _nbytes_arr(a)
    # crs/transform pesan poco, los ignoramos
    return n

GRID2D_CACHE = LRUCache(maxsize=200 * 1024 * 1024, getsizeof=_nbytes_pkg)

# ---- Operador W  ----
def _nbytes_w_operator(pkg) -> int:
    """
    Calcula tamaño en bytes del operador W (matriz dispersa CSR).
    pkg = {"W": csr_matrix, "metadata": dict}
    """
    W = pkg.get("W")
    if W is None:
        return 0
    
    # Para CSR: data, indices, indptr
    size = 0
    if hasattr(W, 'data'):
        size += W.data.nbytes
    if hasattr(W, 'indices'):
        size += W.indices.nbytes
    if hasattr(W, 'indptr'):
        size += W.indptr.nbytes
    
    return size

# Cache RAM para operador W (300 MB)
W_OPERATOR_CACHE = LRUCache(maxsize=300 * 1024 * 1024, getsizeof=_nbytes_w_operator)

# Directorio para cache en disco
CACHE_DIR = Path(settings.CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_w_operator_cache_path(cache_key: str) -> Path:
    """Retorna la ruta del archivo de cache en disco para un operador W."""
    return CACHE_DIR / f"{cache_key}.npz"

def save_w_operator_to_disk(cache_key: str, W: csr_matrix, metadata: dict):
    """
    Guarda operador W (matriz dispersa CSR) en disco de forma eficiente.
    
    Args:
        cache_key: Clave única de identificación
        W: Matriz dispersa scipy.sparse.csr_matrix
        metadata: Dict con información adicional (shape, roi, weight_func, etc.)
    """
    try:
        cache_path = get_w_operator_cache_path(cache_key)
        
        # Guardar matriz dispersa usando scipy (formato .npz optimizado)
        save_npz(cache_path, W, compressed=True)
        
        # Guardar metadata en archivo separado
        metadata_path = cache_path.with_suffix('.meta.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Operador W guardado en disco: {cache_path} ({W.nnz} elementos)")
        
    except Exception as e:
        logger.error(f"Error guardando operador W en disco: {e}")

def load_w_operator_from_disk(cache_key: str) -> tuple[csr_matrix, dict] | None:
    """
    Carga operador W desde disco.
    
    Returns:
        tuple (W, metadata) si existe, None si no
    """
    try:
        cache_path = get_w_operator_cache_path(cache_key)
        metadata_path = cache_path.with_suffix('.meta.pkl')
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        # Cargar matriz dispersa
        W = load_npz(cache_path)
        
        # Cargar metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Operador W cargado desde disco: {cache_path}")
        return W, metadata
        
    except Exception as e:
        logger.error(f"Error cargando operador W desde disco: {e}")
        return None