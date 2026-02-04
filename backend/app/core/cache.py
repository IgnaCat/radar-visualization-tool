import numpy as np
import pickle
import logging
import threading
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

GRID2D_CACHE = LRUCache(maxsize=100 * 1024 * 1024, getsizeof=_nbytes_pkg)

# Índice secundario: session_id -> set de cache keys
# Permite limpieza eficiente por sesión
SESSION_CACHE_INDEX: dict[str, set[str]] = {}

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

# Límite para intentar guardar en RAM (un poco menos que el maxsize para margen)
W_OPERATOR_MAX_RAM_SIZE_MB = 250

# Locks para evitar construcción duplicada del operador W en concurrencia
# Si múltiples threads piden el mismo operador W, solo uno lo construye
_W_OPERATOR_LOCKS: dict[str, threading.Lock] = {}
_W_OPERATOR_LOCKS_MASTER = threading.Lock()  # Protege el diccionario de locks

# Lock global para lectura de archivos NetCDF/HDF5 (no es thread-safe)
NETCDF_READ_LOCK = threading.Lock()


def get_w_operator_lock(cache_key: str) -> threading.Lock:
    """
    Obtiene o crea un lock para una cache_key específica del operador W.
    Permite que solo un thread construya un operador W a la vez para la misma key.
    """
    with _W_OPERATOR_LOCKS_MASTER:
        if cache_key not in _W_OPERATOR_LOCKS:
            _W_OPERATOR_LOCKS[cache_key] = threading.Lock()
        return _W_OPERATOR_LOCKS[cache_key]

# Índice secundario para W_OPERATOR: session_id -> set de cache keys
W_OPERATOR_SESSION_INDEX: dict[str, set[str]] = {}

# Contador de referencias: cache_key -> número de sesiones usando ese operador
W_OPERATOR_REF_COUNT: dict[str, int] = {}

# Directorio para cache en disco
CACHE_DIR = Path(settings.CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_w_operator_size_mb(W: csr_matrix) -> float:
    """Calcula el tamaño en MB de un operador W (matriz dispersa CSR)."""
    if W is None:
        return 0.0
    
    w_size_bytes = 0
    if hasattr(W, 'data'):
        w_size_bytes += W.data.nbytes
    if hasattr(W, 'indices'):
        w_size_bytes += W.indices.nbytes
    if hasattr(W, 'indptr'):
        w_size_bytes += W.indptr.nbytes
    
    return w_size_bytes / (1024 ** 2)


def try_cache_w_operator_in_ram(
    cache_key: str,
    W: csr_matrix,
    metadata: dict,
    session_id: str | None = None
) -> bool:
    """
    Intenta guardar operador W en cache RAM si no excede el límite de tamaño.
    
    Args:
        cache_key: Clave de cache
        W: Operador W (matriz dispersa)
        metadata: Metadatos del operador
        session_id: ID de sesión (opcional)
    
    Returns:
        bool: True si se guardó en RAM, False si era muy grande
    """
    w_size_mb = get_w_operator_size_mb(W)
    
    if w_size_mb < W_OPERATOR_MAX_RAM_SIZE_MB:
        W_OPERATOR_CACHE[cache_key] = {
            "W": W,
            "metadata": metadata
        }
        logger.info(f"Operador W guardado en cache RAM ({w_size_mb:.2f} MB)")
        
        # Registrar en índice de sesión si existe
        if session_id:
            if session_id not in W_OPERATOR_SESSION_INDEX:
                W_OPERATOR_SESSION_INDEX[session_id] = set()
            
            if cache_key not in W_OPERATOR_SESSION_INDEX[session_id]:
                if cache_key not in W_OPERATOR_REF_COUNT:
                    W_OPERATOR_REF_COUNT[cache_key] = 0
                W_OPERATOR_REF_COUNT[cache_key] += 1
                W_OPERATOR_SESSION_INDEX[session_id].add(cache_key)
        
        return True
    else:
        logger.info(
            f"Operador W ({w_size_mb:.2f} MB) NO se guarda en RAM "
            f"(excede límite de {W_OPERATOR_MAX_RAM_SIZE_MB} MB)"
        )
        return False
    

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
        
        # Convertir índices a int64 antes de guardar para evitar overflow
        if W.indices.dtype != np.int64:
            W.indices = W.indices.astype(np.int64)
        if W.indptr.dtype != np.int64:
            W.indptr = W.indptr.astype(np.int64)
        
        # Guardar matriz dispersa usando scipy (formato .npz optimizado)
        save_npz(cache_path, W, compressed=True)
        
        # Guardar metadata en archivo separado
        metadata_path = cache_path.with_suffix('.meta.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Operador W guardado en disco: {cache_path} ({W.nnz} elementos)")
        
    except (ValueError, OverflowError) as e:
        # Error común: índices demasiado grandes para int32 en matrices dispersas
        error_msg = str(e).lower()
        if "value too large" in error_msg or "overflow" in error_msg:
            logger.error(
                f"Error guardando operador W (matriz demasiado grande): {e}\n"
                f"Cache key: {cache_key[:16]}...\n"
                f"Shape: {W.shape}, NNZ: {W.nnz}\n"
                f"El archivo NetCDF tiene demasiados gates. Considere usar menor resolución."
            )
            print(f"[ERROR] Operador W demasiado grande para guardar: {cache_key[:16]}... - {e}")
        else:
            logger.error(f"Error guardando operador W en disco: {e}")
            print(f"[ERROR] Error guardando operador W: {e}")
    except Exception as e:
        logger.error(f"Error inesperado guardando operador W en disco: {e}")
        print(f"[ERROR] Error inesperado guardando operador W: {e}")


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
        
    except (ValueError, OverflowError) as e:
        # Error común: "value too large" cuando índices exceden int32
        error_msg = str(e).lower()
        if "value too large" in error_msg or "overflow" in error_msg:
            logger.error(
                f"Error cargando operador W desde disco (matriz demasiado grande para RAM): {e}\n"
                f"Cache key: {cache_key[:16]}...\n"
                f"Archivo: {get_w_operator_cache_path(cache_key)}\n"
                f"El archivo NetCDF puede ser demasiado grande o tener demasiados gates."
            )
            print(f"[ERROR] Operador W demasiado grande para RAM: {cache_key[:16]}... - {e}")
        else:
            logger.error(f"Error cargando operador W desde disco: {e}")
            print(f"[ERROR] Error cargando operador W: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado cargando operador W desde disco: {e}")
        print(f"[ERROR] Error inesperado cargando operador W: {e}")
        return None