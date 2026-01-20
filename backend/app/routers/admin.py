"""
Endpoints de administración para mantenimiento del sistema.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
import time
import datetime

from ..core.cache import (
    GRID2D_CACHE, 
    W_OPERATOR_CACHE,
    get_w_operator_cache_path,
    CACHE_DIR,
    SESSION_CACHE_INDEX,
    W_OPERATOR_SESSION_INDEX,
    W_OPERATOR_REF_COUNT
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/clear-cache")
def clear_cache(
    cache_type: Optional[str] = None,
    max_age_hours: Optional[float] = None
):
    """
    Limpia cache del sistema.
    
    Args:
        cache_type: Tipo de cache a limpiar ('grid2d', 'w_operator', 'all'). Default: 'all'
        max_age_hours: Solo limpiar operadores W con edad mayor a X horas. None = limpiar todo.
    
    Returns:
        Estadísticas de limpieza
    """
    if cache_type is None:
        cache_type = 'all'
    
    cache_type = cache_type.lower()
    if cache_type not in ['grid2d', 'w_operator', 'w_operator_ram', 'w_operator_disk', 'all']:
        raise HTTPException(400, "cache_type debe ser 'grid2d', 'w_operator', 'w_operator_ram', 'w_operator_disk' o 'all'")
    
    cleared = {
        "grid2d_entries": 0,
        "w_operator_ram_entries": 0,
        "w_operator_disk_files": 0,
        "disk_space_freed_mb": 0.0
    }
    
    # Limpiar GRID2D_CACHE
    if cache_type in ['grid2d', 'all']:
        size_before = len(GRID2D_CACHE)
        GRID2D_CACHE.clear()
        cleared["grid2d_entries"] = size_before
    
    # Limpiar W_OPERATOR_CACHE (RAM + Disco)
    if cache_type in ['w_operator', 'w_operator_ram', 'all']:
        clear_ram = cache_type in ['w_operator', 'w_operator_ram', 'all']
        clear_disk = cache_type in ['w_operator', 'w_operator_disk', 'all']
        cleared_w = _clear_w_operator_cache(max_age_hours, clear_ram=clear_ram, clear_disk=clear_disk)
        cleared["w_operator_ram_entries"] = cleared_w["ram"]
        cleared["w_operator_disk_files"] = cleared_w["disk"]
        cleared["disk_space_freed_mb"] = cleared_w["disk_mb"]
    elif cache_type == 'w_operator_disk':
        cleared_w = _clear_w_operator_cache(max_age_hours, clear_ram=False, clear_disk=True)
        cleared["w_operator_ram_entries"] = cleared_w["ram"]
        cleared["w_operator_disk_files"] = cleared_w["disk"]
        cleared["disk_space_freed_mb"] = cleared_w["disk_mb"]
    
    return {
        "success": True,
        "cleared": cleared,
        "message": f"Cache '{cache_type}' limpiada exitosamente"
    }


def _clear_w_operator_cache(max_age_hours: Optional[float] = None, clear_ram: bool = True, clear_disk: bool = True) -> dict:
    """
    Limpia cache de operadores W en RAM y/o disco.
    
    Args:
        max_age_hours: Solo limpiar operadores más viejos que X horas. None = limpiar todo.
        clear_ram: Si True, limpia la cache en RAM
        clear_disk: Si True, limpia los archivos del disco
    
    Returns:
        Dict con estadísticas: {"ram": int, "disk": int, "disk_mb": float}
    """
    import pickle
    
    cleared = {"ram": 0, "disk": 0, "disk_mb": 0.0}
    current_time = time.time()
    cutoff_time = current_time - (max_age_hours * 3600) if max_age_hours else 0
    
    # Limpiar RAM
    if clear_ram and max_age_hours is None:
        # Limpiar todo
        cleared["ram"] = len(W_OPERATOR_CACHE)
        W_OPERATOR_CACHE.clear()
        # Limpiar contadores e índices también
        W_OPERATOR_REF_COUNT.clear()
        W_OPERATOR_SESSION_INDEX.clear()
    elif clear_ram:
        # Limpiar selectivamente por edad
        keys_to_delete = []
        for cache_key, pkg in W_OPERATOR_CACHE.items():
            metadata = pkg.get("metadata", {})
            created_at = metadata.get("created_at", 0)
            if created_at < cutoff_time:
                keys_to_delete.append(cache_key)
        
        for key in keys_to_delete:
            try:
                del W_OPERATOR_CACHE[key]
                cleared["ram"] += 1
                # Limpiar contador de referencias
                if key in W_OPERATOR_REF_COUNT:
                    del W_OPERATOR_REF_COUNT[key]
                # Limpiar de índices de sesión
                for session_id in list(W_OPERATOR_SESSION_INDEX.keys()):
                    W_OPERATOR_SESSION_INDEX[session_id].discard(key)
                    if not W_OPERATOR_SESSION_INDEX[session_id]:
                        del W_OPERATOR_SESSION_INDEX[session_id]
            except Exception:
                pass
    
    # Limpiar disco
    if not clear_disk or not CACHE_DIR.exists():
        return cleared
    
    for cache_file in CACHE_DIR.glob("W_*.npz"):
        try:
            # Verificar edad del archivo
            if max_age_hours is not None:
                metadata_file = cache_file.with_suffix('.meta.pkl')
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                        created_at = metadata.get("created_at", 0)
                        if created_at >= cutoff_time:
                            continue  # No borrar, es reciente
            
            # Calcular tamaño antes de borrar
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            metadata_file = cache_file.with_suffix('.meta.pkl')
            if metadata_file.exists():
                size_mb += metadata_file.stat().st_size / (1024 * 1024)
            
            # Borrar archivos
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            
            cleared["disk"] += 1
            cleared["disk_mb"] += size_mb
            
        except Exception as e:
            print(f"Error limpiando {cache_file.name}: {e}")
    
    return cleared


@router.get("/cache-stats")
def get_cache_stats():
    """
    Obtiene estadísticas del uso de cache.
    
    Returns:
        Información sobre el estado actual de las caches
    """
    import pickle
    
    # Estadísticas GRID2D_CACHE
    grid2d_count = len(GRID2D_CACHE)
    grid2d_size_mb = sum(
        GRID2D_CACHE.getsizeof(pkg) for pkg in GRID2D_CACHE.values()
    ) / (1024 * 1024)
    
    # Estadísticas W_OPERATOR_CACHE (RAM)
    w_ram_count = len(W_OPERATOR_CACHE)
    w_ram_size_mb = sum(
        W_OPERATOR_CACHE.getsizeof(pkg) for pkg in W_OPERATOR_CACHE.values()
    ) / (1024 * 1024)
    
    # Estadísticas W_OPERATOR_CACHE (Disco)
    w_disk_count = 0
    w_disk_size_mb = 0.0
    w_disk_oldest = None
    w_disk_newest = None
    cache_files_list = []
    
    if CACHE_DIR.exists():
        # Recolectar todos los archivos en cache
        for cache_file in CACHE_DIR.glob("*"):
            if cache_file.is_file():
                try:
                    size = cache_file.stat().st_size
                    created_time = cache_file.stat().st_ctime
                    modified_time = cache_file.stat().st_mtime
                    
                    # Solo agregar archivos .npz a la lista
                    if cache_file.suffix == ".npz":
                        file_info = {
                            "name": cache_file.name,
                            "size_bytes": size,
                            "size_mb": round(size / (1024 * 1024), 3),
                            "created_at": created_time,
                            "modified_at": modified_time
                        }
                        cache_files_list.append(file_info)
                    
                    # Estadísticas específicas para archivos W_*.npz
                    if cache_file.name.startswith("W_") and cache_file.suffix == ".npz":
                        w_disk_count += 1
                        w_disk_size_mb += size / (1024 * 1024)
                        
                        # Leer metadata para timestamps
                        metadata_file = cache_file.with_suffix('.meta.pkl')
                        if metadata_file.exists():
                            with open(metadata_file, 'rb') as f:
                                metadata = pickle.load(f)
                                created_at = metadata.get("created_at")
                                if created_at:
                                    if w_disk_oldest is None or created_at < w_disk_oldest:
                                        w_disk_oldest = created_at
                                    if w_disk_newest is None or created_at > w_disk_newest:
                                        w_disk_newest = created_at
                except Exception:
                    pass
        
        # Ordenar por fecha de modificación (más reciente primero)
        cache_files_list.sort(key=lambda x: x["modified_at"], reverse=True)
    
    return {
        "grid2d_cache": {
            "entries": grid2d_count,
            "size_mb": round(grid2d_size_mb, 2),
            "max_size_mb": 200
        },
        "w_operator_cache_ram": {
            "entries": w_ram_count,
            "size_mb": round(w_ram_size_mb, 2),
            "max_size_mb": 300
        },
        "w_operator_cache_disk": {
            "files": w_disk_count,
            "size_mb": round(w_disk_size_mb, 2),
            "oldest_timestamp": w_disk_oldest,
            "newest_timestamp": w_disk_newest
        },
        "cache_files": cache_files_list
    }
