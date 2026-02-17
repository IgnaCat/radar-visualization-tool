from fastapi import APIRouter
from pathlib import Path
from typing import Iterable
import shutil
import os

from ..core.config import settings
from ..core.cache import (
    GRID2D_CACHE, 
    SESSION_CACHE_INDEX, 
    W_OPERATOR_CACHE, 
    W_OPERATOR_SESSION_INDEX, 
    W_OPERATOR_REF_COUNT,
    _W_OPERATOR_LOCKS,
    _W_OPERATOR_LOCKS_MASTER
)
from ..models import CleanupRequest, FileCleanupRequest

router = APIRouter(prefix="/cleanup", tags=["cleanup"])

# Directorios “oficiales” de tu app
UPLOAD_DIR = Path(settings.UPLOAD_DIR).resolve()       # app/storage/uploads
TMP_DIR = Path(settings.IMAGES_DIR).resolve()          # app/storage/tmp
BASE_DIR = Path("app/storage").resolve()


def _first_safe_under(path_str: str, roots: Iterable[Path], session_id: str | None = None) -> Path | None:
    """
    Normaliza una ruta (absoluta o relativa) y devuelve la primera que
    caiga bajo alguno de los roots permitidos. Si no matchea, devuelve None.
    
    Args:
        path_str: Ruta del archivo (puede ser solo nombre o path completo)
        roots: Directorios raíz permitidos
        session_id: ID de sesión opcional para buscar en subdirectorios de sesión
    """
    s = str(path_str).replace("\\", "/").strip()
    candidates: list[Path] = []

    p = Path(s)
    if p.is_absolute():
        candidates.append(p)
    else:
        # prefijos “lógicos” del proyecto -> FS real
        if s.startswith("app/storage/tmp/"):
            rel = s.split("app/storage/tmp/", 1)[1]
            candidates.append(TMP_DIR / rel)
        elif s.startswith("app/storage/uploads/"):
            rel = s.split("app/storage/uploads/", 1)[1]
            candidates.append(UPLOAD_DIR / rel)
        elif s.startswith("static/tmp/"):
            rel = s.split("static/tmp/", 1)[1]
            candidates.append(TMP_DIR / rel)
        elif s.startswith("tmp/"):
            rel = s.split("tmp/", 1)[1]
            candidates.append(TMP_DIR / rel)
        else:
            # fallback: tratarla como relativa a las carpetas conocidas
            # Si hay session_id, agregar candidatos con subdirectorio de sesión primero
            if session_id:
                candidates.extend([
                    UPLOAD_DIR / session_id / s,
                    TMP_DIR / session_id / s,
                    UPLOAD_DIR / s,
                    TMP_DIR / s,
                    BASE_DIR / s
                ])
            else:
                candidates.extend([UPLOAD_DIR / s, TMP_DIR / s, BASE_DIR / s])

    # resolver y verificar que quede dentro de un root permitido
    for cand in candidates:
        try:
            rp = cand.resolve(strict=False)
        except Exception:
            continue
        for root in roots:
            try:
                # py>=3.9
                if rp == root or rp.is_relative_to(root):  # type: ignore[attr-defined]
                    return rp
            except AttributeError:
                # py<3.9
                from os.path import commonpath
                if commonpath([str(rp), str(root)]) == str(root):
                    return rp
    return None

def _delete_path(p: Path) -> bool:
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            return True
        else:
            # Py3.8+: missing_ok para no fallar si ya no existe
            try:
                p.unlink(missing_ok=True)  # type: ignore[arg-type]
            except TypeError:
                # compat py<3.8
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            return True
    except Exception:
        return False

@router.post("/files")
def cleanup_files(req: FileCleanupRequest):
    """
    Elimina archivos específicos subidos por el usuario.
    Borra el NetCDF, sus COGs asociados y entradas de cache en memoria.
    Devuelve lista de archivos efectivamente eliminados para que el frontend
    actualice su estado.
    """
    deleted_files: list[str] = []
    file_hashes: set[str] = set()
    deleted = {"uploads": 0, "cogs": 0, "cache_entries": 0}

    for filepath in req.filepaths:
        rp = _first_safe_under(
            filepath, [UPLOAD_DIR, TMP_DIR, BASE_DIR], session_id=req.session_id
        )
        if rp and rp.exists():
            # Calcular hash antes de borrar (para limpiar COGs y cache)
            try:
                from ..services.radar_common import md5_file
                file_hash = md5_file(str(rp))[:12]
                file_hashes.add(file_hash)
            except Exception:
                pass

            if _delete_path(rp):
                deleted["uploads"] += 1
                deleted_files.append(filepath)

    # Borrar COGs y cache relacionados
    if file_hashes:
        deleted["cogs"] = _delete_related_cogs(file_hashes, session_id=req.session_id)
        deleted["cache_entries"] = _cleanup_cache_entries(file_hashes, session_id=req.session_id)

    # Limpiar carpetas vacías
    if req.session_id:
        _cleanup_empty_session_dirs(req.session_id)

    return {"deleted": deleted, "removed_files": deleted_files}


@router.post("/close")
def cleanup_close(req: CleanupRequest):
    """
    Borra archivos indicados por el front al cerrar la app.
    - Siempre intenta borrar 'uploads' (NetCDF)
    - Si delete_cache=True: borra COGs relacionados por file_hash
    - Limpia entradas de cache en memoria relacionadas con archivos borrados
    Acepta rutas absolutas o relativas (sanitiza contra traversal).
    """
    deleted = {"uploads": 0, "cogs": 0, "cache_entries": 0}

    # Recopilar nombres de archivos a borrar para limpieza de cache
    upload_filenames = set()
    file_hashes = set()

    # uploads (NetCDF subidos / temporales del usuario)
    for s in req.uploads:
        rp = _first_safe_under(s, [UPLOAD_DIR, TMP_DIR, BASE_DIR], session_id=req.session_id)
        if rp and rp.exists():
            # Guardar nombre del archivo para limpieza de cache
            upload_filenames.add(rp.name)
            # Calcular hash antes de borrar
            try:
                from ..services.radar_common import md5_file
                file_hash = md5_file(str(rp))[:12]
                file_hashes.add(file_hash)
            except Exception:
                pass
            # Borrar archivo
            if _delete_path(rp):
                deleted["uploads"] += 1

    # Borrar COGs relacionados con los uploads eliminados (por file_hash matching)
    if req.delete_cache and file_hashes:
        deleted["cogs"] = _delete_related_cogs(file_hashes, session_id=req.session_id)

    # Limpiar entradas de cache en memoria relacionadas con archivos borrados
    if file_hashes:
        deleted["cache_entries"] = _cleanup_cache_entries(file_hashes, session_id=req.session_id)
        
        # Limpiar W_OPERATOR_CACHE por sesión
        deleted["w_operator_entries"] = _cleanup_w_operator_entries(session_id=req.session_id)
    
    # Si hay session_id, verificar si la carpeta de uploads quedó vacía después de borrar archivos
    if req.session_id and deleted["uploads"] > 0:
        try:
            upload_session_dir = UPLOAD_DIR / req.session_id
            if upload_session_dir.exists() and upload_session_dir.is_dir():
                if not any(upload_session_dir.iterdir()):
                    upload_session_dir.rmdir()
                    print(f"Eliminada carpeta vacía de sesión en UPLOADS: {req.session_id}")
        except Exception as e:
            print(f"Error al verificar carpeta de sesión en UPLOADS: {e}")
    
    # Limpiar carpetas vacías de sesión en TMP (ya se hace en _delete_related_cogs)
    if req.session_id:
        _cleanup_empty_session_dirs(req.session_id)

    return {"deleted": deleted}


def _delete_related_cogs(file_hashes: set[str], session_id: str | None = None) -> int:
    """
    Borra todos los COGs en /tmp que contengan alguno de los file_hashes en su nombre.
    Si session_id está presente, busca solo en el subdirectorio de sesión.
    
    Los COGs se nombran como: radar_FIELD_product_FIELD_vmin_vmax_elevation_HASH.tif
    
    Args:
        file_hashes: Set de hashes de archivos (12 caracteres)
        session_id: ID de sesión (opcional) para limpieza scoped
    
    Returns:
        Número de COGs eliminados
    """
    count = 0
    
    try:
        # Si hay session_id, buscar en subdirectorio de sesión
        if session_id:
            search_dir = TMP_DIR / session_id
        else:
            search_dir = TMP_DIR
            
        if not search_dir.exists():
            return 0
        
        # Buscar recursivamente si no hay session_id (legacy)
        if session_id:
            files_to_check = list(search_dir.iterdir())
        else:
            # Buscar en TMP_DIR y todos los subdirectorios (legacy + session dirs)
            files_to_check = list(search_dir.rglob('*'))
            
        for file_path in files_to_check:
            if not file_path.is_file():
                continue
            
            # Verificar si el nombre contiene alguno de los hashes
            filename = file_path.name
            for file_hash in file_hashes:
                if file_hash in filename:
                    try:
                        file_path.unlink(missing_ok=True)
                        count += 1
                        break  # No seguir buscando otros hashes en este archivo
                    except Exception as e:
                        print(f"Error borrando COG {filename}: {e}")
                        
    except Exception as e:
        print(f"Error limpiando COGs: {e}")
    
    # Si hay session_id y se borraron archivos, verificar si la carpeta quedó vacía
    if session_id and count > 0:
        try:
            session_dir = TMP_DIR / session_id
            if session_dir.exists() and session_dir.is_dir():
                # Si no tiene archivos, eliminar la carpeta
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
                    print(f"Eliminada carpeta vacía de sesión en TMP: {session_id}")
        except Exception as e:
            print(f"Error al verificar carpeta de sesión en TMP: {e}")
    
    if count > 0:
        print(f"Eliminados {count} COG(s) relacionados con {len(file_hashes)} archivo(s) {'en sesión ' + session_id if session_id else ''}")
    
    return count


def _cleanup_cache_entries(file_hashes: set[str], session_id: str | None = None) -> int:
    """
    Limpia entradas de GRID2D_CACHE relacionadas con archivos específicos.
    Si session_id está presente, solo elimina entradas que coincidan con esa sesión.
    
    W_OPERATOR_CACHE NO se limpia aquí porque:
    - Es compartido globalmente (300 MB para todas las sesiones)
    - Se limpia automáticamente por LRU al alcanzar límite
    - Para limpieza manual, usar /admin/clear-cache
    
    Args:
        file_hashes: Set de hashes de archivos (12 caracteres)
        session_id: ID de sesión (opcional) para limpieza scoped
    
    Returns:
        Número de entradas de cache eliminadas
    """
    count = 0
    
    if session_id and session_id in SESSION_CACHE_INDEX:
        # Limpiar usando el índice de sesión (más eficiente)
        keys_to_delete = list(SESSION_CACHE_INDEX[session_id])
        
        for cache_key in keys_to_delete:
            try:
                # Verificar que la key exista en la cache antes de eliminar
                if cache_key in GRID2D_CACHE:
                    del GRID2D_CACHE[cache_key]
                    count += 1
                # Eliminar del índice
                SESSION_CACHE_INDEX[session_id].discard(cache_key)
            except Exception as e:
                print(f"Error eliminando cache key {cache_key}: {e}")
        
        # Limpiar entrada de sesión si está vacía
        if not SESSION_CACHE_INDEX[session_id]:
            del SESSION_CACHE_INDEX[session_id]
    else:
        # Sin session_id o sesión no encontrada: buscar por file_hash en todas las keys
        # Esto es menos eficiente pero funciona para casos legacy
        keys_to_delete = []
        for cache_key in list(GRID2D_CACHE.keys()):
            # Las cache keys contienen el file_hash hasheado en su contenido
            # Como no podemos deshacer el hash, eliminamos todas las keys de esa sesión
            # o si no hay sesión, no hacemos nada (para evitar borrar cache de otras sesiones)
            if not session_id:
                # Sin session_id: no limpiar para evitar afectar otras sesiones
                break
        
        for key in keys_to_delete:
            try:
                del GRID2D_CACHE[key]
                count += 1
            except Exception:
                pass
    
    if count > 0:
        print(f"Limpiadas {count} entradas de GRID2D_CACHE {'para sesión ' + session_id if session_id else 'globales'}")
    
    return count


def _cleanup_w_operator_entries(session_id: str | None = None) -> int:
    """
    Limpia entradas de W_OPERATOR_CACHE por sesión usando contador de referencias.
    Solo elimina operadores cuando ninguna otra sesión los está usando.
    No toca disco (los .npz en disco son compartidos).
    
    Args:
        session_id: ID de sesión
    
    Returns:
        Número de entradas de cache eliminadas de RAM
    """
    count = 0
    
    if session_id and session_id in W_OPERATOR_SESSION_INDEX:
        keys_to_delete = list(W_OPERATOR_SESSION_INDEX[session_id])
        
        for cache_key in keys_to_delete:
            try:
                # Decrementar contador de referencias
                if cache_key in W_OPERATOR_REF_COUNT:
                    W_OPERATOR_REF_COUNT[cache_key] -= 1
                    
                    # Solo eliminar de RAM si nadie más lo está usando
                    if W_OPERATOR_REF_COUNT[cache_key] <= 0:
                        if cache_key in W_OPERATOR_CACHE:
                            del W_OPERATOR_CACHE[cache_key]
                            count += 1
                        # Limpiar lock asociado
                        cleanup_w_operator_lock(cache_key)
                        # Limpiar contador
                        del W_OPERATOR_REF_COUNT[cache_key]
                
                # Siempre remover de índice de esta sesión
                W_OPERATOR_SESSION_INDEX[session_id].discard(cache_key)
            except Exception as e:
                print(f"Error eliminando W_OPERATOR cache key {cache_key}: {e}")
        
        # Limpiar entrada de sesión si está vacía
        if not W_OPERATOR_SESSION_INDEX[session_id]:
            del W_OPERATOR_SESSION_INDEX[session_id]
    
    if count > 0:
        print(f"Limpiadas {count} entradas de W_OPERATOR_CACHE para sesión {session_id}")
    
    return count


def cleanup_w_operator_lock(cache_key: str) -> None:
    """
    Limpia el lock asociado a una cache_key cuando ya no se necesita.

    Args:
        cache_key: Clave de cache del operador W
    """
    with _W_OPERATOR_LOCKS_MASTER:
        if cache_key in _W_OPERATOR_LOCKS:
            del _W_OPERATOR_LOCKS[cache_key]


def _cleanup_empty_session_dirs(session_id: str) -> None:
    """
    Elimina carpetas de sesión si están vacías en UPLOAD_DIR y TMP_DIR.
    
    Args:
        session_id: ID de sesión a limpiar
    """
    dirs_to_check = [
        UPLOAD_DIR / session_id,
        TMP_DIR / session_id,
    ]
    
    for session_dir in dirs_to_check:
        try:
            if not session_dir.exists():
                continue
            
            # Verificar si está vacía (no tiene archivos ni subdirectorios)
            if session_dir.is_dir() and not any(session_dir.iterdir()):
                session_dir.rmdir()
                print(f"Eliminada carpeta vacía de sesión: {session_dir}")
        except Exception as e:
            print(f"Error limpiando carpeta de sesión {session_dir}: {e}")
