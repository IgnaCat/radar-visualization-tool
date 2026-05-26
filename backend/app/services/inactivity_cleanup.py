"""
inactivity_cleanup.py
─────────────────────
Background task que libera recursos RAM de sesiones inactivas.

Problema que resuelve:
    Si el usuario cierra la pestaña sin disparar beforeunload/pagehide
    (browser crash, kill del proceso, red caída, mobile), el frontend
    nunca llega a llamar /cleanup/close y los caches GRID2D y W_OPERATOR
    quedan en RAM indefinidamente hasta que el LRU los evicte.

Solución:
    Un loop asyncio corre cada CHECK_INTERVAL segundos. Para cada sesión
    en SESSION_LAST_ACTIVITY, si pasó más de SESSION_TTL_SECONDS sin
    actividad, ejecuta el mismo cleanup que haría /cleanup/close:
        - Elimina entradas de GRID2D_CACHE vía SESSION_CACHE_INDEX
        - Decrementa REF_COUNT de W_OPERATOR_CACHE y elimina si llega a 0
        - Llama malloc_trim para devolver memoria al OS

Parámetros (ajustables):
    SESSION_TTL_SECONDS  – minutos de inactividad antes de limpiar (default: 30 min)
    CHECK_INTERVAL       – cada cuántos segundos corre el checker (default: 5 min)
"""

import asyncio
import ctypes
import gc
import logging
import time

from ..core.cache import (
    GRID2D_CACHE,
    SESSION_CACHE_INDEX,
    SESSION_LAST_ACTIVITY,
    W_OPERATOR_CACHE,
    W_OPERATOR_SESSION_INDEX,
    W_OPERATOR_REF_COUNT,
    _W_OPERATOR_LOCKS,
    _W_OPERATOR_LOCKS_MASTER,
)

logger = logging.getLogger(__name__)

# Sesión sin actividad por más de este tiempo → limpiar.
# 2 horas es razonable: una sesión real nunca estará 2 h sin tocar nada;
# si el usuario cerró la pestaña y no disparó beforeunload, en 2 h se libera.
SESSION_TTL_SECONDS: float = 2 * 60 * 60  # 2 horas

# Frecuencia del check. No hace falta que sea frecuente: el LRU (100 MB
# GRID2D + 300 MB W_OPERATOR) ya maneja la presión de memoria por sí solo.
# Este cleanup es solo la red de seguridad para sesiones totalmente abandonadas.
CHECK_INTERVAL: float = 60 * 60  # cada 1 hora


def _evict_session(session_id: str, decided_at: float) -> dict:
    """
    Elimina de RAM todos los recursos asociados a una sesión.
    Misma lógica que /cleanup/close, extraída para reutilización.

    Args:
        session_id: sesión a limpiar.
        decided_at: timestamp en que se tomó la decisión de evictar.
            Si el processor actualizó SESSION_LAST_ACTIVITY entre esa
            decisión y ahora, la sesión volvió a estar activa → abortar.

    Returns:
        dict con contadores de entradas eliminadas, o {"aborted": True}.
    """
    # Re-verificar: el radar_processor corre en threads y puede haber
    # actualizado SESSION_LAST_ACTIVITY justo mientras decidíamos evictar.
    current_last = SESSION_LAST_ACTIVITY.get(session_id)
    if current_last is not None and current_last > decided_at:
        # La sesión se activó entre el check y la evicción → no borrar nada
        logger.debug(
            "inactivity_cleanup: sesión '%s' se reactivó antes de evictar, abortando",
            session_id,
        )
        return {"aborted": True}

    removed = {"grid2d": 0, "w_operator": 0}

    # --- GRID2D_CACHE ---
    if session_id in SESSION_CACHE_INDEX:
        keys = list(SESSION_CACHE_INDEX[session_id])
        for key in keys:
            try:
                if key in GRID2D_CACHE:
                    del GRID2D_CACHE[key]
                    removed["grid2d"] += 1
                SESSION_CACHE_INDEX[session_id].discard(key)
            except Exception as exc:
                logger.warning("inactivity_cleanup: error borrando GRID2D key %s: %s", key, exc)
        if not SESSION_CACHE_INDEX.get(session_id):
            SESSION_CACHE_INDEX.pop(session_id, None)

    # --- W_OPERATOR_CACHE (ref-counting) ---
    if session_id in W_OPERATOR_SESSION_INDEX:
        keys = list(W_OPERATOR_SESSION_INDEX[session_id])
        for key in keys:
            try:
                if key in W_OPERATOR_REF_COUNT:
                    W_OPERATOR_REF_COUNT[key] -= 1
                    if W_OPERATOR_REF_COUNT[key] <= 0:
                        W_OPERATOR_CACHE.pop(key, None)
                        # Limpiar lock asociado
                        with _W_OPERATOR_LOCKS_MASTER:
                            _W_OPERATOR_LOCKS.pop(key, None)
                        del W_OPERATOR_REF_COUNT[key]
                        removed["w_operator"] += 1
                W_OPERATOR_SESSION_INDEX[session_id].discard(key)
            except Exception as exc:
                logger.warning("inactivity_cleanup: error borrando W_OPERATOR key %s: %s", key, exc)
        if not W_OPERATOR_SESSION_INDEX.get(session_id):
            W_OPERATOR_SESSION_INDEX.pop(session_id, None)

    # --- Limpiar registro de actividad ---
    SESSION_LAST_ACTIVITY.pop(session_id, None)

    return removed


def _release_memory_to_os() -> None:
    """gc.collect() + malloc_trim para devolver páginas libres al OS."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def run_inactivity_cleanup() -> list[str]:
    """
    Revisa todas las sesiones conocidas y evicta las inactivas.
    Devuelve lista de session_ids limpiados (útil para tests y logs).
    """
    now = time.time()
    evicted: list[str] = []

    # Snapshot para no iterar mientras modificamos
    sessions = list(SESSION_LAST_ACTIVITY.keys())

    for session_id in sessions:
        last = SESSION_LAST_ACTIVITY.get(session_id)
        if last is None:
            continue
        idle_seconds = now - last
        if idle_seconds >= SESSION_TTL_SECONDS:
            removed = _evict_session(session_id, decided_at=now)
            if removed.get("aborted"):
                continue
            logger.info(
                "inactivity_cleanup: sesión '%s' inactiva %.0f min → "
                "eliminadas %d entradas GRID2D, %d operadores W",
                session_id,
                idle_seconds / 60,
                removed["grid2d"],
                removed["w_operator"],
            )
            evicted.append(session_id)

    if evicted:
        _release_memory_to_os()

    return evicted


async def inactivity_cleanup_loop() -> None:
    """
    Loop asyncio que corre indefinidamente.
    Se lanza como background task en el startup de FastAPI.
    """
    logger.info(
        "inactivity_cleanup: iniciado (TTL=%.0f min, check cada %.0f min)",
        SESSION_TTL_SECONDS / 60,
        CHECK_INTERVAL / 60,
    )
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        try:
            evicted = run_inactivity_cleanup()
            if evicted:
                logger.info(
                    "inactivity_cleanup: %d sesión(es) limpiada(s): %s",
                    len(evicted),
                    evicted,
                )
        except Exception as exc:
            # Nunca dejar que el loop muera por un error puntual
            logger.error("inactivity_cleanup: error inesperado: %s", exc, exc_info=True)
