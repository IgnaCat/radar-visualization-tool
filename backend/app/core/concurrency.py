"""
concurrency.py
──────────────
Semáforo de concurrencia para limitar procesamiento pesado simultáneo.

Problema: sin límite, N usuarios pidiendo /process al mismo tiempo lanzan
N pipelines completos (cada uno con multiprocessing.Pool interno para el
operador W, arrays numpy grandes, reproyecciones rasterio, etc.).
Con 5 usuarios esto puede saturar CPU y RAM fácilmente.

Solución: un asyncio.Semaphore que limita cuántos procesamientos pesados
corren en paralelo. Los requests que exceden el límite esperan en cola
(no se rechazan), manteniendo el servidor responsive para requests livianos
(tiles, health, colormaps, etc.).

Nota: run_in_threadpool ya es suficiente para nuestro caso porque:
  - build_W_operator usa multiprocessing.Pool internamente (bypasea GIL)
  - numpy/scipy/rasterio liberan el GIL en operaciones pesadas
  - ProcessPoolExecutor NO sirve aquí porque las caches en RAM
    (GRID2D_CACHE, W_OPERATOR_CACHE) y el NETCDF_READ_LOCK son
    in-process y no serían visibles desde procesos hijos.
"""

import asyncio
import logging
from functools import wraps
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# ── Semáforo global ──────────────────────────────────────────────────
# Límite de procesamientos pesados simultáneos.
# Con 2-5 usuarios y un servidor de 4-8 cores:
#   - 2 slots: conservador, garantiza que siempre hay CPU para tiles/health
#   - 3 slots: buen balance para 4+ cores
# Cada slot puede internamente lanzar multiprocessing.Pool con N workers.
MAX_CONCURRENT_HEAVY = 2

_semaphore: asyncio.Semaphore | None = None


def get_processing_semaphore() -> asyncio.Semaphore:
    """
    Obtiene (o crea) el semáforo de concurrencia.
    Se crea lazy porque necesita un event loop activo.
    """
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_HEAVY)
        logger.info(
            f"Semáforo de concurrencia inicializado: "
            f"max {MAX_CONCURRENT_HEAVY} procesamientos pesados simultáneos"
        )
    return _semaphore


# Timeout en segundos para esperar un slot del semáforo.
# Si un usuario espera más de esto, recibe 503 en vez de quedar colgado.
SEMAPHORE_TIMEOUT = 300  # 5 minutos


async def acquire_processing_slot():
    """
    Adquiere un slot de procesamiento pesado.
    Si no hay slots disponibles, espera hasta SEMAPHORE_TIMEOUT.
    Raises HTTPException 503 si el timeout expira.
    """
    sem = get_processing_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=SEMAPHORE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(
            f"Timeout esperando slot de procesamiento "
            f"({SEMAPHORE_TIMEOUT}s). Servidor saturado."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El servidor está procesando demasiadas solicitudes. "
                "Por favor, intente nuevamente en unos momentos."
            ),
        )


def release_processing_slot():
    """Libera un slot de procesamiento pesado."""
    sem = get_processing_semaphore()
    sem.release()
