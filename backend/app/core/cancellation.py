"""
cancellation.py
───────────────
Token de cancelación cooperativa para requests de procesamiento pesado.

Problema:
    run_in_threadpool no se puede cancelar desde afuera (Python threads
    no tienen kill). Si el cliente cierra la pestaña o hace F5 durante
    una request /process, el thread sigue corriendo y bloquea el semáforo
    de concurrencia hasta que termine.

Solución:
    threading.Event por sesión. El router lo activa cuando detecta que
    el cliente se desconectó (via request.is_disconnected()). El pipeline
    lo chequea en puntos estratégicos (antes de construir W, entre archivos,
    entre campos) y lanza CancelledException para salir limpiamente.

    El multiprocessing.Pool interno de build_W_operator no puede
    interrumpirse mid-flight (procesos separados sin acceso al Event), pero:
      - Si el W todavía no empezó → se corta antes de lanzar el Pool
      - Si ya está corriendo → termina solo sin leak; el semáforo se libera
        igual porque el thread principal salió por CancelledException
      - Todo lo que sigue al W (COG, warp, caché) se corta inmediatamente

Uso en el pipeline (en cualquier punto que tenga session_id):
    from ..core.cancellation import raise_if_cancelled
    raise_if_cancelled(session_id)
"""

import threading
import logging

logger = logging.getLogger(__name__)


class CancelledException(Exception):
    """Lanzada cuando el cliente se desconectó y el pipeline debe abortar."""
    pass


class CancellationToken:
    """threading.Event con interfaz de cancelación."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


# session_id → CancellationToken para requests de procesamiento activas.
# Se crea al inicio de cada /process y se elimina en el finally del router.
SESSION_ACTIVE_TOKENS: dict[str, CancellationToken] = {}


def raise_if_cancelled(session_id: str | None) -> None:
    """
    Lanza CancelledException si la sesión fue cancelada.
    No-op si session_id es None o no tiene token activo.
    Diseñado para llamarse en los puntos de corte del pipeline.
    """
    if not session_id:
        return
    token = SESSION_ACTIVE_TOKENS.get(session_id)
    if token and token.is_cancelled():
        raise CancelledException(
            f"Request cancelada por desconexión del cliente (sesión {session_id})"
        )
