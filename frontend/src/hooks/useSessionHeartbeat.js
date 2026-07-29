import { useEffect, useRef } from "react";
import api from "../api/backend";

const HEARTBEAT_INTERVAL_MS = 5 * 60 * 1000; // cada 5 minutos

/**
 * Envía un heartbeat al backend cada 5 minutos para mantener la sesión activa en DB.
 * Esto permite que el servidor detecte sesiones zombie (browser cerrado sin logout):
 * si una sesión deja de latir por más de SESSION_TTL (2h), el cleanup la marca inactiva.
 *
 * Solo envía heartbeat cuando hay token (usuario autenticado).
 */
export function useSessionHeartbeat(sessionId, token) {
  const sessionIdRef = useRef(sessionId);
  const tokenRef = useRef(token);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    tokenRef.current = token;
  }, [token]);

  useEffect(() => {
    if (!token || !sessionId) return;

    const ping = async () => {
      try {
        await api.post(
          "/auth/heartbeat",
          { session_id: sessionIdRef.current },
          { headers: { Authorization: `Bearer ${tokenRef.current}` } },
        );
      } catch {
        // best effort — no molestar si falla
      }
    };

    ping(); // heartbeat inmediato al montar

    const id = setInterval(ping, HEARTBEAT_INTERVAL_MS);
    return () => clearInterval(id);
  }, [token, sessionId]); // reinicia si cambia token o sessionId
}
