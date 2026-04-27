import {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
} from "react";
import { loginApi, logoutApi } from "../api/auth";

const AuthContext = createContext(null);

// El backend token dura 24 horas. (JWT_EXPIRE_HOURS = 24)
const TOKEN_EXPIRY_MS = 24 * 60 * 60 * 1000;

export function AuthProvider({ children, sessionId }) {
  // Inicializamos el estado desde localStorage, comprobando expiración
  const [token, setToken] = useState(() => {
    const savedToken = localStorage.getItem("token");
    const expiresAt = localStorage.getItem("tokenExpiresAt");

    if (savedToken && expiresAt) {
      if (Date.now() > Number(expiresAt)) {
        // Expirado, limpiamos localStorage
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        localStorage.removeItem("tokenExpiresAt");
        return null;
      }
      return savedToken;
    }
    return null;
  });

  const [user, setUser] = useState(() => {
    const savedUser = localStorage.getItem("user");
    const expiresAt = localStorage.getItem("tokenExpiresAt");
    if (savedUser && expiresAt && Date.now() <= Number(expiresAt)) {
      try {
        return JSON.parse(savedUser);
      } catch {
        return null;
      }
    }
    return null;
  });

  const login = useCallback(
    async (username, password) => {
      const data = await loginApi({
        username,
        password,
        session_id: sessionId,
      });

      // Actualizamos estado en memoria
      setToken(data.access_token);
      setUser(data.user);

      // Guardamos en localStorage con expiración
      const expiresAt = Date.now() + TOKEN_EXPIRY_MS;
      localStorage.setItem("token", data.access_token);
      localStorage.setItem("user", JSON.stringify(data.user));
      localStorage.setItem("tokenExpiresAt", expiresAt.toString());

      return data;
    },
    [sessionId],
  );

  const logout = useCallback(async () => {
    try {
      if (token) {
        await logoutApi({ session_id: sessionId, token });
      }
    } catch {
      // best effort — ignore errors on logout
    }
    // Limpiamos la memoria
    setToken(null);
    setUser(null);

    // Limpiamos localStorage
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    localStorage.removeItem("tokenExpiresAt");
  }, [token, sessionId]);

  const value = useMemo(
    () => ({
      token,
      user,
      login,
      logout,
      isAuthenticated: !!token,
      isAdmin: user?.role === "admin",
    }),
    [token, user, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
