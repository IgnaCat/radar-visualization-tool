import { createContext, useContext, useState, useCallback, useMemo } from "react";
import { loginApi, logoutApi } from "../api/auth";

const AuthContext = createContext(null);

export function AuthProvider({ children, sessionId }) {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);

  const login = useCallback(
    async (username, password) => {
      const data = await loginApi({ username, password, session_id: sessionId });
      setToken(data.access_token);
      setUser(data.user);
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
    setToken(null);
    setUser(null);
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
