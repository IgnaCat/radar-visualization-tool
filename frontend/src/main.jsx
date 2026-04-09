import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { SnackbarProvider } from "notistack";

import "./index.css";
import { AuthProvider } from "./contexts/AuthContext";
import { generateSessionId } from "./utils/session";
import RequireAuth from "./components/guards/RequireAuth";
import RequireAdmin from "./components/guards/RequireAdmin";
import LoginPage from "./views/LoginPage";
import AdminDashboard from "./views/AdminDashboard";
import CacheStats from "./views/CacheStats";
import App from "./App.jsx";

// One stable session ID for the lifetime of this browser tab.
// Generated here so it can be shared between AuthProvider and App.
const sessionId = generateSessionId();

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <SnackbarProvider
      maxSnack={3}
      anchorOrigin={{
        vertical: "top",
        horizontal: "center",
      }}
      autoHideDuration={4000}
    >
      <BrowserRouter>
        <AuthProvider sessionId={sessionId}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/cache" element={<CacheStats />} />
            <Route
              path="/admin"
              element={
                <RequireAdmin>
                  <AdminDashboard />
                </RequireAdmin>
              }
            />
            <Route
              path="*"
              element={
                <RequireAuth>
                  <App sessionId={sessionId} />
                </RequireAuth>
              }
            />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </SnackbarProvider>
  </StrictMode>,
);
