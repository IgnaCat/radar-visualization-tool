import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { SnackbarProvider } from "notistack";

import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <SnackbarProvider
      maxSnack={3}          // máximo de snackbars visibles
      anchorOrigin={{
        vertical: "top", // posición en pantalla
        horizontal: "center"
      }}
      autoHideDuration={4000} // ms
    >
      <App />
    </SnackbarProvider>
  </StrictMode>,
)
