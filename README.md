# radar-visualization-tool

Herramienta web para la visualización interactiva de datos de radar meteorológico a partir de archivos NetCDF, desarrollada para el **Trabajo Especial de FAMAF UNC**.

---

**Puedes correr este proyecto de dos formas:**

- **Modo local:** Instalando dependencias manualmente.
- **Usando Docker**

Ambas opciones están documentadas abajo.

---

## 🚀 Installation Guide

Este proyecto tiene dos partes principales:

- **Backend**: API en FastAPI para procesar y servir datos de radar.
- **Frontend**: Interfaz web en React para visualizar los datos.

---

## 🎨 Frontend

1. Instalar Node.js con NVM (Linux/macOS)

   ```bash
   curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
   nvm install 20
   nvm use 20
   ```

2. Instalar dependencias

   ```bash
   npm install
   ```

3. Levantar el servidor
   ```bash
   npm run dev
   ```

## 📦 Backend

### 🔹 Linux

1. Crear entorno virtual

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Instalar dependencias

   ```bash
   sudo apt install -y build-essential gdal-bin libgdal-dev libhdf5-dev libnetcdf-dev
   pip install -r requirements.txt
   ```

3. Levantar el servidor
   ```bash
   make run
   ```

### 🔹 Windows

1. Crear entorno con conda

   ```bash
   conda create -n radar-env -c conda-forge python=3.11 gdal rasterio pyproj shapely
   conda activate radar-env
   ```

2. Instalar dependencias

   ```bash
   pip install -r requirements.txt
   ```

3. Levantar el servidor
   ```bash
   make run
   ```

---

## 🐳 Docker

### Levantar toda la app

```bash
# Desde la raíz del proyecto
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- Docs API: http://localhost:8000/docs

### Detener servicios

```bash
docker compose down
```

### Variables de entorno

- Copia `.env.example` a `.env` y ajusta según tu entorno.

---
