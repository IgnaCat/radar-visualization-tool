# Guía de Desarrollo

## Prerequisitos

- **Python 3.11** (backend)
- **Node.js 20+** (frontend)
- **GDAL** (librería geoespacial requerida por rasterio)
- **Git**

---

## Backend

### Instalación en Windows (recomendado: conda)

Conda es el método preferido en Windows porque gestiona automáticamente las dependencias nativas de GDAL, HDF5 y NetCDF.

```bash
# 1. Crear el entorno conda
conda create -n radar-env -c conda-forge python=3.11 gdal rasterio pyproj shapely
conda activate radar-env

# 2. Instalar dependencias Python
cd backend
pip install -r requirements.txt

# 3. Levantar el servidor (desarrollo, con hot-reload)
make run
# Equivalente a: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Instalación en Linux

```bash
# 1. Instalar dependencias del sistema
sudo apt install -y build-essential gdal-bin libgdal-dev libhdf5-dev libnetcdf-dev

# 2. Crear entorno virtual
cd backend
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias Python
pip install -r requirements.txt

# 4. Levantar el servidor
make run
```

### Modos de ejecución

| Comando         | Workers        | Uso                                                |
| --------------- | -------------- | -------------------------------------------------- |
| `make run`      | 1 (con reload) | Desarrollo — recarga automática al guardar cambios |
| `make run-prod` | 4              | Producción — mayor concurrencia, sin reload        |

El servidor corre en **http://localhost:8000**. La documentación interactiva de la API (Swagger) está en **http://localhost:8000/docs**.

---

## Frontend

### Instalación

```bash
# 1. Instalar Node.js (con NVM en Linux/macOS)
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 20
nvm use 20

# En Windows: descargar Node.js 20 desde https://nodejs.org

# 2. Instalar dependencias
cd frontend
npm install

# 3. Levantar el dev server
npm run dev
```

El frontend corre en **http://localhost:3000** con HMR (Hot Module Replacement) de Vite.

### Variable de entorno

| Variable       | Default                 | Descripción                                                         |
| -------------- | ----------------------- | ------------------------------------------------------------------- |
| `VITE_API_URL` | `http://localhost:8000` | URL del backend. Modificar si el backend corre en otro host/puerto. |

---

## Docker

### Levantar toda la aplicación

```bash
# Desde la raíz del proyecto
docker compose up --build
```

Esto levanta:

- **radar-backend** en puerto 8000 (FastAPI + TiTiler)
- **radar-frontend** en puerto 3000 (Nginx sirviendo el build de React)

### Detener servicios

```bash
docker compose down
```

### Volúmenes

El `docker-compose.yml` monta dos volúmenes en el backend:

- `./backend/app/storage/data` → datos estáticos (DEMs, shapefiles) — solo lectura
- `./backend/app/storage/cache` → caché de operadores W — lectura/escritura

Los archivos subidos y los COGs generados son efímeros dentro del contenedor.

### Variables de entorno (Docker)

Las variables de entorno del backend se configuran dentro del `docker-compose.yml`. Las principales incluyen optimizaciones de GDAL:

| Variable           | Valor | Propósito                          |
| ------------------ | ----- | ---------------------------------- |
| `GDAL_CACHEMAX`    | 256   | Caché de GDAL en MB                |
| `GDAL_NUM_THREADS` | 4     | Threads para operaciones GDAL      |
| `VSI_CACHE`        | TRUE  | Habilita caché VSI para I/O        |
| `PROJ_NETWORK`     | OFF   | Desactiva descargas de red de PROJ |

Para configuración personalizada, se puede crear un archivo `.env` en la raíz del proyecto.

---

## Convención de nombres de archivo

Los archivos NetCDF deben seguir la convención:

```
{RADAR}_{STRATEGY}_{VOLUME}_{TIMESTAMP}Z.nc
```

Ejemplo: `RMA1_0315_01_20250819T001715Z.nc`

| Parte       | Significado                          | Ejemplo           |
| ----------- | ------------------------------------ | ----------------- |
| `RADAR`     | Identificador del radar              | `RMA1`            |
| `STRATEGY`  | Código de estrategia de escaneo      | `0315`            |
| `VOLUME`    | Número de volumen                    | `01`              |
| `TIMESTAMP` | Fecha y hora UTC (ISO 8601 compacto) | `20250819T001715` |

Esta convención es **obligatoria** — el sistema parsea el nombre para identificar el radar, agrupar volúmenes y ordenar temporalmente las imágenes para la animación.

---

## Dependencias notables

### PyART (fork personalizado)

El proyecto usa un fork de PyART con modificaciones específicas:

```
arm_pyart @ git+https://github.com/IgnaCat/pyart.git@tesis
```

Este fork puede contener colormaps adicionales, correcciones específicas o funcionalidades extendidas necesarias para el proyecto.

### GDAL en Windows

GDAL es la dependencia más problemática de instalar en Windows. La carpeta `backend/` incluye wheels pre-compilados como fallback:

- `GDAL-3.9.2-cp311-cp311-win_amd64.whl` (Python 3.11)
- `GDAL-3.9.2-cp313-cp313-win_amd64.whl` (Python 3.13)

Sin embargo, **conda es la vía recomendada** porque resuelve automáticamente GDAL y todas sus dependencias nativas.

### TiTiler

TiTiler se integra directamente en FastAPI (no como servicio externo). Se importa como `TilerFactory` de `titiler.core` y se monta en el prefijo `/cog`.

---

## Estructura de almacenamiento

```
backend/app/storage/
├── uploads/       # Archivos NetCDF subidos por el usuario
├── tmp/           # COGs generados (servidos como estáticos en /static/tmp)
├── cache/         # Operadores W serializados a disco (.npz + .meta.pkl)
└── data/          # Datos estáticos: DEMs para perfiles de elevación
```

---

## Troubleshooting

### GDAL no se instala en Windows

- Usar conda: `conda install -c conda-forge gdal rasterio`
- Si pip falla, instalar manualmente el wheel: `pip install GDAL-3.9.2-cp311-cp311-win_amd64.whl`

### Errores de TileJSON 404

- Verificar que el archivo COG existe en la ruta indicada en `tilejson_url`
- Revisar los logs del backend para errores de GDAL/CRS
- Las rutas de COG deben ser **rutas absolutas POSIX** (`Path.resolve().as_posix()`)

### Cache hits/misses inesperados

- Las claves de caché incluyen: hash del archivo, producto, campo, elevación/altura, volumen, método de interpolación
- Los filtros **no** son parte de la clave (se aplican post-caché)
- Revisar `grid2d_cache_key()` en `radar_common.py` para entender la generación de claves

### NetCDF/HDF5 errores de concurrencia

- Las librerías NetCDF y HDF5 no son thread-safe
- El sistema usa `NETCDF_READ_LOCK` para serializar lecturas
- Si aparecen errores de corrupción, verificar que el lock se está usando correctamente

### El frontend no conecta al backend

- Verificar que `VITE_API_URL` apunta al host/puerto correcto
- En Docker, el frontend accede al backend desde el navegador (no desde el contenedor), por lo que la URL debe ser accesible desde la máquina del usuario
- Revisar CORS: el backend debe incluir el origen del frontend en `FRONTEND_ORIGINS`

### Volumen 03 da error en PPI

- El volumen 03 no es válido para el producto PPI (genera un warning)
- Usar CAPPI o COLMAX con volumen 03
