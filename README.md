# radar-visualization-tool

Herramienta web para la visualización interactiva de datos de radar meteorológico a partir de archivos NetCDF, desarrollada como **Trabajo Especial de Licenciatura en Ciencias de la Computación — FAMAF, UNC**.

---

## Overview

Este sistema permite cargar archivos crudos de radar meteorológico en formato NetCDF y generar productos de visualización estándar sobre un mapa web interactivo. El flujo completo abarca desde la subida de archivos hasta la renderización de tiles en el navegador:

1. **Carga de archivos**: el usuario sube uno o más archivos NetCDF (`.nc`) correspondientes a distintos radares o volúmenes de escaneo.
2. **Procesamiento**: el backend interpola los datos polares del radar a una grilla cartesiana 3D y un operador de interpolación Barnes2. A partir de esa grilla se generan productos bidimensionales: **PPI** (corte por elevación), **CAPPI** (corte a altura constante), **COLMAX** (máximo en columna) y **Pseudo-RHI** (transecta vertical entre dos puntos).
3. **Generación de tiles**: el resultado 2D se escribe como un GeoTIFF optimizado para la nube (COG) y se sirve dinámicamente a través de TiTiler, que produce tiles en formato Web Mercator listos para consumir desde el mapa.
4. **Visualización**: el frontend, construido con React y Leaflet, renderiza las capas sobre un mapa interactivo con controles de animación temporal, selección de campos meteorológicos (reflectividad, velocidad radial, fase diferencial, etc.), filtros de calidad (QC), perfiles de elevación del terreno, estadísticas por área y consulta de valores puntuales.

El sistema soporta la visualización simultánea de **múltiples campos** y **múltiples radares** fusionando sus frames por proximidad temporal, y emplea un sistema de caché multinivel (grillas 2D en memoria, operadores de interpolación en RAM y disco) para evitar reprocesamiento.

Para más detalles sobre la arquitectura, los endpoints de la API, los componentes del frontend y los productos radar disponibles, consultar la carpeta [`docs/`](docs/).

| Documento                                         | Contenido                                                    |
| ------------------------------------------------- | ------------------------------------------------------------ |
| [Arquitectura](docs/architecture.md)              | Flujo de datos, estructura de módulos, caché y concurrencia  |
| [Backend](docs/backend.md)                        | Endpoints de la API, servicios, modelos de datos             |
| [Frontend](docs/frontend.md)                      | Componentes, mapa, controles, diálogos                       |
| [Productos y campos](docs/products-and-fields.md) | Productos radar, campos meteorológicos, filtros QC           |
| [Guía de desarrollo](docs/development-guide.md)   | Instalación detallada, Docker, dependencias, troubleshooting |

---

## Inicio rápido

### Con Docker (recomendado)

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- Docs API: http://localhost:8000/docs

### Modo local

Consultar la [guía de desarrollo](docs/development-guide.md) para instrucciones completas.

**Backend** (requiere GDAL, conda recomendado en Windows):

```bash
cd backend
conda activate radar-env
make run
```

**Frontend**:

```bash
cd frontend
npm install && npm run dev
```

---
