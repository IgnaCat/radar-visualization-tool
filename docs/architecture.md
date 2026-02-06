# Arquitectura del Sistema

## Visión general

El sistema se divide en dos grandes componentes: un **backend** en Python (FastAPI) que procesa datos de radar meteorológico, y un **frontend** en JavaScript (React + Leaflet) que renderiza los resultados sobre un mapa interactivo. La comunicación entre ambos es mediante una API REST.

---

## Flujo de datos

El recorrido de un dato desde el archivo crudo hasta el pixel en pantalla sigue estos pasos:

1. **Subida**: el usuario sube archivos NetCDF (`.nc`) desde el frontend. El backend los almacena en `backend/app/storage/uploads/` organizados opcionalmente por sesión.

2. **Lectura y metadata**: se extraen metadatos del nombre de archivo (radar, estrategia, volumen, timestamp) usando la convención `{RADAR}_{STRATEGY}_{VOLUME}_{TIMESTAMP}Z.nc`. PyART lee el contenido del NetCDF.

3. **Grillado 3D**: los datos polares del radar se interpolan a una grilla cartesiana 3D regular. Este proceso tiene dos etapas:
   - **Generación del operador W**: se construye una matriz dispersa (CSR) de pesos que relaciona cada punto de la grilla cartesiana con los gates del radar según su distancia geométrica. Este cálculo es costoso porque involucra cálculos de distancia entre miles de puntos, por eso el operador W se **cachea en RAM o disco** para reutilizarse en procesamiento posteriores con la misma geometría de grilla. Se calcula una sola vez por radar/volumen/estrategia.
   - **Aplicación de la interpolación**: una vez precargado el operador W desde la caché, la interpolación se reduce a una multiplicación matricial dispersa `grilla_3D = W × datos_polares`, que es mucho más rápida. Esto permite cambiar de campo meteorológico (ej. de DBZH a ZDR) sin recalcular la geometría.

4. **Colapso a 2D**: según el producto solicitado, la grilla 3D se reduce a una imagen 2D:
   - **PPI**: sigue el haz del radar a una elevación dada.
   - **CAPPI**: corte horizontal a una altura constante.
   - **COLMAX**: máximo valor en la columna vertical.

5. **Aplicación de filtros**: sobre la grilla 2D se aplican filtros de calidad (ej. enmascarar donde RHOHV < 0.7) y filtros visuales (ej. mostrar solo reflectividad > 10 dBZ). Los filtros visuales son post-caché: la grilla sin filtrar se cachea y los filtros se aplican como máscaras. Los filtros de calidad se realizan antes de la interpolacion y generación de la grilla 3D.

6. **Generación de COG**: la grilla 2D coloreada se escribe como GeoTIFF y se convierte a Cloud Optimized GeoTIFF (COG) con overviews (factores 2, 4, 8, 16) y compresión DEFLATE.

7. **Servicio de tiles**: TiTiler (montado en la ruta `/cog` de FastAPI) lee el COG y genera tiles dinámicamente siguiendo el esquema TileJSON → tile URLs en formato Web Mercator.

8. **Renderizado**: el frontend obtiene el TileJSON de cada capa, extrae la URL template de tiles y la pasa a un `TileLayer` de Leaflet que renderiza los tiles sobre el mapa.

---

## Estructura de módulos

### Backend (`backend/app/`)

```
app/
├── main.py                         # App FastAPI, CORS, TiTiler, montaje de routers
├── schemas.py                      # Re-exporta modelos (shim de compatibilidad)
├── core/
│   ├── config.py                   # Settings (Pydantic BaseSettings, dirs, límites)
│   ├── constants.py                # Alias de campos, parámetros de renderizado, ROI
│   └── cache.py                    # Cachés LRU (grillas 2D, operadores W), locks
├── models/                         # Modelos Pydantic (request/response)
│   ├── process.py                  # ProcessRequest, ProcessResponse, LayerResult
│   ├── pseudo_rhi.py               # PseudoRHIRequest/Response
│   ├── stats.py                    # RadarStatsRequest/Response
│   ├── pixel.py                    # RadarPixelRequest/Response
│   └── elevation.py                # ElevationProfileRequest/Response
├── routers/                        # Endpoints HTTP (delegan a servicios)
│   ├── process.py                  # POST /process
│   ├── upload.py                   # POST /upload
│   ├── pseudo_rhi.py               # POST /process/pseudo_rhi
│   ├── radar_stats.py              # POST /stats/area
│   ├── radar_pixel.py              # POST /stats/pixel
│   ├── elevation_profile.py        # POST /stats/elevation_profile
│   ├── colormap.py                 # GET /colormap/*
│   ├── admin.py                    # GET/POST /admin/*
│   └── cleanup.py                  # POST /cleanup/close
├── services/
│   ├── radar_processor.py          # Lógica central de procesamiento
│   ├── radar_common.py             # Utilidades compartidas (resolve_field, colormaps)
│   ├── grid_generator.py           # Generación de grillas
│   ├── metadata.py                 # Extracción de metadata de archivos
│   ├── pseudo_rhi.py               # Servicio de transectas verticales
│   ├── elevation_profile.py        # Servicio de perfil de elevación
│   ├── orchestrators/
│   │   ├── processing_orchestrator.py  # Orquesta procesamiento multi-archivo
│   │   ├── pixel_orchestrator.py       # Orquesta consulta de pixel
│   │   └── stats_orchestrator.py       # Orquesta estadísticas por área
│   └── radar_processing/
│       ├── grid_builder.py         # Construcción de grillas 3D
│       ├── grid_compute.py         # Cálculos sobre grillas
│       ├── grid_geometry.py        # Geometría de la grilla (resolución, puntos)
│       ├── grid_interpolate.py     # Interpolación Barnes2 con operador sparse
│       ├── product_collapse.py     # Colapso 3D → 2D por producto
│       ├── product_preparation.py  # Preparación de campos y relleno
│       ├── filter_application.py   # Aplicación de filtros QC y visuales
│       ├── cog_generator.py        # Generación de COG (GeoTIFF optimizado)
│       ├── warping.py              # Reproyección a Web Mercator
│       └── geotiff.py              # Escritura de GeoTIFF
```

### Frontend (`frontend/src/`)

```
src/
├── App.jsx                         # Componente principal, estado global, frame merging
├── main.jsx                        # Entry point React
├── api/
│   └── backend.js                  # Cliente Axios para todos los endpoints
├── components/
│   ├── map/
│   │   ├── MapView.jsx             # Mapa Leaflet con COGTile (TileJSON → tiles)
│   │   ├── BaseMapSelector.jsx     # Selector de mapa base
│   │   └── ColorLegend.jsx         # Leyenda de colores
│   ├── controls/
│   │   ├── AnimationControls.jsx   # Reproducción temporal de frames
│   │   ├── MapToolbar.jsx          # Barra de herramientas (screenshot, print, split)
│   │   ├── ActiveLayerPicker.jsx   # Selector de capa activa
│   │   ├── LayerControlList.jsx    # Lista de capas con opacidad
│   │   ├── RadarFilterControls.jsx # Controles de filtros QC
│   │   ├── VerticalToolbar.jsx     # Toolbar vertical lateral
│   │   └── ZoomControls.jsx        # Controles de zoom
│   ├── dialogs/
│   │   ├── ProductSelectorDialog.jsx   # Selección de producto/campo/filtros
│   │   ├── PseudoRHIDialog.jsx         # Transecta vertical (Pseudo-RHI)
│   │   ├── AreaStatsDialog.jsx         # Estadísticas por polígono
│   │   ├── ElevationProfileDialog.jsx  # Perfil de elevación del terreno
│   │   └── LayerManagerDialog.jsx      # Gestión de orden de capas
│   ├── overlays/                   # Overlays de interacción sobre el mapa
│   ├── layout/                     # Componentes de layout
│   └── ui/                         # Componentes UI reutilizables
├── hooks/
│   ├── useMapActions.js            # Acciones del mapa (screenshot, print)
│   ├── useDownloads.js             # Descarga de imágenes/datos
│   └── useSplitScreenSync.js       # Sincronización de mapas duales
└── utils/                          # Utilidades varias
```

---

## Sistema de caché

El backend implementa un sistema de caché multinivel para evitar el reprocesamiento costoso:

| Caché                      | Almacenamiento | Tamaño máx. | Contenido                                                  |
| -------------------------- | -------------- | ----------- | ---------------------------------------------------------- |
| **Grillas 2D**             | RAM (LRU)      | 100 MB      | Arrays 2D colapsados con CRS y transformación              |
| **Operador W**             | RAM (LRU)      | 300 MB      | Matrices dispersas CSR del interpolador Barnes2            |
| **Operador W (spillover)** | Disco (`.npz`) | Sin límite  | Operadores que exceden 250 MB individuales                 |
| **COG en disco**           | Disco          | Sin límite  | GeoTIFFs generados (se reusan si los parámetros coinciden) |

**Claves de caché**: incluyen hash del archivo, producto, campo, elevación/altura, volumen y método de interpolación. Los filtros **no** forman parte de la clave porque se aplican como máscaras post-caché.

**Gestión de sesión**: `SESSION_CACHE_INDEX` mapea cada sesión a sus claves de caché, permitiendo limpieza selectiva cuando el usuario cierra la sesión.

**Concurrencia**: un `NETCDF_READ_LOCK` global serializa lecturas de NetCDF/HDF5 (las librerías subyacentes no son thread-safe). Los operadores W tienen locks individuales por clave para evitar builds duplicados concurrentes.

---

## Procesamiento paralelo

El `ProcessingOrchestrator` ejecuta el procesamiento de múltiples archivos y campos en paralelo usando `ThreadPoolExecutor`. El número de workers se calcula como:

```
max_workers = min(max(4, cpu_count * 2), items * fields)
```

Cada worker procesa un par (archivo, campo) independientemente, accediendo al caché compartido con los locks correspondientes.

---

## Multi-radar y fusión temporal

Cuando se procesan múltiples radares simultáneamente:

1. El backend procesa cada radar independientemente, retornando un `RadarProcessResult` por radar dentro del `ProcessResponse`.
2. El frontend (`mergeRadarFrames()`) agrupa las capas de todos los radares por proximidad temporal (tolerancia de 240 segundos).
3. Los frames fusionados contienen capas de distintos radares ordenadas por el campo `.order`, permitiendo visualización superpuesta.
4. Los controles de animación recorren estos frames fusionados cronológicamente.

---

## TiTiler y servicio de tiles

TiTiler se integra directamente en la aplicación FastAPI (no como servicio separado):

- Se monta como un router en el prefijo `/cog`
- Recibe la ruta absoluta POSIX del COG como parámetro `url`
- Genera tiles en Web Mercator con resampling `nearest` (para preservar los bordes nítidos de los pixeles radar)
- El frontend agrega un timestamp como cache-buster a las URLs de tiles para evitar contaminación entre productos distintos
