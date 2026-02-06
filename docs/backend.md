# Backend — API y Servicios

## Tecnologías principales

- **FastAPI** como framework web
- **PyART** (Py-ART) para lectura y procesamiento de datos de radar
- **Rasterio / GDAL** para generación de GeoTIFF y COG
- **TiTiler** para servicio dinámico de tiles
- **NumPy / SciPy** para operaciones numéricas y matrices dispersas
- **cachetools** para cachés LRU en memoria

---

## Endpoints de la API

### Subida de archivos

| Método | Ruta      | Descripción                                                                                                                                                               |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/upload` | Subida de archivos NetCDF (multipart). Valida extensión (`.nc`) y tamaño (máx. 500 MB). Retorna metadata extraída: radares detectados, volúmenes disponibles, timestamps. |

### Procesamiento

| Método | Ruta                  | Descripción                                                                                                                         |
| ------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/process`            | Procesamiento principal. Recibe archivos, producto, campos, filtros y parámetros → genera COGs → retorna URLs de TileJSON por capa. |
| `POST` | `/process/pseudo_rhi` | Genera una transecta vertical (Pseudo-RHI) entre dos coordenadas geográficas.                                                       |

### Estadísticas y consultas

| Método | Ruta                       | Descripción                                                                                                       |
| ------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `POST` | `/stats/area`              | Calcula estadísticas (min, max, media, desv. estándar, histograma) sobre un polígono GeoJSON dibujado en el mapa. |
| `POST` | `/stats/pixel`             | Extrae el valor de un campo en una coordenada puntual.                                                            |
| `POST` | `/stats/elevation_profile` | Genera un perfil de elevación del terreno a lo largo de una línea definida por coordenadas.                       |

### Colormaps

| Método | Ruta                      | Descripción                                              |
| ------ | ------------------------- | -------------------------------------------------------- |
| `GET`  | `/colormap/options`       | Devuelve los nombres de colormaps disponibles por campo. |
| `GET`  | `/colormap/defaults`      | Devuelve el colormap por defecto de cada campo.          |
| `GET`  | `/colormap/colors/{name}` | Devuelve los valores RGB de un colormap específico.      |

### Administración

| Método | Ruta                 | Descripción                                                    |
| ------ | -------------------- | -------------------------------------------------------------- |
| `GET`  | `/admin/cache-stats` | Estadísticas de uso de caché (tamaño, hits, items).            |
| `POST` | `/admin/clear-cache` | Limpia las cachés de grillas en memoria.                       |
| `POST` | `/cleanup/close`     | Limpia archivos subidos, COGs generados y caché de una sesión. |

### TiTiler (tiles dinámicos)

| Método | Ruta                                            | Descripción                                                     |
| ------ | ----------------------------------------------- | --------------------------------------------------------------- |
| `GET`  | `/cog/WebMercatorQuad/tilejson.json?url={path}` | Retorna TileJSON con la URL template de tiles para un COG dado. |
| `GET`  | `/cog/tiles/WebMercatorQuad/{z}/{x}/{y}`        | Tile individual en formato PNG.                                 |

---

## Modelos de datos principales

### ProcessRequest

Modelo de entrada para el endpoint `/process`:

| Campo                | Tipo                | Descripción                                   |
| -------------------- | ------------------- | --------------------------------------------- |
| `filepaths`          | `list[str]`         | Rutas de los archivos NetCDF a procesar       |
| `product`            | `str`               | Producto: `"PPI"`, `"CAPPI"` o `"COLMAX"`     |
| `fields`             | `list[str]`         | Campos a procesar (ej. `["DBZH", "ZDR"]`)     |
| `height`             | `float`             | Altura en metros (para CAPPI)                 |
| `elevation`          | `float`             | Ángulo de elevación en grados (para PPI)      |
| `filters`            | `list[RangeFilter]` | Filtros QC (campo + rango min/max)            |
| `selectedVolumes`    | `list[str]`         | Volúmenes a incluir (ej. `["01", "02"]`)      |
| `selectedRadars`     | `list[str]`         | Radares a incluir (ej. `["RMA1", "RMA3"]`)    |
| `colormap_overrides` | `dict`              | Colormaps personalizados por campo            |
| `session_id`         | `str`               | Identificador de sesión para gestión de caché |

### LayerResult

Resultado de una capa procesada:

| Campo          | Tipo   | Descripción                                        |
| -------------- | ------ | -------------------------------------------------- |
| `tilejson_url` | `str`  | URL al TileJSON de TiTiler para esta capa          |
| `image_url`    | `str`  | URL al COG estático                                |
| `field`        | `str`  | Campo procesado                                    |
| `order`        | `int`  | Orden de la capa (para visualización)              |
| `bounds`       | `list` | Bounding box geográfico `[minx, miny, maxx, maxy]` |
| `timestamp`    | `str`  | Timestamp ISO del dato                             |
| `radar_name`   | `str`  | Nombre del radar de origen                         |
| `colormap`     | `dict` | Información del colormap aplicado                  |
| `metadata`     | `dict` | Metadata adicional (elevación, altura, producto)   |

### RangeFilter

Filtro de calidad aplicado post-procesamiento:

| Campo   | Tipo    | Descripción                                  |
| ------- | ------- | -------------------------------------------- |
| `field` | `str`   | Campo sobre el que se aplica (ej. `"RHOHV"`) |
| `min`   | `float` | Valor mínimo aceptado                        |
| `max`   | `float` | Valor máximo aceptado                        |

---

## Servicios principales

### `radar_processor.py`

Lógica central de procesamiento. Se encarga de:

- Generar nombres únicos para los COG (basados en hash del archivo + parámetros)
- Coordinar el pipeline: lectura → grillado → colapso → filtrado → COG
- Verificar caché antes de reprocesar
- Construir el resumen de salida (`LayerResult`)

### Orquestadores (`services/orchestrators/`)

- **`ProcessingOrchestrator`**: valida el request, resuelve rutas de archivos, paraleliza el procesamiento de múltiples archivos/campos con `ThreadPoolExecutor`, y agrupa los resultados por radar.
- **`PixelOrchestrator`**: lee el valor de un pixel específico en un COG existente.
- **`StatsOrchestrator`**: calcula estadísticas zonales sobre un polígono GeoJSON.

### Pipeline de procesamiento (`services/radar_processing/`)

El procesamiento se descompone en módulos especializados:

| Módulo                   | Responsabilidad                                                |
| ------------------------ | -------------------------------------------------------------- |
| `grid_builder.py`        | Construye la grilla 3D a partir del radar con operador Barnes2 |
| `grid_compute.py`        | Cálculos auxiliares sobre grillas                              |
| `grid_geometry.py`       | Calcula resolución y puntos de grilla según volumen            |
| `grid_interpolate.py`    | Interpola datos polares a la grilla usando matrices dispersas  |
| `product_collapse.py`    | Colapsa la grilla 3D a 2D según el producto (PPI/CAPPI/COLMAX) |
| `product_preparation.py` | Preparación de campos, relleno de DBZH si falta                |
| `filter_application.py`  | Aplica filtros QC y visuales como máscaras sobre el array 2D   |
| `cog_generator.py`       | Genera el COG con overviews y compresión                       |
| `warping.py`             | Reproyecta la grilla a Web Mercator (EPSG:3857)                |
| `geotiff.py`             | Escritura de GeoTIFF intermedio                                |

---

## Configuración

La configuración se gestiona mediante `Pydantic BaseSettings` en `core/config.py`:

| Variable           | Default                             | Descripción              |
| ------------------ | ----------------------------------- | ------------------------ |
| `APP_NAME`         | `"Radar Visualization"`             | Nombre de la aplicación  |
| `BASE_URL`         | `"http://localhost:8000"`           | URL base del backend     |
| `FRONTEND_ORIGINS` | `["http://localhost:3000"]`         | Orígenes CORS permitidos |
| `ALLOWED_PRODUCTS` | `["PPI", "RHI", "CAPPI", "COLMAX"]` | Productos habilitados    |
| `MAX_UPLOAD_SIZE`  | `500 MB`                            | Tamaño máximo de archivo |

Las variables pueden sobreescribirse mediante un archivo `.env` en la raíz del backend.
