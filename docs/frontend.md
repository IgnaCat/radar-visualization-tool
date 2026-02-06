# Frontend — Interfaz de Visualización

## Tecnologías principales

- **React 18** como framework UI
- **Leaflet** (via `react-leaflet`) para renderizado de mapas
- **Vite** como bundler y dev server
- **Material UI (MUI)** para componentes de interfaz
- **Axios** para comunicación con el backend
- **Notistack** para notificaciones tipo snackbar

---

## Estructura general

La aplicación se organiza en un componente principal (`App.jsx`) que gestiona el estado global y delega la visualización a componentes especializados agrupados por función.

### `App.jsx` — Componente principal

Responsabilidades:

- **Estado global**: archivos subidos, capas activas, frames de animación, producto/campo seleccionado, filtros activos
- **Llamadas a la API**: orquesta `uploadFile`, `processFile`, `generatePseudoRHI`, `generateAreaStats`, `generatePixelStat`, `generateElevationProfile`
- **Fusión multi-radar** (`mergeRadarFrames`): agrupa capas de distintos radares en frames temporales con tolerancia de 240 segundos
- **Deduplicación** (`buildComputeKey`): evita reprocesar si los parámetros no cambiaron

---

## Componentes del mapa

### `MapView.jsx`

Mapa interactivo Leaflet que renderiza:

- **Mapa base** seleccionable (OpenStreetMap, satélite, etc.)
- **Capas COG** dinámicas mediante el componente interno `COGTile`:
  - Obtiene el TileJSON desde TiTiler
  - Extrae la URL template de tiles
  - Renderiza como `TileLayer` de Leaflet con opacidad y z-index configurables
- **Overlays de interacción**: puntos de click, polígonos de área, líneas de transecta
- **Leyenda de colores** (`ColorLegend`)

### `BaseMapSelector.jsx`

Selector de mapa base con opciones predefinidas (calles, satélite, topográfico, oscuro).

### `ColorLegend.jsx`

Leyenda de colores que muestra el colormap activo con los valores mínimo y máximo del campo visualizado.

---

## Controles

### `AnimationControls.jsx`

Controles de reproducción para navegar frames temporales:

- Play/Pause con intervalo de 1300 ms
- Avance/retroceso frame a frame
- Slider para selección directa de frame
- Loop automático al llegar al último frame
- Muestra metadata del frame actual (timestamp, producto, radares)

### `MapToolbar.jsx`

Barra de herramientas del mapa (esquina superior derecha):

- Captura de pantalla (screenshot)
- Impresión del mapa
- Pantalla completa
- Menú de descarga de datos/imágenes
- Modo pantalla dividida (split screen) para comparar dos visualizaciones
- Bloqueo de sincronización entre mapas divididos

### `ActiveLayerPicker.jsx`

Selector rápido de la capa activa. La capa activa determina qué campo responde a las herramientas de consulta (pixel, estadísticas, etc.).

### `LayerControlList.jsx`

Lista de capas con controles de opacidad individual por capa.

### `RadarFilterControls.jsx`

Controles para definir filtros de calidad (QC). Permite agregar filtros por campo con rangos min/max.

### `ZoomControls.jsx`

Controles de zoom del mapa.

### `VerticalToolbar.jsx`

Toolbar lateral con acceso rápido a herramientas.

---

## Diálogos

### `ProductSelectorDialog.jsx`

Diálogo principal de configuración de procesamiento:

- Selección de **producto** (PPI, CAPPI, COLMAX)
- Selección de **campos** a visualizar (DBZH, ZDR, RHOHV, etc.) con sliders de rango
- Configuración de **altura** (CAPPI) o **elevación** (PPI)
- Gestión de **filtros QC** (ej. RHOHV > 0.7)
- Selección de **volúmenes** y **radares** a procesar
- Personalización de **colormaps** por campo

### `PseudoRHIDialog.jsx`

Generación de transectas verticales (Pseudo-RHI):

- El usuario selecciona dos puntos en el mapa
- Elige el campo a visualizar (`DBZH`, `KDP`, `RHOHV`, `ZDR`)
- El backend calcula la sección transversal vertical entre ambos puntos
- Se muestra un gráfico de elevación con los datos superpuestos
- Opcionalmente incluye perfil del terreno
- Soporta descarga de la imagen

### `AreaStatsDialog.jsx`

Estadísticas sobre un polígono dibujado en el mapa:

- El usuario dibuja un polígono
- Se calculan: mínimo, máximo, media, desviación estándar
- Se muestra un histograma de valores
- Soporta múltiples consultas simultáneas

### `ElevationProfileDialog.jsx`

Perfil de elevación del terreno:

- El usuario dibuja una línea en el mapa
- Se genera un gráfico con la altura del terreno a lo largo de la línea
- Permite resaltar puntos del perfil en el mapa

### `LayerManagerDialog.jsx`

Gestión del orden de capas mediante drag-and-drop:

- La primera capa se convierte en la capa activa para herramientas
- Sincroniza con el `ProductSelectorDialog` y `ActiveLayerPicker`

---

## Hooks personalizados

| Hook                 | Función                                                    |
| -------------------- | ---------------------------------------------------------- |
| `useMapActions`      | Acciones del mapa: captura de pantalla, impresión          |
| `useDownloads`       | Descarga de imágenes y datos procesados                    |
| `useSplitScreenSync` | Sincronización de pan/zoom entre dos MapView en modo split |

---

## Cliente API (`api/backend.js`)

Centraliza todas las llamadas al backend usando Axios. La URL base se configura con la variable de entorno `VITE_API_URL` (default: `http://localhost:8000`).

| Función                    | Endpoint                        | Uso                                |
| -------------------------- | ------------------------------- | ---------------------------------- |
| `uploadFile`               | `POST /upload`                  | Subir archivos NetCDF              |
| `processFile`              | `POST /process`                 | Procesar radar (producto + campos) |
| `generatePseudoRHI`        | `POST /process/pseudo_rhi`      | Transecta vertical                 |
| `generateAreaStats`        | `POST /stats/area`              | Estadísticas por polígono          |
| `generatePixelStat`        | `POST /stats/pixel`             | Valor en un punto                  |
| `generateElevationProfile` | `POST /stats/elevation_profile` | Perfil de elevación                |
| `getColormapOptions`       | `GET /colormap/options`         | Opciones de colormap               |
| `getColormapDefaults`      | `GET /colormap/defaults`        | Colormaps por defecto              |
| `getColormapColors`        | `GET /colormap/colors/:name`    | Colores RGB de un colormap         |
| `getCacheStats`            | `GET /admin/cache-stats`        | Estadísticas de caché              |
| `clearCache`               | `POST /admin/clear-cache`       | Limpiar caché                      |
| `cleanupClose`             | `POST /cleanup/close`           | Limpiar sesión                     |
