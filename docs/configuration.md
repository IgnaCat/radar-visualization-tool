# Parámetros Configurables

Esta guía cataloga los parámetros ajustables del backend: dónde vive cada uno, su
valor por defecto, qué hace, un rango razonable y qué efecto tiene cambiarlo. Es el
mapa para quien continúe el desarrollo.

Los parámetros se dividen en **dos familias**:

- **Operativas / de despliegue** — las toca quien opera o despliega el sistema
  (cuánta RAM de caché, cuántos procesamientos simultáneos, timeouts). Viven en
  `config.py` (override-ables por variable de entorno) o como constantes con nombre
  al tope de su módulo.
- **De dominio / ciencia** — las toca quien cambia el comportamiento científico del
  radar (ROI por volumen, resolución de grilla, rangos de color). Viven en
  `core/constants.py`.

> **Advertencia:** algunos cambios invalidan cachés — ver la sección
> [Efectos colaterales](#efectos-colaterales-e-invalidación-de-caché).

---

## Parámetros operativos

| Parámetro                    | Archivo                                     | Default    | Qué hace                                                                                 | Rango sugerido    |
| ---------------------------- | ------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------- | ----------------- |
| `MAX_CONCURRENT_HEAVY`       | `core/concurrency.py`                       | 2          | Procesamientos pesados simultáneos (resto espera en cola).                               | 2-4 según cores   |
| `SEMAPHORE_TIMEOUT`          | `core/concurrency.py`                       | 300 s      | Espera máxima por un slot antes de responder 503.                                        | 60-600 s          |
| GRID2D cache `maxsize`       | `core/cache.py`                             | 100 MB     | Tope del caché LRU de grillas 2D colapsadas.                                             | 50-300 MB         |
| W cache `maxsize`            | `core/cache.py`                             | 300 MB     | Tope del caché LRU de operadores W en RAM.                                               | 100-500 MB        |
| `W_OPERATOR_MAX_RAM_SIZE_MB` | `core/cache.py`                             | 250        | Tamaño máximo de un W individual para cachear en RAM (los más grandes solo van a disco). | < W cache maxsize |
| `SESSION_TTL_SECONDS`        | `services/inactivity_cleanup.py`            | 7200 (2 h) | Inactividad antes de liberar la RAM de una sesión abandonada.                            | 1800-14400 s      |
| `CHECK_INTERVAL`             | `services/inactivity_cleanup.py`            | 3600 (1 h) | Frecuencia del checker de inactividad.                                                   | 600-7200 s        |
| `W_BUILD_MAX_WORKERS`        | `services/radar_processing/grid_compute.py` | 4          | Cap de procesos del Pool que construye W (límite de RAM pico).                           | 1-8 según RAM     |
| `MAX_UPLOAD_MB`              | `core/config.py` (+ `docker-compose.yml`)   | 100        | Tamaño máximo **por archivo** (no hay tope agregado por request — ver nota abajo).       | según disco       |
| `JWT_EXPIRE_HOURS`           | `core/config.py`                            | 2          | Vigencia del token de sesión.                                                            | 1-24 h            |
| Workers de uvicorn           | `Makefile` / `Dockerfile`                   | 1          | Procesos del servidor. **Ver nota multi-worker.**                                        | 1 recomendado hoy |
| Rotación de logs             | `main.py`                                   | 10 MB      | Tamaño por archivo de log antes de rotar.                                                | 5-50 MB           |

> **Nota sobre `MAX_UPLOAD_MB`:** el límite se aplica a **cada archivo por
> separado**, no a la suma de la request. Subir 3 archivos de 90 MB en una sola
> subida pasa (cada uno < 100). Si se define en `docker-compose.yml`, ese valor
> tiene precedencia sobre el default de `config.py` — cambiarlo en un solo lado
> deja el otro desactualizado.

### Qué es un "procesamiento pesado" y cómo funciona el semáforo

Un **procesamiento pesado** es generar un producto (PPI, CAPPI, COLMAX…) desde un
archivo de radar: leer el NetCDF, construir o aplicar el operador W, colapsar a 2D,
reproyectar y escribir el COG. Es intensivo en CPU y RAM (de segundos a ~1 min con W
en frío), a diferencia de los requests livianos (tiles, `/health`, colormaps, consulta
de pixel) que tardan milisegundos.

Para que N pedidos pesados simultáneos no saturen CPU/RAM y dejen al servidor sin
responder los livianos, un **semáforo** (`core/concurrency.py`) limita a
`MAX_CONCURRENT_HEAVY` los procesamientos pesados en curso. Cada uno adquiere un slot
antes de empezar y lo libera al terminar; si no hay slots, el request espera en cola
(no se rechaza) hasta que se libere uno o hasta `SEMAPHORE_TIMEOUT` segundos, tras lo
cual recibe un `503`. Subir el valor permite procesar más pedidos en paralelo, a costa
de más CPU/RAM pico.

### Variables de entorno de GDAL (`main.py` / `docker-compose.yml`)

Se fijan con `setdefault` en `main.py` (en Docker, `docker-compose.yml` tiene
precedencia). Valores actuales:

| Variable                     | Valor     | Propósito                               |
| ---------------------------- | --------- | --------------------------------------- |
| `GDAL_CACHEMAX`              | 512       | Caché de bloques raster de GDAL, en MB. |
| `GDAL_NUM_THREADS`           | ALL_CPUS  | Threads para descompresión de TIFF.     |
| `VSI_CACHE`                  | TRUE      | Caché de bloques de archivo VSI.        |
| `VSI_CACHE_SIZE`             | 262144000 | Tamaño del caché VSI (~250 MB).         |
| `GDAL_MAX_DATASET_POOL_SIZE` | 450       | Máximo de datasets abiertos.            |
| `PROJ_NETWORK`               | OFF       | Sin descargas de red para CRS.          |

---

## Constantes de dominio / ciencia (`core/constants.py`)

| Parámetro                      | Default     | Qué hace                                                                                          |
| ------------------------------ | ----------- | ------------------------------------------------------------------------------------------------- |
| `TOA`                          | 25000       | Top of Atmosphere en metros: altura máxima de la grilla 3D.                                       |
| `DEFAULT_WEIGHT_FUNC`          | `"nearest"` | Función de ponderación de la interpolación (preserva extremos). Alternativa: `"Barnes2"`.         |
| `DEFAULT_MAX_NEIGHBORS`        | 1           | Vecinos usados por voxel. 1 = vecino más cercano.                                                 |
| `GRID_RESOLUTION_XY_DEFAULT_M` | 1000        | Resolución horizontal de la grilla (volúmenes normales), en metros.                               |
| `GRID_RESOLUTION_XY_VOL03_M`   | 300         | Resolución horizontal para vol 03 (bird bath).                                                    |
| `GRID_RESOLUTION_Z_M`          | 600         | Resolución vertical de la grilla, en metros.                                                      |
| `VOL03_GRID_EXTENT_M`          | 40000.0     | Radio horizontal fijo (m) de la grilla para vol 03.                                               |
| `GRID_RANGE_ROUND_TO_KM`       | 20          | Redondeo del alcance máximo (km) para estabilizar el caché.                                       |
| `BIRD_BATH_ELEV_THRESHOLD_DEG` | 80.0        | Elevación (grados) por encima de la cual se trata el scan como vertical.                          |
| `ROI_PARAMS_VOL01..04`         | ver archivo | Parámetros ROI `(h_factor, nb, bsp, min_radius)` por volumen. **Cambiarlos invalida el caché W.** |
| `AFFECTS_INTERP_FIELDS`        | `{"RHOHV"}` | Campos cuyo filtrado se aplica ANTES de interpolar (afectan la grilla y el cache key).            |

Los diccionarios de render por campo también viven acá:

| Diccionario              | Para qué                                                              |
| ------------------------ | --------------------------------------------------------------------- |
| `FIELD_RENDER`           | `vmin`/`vmax`/`cmap` por campo (rango de color y paleta por defecto). |
| `FIELD_ALIASES`          | Nombres alternativos de cada campo en los NetCDF.                     |
| `FIELD_COLORMAP_OPTIONS` | Paletas ofrecidas al usuario por campo.                               |
| `FIELD_LEGEND_VALUES`    | Valores de referencia de la leyenda por campo.                        |
| `VARIABLE_UNITS`         | Unidad mostrada por campo.                                            |

---

## Receta: agregar un volumen de radar nuevo

Hoy existen los volúmenes `01`-`04` con parámetros ROI propios en
`ROI_PARAMS_BY_VOLUME`. Un volumen no mapeado **cae al default `ROI_PARAMS_VOL01`**
(ver `w_operator_cache_key` en `radar_common.py`). Para agregar, por ejemplo, un
volumen `05`:

1. En `core/constants.py`, definir sus parámetros ROI:
   ```python
   ROI_PARAMS_VOL05 = (1.1, 1.6, 1.3, 900.0)  # (h_factor, nb, bsp, min_radius)
   ```
2. Mapearlo en `ROI_PARAMS_BY_VOLUME`:
   ```python
   ROI_PARAMS_BY_VOLUME = {
       "01": ROI_PARAMS_VOL01,
       # ...
       "05": ROI_PARAMS_VOL05,
   }
   ```
3. Si el volumen es un scan vertical (bird bath), replicar el tratamiento especial
   de `"03"`: la resolución XY (`calculate_grid_resolution` en `grid_geometry.py`) y
   el grid extent fijo (`compute_grid_limits`) chequean `volume == "03"`
   explícitamente. Para otro bird bath habría que extender esas condiciones.
4. Los parámetros ROI nuevos no afectan cachés existentes (el cache key del W
   incluye el volumen), pero conviene borrar `storage/cache/` si se reusa un número
   de volumen con parámetros distintos.

---

## Receta: agregar un campo meteorológico nuevo

Para que el sistema reconozca y renderice un campo nuevo, agregar una entrada en
cada diccionario de `core/constants.py` (usar la clave canónica en mayúsculas):

1. `FIELD_ALIASES` — los nombres con que el campo aparece en los NetCDF.
2. `FIELD_RENDER` — `vmin`, `vmax` y `cmap` por defecto.
3. `FIELD_LEGEND_VALUES` — valores de la leyenda, de mayor a menor.
4. `VARIABLE_UNITS` — la unidad a mostrar.
5. `FIELD_COLORMAP_OPTIONS` — las paletas ofrecidas al usuario.
6. Agregarlo a `AFFECTS_INTERP_FIELDS` **solo si** su filtrado debe aplicarse antes
   de interpolar (como RHOHV para control de calidad).

---

## Efectos colaterales e invalidación de caché

El operador W se cachea por geometría (radar, estrategia, volumen, shape y límites
de grilla, ROI, `weight_func`, `max_neighbors`). Cambiar cualquiera de estos
**invalida los W cacheados** y fuerza reconstruirlos (el cold start de ~1 min por
configuración):

- `ROI_PARAMS_*`, `TOA`, `GRID_RESOLUTION_*`, `VOL03_GRID_EXTENT_M`,
  `GRID_RANGE_ROUND_TO_KM`, `DEFAULT_WEIGHT_FUNC`, `DEFAULT_MAX_NEIGHBORS`.

Los COG en disco se reusan si coinciden los parámetros del nombre de archivo;
cambios en `FIELD_RENDER` (colores) regeneran el COG pero no el W.

**Cómo limpiar cachés:**

- Endpoint admin: `POST /admin/clear-cache` (RAM y/o disco).
- Manual: borrar el contenido de `backend/app/storage/cache/` (operadores W en
  disco) y `backend/app/storage/tmp/` (COGs).

---

## Nota multi-worker

**El sistema hoy está diseñado para correr con un solo worker (un solo proceso).**
Toda la aceleración del backend se apoya en estado que vive en la RAM de ese proceso:
los dos cachés (`GRID2D_CACHE`, `W_OPERATOR_CACHE`), el semáforo de concurrencia
`MAX_CONCURRENT_HEAVY`, los locks de construcción del operador W, los TTLs de sesión y
los tokens de cancelación. Nada de eso se comparte entre procesos.

Un solo worker ya maneja concurrencia real: `run_in_threadpool` atiende varios requests
a la vez, el semáforo limita los procesamientos pesados simultáneos, y la construcción de
W usa un `multiprocessing.Pool` interno. Por eso **se recomienda 1 worker**.

Poner **varios workers para escalar NO es solo cambiar un número** — rompe o degrada
varias cosas, porque cada worker es un proceso aislado con su propia RAM:

- **Estado en RAM duplicado y no compartido.** Cada worker tiene su propio caché y su
  propio semáforo. El hit-rate de caché cae a ~1/N (cada request cae en un worker al
  azar) y el límite real de concurrencia pasa a ser `MAX_CONCURRENT_HEAVY × N_workers`.
  Habría que externalizar el estado compartido (p. ej. Redis) o usar routing sticky por
  sesión detrás de un proxy.
- **Tareas de arranque por-worker.** El `startup` (seed del admin, migraciones, limpieza
  de archivos stale, desactivar sesiones) corre en **cada** worker. Si un worker se
  reinicia (p. ej. lo mata el OOM-killer) borra los uploads de los usuarios conectados.
  Habría que moverlas a un paso de init único, antes de levantar los workers.
- **SQLite.** Con varios procesos escribiendo (access logs, heartbeat) aparece
  `database is locked`. Habría que activar WAL + `busy_timeout`, o migrar a PostgreSQL.
- **Caché de W en disco.** El lock que evita construir el mismo W dos veces es un
  `threading.Lock` que no cruza procesos, y la escritura del `.npz` no es atómica. Con N
  workers se duplican builds (los ~1 min de cold start, repetidos) y un worker puede leer
  un archivo a medio escribir. Habría que usar un lock de archivo + escritura atómica.
- **Memoria.** Cada worker carga su propio ~1 GB de cachés → el consumo se multiplica por N.
- **Logs.** El `RotatingFileHandler` no es multiproceso-safe; habría que loguear a stdout
  o usar un handler apto para varios procesos.

En resumen: escalar horizontalmente es trabajo futuro conocido que toca la arquitectura
de estado, no un ajuste de configuración.
