# Productos Radar y Campos Meteorológicos

## Productos de visualización

El sistema genera productos estándar de radar meteorológico a partir de los datos volumétricos contenidos en los archivos NetCDF. Cada producto representa una forma distinta de proyectar la información 3D del radar a una vista 2D.

### PPI — Plan Position Indicator

Muestra los datos a lo largo del haz del radar a un **ángulo de elevación** determinado. Es la vista más común en meteorología operativa: un disco centrado en el radar donde cada punto corresponde a la distancia y acimut originales del escaneo.

- **Parámetro**: ángulo de elevación (grados).
- **Ejemplo**: PPI a 0.5° muestra los datos del barrido más bajo del radar.

### CAPPI — Constant Altitude Plan Position Indicator

Muestra un corte horizontal a una **altura constante** sobre el nivel del mar. Útil para comparar datos de distintos radares a la misma altitud o para analizar la estructura de tormentas a diferentes niveles.

- **Parámetro**: altura en metros (0 - 12000 m).
- **Ejemplo**: CAPPI a 3000 m muestra la reflectividad a 3 km de altura.

### COLMAX — Column Maximum

Muestra el **valor máximo en la columna vertical** para cada punto horizontal de la grilla. Equivale a una proyección "aplastada" de arriba hacia abajo, útil para identificar las zonas con mayor intensidad de precipitación.

- **Parámetro**: ninguno adicional.

### Pseudo-RHI — Range-Height Indicator

Genera una **sección transversal vertical** entre dos puntos geográficos seleccionados por el usuario. A diferencia de un RHI real (que requiere un escaneo específico del radar), este producto se construye a partir de los datos volumétricos interpolados.

- **Parámetros**: coordenadas de inicio y fin (lat/lon).
- **Campos disponibles**: DBZH, KDP, RHOHV, ZDR.
- **Incluye**: opcionalmente, el perfil de elevación del terreno superpuesto.

---

## Campos meteorológicos

El sistema soporta los siguientes campos, identificados por nombres canónicos que se resuelven automáticamente a los nombres reales presentes en cada archivo de radar:

### Campos de reflectividad

| Campo     | Nombre completo                   | Unidad | Rango típico | Descripción                                                                 |
| --------- | --------------------------------- | ------ | ------------ | --------------------------------------------------------------------------- |
| **DBZH**  | Reflectividad horizontal          | dBZ    | -30 a 70     | Reflectividad principal. Indica la intensidad de los ecos de precipitación. |
| **DBZV**  | Reflectividad vertical            | dBZ    | -30 a 70     | Reflectividad en polarización vertical.                                     |
| **DBZHF** | Reflectividad horizontal filtrada | dBZ    | -30 a 70     | Reflectividad con filtrado previo aplicado.                                 |

### Campos de polarimetría diferencial

| Campo     | Nombre completo             | Unidad | Rango típico | Descripción                                                                                                                    |
| --------- | --------------------------- | ------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **ZDR**   | Reflectividad diferencial   | dB     | -5 a 10.5    | Diferencia entre reflectividad horizontal y vertical. Indica la forma de los hidrometeoros.                                    |
| **RHOHV** | Coeficiente de correlación  | adim.  | 0.3 a 1.0    | Correlación entre señales H y V. Valores altos (>0.95) indican precipitación; valores bajos pueden ser ecos no meteorológicos. |
| **KDP**   | Fase diferencial específica | °/km   | 0 a 8        | Tasa de cambio de la fase diferencial. Útil para estimación de lluvia.                                                         |
| **PHIDP** | Fase diferencial            | °      | -180 a 180   | Fase diferencial acumulada entre señales H y V.                                                                                |

### Campos Doppler

| Campo    | Nombre completo  | Unidad | Rango típico | Descripción                                                                                                   |
| -------- | ---------------- | ------ | ------------ | ------------------------------------------------------------------------------------------------------------- |
| **VRAD** | Velocidad radial | m/s    | -35 a 35     | Velocidad de los hidrometeoros hacia/desde el radar. Positivo = alejándose.                                   |
| **WRAD** | Ancho espectral  | m/s    | 0 a 10       | Dispersión de velocidades dentro del volumen de muestreo. Valores altos indican turbulencia o mezcla de ecos. |

---

## Resolución de campos (aliases)

Cada radar puede usar nombres distintos para el mismo campo físico. El sistema define aliases en `backend/app/core/constants.py` que se resuelven automáticamente:

```
DBZH  → ["DBZH", "corrected_reflectivity_horizontal"]
RHOHV → ["RHOHV", "rhohv"]
VRAD  → ["VRAD", "velocity"]
...
```

La función `resolve_field()` en `radar_common.py` busca el primer alias que exista en el archivo de radar cargado.

---

## Filtros de calidad (QC)

Los filtros permiten enmascarar datos que no cumplen ciertos criterios de calidad, eliminando ecos espurios (clutter, insectos, anomalías de propagación).

### Tipos de filtro

**Filtros QC (campos auxiliares)**: usan un campo diferente al visualizado para enmascarar. Se aplican como máscaras 2D post-caché.

- Ejemplo: al visualizar DBZH, filtrar donde RHOHV < 0.7 descarta ecos con baja correlación (probablemente no meteorológicos).

**Filtros visuales (mismo campo)**: enmascaran valores del propio campo que está siendo visualizado.

- Ejemplo: al visualizar DBZH, filtrar donde DBZH < 10 oculta ecos débiles.

### Aplicación

Los filtros se definen como `RangeFilter` (campo + min + max) y se aplican **después** del cacheo de la grilla 2D. Esto permite reutilizar la misma grilla cacheada con distintas combinaciones de filtros sin reprocesar.

Los campos listados en `AFFECTS_INTERP_FIELDS` (actualmente solo `RHOHV`) se cachean por separado como campos auxiliares junto al campo principal, ya que se necesitan para aplicar los filtros QC.

---

## Colormaps

Cada campo tiene un **colormap por defecto** definido en `FIELD_RENDER` (ej. `pyart_NWSRef` para DBZH), junto con **valores mínimo y máximo** de renderizado.

El usuario puede:

- Seleccionar un colormap diferente de la lista disponible por campo (`FIELD_COLORMAP_OPTIONS`)
- Los colormaps se aplican durante la generación del GeoTIFF (paso de colorización previo a la creación del COG)

---

## Parámetros de interpolación por volumen

La grilla cartesiana se construye con parámetros que varían según el volumen de escaneo, optimizados para la resolución espacial de cada estrategia:

| Volumen    | h_factor | nb       | bsp      | min_radius | Uso                               |
| ---------- | -------- | -------- | -------- | ---------- | --------------------------------- |
| 01, 02, 04 | Estándar | Estándar | Estándar | Estándar   | Volúmenes normales (grilla 1000m) |
| 03         | Mayor    | Mayor    | Mayor    | Mayor      | Alta resolución (grilla 300m)     |

El volumen 03, al tener una resolución más fina, utiliza parámetros de interpolación más amplios para el método Barnes2, generando una grilla de mayor detalle pero mayor costo computacional.
