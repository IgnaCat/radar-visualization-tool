/**
 * Utilidades para analizar y clasificar campos de múltiples archivos de radar.
 * Ayuda a identificar campos comunes vs campos específicos de archivo para mejor UX.
 */

/**
 * Extraer información de origen del archivo (radar, estrategia, volumen) desde metadata
 * @param {Object} fileInfo - Objeto de información del archivo con metadata
 * @returns {Object} - Información de origen extraída
 */
function extractFileSource(fileInfo) {
  const metadata = fileInfo?.metadata || {};
  const filepath = fileInfo?.filepath || "";

  // Extraer información del nombre del archivo
  // Formato esperado: {RADAR}_{STRATEGY}_{VOLUME}_{TIMESTAMP}Z.nc
  // Ejemplo: RMA1_0315_01_20250830T203606Z.nc
  const filename = filepath.split("/").pop(); // Obtener solo el nombre del archivo
  const parts = filename.split("_");

  let radar = "Unknown";
  let strategy = "Unknown";
  let volume = "Unknown";

  if (parts.length >= 3) {
    radar = parts[0]; // RMA1
    strategy = parts[1]; // 0315
    volume = parts[2]; // 01
  }

  // Fallback a metadata si no se pudo extraer del filename
  if (radar === "Unknown") {
    radar =
      metadata.radar ||
      metadata.radar_site?.name ||
      metadata.site?.name ||
      "Unknown";
  }
  if (strategy === "Unknown") {
    strategy = metadata.strategy || "Unknown";
  }
  if (volume === "Unknown") {
    volume = metadata.volume || "Unknown";
  }

  return {
    radar,
    strategy,
    volume,
    filepath,
  };
}

/**
 * Crear un identificador único para una fuente de archivo
 * @param {Object} source - Objeto de origen con radar, estrategia, volumen
 * @returns {string} - Identificador único
 */
function createSourceId(source) {
  return `${source.radar}_${source.strategy}_${source.volume}`;
}

/**
 * Analizar campos a través de múltiples archivos y clasificarlos por disponibilidad
 * @param {Array} filesInfo - Array de objetos de información de archivos
 * @returns {Object} Resultado del análisis con campos comunes y específicos
 */
export function analyzeFieldsAcrossFiles(filesInfo) {
  if (!Array.isArray(filesInfo) || filesInfo.length === 0) {
    return {
      commonFields: [],
      specificFields: [],
      allFields: [],
      sources: [],
    };
  }

  // Construir un mapa de campo -> conjunto de IDs de origen que lo tienen
  const fieldToSources = new Map();
  const sourceMap = new Map(); // sourceId -> objeto de origen

  filesInfo.forEach((fileInfo) => {
    const source = extractFileSource(fileInfo);
    const sourceId = createSourceId(source);

    // Almacenar información de origen
    if (!sourceMap.has(sourceId)) {
      sourceMap.set(sourceId, source);
    }

    const fields = fileInfo?.metadata?.fields_present || [];

    fields.forEach((field) => {
      if (!fieldToSources.has(field)) {
        fieldToSources.set(field, new Set());
      }
      fieldToSources.get(field).add(sourceId);
    });
  });

  const allSourceIds = Array.from(sourceMap.keys());
  const totalSources = allSourceIds.length;

  // Detectar si todas las fuentes tienen el mismo radar y estrategia
  const allSources = Array.from(sourceMap.values());
  const firstSource = allSources[0];
  const sameRadarStrategy = allSources.every(
    (s) => s.radar === firstSource.radar && s.strategy === firstSource.strategy,
  );

  // Clasificar campos
  const commonFields = [];
  const specificFieldsMap = new Map(); // sourceId -> array de campos

  // Si solo hay una fuente única (mismo radar/estrategia/volumen), no marcar nada como común
  if (totalSources <= 1) {
    // Todos los campos son "normales" - sin clasificación de común/específico
    // No agregamos nada a commonFields, dejamos que se muestren sin chips
  } else {
    // Hay múltiples fuentes, clasificar campos
    fieldToSources.forEach((sourceIds, field) => {
      if (sourceIds.size === totalSources) {
        // El campo está presente en todos los archivos
        commonFields.push(field);
      } else {
        // El campo es específico de algunos archivos
        sourceIds.forEach((sourceId) => {
          if (!specificFieldsMap.has(sourceId)) {
            specificFieldsMap.set(sourceId, []);
          }
          specificFieldsMap.get(sourceId).push(field);
        });
      }
    });
  }

  // Convertir el mapa de campos específicos a formato de array
  const specificFields = Array.from(specificFieldsMap.entries()).map(
    ([sourceId, fields]) => ({
      source: sourceMap.get(sourceId),
      sourceId,
      fields,
    }),
  );

  // Ordenar campos específicos por número de campos (descendente) para mejor UX
  specificFields.sort((a, b) => b.fields.length - a.fields.length);

  // Obtener todos los campos únicos en orden: comunes primero, luego específicos
  let allFields;
  if (totalSources <= 1) {
    // Si solo hay una fuente, todos los campos sin clasificación
    allFields = Array.from(fieldToSources.keys());
  } else {
    // Múltiples fuentes: comunes primero, luego específicos
    allFields = [
      ...commonFields,
      ...specificFields.flatMap((sf) =>
        sf.fields.filter((f) => !commonFields.includes(f)),
      ),
    ];
  }
  // Eliminar duplicados de allFields
  const uniqueAllFields = Array.from(new Set(allFields));

  return {
    commonFields,
    specificFields,
    allFields: uniqueAllFields,
    sources: Array.from(sourceMap.values()),
    sameRadarStrategy, // Indica si todas las sources tienen mismo radar/estrategia
  };
}

/**
 * Formatear información de origen para mostrar
 * @param {Object} source - Objeto de origen con radar, estrategia, volumen
 * @param {boolean} simplified - Si es true, solo mostrar volumen (cuando todas las sources tienen mismo radar/estrategia)
 * @returns {string} - Cadena formateada para mostrar
 */
export function formatSourceDisplay(source, simplified = false) {
  // Si simplified es true (mismo radar/estrategia en todos), solo mostrar volumen
  if (simplified) {
    if (source.volume && source.volume !== "Unknown") {
      return `Vol ${source.volume}`;
    }
    return "Unknown";
  }

  // Modo completo: mostrar radar · volumen · estrategia
  const parts = [];

  if (source.radar && source.radar !== "Unknown") {
    parts.push(source.radar);
  }

  if (source.volume && source.volume !== "Unknown") {
    parts.push(`Vol ${source.volume}`);
  }

  if (source.strategy && source.strategy !== "Unknown") {
    parts.push(`Strat ${source.strategy}`);
  }

  return parts.length > 0 ? parts.join(" · ") : "Unknown";
}

/**
 * Verificar si un campo está disponible en la selección actual de archivos
 * @param {string} field - Nombre del campo
 * @param {Array} filesInfo - Array de objetos de información de archivos
 * @param {Array} selectedVolumes - Números de volumen seleccionados
 * @param {Array} selectedRadars - Nombres de radar seleccionados
 * @returns {boolean} - True si el campo está disponible en la selección
 */
export function isFieldAvailableInSelection(
  field,
  filesInfo,
  selectedVolumes,
  selectedRadars,
) {
  return filesInfo.some((fileInfo) => {
    const source = extractFileSource(fileInfo);
    const fields = fileInfo?.metadata?.fields_present || [];

    // Verificar si este archivo coincide con la selección
    const volumeMatch =
      !selectedVolumes ||
      selectedVolumes.length === 0 ||
      selectedVolumes.includes(source.volume);

    const radarMatch =
      !selectedRadars ||
      selectedRadars.length === 0 ||
      selectedRadars.includes(source.radar);

    return volumeMatch && radarMatch && fields.includes(field);
  });
}
