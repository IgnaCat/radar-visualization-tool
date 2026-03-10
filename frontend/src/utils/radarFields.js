/**
 * Metadatos compartidos de campos de radar, usados en diálogos y controles.
 */

/** Mínimo y máximo físico de cada campo de radar — usado para sliders y valores por defecto de filtros. */
export const FIELD_LIMITS = {
    DBZH: { min: -30, max: 70 },
    DBZV: { min: -30, max: 70 },
    DBZHF: { min: -30, max: 70 },
    ZDR: { min: -5, max: 10.5 },
    RHOHV: { min: 0.3, max: 1.0 },
    KDP: { min: 0, max: 8 },
    VRAD: { min: -35, max: 35 },
    WRAD: { min: 0, max: 10 },
    PHIDP: { min: -180, max: 180 },
};

/** Marcas del slider para el rango 0–1 (p. ej. sliders de RHOHV). */
export const MARKS_01 = [
    { value: 0, label: "0" },
    { value: 0.25, label: "0.25" },
    { value: 0.5, label: "0.5" },
    { value: 0.75, label: "0.75" },
    { value: 1, label: "1" },
];

/**
 * Devuelve el estado inicial de filtros internos para un campo dado.
 * Forma: { rhohv: { enabled, min }, range: { enabled, min, max } }
 */
export function initFilterForField(field) {
    const lim = FIELD_LIMITS[String(field).toUpperCase()] || { min: 0, max: 1 };
    return {
        rhohv: { enabled: false, min: 0.8 },
        range: { enabled: false, min: lim.min, max: lim.max },
    };
}

/**
 * Convierte el estado interno de filtros por campo (usado en LayerManagerDialog)
 * al formato `filters_per_field` del backend:
 *   { FIELD: [{ field, min, max }, ...] }
 *
 * Forma de localFilters: { FIELD: { rhohv: {enabled, min}, range: {enabled, min, max} } }
 */
export function convertToBackend(localFilters) {
    const result = {};
    for (const [field, f] of Object.entries(localFilters)) {
        if (!f) continue;
        const key = String(field).toUpperCase();
        const arr = [];
        if (f.rhohv?.enabled) {
            arr.push({ field: "RHOHV", min: Number(f.rhohv.min ?? 0.8), max: 1.0 });
        }
        if (f.range?.enabled) {
            const lim = FIELD_LIMITS[key] || { min: -999, max: 999 };
            arr.push({
                field: key,
                min: Number(f.range.min ?? lim.min),
                max: Number(f.range.max ?? lim.max),
            });
        }
        if (arr.length > 0) {
            result[key] = arr;
        }
    }
    return result;
}
