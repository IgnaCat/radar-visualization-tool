"""
Funciones de suavizado para productos 2D de radar.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, generic_filter


def apply_gaussian_smoothing_masked(
    arr: np.ma.MaskedArray,
    sigma: float,
) -> np.ma.MaskedArray:
    """
    Aplica suavizado gaussiano preservando máscaras (sin mezclar nodata con datos).

    Se implementa con convolución normalizada:
    - suaviza el campo de datos con nodata rellenado en 0
    - suaviza por separado una máscara/peso de validez (1 válido, 0 inválido)
    - divide ambos para evitar que nodata contamine bordes
    """
    if sigma <= 0:
        return arr

    original_mask = np.ma.getmaskarray(arr)
    data = np.ma.filled(arr, fill_value=np.nan).astype(np.float32)
    valid = ((~original_mask) & np.isfinite(data)).astype(np.float32)
    data_filled = np.where(np.isfinite(data), data, 0.0).astype(np.float32)

    smoothed_data = gaussian_filter(data_filled, sigma=sigma, mode="nearest")
    smoothed_weights = gaussian_filter(valid, sigma=sigma, mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            smoothed_weights > 1e-6,
            smoothed_data / smoothed_weights,
            np.nan,
        )

    out = out.astype(np.float32)
    final_mask = original_mask | (~np.isfinite(out))
    return np.ma.array(out, mask=final_mask)


def apply_median_smoothing_masked(
    arr: np.ma.MaskedArray,
    size: int,
) -> np.ma.MaskedArray:
    """
    Aplica suavizado por mediana ignorando valores enmascarados/NaN.

    Usa generic_filter + np.nanmedian para no contaminar con nodata.
    """
    if size <= 1:
        return arr

    # Asegurar ventana impar para un centro bien definido.
    if size % 2 == 0:
        size += 1

    original_mask = np.ma.getmaskarray(arr)
    data = np.ma.filled(arr, fill_value=np.nan).astype(np.float32)

    def _nanmedian(window):
        valid = window[np.isfinite(window)]
        if valid.size == 0:
            return np.nan
        return float(np.median(valid))

    out = generic_filter(
        data,
        function=_nanmedian,
        size=size,
        mode="nearest",
    ).astype(np.float32)

    final_mask = original_mask | (~np.isfinite(out))
    return np.ma.array(out, mask=final_mask)


def apply_smoothing_masked(
    arr: np.ma.MaskedArray,
    method: str = "median",
    sigma: float = 0.8,
    median_size: int = 3,
) -> np.ma.MaskedArray:
    """Despacha al método de suavizado configurado."""
    m = (method or "median").lower()
    if m == "median":
        return apply_median_smoothing_masked(arr, int(median_size))
    return apply_gaussian_smoothing_masked(arr, float(sigma))
