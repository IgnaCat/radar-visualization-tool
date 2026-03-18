"""
Funciones de suavizado para productos 2D de radar.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


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
