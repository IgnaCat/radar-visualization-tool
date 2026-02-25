"""
Generación de COG (Cloud Optimized GeoTIFF) con RGB y transparencia.
"""
import numpy as np
import rasterio
from rasterio.shutil import copy
from rasterio.enums import Resampling
from pathlib import Path
import shutil


def create_cog_from_warped_array(
    data_warped, output_path, transform, crs, cmap, vmin, vmax
):
    """
    Crea un COG (Cloud Optimized GeoTIFF) RGB desde un array ya warped a Web Mercator.
    
    Args:
        data_warped: Array numpy 2D ya proyectado en Web Mercator con valores numéricos
        output_path: Path donde guardar el COG
        transform: Affine transform del array warped
        crs: CRS del array warped (debe ser Web Mercator)
        cmap: Matplotlib colormap para RGB
        vmin, vmax: Rango de valores para normalización
    
    Returns:
        Path del COG creado
    """
    # Aplicar colormap para generar RGBA con transparencia
    if not np.ma.is_masked(data_warped):
        data_masked = np.ma.masked_invalid(data_warped)
    else:
        data_masked = data_warped.copy()
    
    # Enmascarar valores inválidos (NaN, inf)
    data_masked = np.ma.masked_invalid(data_masked)
    
    # Normalizar solo los valores válidos
    data_norm = (data_masked - vmin) / (vmax - vmin)
    data_norm = np.clip(data_norm, 0, 1)
    
    # Crear máscara de transparencia ANTES de aplicar colormap
    mask = data_masked.mask
    
    # Aplicar colormap solo a valores válidos (retorna RGBA)
    # Para valores enmascarados, filled() pone 0 temporalmente pero serán reemplazados
    rgba = cmap(data_norm.filled(0))
    data_rgba = (rgba * 255).astype(np.uint8)
    
    # Crear alpha channel
    alpha = np.where(mask, 0, 255).astype(np.uint8)
    
    # Poner pixels enmascarados en negro (0,0,0) para evitar halos de colormap
    # Esto previene que el colormap aplicado a valores enmascarados genere artefactos visibles
    for i in range(3):
        data_rgba[:, :, i] = np.where(mask, 0, data_rgba[:, :, i])
    
    # Configuración del COG optimizado con 4 bandas (RGBA)
    height, width = data_warped.shape
    profile = {
        'driver': 'COG',
        'height': height,
        'width': width,
        'count': 4,  # RGBA = 4 bandas
        'dtype': np.uint8,
        'crs': crs,
        'transform': transform,
        'compress': 'DEFLATE',
        # COG driver options (NOT GTiff — TILED/BLOCKXSIZE/BLOCKYSIZE/PHOTOMETRIC are invalid here)
        'BLOCKSIZE': 512,
        'OVERVIEW_RESAMPLING': 'NEAREST',
        'RESAMPLING': 'NEAREST',
        'NUM_THREADS': 'ALL_CPUS',
        'BIGTIFF': 'IF_SAFER'
    }
    
    # Escribir COG con las 4 bandas RGBA
    with rasterio.open(output_path, 'w', **profile) as dst:
        # Escribir RGB
        for i in range(3):
            dst.write(data_rgba[:, :, i], i + 1)
        # Escribir Alpha (canal de transparencia)
        dst.write(alpha, 4)
        # Marcar banda 4 como alpha
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha
        ]
    
    return output_path


def convert_to_cog(src_path, cog_path):
    """
    Convierte un GeoTIFF existente a un COG (Cloud Optimized GeoTIFF) optimizado para tiling rápido.
    Re-escribe completamente el archivo con estructura tiled y overviews.
    
    NOTA: Esta función se mantiene para compatibilidad legacy. 
    Para nuevos desarrollos usar create_cog_from_warped_array().
    
    Args:
        src_path: Path del GeoTIFF fuente
        cog_path: Path del COG destino
    
    Returns:
        Path del COG creado
    """
    # Si el COG ya existe y es válido, no regenerar
    if cog_path.exists():
        try:
            with rasterio.open(cog_path) as test:
                if (test.profile.get('tiled') and 
                    test.overviews(1) and 
                    test.profile.get('blockxsize', 0) >= 256):
                    return cog_path
        except:
            pass    
    try:
        # Abrir con IGNORE_COG_LAYOUT_BREAK para permitir modificar archivos COG
        with rasterio.open(src_path, IGNORE_COG_LAYOUT_BREAK='YES') as src:
            # Configurar perfil COG optimizado con tiles grandes
            # SIN compresión para máxima velocidad de lectura en tiles
            profile = src.profile.copy()
            profile.update(
                driver='COG',
                compress='DEFLATE',
                # COG driver options (TILED/BLOCKXSIZE/BLOCKYSIZE are GTiff-only)
                BLOCKSIZE=512,
                BIGTIFF='IF_NEEDED',
                NUM_THREADS='ALL_CPUS',
                COPY_SRC_OVERVIEWS='YES',
            )
            # Remove GTiff-specific keys that leaked from src.profile
            for gtiff_key in ('tiled', 'blockxsize', 'blockysize', 'photometric', 'interleave'):
                profile.pop(gtiff_key, None)
            
            # Escribir archivo intermedio tiled
            temp_tiled = Path(cog_path.parent) / f"temp_{cog_path.name}"
            
            with rasterio.open(temp_tiled, 'w', **profile) as dst:
                # Copiar todas las bandas
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
                
                # Copiar colorinterp si existe
                try:
                    dst.colorinterp = src.colorinterp
                except:
                    pass
            
            # Ahora agregar overviews al archivo tiled
            # También necesita IGNORE_COG_LAYOUT_BREAK porque el driver COG ya optimizó el archivo
            with rasterio.open(temp_tiled, 'r+', IGNORE_COG_LAYOUT_BREAK='YES') as dst:
                factors = [2, 4, 8, 16, 32]
                dst.build_overviews(factors, Resampling.nearest)
                dst.update_tags(ns='rio_overview', resampling='nearest')
            
            # Mover archivo temporal al destino final
            # shutil.move elimina el temp_tiled automáticamente
            shutil.move(str(temp_tiled), str(cog_path))
            
    except Exception as e:
        print(f"Error generando COG optimizado: {e}")
        
        # Limpiar archivo temporal si quedó creado
        try:
            if temp_tiled.exists():
                temp_tiled.unlink()
        except:
            pass
        
        # Fallback: copiar el original
        if not cog_path.exists():
            shutil.copy2(src_path, cog_path)
    
    return cog_path
