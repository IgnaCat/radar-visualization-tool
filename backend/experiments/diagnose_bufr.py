"""
Script de diagnóstico para verificar conversión BUFR → NetCDF.
Analiza la cantidad de datos válidos (no-masked) en cada etapa de la conversión.

Uso:
    python -m backend.diagnose_bufr <archivo.BUFR> [archivo.nc]

Ejemplo:
    python -m backend.diagnose_bufr RMA11_0315_01_DBZH_20251020T152828Z.BUFR
"""

import sys
import numpy as np
import pyart
from pathlib import Path


def diagnose_bufr_conversion(bufr_file_path: str, netcdf_file_path: str = None):
    """
    Diagnostica la conversión de BUFR a NetCDF analizando datos válidos.
    
    Args:
        bufr_file_path: Path al archivo BUFR original
        netcdf_file_path: Path al NetCDF convertido (opcional)
    """
    print("=" * 80)
    print("DIAGNÓSTICO DE CONVERSIÓN BUFR → NetCDF")
    print("=" * 80)
    
    # Importar módulos BUFR
    try:
        from app.services.bufr.bufr_decoder import bufr_to_dict
        from app.services.bufr.bufr_to_pyart import bufr_fields_to_pyart_radar
        print("✓ Módulos BUFR importados correctamente")
    except ImportError as e:
        print("❌ ERROR: No se pudieron importar módulos BUFR.")
        print(f"   Detalles: {e}")
        print("\nAsegúrate de ejecutar este script desde la raíz del proyecto:")
        print("  python -m backend.diagnose_bufr <archivo.BUFR>")
        return
    
    # ═══ PASO 1: Decodificar BUFR ═══
    print(f"\n[1/3] Decodificando archivo BUFR: {Path(bufr_file_path).name}")
    print(f"      Path completo: {bufr_file_path}")
    
    if not Path(bufr_file_path).exists():
        print(f"❌ ERROR: Archivo no existe!")
        return
    
    print(f"      Tamaño: {Path(bufr_file_path).stat().st_size / 1024:.1f} KB")
    
    try:
        print("      Llamando a bufr_to_dict()...")
        field_dict = bufr_to_dict(bufr_file_path, root_resources=None, legacy=False)
        
        if field_dict is None:
            print("  ❌ bufr_to_dict() retornó None")
            print("     Esto indica que la decodificación falló completamente.")
            return
        
        print("      ✓ Decodificación exitosa")
        
    except Exception as e:
        print(f"  ❌ Error decodificando BUFR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analizar datos crudos
    data_raw = field_dict['data']
    info = field_dict['info']
    
    print(f"\n  Información del volumen:")
    print(f"    • Radar: {info['nombre_radar']}")
    print(f"    • Producto: {info['tipo_producto']}")
    print(f"    • Estrategia: {info.get('estrategia', {}).get('nombre', 'N/A')}")
    print(f"    • Volumen: {info.get('estrategia', {}).get('volume_number', 'N/A')}")
    print(f"    • Sweeps: {info['nsweeps']}")
    print(f"    • Shape de datos crudos: {data_raw.shape}")
    print(f"    • Dtype: {data_raw.dtype}")
    
    # Detalles de sweeps
    if 'sweeps' in info and hasattr(info['sweeps'], 'shape'):
        sweeps_df = info['sweeps']
        print(f"\n  Configuración de sweeps:")
        print(f"    • Elevaciones: {sweeps_df['elevaciones'].tolist()}")
        print(f"    • Nrays por sweep: {sweeps_df['nrayos'].unique().tolist()}")
        print(f"    • Ngates por sweep: {sweeps_df['ngates'].unique().tolist()}")
        print(f"    • Gate size: {sweeps_df['gate_size'].unique().tolist()} metros")
        print(f"    • Gate offset: {sweeps_df['gate_offset'].unique().tolist()} metros")
        
        # Calcular rango máximo
        max_range_m = (sweeps_df['gate_offset'].iloc[0] + 
                       sweeps_df['gate_size'].iloc[0] * sweeps_df['ngates'].iloc[0])
        print(f"    • Rango máximo: {max_range_m/1000:.1f} km")
    
    # Analizar NaN y máscara
    print(f"\n  Análisis de datos crudos (post-decodificación):")
    
    n_total = data_raw.size
    n_nan = np.isnan(data_raw).sum()
    n_finite = np.isfinite(data_raw).sum()
    pct_nan = (n_nan / n_total) * 100 if n_total > 0 else 0
    pct_finite = (n_finite / n_total) * 100 if n_total > 0 else 0
    
    print(f"    • Total de gates: {n_total:,}")
    print(f"    • Gates con NaN: {n_nan:,} ({pct_nan:.1f}%)")
    print(f"    • Gates con valores finitos: {n_finite:,} ({pct_finite:.1f}%)")
    
    # Mostrar sample de datos
    print(f"\n    Sample de primeros 20 valores:")
    sample = data_raw.ravel()[:20]
    print(f"    {sample}")
    
    if np.ma.is_masked(data_raw):
        n_masked = np.ma.getmaskarray(data_raw).sum()
        n_valid = n_total - n_masked
        pct_valid = (n_valid / n_total) * 100 if n_total > 0 else 0
        
        print(f"\n    • Tipo: MaskedArray (con valores enmascarados)")
        print(f"    • Gates válidos (no-masked): {n_valid:,} ({pct_valid:.1f}%)")
        print(f"    • Gates masked: {n_masked:,} ({(100-pct_valid):.1f}%)")
        
        # Análisis por fila/sweep
        if pct_valid < 100:
            print(f"\n  Análisis por sweep (primeros 5):")
            nrows = min(5, data_raw.shape[0])
            for i in range(nrows):
                row_data = data_raw[i, :]
                if np.ma.is_masked(row_data):
                    row_valid = (~np.ma.getmaskarray(row_data)).sum()
                    row_total = row_data.size
                    row_pct = (row_valid / row_total) * 100 if row_total > 0 else 0
                    print(f"    Fila {i}: {row_valid:,}/{row_total:,} válidos ({row_pct:.1f}%)")
        
        # Estadísticas de valores finitos
        if n_valid > 0 and n_finite > 0:
            valid_data = data_raw[~np.ma.getmaskarray(data_raw)]
            finite_data = valid_data[np.isfinite(valid_data)]
            
            if finite_data.size > 0:
                print(f"\n  Estadísticas de valores finitos (no-NaN):")
                print(f"    • Count: {finite_data.size:,}")
                print(f"    • Min: {finite_data.min():.2f}")
                print(f"    • Max: {finite_data.max():.2f}")
                print(f"    • Mean: {finite_data.mean():.2f}")
                print(f"    • Median: {np.median(finite_data):.2f}")
                print(f"    • Std: {finite_data.std():.2f}")
            else:
                print(f"\n    ⚠️  NO HAY VALORES FINITOS - Todos son NaN/Inf")
    else:
        print(f"    • Tipo: ndarray regular (sin máscara)")
        if n_finite > 0:
            finite_vals = data_raw[np.isfinite(data_raw)]
            print(f"\n  Estadísticas de valores finitos:")
            print(f"    • Count: {finite_vals.size:,}")
            print(f"    • Min: {finite_vals.min():.2f}")
            print(f"    • Max: {finite_vals.max():.2f}")
            print(f"    • Mean: {finite_vals.mean():.2f}")
            print(f"    • Median: {np.median(finite_vals):.2f}")
        else:
            print(f"\n    ⚠️  TODOS los valores son NaN/Inf!")
            print(f"       Este es el problema principal.")
    
    # ═══ PASO 2: Convertir a PyART Radar ═══
    print(f"\n[2/3] Convirtiendo a objeto PyART Radar...")
    try:
        print("      Llamando a bufr_fields_to_pyart_radar()...")
        radar = bufr_fields_to_pyart_radar([field_dict])
        print("      ✓ Conversión exitosa")
    except Exception as e:
        print(f"  ❌ Error creando Radar PyART: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n  Radar PyART creado:")
    print(f"    • nrays: {radar.nrays}")
    print(f"    • ngates: {radar.ngates}")
    print(f"    • nsweeps: {radar.nsweeps}")
    print(f"    • Campos: {list(radar.fields.keys())}")
    print(f"    • Lat/Lon: {radar.latitude['data'][0]:.4f}, {radar.longitude['data'][0]:.4f}")
    print(f"    • Altitud: {radar.altitude['data'][0]:.1f} m")
    
    # Rango máximo
    if radar.range['data'].size > 0:
        max_range = radar.range['data'][-1]
        print(f"    • Rango máximo: {max_range/1000:.1f} km")
    
    # Analizar campo en PyART
    field_name = info['tipo_producto']
    if field_name in radar.fields:
        field_data = radar.fields[field_name]['data']
        
        print(f"\n  Análisis del campo '{field_name}' en PyART Radar:")
        print(f"    • Shape: {field_data.shape}")
        print(f"    • Dtype: {field_data.dtype}")
        
        # Analizar NaN
        n_total_pyart = field_data.size
        n_nan_pyart = np.isnan(field_data).sum()
        n_finite_pyart = np.isfinite(field_data).sum()
        pct_nan_pyart = (n_nan_pyart / n_total_pyart) * 100 if n_total_pyart > 0 else 0
        
        print(f"    • NaN en PyART: {n_nan_pyart:,} ({pct_nan_pyart:.1f}%)")
        print(f"    • Valores finitos: {n_finite_pyart:,} ({100-pct_nan_pyart:.1f}%)")
        
        if n_finite_pyart > 0:
            finite_pyart = field_data[np.isfinite(field_data)]
            print(f"    • Min (finito): {finite_pyart.min():.2f}")
            print(f"    • Max (finito): {finite_pyart.max():.2f}")
            print(f"    • Mean (finito): {finite_pyart.mean():.2f}")
        else:
            print(f"    ⚠️  TODOS son NaN en PyART también")
    else:
        print(f"\n  ⚠️  Campo '{field_name}' no encontrado en radar PyART!")
        print(f"      Campos disponibles: {list(radar.fields.keys())}")
    
    # ═══ PASO 3: Analizar NetCDF si existe ═══
    if netcdf_file_path and Path(netcdf_file_path).exists():
        print(f"\n[3/3] Analizando archivo NetCDF: {Path(netcdf_file_path).name}")
        try:
            radar_nc = pyart.io.read(netcdf_file_path)
            print(f"      ✓ NetCDF cargado")
            
            for fname in radar_nc.fields.keys():
                fdata = radar_nc.fields[fname]['data']
                n_nan_nc = np.isnan(fdata).sum()
                n_finite_nc = np.isfinite(fdata).sum()
                pct_nan_nc = (n_nan_nc / fdata.size) * 100
                
                print(f"\n  Campo '{fname}':")
                print(f"    • NaN: {n_nan_nc:,} ({pct_nan_nc:.1f}%)")
                print(f"    • Finitos: {n_finite_nc:,} ({100-pct_nan_nc:.1f}%)")
                
        except Exception as e:
            print(f"  ❌ Error leyendo NetCDF: {e}")
    else:
        print(f"\n[3/3] Archivo NetCDF no especificado")
    
    # ═══ RESUMEN ═══
    print("\n" + "=" * 80)
    print("RESUMEN Y DIAGNÓSTICO")
    print("=" * 80)
    
    if pct_nan > 50:
        print(f"\n❌ PROBLEMA CRÍTICO: {pct_nan:.1f}% de valores son NaN!")
        print("\n   CAUSA RAÍZ:")
        print("   • El decodificador BUFR produce NaN para todos/casi todos los gates")
        print("\n   POSIBLES RAZONES:")
        print("   1. Valores 'missing' (-1.797e308) no se convierten correctamente a masked")
        print("   2. Error en decompress_sweep() - la descompresión zlib produce basura")
        print("   3. El archivo BUFR está corrupto")
        print("   4. Error en np.frombuffer() - dtype incorrecto")
        print("\n   SIGUIENTE PASO:")
        print("   • Revisar backend/app/services/bufr/bufr_decoder.py línea ~315")
        print("   • Verificar que decompress_sweep() funciona:")
        print("       dec_data = zlib.decompress(...)")
        print("       arr = np.frombuffer(dec_data, dtype=np.float64)")
        print("   • Comparar con radarlib original")
        print("   • Probar con OTRO archivo BUFR diferente")
        
    elif pct_finite < 20 and np.ma.is_masked(data_raw):
        print(f"\n❌ PROBLEMA: Solo {pct_finite:.1f}% de datos válidos")
        print("   • Muchos gates están masked (sin datos)")
        print("   • Revisar filtros y gate masks en el decoder")
        
    elif pct_finite >= 70:
        print(f"\n✓ DATOS BUENOS: {pct_finite:.1f}% de valores finitos")
        print("   • La conversión BUFR funciona correctamente")
        print("   • Si hay problemas en el GeoTIFF, revisar interpolación")
    else:
        print(f"\n⚠️  DATOS MEDIOCRES: {pct_finite:.1f}% de valores finitos")
        print("   • Puede ser normal para sweeps altos sin ecos")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 80)
        print("Script de diagnóstico BUFR → NetCDF")
        print("=" * 80)
        print("\nUso:")
        print("  python -m backend.diagnose_bufr <archivo.BUFR> [archivo.nc]")
        print("\nEjemplos:")
        print("  python -m backend.diagnose_bufr backend/app/storage/RMA1_DBZH.BUFR")
        print("  python -m backend.diagnose_bufr backend/app/storage/RMA1_DBZH.BUFR backend/app/storage/RMA1.nc")
        print("\n" + "=" * 80)
        sys.exit(1)
    
    bufr_path = sys.argv[1]
    nc_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    diagnose_bufr_conversion(bufr_path, nc_path)
