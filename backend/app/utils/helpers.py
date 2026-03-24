import time
import os
import re
from datetime import datetime


def cleanup_tmp(directory="app/storage/tmp", max_age_seconds=20000):
    """
    Elimina archivos antiguos en el directorio especificado.
    Archivos más viejos que max_age_seconds serán eliminados.
    """
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if now - os.path.getmtime(filepath) > max_age_seconds:
                os.remove(filepath)


def extract_metadata_from_filename(filename):
    """
    Extrae radar, estrategia, volumen y timestamp desde un nombre de archivo como:
    RMA1_0303_01_20221209T230832Z.nc
    """
    base = os.path.basename(filename)
    match = re.match(r"(RMA\d+)_(\d+)_(\d+)_(\d{8}T\d{6})Z", base)
    
    if not match:
        return None, None, None, None

    radar = match.group(1)
    estrategia = match.group(2)
    volumen = match.group(3)
    timestamp_str = match.group(4)

    timestamp = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
    return radar, estrategia, volumen, timestamp

def extract_volume_from_filename(filename):
    _, _, volume, _ = extract_metadata_from_filename(filename)
    return str(volume) if volume else None

def should_animate(results, max_minutes_diff=30):
    """
    Determina si todos los archivos son del mismo radar y están cerca en el tiempo.
    `results` debe contener un campo `source_file` (filepath original).
    """
    radars = set()
    timestamps = []

    if not results or len(results) <= 1:
        return False

    for result in results:
        filepath = result.get("source_file")
        if not filepath:
            return False

        radar, _ , _ , timestamp = extract_metadata_from_filename(filepath)
        if not radar or not timestamp:
            return False

        radars.add(radar)
        timestamps.append(timestamp)

    if len(radars) != 1:
        return False

    timestamps.sort()
    for i in range(1, len(timestamps)):
        diff = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60
        if diff > max_minutes_diff:
            print(f"Diferencia de tiempo entre archivos {i-1} y {i} es mayor a {max_minutes_diff} minutos: {diff} minutos.")
            return False

    return True

