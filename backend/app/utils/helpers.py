import time
import os
import re
from datetime import datetime
import imageio.v2 as imageio
import uuid


def cleanup_tmp(directory="app/storage/tmp", max_age_seconds=10800):
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


def create_animation(image_paths):
    """
    Crea un GIF animado a partir de las imágenes procesadas.
    Retorna la URL o path del archivo generado.
    """
    static_dir = os.path.join(os.getcwd(), "static", "tmp")
    paths = [os.path.join(static_dir, os.path.basename(p)) for p in image_paths]

    images = [imageio.imread(p) for p in paths if os.path.exists(p)]

    if not images:
        raise ValueError("No se pudieron abrir las imágenes para la animación")

    output_dir = "static/tmp"
    os.makedirs(output_dir, exist_ok=True)

    gif_name = f"anim_{uuid.uuid4().hex[:8]}.gif"
    gif_path = os.path.join(output_dir, gif_name)

    imageio.mimsave(gif_path, images, fps=1)

    return f"/static/tmp/{gif_name}"

