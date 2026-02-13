"""
Decodificador de archivos BUFR de radar usando librería C (libdecbufr).

Adaptado de radarlib.io.bufr.bufr — se eliminaron las dependencias a
radarlib.config y radarlib.resources, reemplazadas por config local.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import zlib
from contextlib import contextmanager
from ctypes import CDLL, POINTER, Structure, c_char_p, c_double, c_int, cdll
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.services.bufr.config import BUFR_RESOURCES_PATH

logger = logging.getLogger(__name__)


class SweepConsistencyException(Exception):
    pass


class point_t(Structure):
    _fields_ = [("lat", c_double), ("lon", c_double)]


class meta_t(Structure):
    _fields_ = [
        ("year", c_int),
        ("month", c_int),
        ("day", c_int),
        ("hour", c_int),
        ("min", c_int),
        ("radar", point_t),
        ("radar_height", c_double),
    ]


@contextmanager
def decbufr_library_context(root_resources: str | None = None):
    """Context manager para cargar la librería C de decodificación BUFR."""
    if root_resources is None:
        root_resources = BUFR_RESOURCES_PATH
    lib = load_decbufr_library(root_resources)
    try:
        yield lib
    finally:
        pass


@contextmanager
def safe_c_call():
    """
    Context manager que redirige stderr de la librería C para capturar
    mensajes de error sin terminar el proceso Python.
    """
    original_stderr = os.dup(2)
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".err") as tmp:
            temp_file_path = tmp.name
        stderr_file = open(temp_file_path, "w")
        os.dup2(stderr_file.fileno(), 2)
        yield stderr_file, temp_file_path
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        try:
            stderr_file.close()
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception:
            pass


def bufr_name_metadata(bufr_filename: str) -> dict:
    """
    Extrae información estructural del nombre de archivo BUFR.
    Patrón: <RADAR>_<ESTRATEGIA>_<NVOL>_<TIPO>_<TIMESTAMP>.BUFR
    """
    filename = bufr_filename.split("/")[-1]
    base_name = filename.split(".")[0]
    parts = base_name.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected BUFR filename format: {bufr_filename}")
    return {
        "radar_name": parts[0],
        "estrategia_nombre": parts[1],
        "estrategia_nvol": parts[2],
        "tipo_producto": parts[3],
        "filename": filename,
    }


def load_decbufr_library(root_resources: str) -> CDLL:
    """Carga la librería dinámica compartida (libdecbufr.so)."""
    if root_resources is None:
        root_resources = BUFR_RESOURCES_PATH
    lib_path = os.path.join(root_resources, "dynamic_library/libdecbufr.so")
    return cdll.LoadLibrary(lib_path)


def get_metadata(lib: CDLL, bufr_path: str, root_resources: str | None = None) -> Dict[str, Any]:
    """Extrae metadatos básicos del archivo BUFR mediante la función C."""
    if root_resources is None:
        root_resources = BUFR_RESOURCES_PATH

    if not os.path.exists(bufr_path):
        raise FileNotFoundError(f"BUFR file not found: {bufr_path}")

    get_meta_data = lib.get_meta_data
    get_meta_data.argtypes = [c_char_p, c_char_p]
    get_meta_data.restype = POINTER(meta_t)

    tables_path = os.path.join(root_resources, "bufr_tables")
    if not os.path.exists(tables_path):
        raise FileNotFoundError(f"BUFR tables directory not found: {tables_path}")

    try:
        with safe_c_call() as (stderr_file, temp_file_path):
            metadata = get_meta_data(
                bufr_path.encode("utf-8"), tables_path.encode("utf-8")
            )
        try:
            with open(temp_file_path, "r") as f:
                stderr_content = f.read().strip()
                if stderr_content:
                    raise RuntimeError(f"C library error: {stderr_content}")
        except FileNotFoundError:
            pass

        if metadata is None:
            raise RuntimeError(f"get_meta_data returned NULL for {bufr_path}")
        if metadata.contents.year < 1900 or metadata.contents.year > 2100:
            raise ValueError(
                f"Invalid year from BUFR file: {metadata.contents.year}"
            )

        return {
            "year": metadata.contents.year,
            "month": metadata.contents.month,
            "day": metadata.contents.day,
            "hour": metadata.contents.hour,
            "min": metadata.contents.min,
            "lat": metadata.contents.radar.lat,
            "lon": metadata.contents.radar.lon,
            "radar_height": metadata.contents.radar_height,
        }
    except Exception as e:
        raise RuntimeError(f"C library error in get_meta_data: {e}") from e


def get_elevations(
    lib: CDLL,
    bufr_path: str,
    max_elev: int = 30,
    root_resources: str | None = None,
) -> np.ndarray:
    """Recupera las elevaciones de los barridos desde la librería C."""
    if root_resources is None:
        root_resources = BUFR_RESOURCES_PATH

    get_elevation_data = lib.get_elevation_data
    get_elevation_data.argtypes = [c_char_p, c_char_p]
    array_shape = c_double * max_elev
    get_elevation_data.restype = POINTER(array_shape)
    tables_path = os.path.join(root_resources, "bufr_tables")

    try:
        arr = get_elevation_data(
            bufr_path.encode("utf-8"), tables_path.encode("utf-8")
        )
        if arr is None:
            raise RuntimeError(
                f"get_elevation_data returned NULL for {bufr_path}"
            )
        result = np.asarray(list(arr.contents))
        valid_elevs = result[result > 0]
        if len(valid_elevs) == 0:
            raise ValueError("No valid elevations found in BUFR file")
        if np.any((valid_elevs < -1) | (valid_elevs > 90)):
            raise ValueError(f"Invalid elevation values found")
        return result
    except Exception as e:
        raise RuntimeError(
            f"C library error in get_elevation_data: {e}"
        ) from e


def get_raw_volume(
    lib: CDLL,
    bufr_path: str,
    size: int,
    root_resources: str | None = None,
) -> np.ndarray:
    """Recupera el bloque de datos crudo del archivo BUFR."""
    if root_resources is None:
        root_resources = BUFR_RESOURCES_PATH

    if size <= 0:
        raise ValueError(f"Invalid size: {size}")

    get_data = lib.get_data
    get_data.argtypes = [c_char_p, c_char_p]
    array_shape = c_int * size
    get_data.restype = POINTER(array_shape)
    tables_path = os.path.join(root_resources, "bufr_tables")

    try:
        raw = get_data(bufr_path.encode("utf-8"), tables_path.encode("utf-8"))
        if raw is None:
            raise RuntimeError(f"get_data returned NULL for {bufr_path}")
        result = np.asarray(list(raw.contents))
        if len(result) != size:
            raise ValueError(
                f"Data size mismatch: expected {size}, got {len(result)}"
            )
        return result
    except Exception as e:
        raise RuntimeError(f"C library error in get_raw_volume: {e}") from e


def get_size_data(
    lib: CDLL, bufr_path: str, root_resources: str | None = None
) -> int:
    """Devuelve el tamaño del bloque de datos bruto del archivo BUFR."""
    if root_resources is None:
        root_resources = BUFR_RESOURCES_PATH

    get_size_data_fn = lib.get_size_data
    get_size_data_fn.argtypes = [c_char_p, c_char_p]
    get_size_data_fn.restype = c_int
    tables_path = os.path.join(root_resources, "bufr_tables")

    try:
        size = get_size_data_fn(
            bufr_path.encode("utf-8"), tables_path.encode("utf-8")
        )
        if size <= 0:
            raise ValueError(f"Invalid data size: {size}")
        if size > 26_000_000:
            raise ValueError(f"Data size too large: {size} elements")
        return size
    except Exception as e:
        raise RuntimeError(
            f"C library error in get_size_data: {e}"
        ) from e


def parse_sweeps(
    vol: np.ndarray, nsweeps: int, elevs: np.ndarray
) -> list[dict]:
    """Parsea el buffer de enteros y extrae una lista de barridos (sweeps)."""
    sweeps = []
    u = 1
    for sweep_idx in range(nsweeps):
        (
            year_ini, month_ini, day_ini, hour_ini, min_ini, sec_ini,
            year, month, day, hour, minute, sec, product_type,
        ) = vol[u : u + 13]
        u += 13

        elevation = elevs[sweep_idx] if sweep_idx < len(elevs) else None
        u += 1
        ngates, range_size, range_offset, nrays, azimuth = vol[u : u + 5]
        u += 5
        u += 3

        multi_pri = vol[u]
        u += 1
        comp_chunks = []
        for _ in range(multi_pri):
            multi_sec = vol[u]
            u += 1
            data_chunk = vol[u : u + multi_sec]
            u += multi_sec
            data_chunk = np.where(data_chunk == 99999, 255, data_chunk)
            comp_chunks.append(data_chunk)
        compress_data = bytearray(
            np.concatenate(comp_chunks).astype(np.uint8)
        )

        sweeps.append({
            "year_ini": year_ini, "month_ini": month_ini,
            "day_ini": day_ini, "hour_ini": hour_ini,
            "min_ini": min_ini, "sec_ini": sec_ini,
            "year": year, "month": month, "day": day,
            "hour": hour, "min": minute, "sec": sec,
            "product_type": product_type, "elevation": elevation,
            "ngates": ngates, "range_size": range_size,
            "range_offset": range_offset, "nrays": nrays,
            "antenna_beam_az": azimuth,
            "compress_data": compress_data,
        })
    return sweeps


def decompress_sweep(sweep: dict) -> np.ndarray:
    """Descomprime y reconstruye los datos de un solo barrido."""
    if sweep["ngates"] > 8400:
        raise SweepConsistencyException(
            f"Barrido con ngates > 8400: {sweep['ngates']}"
        )

    dec_data = zlib.decompress(memoryview(sweep["compress_data"]))
    # frombuffer retorna read-only array, necesitamos copia para modificar
    arr = np.frombuffer(dec_data, dtype=np.float64).copy()
    
    # Valor "missing" en BUFR: -1.797693134862315708e308 (cerca de -float64_max)
    # Reemplazar por NaN en lugar de masked array para evitar perder la info después
    # Umbral: valores con magnitud > 1e100 son considerados "missing"
    arr[np.abs(arr) > 1e100] = np.nan

    expected = sweep["nrays"] * sweep["ngates"]
    if arr.size != expected:
        raise ValueError(
            f"Data de barrido inconsistente: obtenido {arr.size}, "
            f"esperado {expected}"
        )
    return arr.reshape((sweep["nrays"], sweep["ngates"]))


def uniformize_sweeps(sweeps: list[dict]) -> list[dict]:
    """Normaliza todos los barridos para que compartan el mismo número de gates."""
    max_gates = max(sweep["data"].shape[1] for sweep in sweeps)
    for sw in sweeps:
        nr, ng = sw["data"].shape
        if ng < max_gates:
            pad = np.full((nr, max_gates), np.nan, dtype=np.float64)
            pad[:, :ng] = sw["data"]
            sw["data"] = pad
            sw["ngates"] = max_gates
    return sweeps


def assemble_volume(sweeps: list[dict]) -> np.ndarray:
    """Concatena los arrays de cada barrido para formar el volumen final."""
    return np.vstack([sw["data"] for sw in sweeps])


def validate_sweeps_df(sweeps_df: pd.DataFrame) -> pd.DataFrame:
    """Valida consistencia básica entre los barridos del volumen."""
    assert sweeps_df["nrayos"].nunique() == 1, \
        "Número de rayos inconsistente entre sweeps"
    assert sweeps_df["gate_size"].nunique() == 1, \
        "Gate size inconsistente entre sweeps"
    max_offset = sweeps_df["gate_offset"].iloc[0] // 2
    assert all(
        abs(sweeps_df["gate_offset"] - sweeps_df["gate_offset"].iloc[0])
        <= max_offset
    ), "Desplazamiento excesivo en gate_offset entre sweeps"
    return sweeps_df


def build_metadata(filename: str, info: dict) -> dict:
    """Construye metadatos estandarizados para el pipeline."""
    dia_sweep = int(info["sweeps"]["dia_sweep"].iloc[0])
    mes_sweep = int(info["sweeps"]["mes_sweep"].iloc[0])
    ano_sweep = int(info["sweeps"]["ano_sweep"].iloc[0])
    hora_sweep = int(info["sweeps"]["hora_sweep"].iloc[0])
    min_sweep = int(info["sweeps"]["min_sweep"].iloc[0])
    seg_sweep = int(info["sweeps"]["seg_sweep"].iloc[0])
    return {
        "comment": "-",
        "instrument_type": "Radar",
        "site_name": "-",
        "Sub_conventions": "-",
        "references": "-",
        "volume_number": info["estrategia"]["volume_number"],
        "scan_id": info["estrategia"]["nombre"],
        "title": "-",
        "source": "-",
        "version": "-",
        "instrument_name": info["nombre_radar"],
        "ray_times_increase": "-",
        "platform_is_mobile": "false",
        "driver": "-",
        "institution": "SiNaRaMe",
        "n_gates_vary": "-",
        "primary_axis": "-",
        "created": (
            f"Fecha:{dia_sweep}/{mes_sweep}/{ano_sweep} "
            f"Hora:{hora_sweep}:{min_sweep}:{seg_sweep}"
        ),
        "scan_name": "-",
        "author": "Grupo Radar Cordoba (GRC) - Extractor/Conversor de Datos de Radar ",
        "Conventions": "-",
        "platform_type": "Base Fija",
        "history": "-",
        "filename": (
            filename.split("_")[0] + "_"
            + filename.split("_")[1] + "_"
            + filename.split("_")[2] + "_"
            + filename.split("_")[4].split(".")[0] + ".nc"
        ),
    }


def build_info_dict(meta_vol: dict, meta_sweeps: list[dict]) -> dict:
    """Ensambla el diccionario 'info' con metadatos del volumen y barridos."""
    nsweeps = meta_vol["nsweeps"]
    info = {
        "nombre_radar": meta_vol["radar_name"],
        "estrategia": {
            "nombre": meta_vol["estrategia_nombre"],
            "volume_number": meta_vol["estrategia_nvol"],
        },
        "tipo_producto": meta_vol["tipo_producto"],
        "filename": meta_vol["filename"],
        "ano_vol": meta_vol["year"],
        "mes_vol": meta_vol["month"],
        "dia_vol": meta_vol["day"],
        "hora_vol": meta_vol["hour"],
        "min_vol": meta_vol["min"],
        "lat": meta_vol["lat"],
        "lon": meta_vol["lon"],
        "altura": meta_vol["radar_height"],
        "nsweeps": nsweeps,
    }

    drop_cols = ["data", "compress_data", "product_type"]
    sweeps_df = pd.DataFrame.from_dict(meta_sweeps).drop(columns=drop_cols)
    sweeps_df = sweeps_df.rename(columns={
        "year_ini": "ano_sweep_ini",
        "year": "ano_sweep",
        "month_ini": "mes_sweep_ini",
        "month": "mes_sweep",
        "day_ini": "dia_sweep_ini",
        "day": "dia_sweep",
        "hour_ini": "hora_sweep_ini",
        "hour": "hora_sweep",
        "min_ini": "min_sweep_ini",
        "min": "min_sweep",
        "sec_ini": "seg_sweep_ini",
        "sec": "seg_sweep",
        "elevation": "elevaciones",
        "ngates": "ngates",
        "range_size": "gate_size",
        "range_offset": "gate_offset",
        "nrays": "nrayos",
        "antenna_beam_az": "rayo_inicial",
    })

    info["sweeps"] = validate_sweeps_df(sweeps_df)
    info["metadata"] = build_metadata(meta_vol["filename"], info)
    return info


def dec_bufr_file(
    bufr_filename: str,
    root_resources: str | None = None,
    parallel: bool = True,
) -> Tuple[Dict[str, Any], List[dict], np.ndarray, List[List[Any]]]:
    """
    Decodifica un archivo BUFR: metadatos, barridos descomprimidos,
    volumen concatenado y log de ejecución.
    """
    run_log: List[List[Any]] = []

    try:
        with decbufr_library_context(root_resources) as lib:
            vol_metadata = get_metadata(lib, bufr_filename, root_resources)
            size_data = get_size_data(lib, bufr_filename, root_resources)
            vol = get_raw_volume(
                lib, bufr_filename, size_data, root_resources=root_resources
            )
            elevs = get_elevations(
                lib, bufr_filename, max_elev=vol[0],
                root_resources=root_resources,
            )

            nsweeps = int(vol[0])
            vol_metadata["nsweeps"] = nsweeps
            sweeps = parse_sweeps(vol, nsweeps, elevs)

            def decompress_wrapper(sw, idx):
                try:
                    sw["data"] = decompress_sweep(sw)
                    return sw, None
                except SweepConsistencyException:
                    vol_name = bufr_filename.split("/")[-1].split(".")[0][:-5]
                    product_type = sw.get("product_type", "N/A")
                    message = (
                        f"{vol_name}: Se descarta barrido inconsistente "
                        f"({product_type} / Sw: {idx}) (ngates fuera de limites)"
                    )
                    logger.warning(message)
                    return None, [2, message]
                except Exception as exc:
                    logger.warning(
                        "Descartado barrido inconsistente en sweep %d: %s",
                        idx, exc,
                    )
                    return None, [2, f"Descartado barrido sweep {idx}: {exc}"]

            results = []
            if parallel:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(decompress_wrapper, sw, idx): idx
                        for idx, sw in enumerate(sweeps)
                    }
                    for future in as_completed(futures):
                        sw, log_entry = future.result()
                        if sw is not None:
                            results.append(sw)
                        if log_entry:
                            run_log.append(log_entry)
                results.sort(key=lambda sw: sweeps.index(sw))
            else:
                for idx, sw in enumerate(sweeps):
                    sw_out, log_entry = decompress_wrapper(sw, idx)
                    if sw_out is not None:
                        results.append(sw_out)
                    if log_entry:
                        run_log.append(log_entry)
            sweeps = results

            vol_metadata["nsweeps"] = len(sweeps)
            name_dict = bufr_name_metadata(bufr_filename)
            vol_metadata = dict(vol_metadata, **name_dict)

            sweeps = uniformize_sweeps(sweeps)
            vol_data = assemble_volume(sweeps)

            return vol_metadata, sweeps, vol_data, run_log

    except Exception as exc:
        msg = f"Error en la decodificacion del archivo BUFR: {exc}"
        logger.error(msg, exc_info=True)
        run_log.append([3, str(exc)])
        raise ValueError(msg)


def bufr_to_dict(
    bufr_filename: str,
    root_resources: str | None = None,
    legacy: bool = False,
) -> Optional[dict]:
    """
    Procesa un archivo BUFR y devuelve un diccionario con:
      - 'data': ndarray 2-D con todos los datos concatenados.
      - 'info': diccionario con metadatos y listas por barrido.

    Retorna None en caso de fallo (error loggeado).
    """
    max_attempts = 3
    base_delay = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            meta_vol, meta_sweeps, vol_data, run_log = dec_bufr_file(
                bufr_filename=bufr_filename,
                root_resources=root_resources,
            )

            vol: Dict[str, Any] = {"data": vol_data}
            vol["info"] = build_info_dict(meta_vol, meta_sweeps)

            if legacy:
                vol["info"] = dict(
                    vol["info"], **vol["info"]["sweeps"].to_dict(orient="list")
                )
                del vol["info"]["sweeps"]

            return vol

        except Exception as e:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt, max_attempts, bufr_filename, e,
            )
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                delay = delay * (0.8 + 0.4 * np.random.random())
                time.sleep(delay)
                continue
            else:
                logger.error(
                    "Error en bufr_to_dict (final): %s", e, exc_info=True
                )
                return None
