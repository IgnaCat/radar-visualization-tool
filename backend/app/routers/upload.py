from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from pathlib import Path
from werkzeug.utils import secure_filename
from typing import Optional
import os

from ..core.config import settings
from ..utils import helpers
from ..services.metadata import extract_radar_metadata

router = APIRouter(prefix="/upload", tags=["upload"])

CHUNK_SIZE = 1024 * 1024  # 1 MB

def _ext_ok(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in {e.lower() for e in settings.ALLOWED_EXTENSIONS}

def _max_size_ok(size_bytes: int) -> bool:
    return size_bytes <= settings.MAX_UPLOAD_MB * 1024 * 1024


@router.post("", status_code=201)
async def upload(files: list[UploadFile] = File(...), session_id: Optional[str] = Form(None)):
    """
    Endpoint para subir múltiples archivos NetCDF.
    Si session_id está presente, los archivos se guardan en uploads/{session_id}/
    Si un archivo ya existe, no se sobrescribe pero se devuelve en la respuesta con warning.
    Devuelve paths + metadata del radar and warnings.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos.")
    
    # Crear subdirectorio de sesión si se proporciona session_id
    UPLOAD_DIR = Path(settings.UPLOAD_DIR)
    if session_id:
        UPLOAD_DIR = UPLOAD_DIR / session_id
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    warnings: list[str] = []
    saved_files: list[dict] = []
    volumes: set[int] = set()
    radars: set[str] = set()

    try:
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="Archivo sin nombre.")
            if not _ext_ok(file.filename):
                raise HTTPException(
                    status_code=415,
                    detail=f"Extensión no permitida: {Path(file.filename).suffix}. Solo {settings.ALLOWED_EXTENSIONS}"
                )

            unique_name = secure_filename(file.filename)
            target = UPLOAD_DIR / unique_name

            if target.exists():
                warnings.append(f"El archivo '{file.filename}' ya existe")
                meta = extract_radar_metadata(str(target))
                radar, _, volume, _ = helpers.extract_metadata_from_filename(str(target))
                if volume is not None:
                    volumes.add(volume)
                if radar is not None:
                    radars.add(radar)
                saved_files.append({
                    "filepath": unique_name,  # Solo el nombre del archivo, no el path completo
                    "filename": file.filename,
                    "size_bytes": os.path.getsize(target) if target.exists() else None,
                    "metadata": meta,
                })
                continue

            size = 0
            with open(target, "wb") as out:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if not _max_size_ok(size):
                        out.close()
                        try:
                            os.remove(target)
                        except Exception:
                            pass
                        while await file.read(CHUNK_SIZE):
                            pass
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"'{file.filename}' excede {settings.MAX_UPLOAD_MB} MB"
                        )
                    out.write(chunk)
                
                # Asegurar que el archivo está completamente escrito al disco
                out.flush()
                os.fsync(out.fileno())

            if size == 0:
                try:
                    os.remove(target)
                except Exception:
                    pass
                raise HTTPException(status_code=400, detail=f"'{file.filename}' está vacío.")

            # Extraer radar metadata
            meta = extract_radar_metadata(str(target))

            # Extraer volumen/radar del nombre de archivo
            radar, _, volume, _ = helpers.extract_metadata_from_filename(str(target))
            if volume is not None:
                volumes.add(volume)
            if radar is not None:
                radars.add(radar)

            saved_files.append({
                "filepath": unique_name,  # Solo el nombre del archivo, no el path completo
                "filename": file.filename,
                "size_bytes": size,
                "metadata": meta,
            })

        return {"files": saved_files, "warnings": warnings, "volumes": list(volumes), "radars": list(radars)}

    except HTTPException as exc:
        # rollback de los que se alcanzaron a guardar
        for p in saved_files:
            try:
                fp = p.get("filepath")
                if fp and Path(fp).exists():
                    os.remove(fp)
            except Exception:
                pass
        raise exc

    finally:
        for f in files:
            try:
                await f.close()
            except Exception:
                pass
