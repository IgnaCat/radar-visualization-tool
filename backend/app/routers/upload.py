from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pathlib import Path
from werkzeug.utils import secure_filename
import os

from ..core.config import settings

router = APIRouter(prefix="/upload", tags=["upload"])

CHUNK_SIZE = 1024 * 1024  # 1 MB

def _ext_ok(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in {e.lower() for e in settings.ALLOWED_EXTENSIONS}

def _max_size_ok(size_bytes: int) -> bool:
    return size_bytes <= settings.MAX_UPLOAD_MB * 1024 * 1024


@router.post("", status_code=201)
async def upload(files: list[UploadFile] = File(...)):
    """
    Endpoint para subir múltiples archivos NetCDF.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos.")
    
    UPLOAD_DIR = settings.UPLOAD_DIR
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Validaciones previas
    for f in files:
        if not f.filename:
            raise HTTPException(status_code=400, detail="Archivo sin nombre.")
        if not _ext_ok(f.filename):
            raise HTTPException(status_code=415, detail=f"Extensión no permitida: {Path(f.filename).suffix}. Solo {settings.ALLOWED_EXTENSIONS}")

        unique_name = secure_filename(f.filename)
        filepath = os.path.join(UPLOAD_DIR, unique_name)
        
        if os.path.exists(filepath):
            raise HTTPException(status_code=409, detail=f"El archivo '{f.filename}' ya existe.")

    saved_paths: list[Path] = []

    try:
        for file in files:
            unique_name = secure_filename(file.filename)
            target = Path(UPLOAD_DIR) / unique_name
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
                        # consumir el resto del stream por higiene
                        while await file.read(CHUNK_SIZE):
                            pass
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"'{file.filename}' excede {settings.MAX_UPLOAD_MB} MB"
                        )
                    out.write(chunk)

            if size == 0:
                try:
                    os.remove(target)
                except Exception:
                    pass
                raise HTTPException(status_code=400, detail=f"'{file.filename}' está vacío.")

            saved_paths.append(file.filename)

        # Todo ok
        return {"filepaths": [str(p) for p in saved_paths]}

    except HTTPException as exc:
        # rollback: borrar lo que se haya guardado antes del fallo
        for p in saved_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        raise exc

    finally:
        # cerrar streams
        for f in files:
            try:
                await f.close()
            except Exception:
                pass
