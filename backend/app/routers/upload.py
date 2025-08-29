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
    Si un archivo ya existe, no se sobrescribe pero se devuelve en la respuesta con warning.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos.")
    
    UPLOAD_DIR = settings.UPLOAD_DIR
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_paths: list[str] = []
    warnings: list[str] = []

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
            target = Path(UPLOAD_DIR) / unique_name

            if target.exists():
                warnings.append(f"El archivo '{file.filename}' ya existe")
                #saved_paths.append(str(target))
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

            if size == 0:
                try:
                    os.remove(target)
                except Exception:
                    pass
                raise HTTPException(status_code=400, detail=f"'{file.filename}' está vacío.")

            saved_paths.append(str(target))

        return {"filepaths": saved_paths, "warnings": warnings}

    except HTTPException as exc:
        # rollback de los que se alcanzaron a guardar
        for p in saved_paths:
            try:
                if Path(p).exists():
                    os.remove(p)
            except Exception:
                pass
        raise exc

    finally:
        for f in files:
            try:
                await f.close()
            except Exception:
                pass
