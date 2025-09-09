from fastapi import APIRouter
from pathlib import Path
from typing import Iterable
import shutil
import os

from ..core.config import settings
from ..schemas import CleanupRequest

router = APIRouter(prefix="/cleanup", tags=["cleanup"])

# Directorios “oficiales” de tu app
UPLOAD_DIR = Path(settings.UPLOAD_DIR).resolve()       # app/storage/uploads
TMP_DIR = Path(settings.IMAGES_DIR).resolve()          # app/storage/tmp
BASE_DIR = Path("app/storage").resolve()


def _first_safe_under(path_str: str, roots: Iterable[Path]) -> Path | None:
    """
    Normaliza una ruta (absoluta o relativa) y devuelve la primera que
    caiga bajo alguno de los roots permitidos. Si no matchea, devuelve None.
    """
    s = str(path_str).replace("\\", "/").strip()
    candidates: list[Path] = []

    p = Path(s)
    if p.is_absolute():
        candidates.append(p)
    else:
        # prefijos “lógicos” del proyecto -> FS real
        if s.startswith("app/storage/tmp/"):
            rel = s.split("app/storage/tmp/", 1)[1]
            candidates.append(TMP_DIR / rel)
        elif s.startswith("app/storage/uploads/"):
            rel = s.split("app/storage/uploads/", 1)[1]
            candidates.append(UPLOAD_DIR / rel)
        elif s.startswith("static/tmp/"):
            rel = s.split("static/tmp/", 1)[1]
            candidates.append(TMP_DIR / rel)
        elif s.startswith("tmp/"):
            rel = s.split("tmp/", 1)[1]
            candidates.append(TMP_DIR / rel)
        else:
            # fallback: tratarla como relativa a las carpetas conocidas
            candidates.extend([UPLOAD_DIR / s, TMP_DIR / s, BASE_DIR / s])

    # resolver y verificar que quede dentro de un root permitido
    for cand in candidates:
        try:
            rp = cand.resolve(strict=False)
        except Exception:
            continue
        for root in roots:
            try:
                # py>=3.9
                if rp == root or rp.is_relative_to(root):  # type: ignore[attr-defined]
                    return rp
            except AttributeError:
                # py<3.9
                from os.path import commonpath
                if commonpath([str(rp), str(root)]) == str(root):
                    return rp
    return None

def _delete_path(p: Path) -> bool:
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            return True
        else:
            # Py3.8+: missing_ok para no fallar si ya no existe
            try:
                p.unlink(missing_ok=True)  # type: ignore[arg-type]
            except TypeError:
                # compat py<3.8
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            return True
    except Exception:
        return False

@router.post("/close")
def cleanup_close(req: CleanupRequest):
    """
    Borra archivos indicados por el front al cerrar la app.
    - Siempre intenta borrar 'uploads'
    - Sólo borra 'cogs' si delete_cache=True
    Acepta rutas absolutas o relativas (sanitiza contra traversal).
    """
    deleted = {"uploads": 0, "cogs": 0}

    # uploads (NetCDF subidos / temporales del usuario)
    for s in req.uploads:
        rp = _first_safe_under(s, [UPLOAD_DIR, TMP_DIR, BASE_DIR])
        if rp and _delete_path(rp):
            deleted["uploads"] += 1

    # cogs (COGs/GeoTIFFs); por defecto NO se borran, salvo que lo pidas
    if req.delete_cache:
        for s in req.cogs:
            rp = _first_safe_under(s, [TMP_DIR, BASE_DIR])
            if rp and _delete_path(rp):
                deleted["cogs"] += 1

    return {"deleted": deleted}
