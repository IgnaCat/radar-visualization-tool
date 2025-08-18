from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from titiler.core.factory import TilerFactory

from .core.config import settings
from .routers import process, upload

app = FastAPI(title=settings.APP_NAME)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Todo lo que guardemos en app/storage/tmp será accesible en /static/tmp/...
images_dir = Path(settings.IMAGES_DIR); images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/tmp", StaticFiles(directory=images_dir), name="tmp")

# COGs estáticos (para construir la URL pública del .tif)
cog_dir = Path(settings.COG_DIR); cog_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/cogs", StaticFiles(directory=cog_dir), name="static_cogs")

cog = TilerFactory()
app.include_router(cog.router, prefix="/cog", tags=["cog"])
app.include_router(upload.router)
app.include_router(process.router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/cog/ping")
def ping():
    return {"ok": True}

