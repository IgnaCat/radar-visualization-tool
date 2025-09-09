from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from titiler.core.factory import TilerFactory

from .core.config import settings
from .routers import process, upload, cleanup

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
# COG, png, etc
images_dir = Path(settings.IMAGES_DIR); images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/tmp", StaticFiles(directory=images_dir), name="tmp")

cog = TilerFactory()
app.include_router(cog.router, prefix="/cog", tags=["cog"])
app.include_router(upload.router)
app.include_router(process.router)
app.include_router(cleanup.router)

@app.get("/health")
def health():
    return {"status": "ok"}

