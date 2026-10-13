"""
SQLAlchemy database setup.
SQLite with scoped sessions for FastAPI's sync endpoints.
"""
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


def _env_int(name: str, default: int) -> int:
    """Entero positivo desde el entorno; ausente o invalido -> default."""
    try:
        v = int(os.getenv(name, ""))
        return v if v >= 1 else default
    except ValueError:
        return default


def get_engine():
    db_dir = Path(settings.DB_DIR)
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "radar.db"
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
        # Cada request retiene una conexion durante todo su ciclo (get_db hace yield).
        # Defaults = los de SQLAlchemy (5 + 10, 30 s). Override por entorno.
        pool_size=_env_int("DB_POOL_SIZE", 5),
        max_overflow=_env_int("DB_MAX_OVERFLOW", 10),
        pool_timeout=_env_int("DB_POOL_TIMEOUT", 30),
    )


engine = get_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency: yields a DB session, closes after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Safe to call multiple times (CREATE IF NOT EXISTS)."""
    Base.metadata.create_all(bind=engine)
