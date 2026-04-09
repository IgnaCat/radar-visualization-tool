"""
SQLAlchemy database setup.
SQLite with scoped sessions for FastAPI's sync endpoints.
"""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


def get_engine():
    db_dir = Path(settings.DB_DIR)
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "radar.db"
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
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
