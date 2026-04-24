"""
Lightweight column migrations for SQLite.
Runs on startup after init_db(). Each migration is idempotent (skips if column exists).
"""
import logging
from sqlalchemy import text, inspect
from .database import engine

logger = logging.getLogger(__name__)


def _add_column_if_missing(table: str, column: str, col_type: str) -> None:
    """Add a column to a table if it doesn't already exist."""
    insp = inspect(engine)
    existing = [c["name"] for c in insp.get_columns(table)]
    if column in existing:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
    logger.info("Added column %s.%s", table, column)


def run_migrations() -> None:
    """Run all pending column migrations."""
    _add_column_if_missing("access_logs", "latitude", "FLOAT")
    _add_column_if_missing("access_logs", "longitude", "FLOAT")
    _add_column_if_missing("access_logs", "address", "VARCHAR(500)")
    _add_column_if_missing("access_logs", "location_source", "VARCHAR(20)")
