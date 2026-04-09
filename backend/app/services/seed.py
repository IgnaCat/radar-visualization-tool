"""Seed initial admin user on first startup."""
import logging
from ..core.database import SessionLocal
from ..core.config import settings
from ..core.security import hash_password
from ..models.db.user import User, UserRole

logger = logging.getLogger(__name__)


def seed_admin():
    """Create admin user if no admin exists yet."""
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.role == UserRole.admin).first()
        if existing:
            logger.info("Admin user already exists: %s", existing.username)
            return

        admin = User(
            username=settings.ADMIN_USERNAME,
            hashed_password=hash_password(settings.ADMIN_PASSWORD),
            role=UserRole.admin,
        )
        db.add(admin)
        db.commit()
        logger.info("Admin user created: %s", settings.ADMIN_USERNAME)
    finally:
        db.close()
