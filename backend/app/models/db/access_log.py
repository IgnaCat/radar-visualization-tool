from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey

from ...core.database import Base


class AccessLog(Base):
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ip_address = Column(String(45), nullable=False)  # IPv6 max length
    city = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    user_agent = Column(String(500), nullable=True)
    logged_in_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    # Geolocation fields
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    address = Column(String(500), nullable=True)
    location_source = Column(String(20), nullable=True)  # "browser" or "geoip"
