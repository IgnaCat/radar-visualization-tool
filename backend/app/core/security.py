"""JWT token and password hashing utilities."""
import bcrypt
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError, ExpiredSignatureError

from .config import settings


def hash_password(password: str) -> str:
    # Use bcrypt directly to avoid passlib compatibility issues with bcrypt 4.0+
    # Ensure password is not longer than 72 bytes
    pwd_bytes = password.encode('utf-8')[:72]
    hashed_bytes = bcrypt.hashpw(pwd_bytes, bcrypt.gensalt())
    return hashed_bytes.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pwd_bytes = plain_password.encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(pwd_bytes, hashed_bytes)


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=settings.JWT_EXPIRE_HOURS)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except ExpiredSignatureError:
        raise ValueError("Token expirado")
    except JWTError:
        raise ValueError("Token inválido")
