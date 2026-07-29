"""Tests for JWT and password hashing utilities."""
import pytest
from datetime import timedelta


def test_hash_password_returns_bcrypt_string():
    from app.core.security import hash_password
    hashed = hash_password("mysecret")
    assert hashed.startswith("$2b$")
    assert hashed != "mysecret"


def test_verify_password_correct():
    from app.core.security import hash_password, verify_password
    hashed = hash_password("mysecret")
    assert verify_password("mysecret", hashed) is True


def test_verify_password_wrong():
    from app.core.security import hash_password, verify_password
    hashed = hash_password("mysecret")
    assert verify_password("wrongpass", hashed) is False


def test_create_token_returns_string():
    from app.core.security import create_access_token
    token = create_access_token({"user_id": 1, "username": "admin", "role": "admin"})
    assert isinstance(token, str)
    assert len(token) > 20


def test_decode_token_roundtrip():
    from app.core.security import create_access_token, decode_access_token
    payload = {"user_id": 42, "username": "testuser", "role": "user"}
    token = create_access_token(payload)
    decoded = decode_access_token(token)
    assert decoded["user_id"] == 42
    assert decoded["username"] == "testuser"
    assert decoded["role"] == "user"
    assert "exp" in decoded


def test_decode_token_expired_raises():
    from app.core.security import create_access_token, decode_access_token
    token = create_access_token(
        {"user_id": 1, "username": "x", "role": "user"},
        expires_delta=timedelta(seconds=-1),
    )
    with pytest.raises(ValueError, match="Token expirado"):
        decode_access_token(token)


def test_decode_token_invalid_raises():
    from app.core.security import decode_access_token
    with pytest.raises(ValueError, match="Token inválido"):
        decode_access_token("not.a.valid.token")
