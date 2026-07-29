"""Tests for auth FastAPI dependencies."""
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from app.core.security import create_access_token


def _make_db_session(user=None):
    """Create a mock DB session that returns a user (or None) on query."""
    db = MagicMock()
    query = db.query.return_value
    query.filter.return_value.first.return_value = user
    return db


def _make_user(user_id=1, username="testuser", role="user", is_active=True):
    """Create a mock User object."""
    user = MagicMock()
    user.id = user_id
    user.username = username
    user.role = MagicMock()
    user.role.value = role
    user.is_active = is_active
    return user


def test_get_current_user_valid_token():
    from app.dependencies.auth import get_current_user
    user = _make_user()
    db = _make_db_session(user)
    token = create_access_token({"user_id": 1, "username": "testuser", "role": "user"})
    result = get_current_user(token=token, db=db)
    assert result.id == 1


def test_get_current_user_no_token_raises_401():
    from app.dependencies.auth import get_current_user
    db = _make_db_session()
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token=None, db=db)
    assert exc_info.value.status_code == 401


def test_get_current_user_expired_token_raises_401():
    from app.dependencies.auth import get_current_user
    from datetime import timedelta
    db = _make_db_session()
    token = create_access_token(
        {"user_id": 1, "username": "x", "role": "user"},
        expires_delta=timedelta(seconds=-1),
    )
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token=token, db=db)
    assert exc_info.value.status_code == 401


def test_get_current_user_inactive_user_raises_403():
    from app.dependencies.auth import get_current_user
    user = _make_user(is_active=False)
    db = _make_db_session(user)
    token = create_access_token({"user_id": 1, "username": "testuser", "role": "user"})
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token=token, db=db)
    assert exc_info.value.status_code == 403


def test_require_admin_with_admin_user():
    from app.dependencies.auth import require_admin
    user = _make_user(role="admin")
    result = require_admin(current_user=user)
    assert result.id == 1


def test_require_admin_with_regular_user_raises_403():
    from app.dependencies.auth import require_admin
    user = _make_user(role="user")
    with pytest.raises(HTTPException) as exc_info:
        require_admin(current_user=user)
    assert exc_info.value.status_code == 403
