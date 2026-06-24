"""
Tests para POST /location — recepción de geolocalización del browser.

Usa dependency_overrides para inyectar mocks de auth y DB,
porque @patch no funciona con FastAPI Depends().
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.routers.auth import get_current_user
from app.core.database import get_db


def _mock_db_with_log(user_id=1):
    """Create a mock DB session that returns a matching AccessLog."""
    mock_db = MagicMock()

    mock_session = MagicMock()
    mock_session.user_id = user_id
    mock_session.is_active = True

    mock_log = MagicMock()
    mock_log.user_id = user_id
    mock_log.latitude = None

    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_order = MagicMock()
    mock_order.first.return_value = mock_log
    mock_filter.first.return_value = mock_session
    mock_filter.order_by.return_value = mock_order
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query

    return mock_db, mock_log


@pytest.fixture(autouse=True)
def override_auth():
    """Inyecta un usuario mock en el DI de FastAPI para todos los tests."""
    mock_user = MagicMock(id=1)
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
def override_db():
    """Inyecta una DB mock en el DI de FastAPI."""
    mock_db, mock_log = _mock_db_with_log()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db, mock_log
    app.dependency_overrides.pop(get_db, None)


client = TestClient(app)


@patch("app.routers.location.reverse_geocode", new_callable=AsyncMock)
def test_location_success(mock_geocode, override_db):
    """Valid coordinates → updates AccessLog with browser location."""
    _, mock_log = override_db
    mock_geocode.return_value = {
        "address": "Centro, Córdoba, Argentina",
        "city": "Córdoba",
        "country": "Argentina",
    }

    response = client.post(
        "/location",
        json={"session_id": "session-123", "latitude": -31.4135, "longitude": -64.1811},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert mock_log.latitude == -31.4135
    assert mock_log.longitude == -64.1811
    assert mock_log.location_source == "browser"


def test_location_invalid_latitude():
    """Latitude out of range → 422."""
    response = client.post(
        "/location",
        json={"session_id": "session-123", "latitude": 999, "longitude": -64.18},
    )
    assert response.status_code == 422


def test_location_invalid_longitude():
    """Longitude out of range → 422."""
    response = client.post(
        "/location",
        json={"session_id": "session-123", "latitude": -31.41, "longitude": 999},
    )
    assert response.status_code == 422
