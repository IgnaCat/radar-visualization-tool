import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def _auth_header(token="test-token"):
    return {"Authorization": f"Bearer {token}"}


def _mock_db_with_log(user_id=1, session_id="session-123"):
    """Create a mock DB session that returns a matching AccessLog."""
    mock_db = MagicMock()

    # Mock session lookup
    mock_session = MagicMock()
    mock_session.user_id = user_id
    mock_session.is_active = True

    # Mock access log
    mock_log = MagicMock()
    mock_log.user_id = user_id
    mock_log.latitude = None

    # query().filter().first() chain — called twice: once for UserSession, once for AccessLog
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_order = MagicMock()
    mock_order.first.return_value = mock_log
    mock_filter.first.return_value = mock_session
    mock_filter.order_by.return_value = mock_order
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query

    return mock_db, mock_log


@patch("app.routers.location.reverse_geocode", new_callable=AsyncMock)
@patch("app.routers.location.get_current_user")
@patch("app.core.database.get_db")
def test_location_success(mock_get_db, mock_get_user, mock_geocode):
    """Valid coordinates → updates AccessLog with browser location."""
    mock_db, mock_log = _mock_db_with_log()
    mock_get_db.return_value = iter([mock_db])
    mock_get_user.return_value = MagicMock(id=1)
    mock_geocode.return_value = {
        "address": "Centro, Córdoba, Argentina",
        "city": "Córdoba",
        "country": "Argentina",
    }

    response = client.post(
        "/location",
        json={"session_id": "session-123", "latitude": -31.4135, "longitude": -64.1811},
        headers=_auth_header(),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert mock_log.latitude == -31.4135
    assert mock_log.longitude == -64.1811
    assert mock_log.location_source == "browser"


@patch("app.routers.location.get_current_user")
def test_location_invalid_latitude(mock_get_user):
    """Latitude out of range → 422."""
    mock_get_user.return_value = MagicMock(id=1)

    response = client.post(
        "/location",
        json={"session_id": "session-123", "latitude": 999, "longitude": -64.18},
        headers=_auth_header(),
    )

    assert response.status_code == 422


@patch("app.routers.location.get_current_user")
def test_location_invalid_longitude(mock_get_user):
    """Longitude out of range → 422."""
    mock_get_user.return_value = MagicMock(id=1)

    response = client.post(
        "/location",
        json={"session_id": "session-123", "latitude": -31.41, "longitude": 999},
        headers=_auth_header(),
    )

    assert response.status_code == 422
