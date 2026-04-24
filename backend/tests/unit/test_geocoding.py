import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.geocoding import reverse_geocode


@pytest.mark.asyncio
async def test_reverse_geocode_success():
    """Nominatim returns valid response → parsed correctly."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "display_name": "Centro, Córdoba, Departamento Capital, Córdoba, Argentina",
        "address": {
            "suburb": "Centro",
            "city": "Córdoba",
            "state": "Córdoba",
            "country": "Argentina",
        },
    }

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("app.services.geocoding.httpx.AsyncClient", return_value=mock_client):
        result = await reverse_geocode(-31.4135, -64.1811)

    assert result is not None
    assert result["address"] == "Centro, Córdoba, Departamento Capital, Córdoba, Argentina"
    assert result["city"] == "Córdoba"
    assert result["country"] == "Argentina"


@pytest.mark.asyncio
async def test_reverse_geocode_timeout():
    """Nominatim times out → returns None."""
    import httpx

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    with patch("app.services.geocoding.httpx.AsyncClient", return_value=mock_client):
        result = await reverse_geocode(-31.4135, -64.1811)

    assert result is None


@pytest.mark.asyncio
async def test_reverse_geocode_no_city_falls_back_to_town():
    """Response has 'town' instead of 'city' → uses town."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "display_name": "Villa Carlos Paz, Punilla, Córdoba, Argentina",
        "address": {
            "town": "Villa Carlos Paz",
            "state": "Córdoba",
            "country": "Argentina",
        },
    }

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("app.services.geocoding.httpx.AsyncClient", return_value=mock_client):
        result = await reverse_geocode(-31.42, -64.50)

    assert result["city"] == "Villa Carlos Paz"


@pytest.mark.asyncio
async def test_reverse_geocode_http_error():
    """Nominatim returns non-200 → returns None."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("app.services.geocoding.httpx.AsyncClient", return_value=mock_client):
        result = await reverse_geocode(-31.4135, -64.1811)

    assert result is None
