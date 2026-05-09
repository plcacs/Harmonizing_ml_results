from collections.abc import Generator
from unittest.mock import AsyncMock, patch
import pytest
import requests_mock
from homeassistant.components.plex.const import DOMAIN, PLEX_SERVER_CONFIG, SERVERS
from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant
from .const import DEFAULT_DATA, DEFAULT_OPTIONS, PLEX_DIRECT_URL
from .helpers import websocket_connected
from .mock_classes import MockGDM
from tests.common import MockConfigEntry, load_fixture

def plex_server_url(entry: MockConfigEntry) -> str:
    """Return a protocol-less URL from a config entry."""
    return entry.data[PLEX_SERVER_CONFIG][CONF_URL].split(':', 1)[-1]

@pytest.fixture
def mock_setup_entry() -> AsyncMock:
    """Override async_setup_entry."""
    with patch('homeassistant.components.plex.async_setup_entry', return_value=True) as mock_setup_entry:
        yield mock_setup_entry

# ... (rest of the code remains the same)
