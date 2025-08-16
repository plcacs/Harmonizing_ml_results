from typing import Any, Dict, Generator
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
    return entry.data[PLEX_SERVER_CONFIG][CONF_URL].split(':', 1)[-1]

@pytest.fixture
def mock_setup_entry() -> Generator:
    ...

@pytest.fixture(name='album', scope='package') -> Generator:
    ...

@pytest.fixture(name='artist_albums', scope='package') -> Generator:
    ...

# Add type annotations for the remaining fixtures as well

@pytest.fixture
def mock_websocket() -> Generator:
    ...

@pytest.fixture
def mock_plex_calls(entry: MockConfigEntry, requests_mock: Any, children_20: str, children_30: str, children_200: str, children_300: str, empty_library: str, empty_payload: str, grandchildren_300: str, library: str, library_sections: str, library_movies_all: str, library_movies_collections: str, library_movies_metadata: str, library_movies_sort: str, library_music_all: str, library_music_collections: str, library_music_metadata: str, library_music_sort: str, library_tvshows_all: str, library_tvshows_collections: str, library_tvshows_metadata: str, library_tvshows_sort: str, media_1: str, media_30: str, media_100: str, media_200: str, playlists: str, playlist_500: str, plextv_account: str, plextv_resources: str, plextv_shared_users: str, plex_server_accounts: str, plex_server_clients: str, plex_server_default: str, security_token: str, update_check_nochange: str) -> Generator:
    ...

@pytest.fixture
def setup_plex_server(hass: HomeAssistant, entry: MockConfigEntry, livetv_sessions: str, mock_websocket: Any, mock_plex_calls: Any, requests_mock: Any, empty_payload: str, session_default: str, session_live_tv: str, session_photo: str, session_plexweb: str, session_transient: str, session_unknown: str) -> Generator:
    ...

@pytest.fixture
async def mock_plex_server(entry: MockConfigEntry, setup_plex_server: Any) -> Any:
    ...
