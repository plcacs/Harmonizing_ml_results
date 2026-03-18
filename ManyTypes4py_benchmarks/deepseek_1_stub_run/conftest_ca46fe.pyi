```python
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock
import pytest
import requests_mock
from homeassistant.components.plex.const import DOMAIN, PLEX_SERVER_CONFIG, SERVERS
from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant
from .const import DEFAULT_DATA, DEFAULT_OPTIONS, PLEX_DIRECT_URL
from .helpers import websocket_connected
from .mock_classes import MockGDM
from tests.common import MockConfigEntry

def plex_server_url(entry: Any) -> str: ...

@pytest.fixture
def mock_setup_entry() -> Generator[Any, None, None]: ...

@pytest.fixture(name='album', scope='package')
def album_fixture() -> Any: ...

@pytest.fixture(name='artist_albums', scope='package')
def artist_albums_fixture() -> Any: ...

@pytest.fixture(name='children_20', scope='package')
def children_20_fixture() -> Any: ...

@pytest.fixture(name='children_30', scope='package')
def children_30_fixture() -> Any: ...

@pytest.fixture(name='children_200', scope='package')
def children_200_fixture() -> Any: ...

@pytest.fixture(name='children_300', scope='package')
def children_300_fixture() -> Any: ...

@pytest.fixture(name='empty_library', scope='package')
def empty_library_fixture() -> Any: ...

@pytest.fixture(name='empty_payload', scope='package')
def empty_payload_fixture() -> Any: ...

@pytest.fixture(name='grandchildren_300', scope='package')
def grandchildren_300_fixture() -> Any: ...

@pytest.fixture(name='library_movies_all', scope='package')
def library_movies_all_fixture() -> Any: ...

@pytest.fixture(name='library_movies_metadata', scope='package')
def library_movies_metadata_fixture() -> Any: ...

@pytest.fixture(name='library_movies_collections', scope='package')
def library_movies_collections_fixture() -> Any: ...

@pytest.fixture(name='library_tvshows_all', scope='package')
def library_tvshows_all_fixture() -> Any: ...

@pytest.fixture(name='library_tvshows_metadata', scope='package')
def library_tvshows_metadata_fixture() -> Any: ...

@pytest.fixture(name='library_tvshows_collections', scope='package')
def library_tvshows_collections_fixture() -> Any: ...

@pytest.fixture(name='library_music_all', scope='package')
def library_music_all_fixture() -> Any: ...

@pytest.fixture(name='library_music_metadata', scope='package')
def library_music_metadata_fixture() -> Any: ...

@pytest.fixture(name='library_music_collections', scope='package')
def library_music_collections_fixture() -> Any: ...

@pytest.fixture(name='library_movies_sort', scope='package')
def library_movies_sort_fixture() -> Any: ...

@pytest.fixture(name='library_tvshows_sort', scope='package')
def library_tvshows_sort_fixture() -> Any: ...

@pytest.fixture(name='library_music_sort', scope='package')
def library_music_sort_fixture() -> Any: ...

@pytest.fixture(name='library_movies_filtertypes', scope='package')
def library_movies_filtertypes_fixture() -> Any: ...

@pytest.fixture(name='library', scope='package')
def library_fixture() -> Any: ...

@pytest.fixture(name='library_movies_size', scope='package')
def library_movies_size_fixture() -> Any: ...

@pytest.fixture(name='library_music_size', scope='package')
def library_music_size_fixture() -> Any: ...

@pytest.fixture(name='library_tvshows_size', scope='package')
def library_tvshows_size_fixture() -> Any: ...

@pytest.fixture(name='library_tvshows_size_episodes', scope='package')
def library_tvshows_size_episodes_fixture() -> Any: ...

@pytest.fixture(name='library_tvshows_size_seasons', scope='package')
def library_tvshows_size_seasons_fixture() -> Any: ...

@pytest.fixture(name='library_sections', scope='package')
def library_sections_fixture() -> Any: ...

@pytest.fixture(name='media_1', scope='package')
def media_1_fixture() -> Any: ...

@pytest.fixture(name='media_30', scope='package')
def media_30_fixture() -> Any: ...

@pytest.fixture(name='media_100', scope='package')
def media_100_fixture() -> Any: ...

@pytest.fixture(name='media_200', scope='package')
def media_200_fixture() -> Any: ...

@pytest.fixture(name='player_plexweb_resources', scope='package')
def player_plexweb_resources_fixture() -> Any: ...

@pytest.fixture(name='player_plexhtpc_resources', scope='package')
def player_plexhtpc_resources_fixture() -> Any: ...

@pytest.fixture(name='playlists', scope='package')
def playlists_fixture() -> Any: ...

@pytest.fixture(name='playlist_500', scope='package')
def playlist_500_fixture() -> Any: ...

@pytest.fixture(name='playqueue_created', scope='package')
def playqueue_created_fixture() -> Any: ...

@pytest.fixture(name='playqueue_1234', scope='package')
def playqueue_1234_fixture() -> Any: ...

@pytest.fixture(name='plex_server_accounts', scope='package')
def plex_server_accounts_fixture() -> Any: ...

@pytest.fixture(name='plex_server_base', scope='package')
def plex_server_base_fixture() -> Any: ...

@pytest.fixture(name='plex_server_default', scope='package')
def plex_server_default_fixture(plex_server_base: Any) -> Any: ...

@pytest.fixture(name='plex_server_clients', scope='package')
def plex_server_clients_fixture() -> Any: ...

@pytest.fixture(name='plextv_account', scope='package')
def plextv_account_fixture() -> Any: ...

@pytest.fixture(name='plextv_resources', scope='package')
def plextv_resources_fixture() -> Any: ...

@pytest.fixture(name='plextv_resources_two_servers', scope='package')
def plextv_resources_two_servers_fixture() -> Any: ...

@pytest.fixture(name='plextv_shared_users', scope='package')
def plextv_shared_users_fixture() -> Any: ...

@pytest.fixture(name='session_base', scope='package')
def session_base_fixture() -> Any: ...

@pytest.fixture(name='session_default', scope='package')
def session_default_fixture(session_base: Any) -> Any: ...

@pytest.fixture(name='session_new_user', scope='package')
def session_new_user_fixture(session_base: Any) -> Any: ...

@pytest.fixture(name='session_photo', scope='package')
def session_photo_fixture() -> Any: ...

@pytest.fixture(name='session_plexweb', scope='package')
def session_plexweb_fixture() -> Any: ...

@pytest.fixture(name='session_transient', scope='package')
def session_transient_fixture() -> Any: ...

@pytest.fixture(name='session_unknown', scope='package')
def session_unknown_fixture() -> Any: ...

@pytest.fixture(name='session_live_tv', scope='package')
def session_live_tv_fixture() -> Any: ...

@pytest.fixture(name='livetv_sessions', scope='package')
def livetv_sessions_fixture() -> Any: ...

@pytest.fixture(name='security_token', scope='package')
def security_token_fixture() -> Any: ...

@pytest.fixture(name='show_seasons', scope='package')
def show_seasons_fixture() -> Any: ...

@pytest.fixture(name='sonos_resources', scope='package')
def sonos_resources_fixture() -> Any: ...

@pytest.fixture(name='hubs', scope='package')
def hubs_fixture() -> Any: ...

@pytest.fixture(name='hubs_music_library', scope='package')
def hubs_music_library_fixture() -> Any: ...

@pytest.fixture(name='update_check_nochange', scope='package')
def update_check_fixture_nochange() -> Any: ...

@pytest.fixture(name='update_check_new', scope='package')
def update_check_fixture_new() -> Any: ...

@pytest.fixture(name='update_check_new_not_updatable', scope='package')
def update_check_fixture_new_not_updatable() -> Any: ...

@pytest.fixture(name='entry')
async def mock_config_entry() -> MockConfigEntry: ...

@pytest.fixture
def mock_websocket() -> Generator[Any, None, None]: ...

@pytest.fixture
def mock_plex_calls(
    entry: Any,
    requests_mock: Any,
    children_20: Any,
    children_30: Any,
    children_200: Any,
    children_300: Any,
    empty_library: Any,
    empty_payload: Any,
    grandchildren_300: Any,
    library: Any,
    library_sections: Any,
    library_movies_all: Any,
    library_movies_collections: Any,
    library_movies_metadata: Any,
    library_movies_sort: Any,
    library_music_all: Any,
    library_music_collections: Any,
    library_music_metadata: Any,
    library_music_sort: Any,
    library_tvshows_all: Any,
    library_tvshows_collections: Any,
    library_tvshows_metadata: Any,
    library_tvshows_sort: Any,
    media_1: Any,
    media_30: Any,
    media_100: Any,
    media_200: Any,
    playlists: Any,
    playlist_500: Any,
    plextv_account: Any,
    plextv_resources: Any,
    plextv_shared_users: Any,
    plex_server_accounts: Any,
    plex_server_clients: Any,
    plex_server_default: Any,
    security_token: Any,
    update_check_nochange: Any
) -> None: ...

@pytest.fixture
def setup_plex_server(
    hass: HomeAssistant,
    entry: Any,
    livetv_sessions: Any,
    mock_websocket: Any,
    mock_plex_calls: Any,
    requests_mock: Any,
    empty_payload: Any,
    session_default: Any,
    session_live_tv: Any,
    session_photo: Any,
    session_plexweb: Any,
    session_transient: Any,
    session_unknown: Any
) -> Any: ...

@pytest.fixture
async def mock_plex_server(
    entry: Any,
    setup_plex_server: Any
) -> Any: ...
```