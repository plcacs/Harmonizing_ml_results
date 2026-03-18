```pyi
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry

def plex_server_url(entry: MockConfigEntry) -> str: ...

@pytest.fixture
def mock_setup_entry() -> Generator[MagicMock, None, None]: ...

@pytest.fixture(name='album', scope='package')
def album_fixture() -> str: ...

@pytest.fixture(name='artist_albums', scope='package')
def artist_albums_fixture() -> str: ...

@pytest.fixture(name='children_20', scope='package')
def children_20_fixture() -> str: ...

@pytest.fixture(name='children_30', scope='package')
def children_30_fixture() -> str: ...

@pytest.fixture(name='children_200', scope='package')
def children_200_fixture() -> str: ...

@pytest.fixture(name='children_300', scope='package')
def children_300_fixture() -> str: ...

@pytest.fixture(name='empty_library', scope='package')
def empty_library_fixture() -> str: ...

@pytest.fixture(name='empty_payload', scope='package')
def empty_payload_fixture() -> str: ...

@pytest.fixture(name='grandchildren_300', scope='package')
def grandchildren_300_fixture() -> str: ...

@pytest.fixture(name='library_movies_all', scope='package')
def library_movies_all_fixture() -> str: ...

@pytest.fixture(name='library_movies_metadata', scope='package')
def library_movies_metadata_fixture() -> str: ...

@pytest.fixture(name='library_movies_collections', scope='package')
def library_movies_collections_fixture() -> str: ...

@pytest.fixture(name='library_tvshows_all', scope='package')
def library_tvshows_all_fixture() -> str: ...

@pytest.fixture(name='library_tvshows_metadata', scope='package')
def library_tvshows_metadata_fixture() -> str: ...

@pytest.fixture(name='library_tvshows_collections', scope='package')
def library_tvshows_collections_fixture() -> str: ...

@pytest.fixture(name='library_music_all', scope='package')
def library_music_all_fixture() -> str: ...

@pytest.fixture(name='library_music_metadata', scope='package')
def library_music_metadata_fixture() -> str: ...

@pytest.fixture(name='library_music_collections', scope='package')
def library_music_collections_fixture() -> str: ...

@pytest.fixture(name='library_movies_sort', scope='package')
def library_movies_sort_fixture() -> str: ...

@pytest.fixture(name='library_tvshows_sort', scope='package')
def library_tvshows_sort_fixture() -> str: ...

@pytest.fixture(name='library_music_sort', scope='package')
def library_music_sort_fixture() -> str: ...

@pytest.fixture(name='library_movies_filtertypes', scope='package')
def library_movies_filtertypes_fixture() -> str: ...

@pytest.fixture(name='library', scope='package')
def library_fixture() -> str: ...

@pytest.fixture(name='library_movies_size', scope='package')
def library_movies_size_fixture() -> str: ...

@pytest.fixture(name='library_music_size', scope='package')
def library_music_size_fixture() -> str: ...

@pytest.fixture(name='library_tvshows_size', scope='package')
def library_tvshows_size_fixture() -> str: ...

@pytest.fixture(name='library_tvshows_size_episodes', scope='package')
def library_tvshows_size_episodes_fixture() -> str: ...

@pytest.fixture(name='library_tvshows_size_seasons', scope='package')
def library_tvshows_size_seasons_fixture() -> str: ...

@pytest.fixture(name='library_sections', scope='package')
def library_sections_fixture() -> str: ...

@pytest.fixture(name='media_1', scope='package')
def media_1_fixture() -> str: ...

@pytest.fixture(name='media_30', scope='package')
def media_30_fixture() -> str: ...

@pytest.fixture(name='media_100', scope='package')
def media_100_fixture() -> str: ...

@pytest.fixture(name='media_200', scope='package')
def media_200_fixture() -> str: ...

@pytest.fixture(name='player_plexweb_resources', scope='package')
def player_plexweb_resources_fixture() -> str: ...

@pytest.fixture(name='player_plexhtpc_resources', scope='package')
def player_plexhtpc_resources_fixture() -> str: ...

@pytest.fixture(name='playlists', scope='package')
def playlists_fixture() -> str: ...

@pytest.fixture(name='playlist_500', scope='package')
def playlist_500_fixture() -> str: ...

@pytest.fixture(name='playqueue_created', scope='package')
def playqueue_created_fixture() -> str: ...

@pytest.fixture(name='playqueue_1234', scope='package')
def playqueue_1234_fixture() -> str: ...

@pytest.fixture(name='plex_server_accounts', scope='package')
def plex_server_accounts_fixture() -> str: ...

@pytest.fixture(name='plex_server_base', scope='package')
def plex_server_base_fixture() -> str: ...

@pytest.fixture(name='plex_server_default', scope='package')
def plex_server_default_fixture(plex_server_base: str) -> str: ...

@pytest.fixture(name='plex_server_clients', scope='package')
def plex_server_clients_fixture() -> str: ...

@pytest.fixture(name='plextv_account', scope='package')
def plextv_account_fixture() -> str: ...

@pytest.fixture(name='plextv_resources', scope='package')
def plextv_resources_fixture() -> str: ...

@pytest.fixture(name='plextv_resources_two_servers', scope='package')
def plextv_resources_two_servers_fixture() -> str: ...

@pytest.fixture(name='plextv_shared_users', scope='package')
def plextv_shared_users_fixture() -> str: ...

@pytest.fixture(name='session_base', scope='package')
def session_base_fixture() -> str: ...

@pytest.fixture(name='session_default', scope='package')
def session_default_fixture(session_base: str) -> str: ...

@pytest.fixture(name='session_new_user', scope='package')
def session_new_user_fixture(session_base: str) -> str: ...

@pytest.fixture(name='session_photo', scope='package')
def session_photo_fixture() -> str: ...

@pytest.fixture(name='session_plexweb', scope='package')
def session_plexweb_fixture() -> str: ...

@pytest.fixture(name='session_transient', scope='package')
def session_transient_fixture() -> str: ...

@pytest.fixture(name='session_unknown', scope='package')
def session_unknown_fixture() -> str: ...

@pytest.fixture(name='session_live_tv', scope='package')
def session_live_tv_fixture() -> str: ...

@pytest.fixture(name='livetv_sessions', scope='package')
def livetv_sessions_fixture() -> str: ...

@pytest.fixture(name='security_token', scope='package')
def security_token_fixture() -> str: ...

@pytest.fixture(name='show_seasons', scope='package')
def show_seasons_fixture() -> str: ...

@pytest.fixture(name='sonos_resources', scope='package')
def sonos_resources_fixture() -> str: ...

@pytest.fixture(name='hubs', scope='package')
def hubs_fixture() -> str: ...

@pytest.fixture(name='hubs_music_library', scope='package')
def hubs_music_library_fixture() -> str: ...

@pytest.fixture(name='update_check_nochange', scope='package')
def update_check_fixture_nochange() -> str: ...

@pytest.fixture(name='update_check_new', scope='package')
def update_check_fixture_new() -> str: ...

@pytest.fixture(name='update_check_new_not_updatable', scope='package')
def update_check_fixture_new_not_updatable() -> str: ...

@pytest.fixture(name='entry')
async def mock_config_entry() -> MockConfigEntry: ...

@pytest.fixture
def mock_websocket() -> Generator[MagicMock, None, None]: ...

@pytest.fixture
def mock_plex_calls(
    entry: MockConfigEntry,
    requests_mock: Any,
    children_20: str,
    children_30: str,
    children_200: str,
    children_300: str,
    empty_library: str,
    empty_payload: str,
    grandchildren_300: str,
    library: str,
    library_sections: str,
    library_movies_all: str,
    library_movies_collections: str,
    library_movies_metadata: str,
    library_movies_sort: str,
    library_music_all: str,
    library_music_collections: str,
    library_music_metadata: str,
    library_music_sort: str,
    library_tvshows_all: str,
    library_tvshows_collections: str,
    library_tvshows_metadata: str,
    library_tvshows_sort: str,
    media_1: str,
    media_30: str,
    media_100: str,
    media_200: str,
    playlists: str,
    playlist_500: str,
    plextv_account: str,
    plextv_resources: str,
    plextv_shared_users: str,
    plex_server_accounts: str,
    plex_server_clients: str,
    plex_server_default: str,
    security_token: str,
    update_check_nochange: str,
) -> None: ...

@pytest.fixture
def setup_plex_server(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    livetv_sessions: str,
    mock_websocket: MagicMock,
    mock_plex_calls: None,
    requests_mock: Any,
    empty_payload: str,
    session_default: str,
    session_live_tv: str,
    session_photo: str,
    session_plexweb: str,
    session_transient: str,
    session_unknown: str,
) -> Any: ...

@pytest.fixture
async def mock_plex_server(entry: MockConfigEntry, setup_plex_server: Any) -> Any: ...
```