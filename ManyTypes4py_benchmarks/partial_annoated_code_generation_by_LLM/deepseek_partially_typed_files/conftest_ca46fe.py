"""Fixtures for Plex tests."""
from collections.abc import Generator
from typing import Any
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
def mock_setup_entry() -> Generator[Any, None, None]:
    """Override async_setup_entry."""
    with patch('homeassistant.components.plex.async_setup_entry', return_value=True) as mock_setup_entry:
        yield mock_setup_entry

@pytest.fixture(name='album', scope='package')
def album_fixture() -> str:
    """Load album payload and return it."""
    return load_fixture('plex/album.xml')

@pytest.fixture(name='artist_albums', scope='package')
def artist_albums_fixture() -> str:
    """Load artist's albums payload and return it."""
    return load_fixture('plex/artist_albums.xml')

@pytest.fixture(name='children_20', scope='package')
def children_20_fixture() -> str:
    """Load children payload for item 20 and return it."""
    return load_fixture('plex/children_20.xml')

@pytest.fixture(name='children_30', scope='package')
def children_30_fixture() -> str:
    """Load children payload for item 30 and return it."""
    return load_fixture('plex/children_30.xml')

@pytest.fixture(name='children_200', scope='package')
def children_200_fixture() -> str:
    """Load children payload for item 200 and return it."""
    return load_fixture('plex/children_200.xml')

@pytest.fixture(name='children_300', scope='package')
def children_300_fixture() -> str:
    """Load children payload for item 300 and return it."""
    return load_fixture('plex/children_300.xml')

@pytest.fixture(name='empty_library', scope='package')
def empty_library_fixture() -> str:
    """Load an empty library payload and return it."""
    return load_fixture('plex/empty_library.xml')

@pytest.fixture(name='empty_payload', scope='package')
def empty_payload_fixture() -> str:
    """Load an empty payload and return it."""
    return load_fixture('plex/empty_payload.xml')

@pytest.fixture(name='grandchildren_300', scope='package')
def grandchildren_300_fixture() -> str:
    """Load grandchildren payload for item 300 and return it."""
    return load_fixture('plex/grandchildren_300.xml')

@pytest.fixture(name='library_movies_all', scope='package')
def library_movies_all_fixture() -> str:
    """Load payload for all items in the movies library and return it."""
    return load_fixture('plex/library_movies_all.xml')

@pytest.fixture(name='library_movies_metadata', scope='package')
def library_movies_metadata_fixture() -> str:
    """Load payload for metadata in the movies library and return it."""
    return load_fixture('plex/library_movies_metadata.xml')

@pytest.fixture(name='library_movies_collections', scope='package')
def library_movies_collections_fixture() -> str:
    """Load payload for collections in the movies library and return it."""
    return load_fixture('plex/library_movies_collections.xml')

@pytest.fixture(name='library_tvshows_all', scope='package')
def library_tvshows_all_fixture() -> str:
    """Load payload for all items in the tvshows library and return it."""
    return load_fixture('plex/library_tvshows_all.xml')

@pytest.fixture(name='library_tvshows_metadata', scope='package')
def library_tvshows_metadata_fixture() -> str:
    """Load payload for metadata in the TV shows library and return it."""
    return load_fixture('plex/library_tvshows_metadata.xml')

@pytest.fixture(name='library_tvshows_collections', scope='package')
def library_tvshows_collections_fixture() -> str:
    """Load payload for collections in the TV shows library and return it."""
    return load_fixture('plex/library_tvshows_collections.xml')

@pytest.fixture(name='library_music_all', scope='package')
def library_music_all_fixture() -> str:
    """Load payload for all items in the music library and return it."""
    return load_fixture('plex/library_music_all.xml')

@pytest.fixture(name='library_music_metadata', scope='package')
def library_music_metadata_fixture() -> str:
    """Load payload for metadata in the music library and return it."""
    return load_fixture('plex/library_music_metadata.xml')

@pytest.fixture(name='library_music_collections', scope='package')
def library_music_collections_fixture() -> str:
    """Load payload for collections in the music library and return it."""
    return load_fixture('plex/library_music_collections.xml')

@pytest.fixture(name='library_movies_sort', scope='package')
def library_movies_sort_fixture() -> str:
    """Load sorting payload for movie library and return it."""
    return load_fixture('plex/library_movies_sort.xml')

@pytest.fixture(name='library_tvshows_sort', scope='package')
def library_tvshows_sort_fixture() -> str:
    """Load sorting payload for tvshow library and return it."""
    return load_fixture('plex/library_tvshows_sort.xml')

@pytest.fixture(name='library_music_sort', scope='package')
def library_music_sort_fixture() -> str:
    """Load sorting payload for music library and return it."""
    return load_fixture('plex/library_music_sort.xml')

@pytest.fixture(name='library_movies_filtertypes', scope='package')
def library_movies_filtertypes_fixture() -> str:
    """Load filtertypes payload for movie library and return it."""
    return load_fixture('plex/library_movies_filtertypes.xml')

@pytest.fixture(name='library', scope='package')
def library_fixture() -> str:
    """Load library payload and return it."""
    return load_fixture('plex/library.xml')

@pytest.fixture(name='library_movies_size', scope='package')
def library_movies_size_fixture() -> str:
    """Load movie library size payload and return it."""
    return load_fixture('plex/library_movies_size.xml')

@pytest.fixture(name='library_music_size', scope='package')
def library_music_size_fixture() -> str:
    """Load music library size payload and return it."""
    return load_fixture('plex/library_music_size.xml')

@pytest.fixture(name='library_tvshows_size', scope='package')
def library_tvshows_size_fixture() -> str:
    """Load tvshow library size payload and return it."""
    return load_fixture('plex/library_tvshows_size.xml')

@pytest.fixture(name='library_tvshows_size_episodes', scope='package')
def library_tvshows_size_episodes_fixture() -> str:
    """Load tvshow library size in episodes payload and return it."""
    return load_fixture('plex/library_tvshows_size_episodes.xml')

@pytest.fixture(name='library_tvshows_size_seasons', scope='package')
def library_tvshows_size_seasons_fixture() -> str:
    """Load tvshow library size in seasons payload and return it."""
    return load_fixture('plex/library_tvshows_size_seasons.xml')

@pytest.fixture(name='library_sections', scope='package')
def library_sections_fixture() -> str:
    """Load library sections payload and return it."""
    return load_fixture('plex/library_sections.xml')

@pytest.fixture(name='media_1', scope='package')
def media_1_fixture() -> str:
    """Load media payload for item 1 and return it."""
    return load_fixture('plex/media_1.xml')

@pytest.fixture(name='media_30', scope='package')
def media_30_fixture() -> str:
    """Load media payload for item 30 and return it."""
    return load_fixture('plex/media_30.xml')

@pytest.fixture(name='media_100', scope='package')
def media_100_fixture() -> str:
    """Load media payload for item 100 and return it."""
    return load_fixture('plex/media_100.xml')

@pytest.fixture(name='media_200', scope='package')
def media_200_fixture() -> str:
    """Load media payload for item 200 and return it."""
    return load_fixture('plex/media_200.xml')

@pytest.fixture(name='player_plexweb_resources', scope='package')
def player_plexweb_resources_fixture() -> str:
    """Load resources payload for a Plex Web player and return it."""
    return load_fixture('plex/player_plexweb_resources.xml')

@pytest.fixture(name='player_plexhtpc_resources', scope='package')
def player_plexhtpc_resources_fixture() -> str:
    """Load resources payload for a Plex HTPC player and return it."""
    return load_fixture('plex/player_plexhtpc_resources.xml')

@pytest.fixture(name='playlists', scope='package')
def playlists_fixture() -> str:
    """Load payload for all playlists and return it."""
    return load_fixture('plex/playlists.xml')

@pytest.fixture(name='playlist_500', scope='package')
def playlist_500_fixture() -> str:
    """Load payload for playlist 500 and return it."""
    return load_fixture('plex/playlist_500.xml')

@pytest.fixture(name='playqueue_created', scope='package')
def playqueue_created_fixture() -> str:
    """Load payload for playqueue creation response and return it."""
    return load_fixture('plex/playqueue_created.xml')

@pytest.fixture(name='playqueue_1234', scope='package')
def playqueue_1234_fixture() -> str:
    """Load payload for playqueue 1234 and return it."""
    return load_fixture('plex/playqueue_1234.xml')

@pytest.fixture(name='plex_server_accounts', scope='package')
def plex_server_accounts_fixture() -> str:
    """Load payload accounts on the Plex server and return it."""
    return load_fixture('plex/plex_server_accounts.xml')

@pytest.fixture(name='plex_server_base', scope='package')
def plex_server_base_fixture() -> str:
    """Load base payload for Plex server info and return it."""
    return load_fixture('plex/plex_server_base.xml')

@pytest.fixture(name='plex_server_default', scope='package')
def plex_server_default_fixture(plex_server_base: str) -> str:
    """Load default payload for Plex server info and return it."""
    return plex_server_base.format(name='Plex Server 1', machine_identifier='unique_id_123')

@pytest.fixture(name='plex_server_clients', scope='package')
def plex_server_clients_fixture() -> str:
    """Load available clients payload for Plex server and return it."""
    return load_fixture('plex/plex_server_clients.xml')

@pytest.fixture(name='plextv_account', scope='package')
def plextv_account_fixture() -> str:
    """Load account info from plex.tv and return it."""
    return load_fixture('plex/plextv_account.xml')

@pytest.fixture(name='plextv_resources', scope='package')
def plextv_resources_fixture() -> str:
    """Load single-server payload for plex.tv resources and return it."""
    return load_fixture('plex/plextv_resources_one_server.xml')

@pytest.fixture(name='plextv_resources_two_servers', scope='package')
def plextv_resources_two_servers_fixture() -> str:
    """Load two-server payload for plex.tv resources and return it."""
    return load_fixture('plex/plextv_resources_two_servers.xml')

@pytest.fixture(name='plextv_shared_users', scope='package')
def plextv_shared_users_fixture() -> str:
    """Load payload for plex.tv shared users and return it."""
    return load_fixture('plex/plextv_shared_users.xml')

@pytest.fixture(name='session_base', scope='package')
def session_base_fixture() -> str:
    """Load the base session payload and return it."""
    return load_fixture('plex/session_base.xml')

@pytest.fixture(name='session_default', scope='package')
def session_default_fixture(session_base: str) -> str:
    """Load the default session payload and return it."""
    return session_base.format(user_id=1)

@pytest.fixture(name='session_new_user', scope='package')
def session_new_user_fixture(session_base: str) -> str:
    """Load the new user session payload and return it."""
    return session_base.format(user_id=1001)

@pytest.fixture(name='session_photo', scope='package')
def session_photo_fixture() -> str:
    """Load a photo session payload and return it."""
    return load_fixture('plex/session_photo.xml')

@pytest.fixture(name='session_plexweb', scope='package')
def session_plexweb_fixture() -> str:
    """Load a Plex Web session payload and return it."""
    return load_fixture('plex/session_plexweb.xml')

@pytest.fixture(name='session_transient', scope='package')
def session_transient_fixture() -> str:
    """Load a transient session payload and return it."""
    return load_fixture('plex/session_transient.xml')

@pytest.fixture(name='session_unknown', scope='package')
def session_unknown_fixture() -> str:
    """Load a hypothetical unknown session payload and return it."""
    return load_fixture('plex/session_unknown.xml')

@pytest.fixture(name='session_live_tv', scope='package')
def session_live_tv_fixture() -> str:
    """Load a Live TV session payload and return it."""
    return load_fixture('plex/session_live_tv.xml')

@pytest.fixture(name='livetv_sessions', scope='package')
def livetv_sessions_fixture() -> str:
    """Load livetv/sessions payload and return it."""
    return load_fixture('plex/livetv_sessions.xml')

@pytest.fixture(name='security_token', scope='package')
def security_token_fixture() -> str:
    """Load a security token payload and return it."""
    return load_fixture('plex/security_token.xml')

@pytest.fixture(name='show_seasons', scope='package')
def show_seasons_fixture() -> str:
    """Load a show's seasons payload and return it."""
    return load_fixture('plex/show_seasons.xml')

@pytest.fixture(name='sonos_resources', scope='package')
def sonos_resources_fixture() -> str:
    """Load Sonos resources payload and return it."""
    return load_fixture('plex/sonos_resources.xml')

@pytest.fixture(name='hubs', scope='package')
def hubs_fixture() -> str:
    """Load hubs resource payload and return it."""
    return load_fixture('plex/hubs.xml')

@pytest.fixture(name='hubs_music_library', scope='package')
def hubs_music_library_fixture() -> str:
    """Load music library hubs resource payload and return it."""
    return load_fixture('plex/hubs_library_section.xml')

@pytest.fixture(name='update_check_nochange', scope='package')
def update_check_fixture_nochange() -> str:
    """Load a no-change update resource payload and return it."""
    return load_fixture('plex/release_nochange.xml')

@pytest.fixture(name='update_check_new', scope='package')
def update_check_fixture_new() -> str:
    """Load a changed update resource payload and return it."""
    return load_fixture('plex/release_new.xml')

@pytest.fixture(name='update_check_new_not_updatable', scope='package')
def update_check_fixture_new_not_updatable() -> str:
    """Load a changed update resource payload (not updatable) and return it."""
    return load_fixture('plex/release_new_not_updatable.xml')

@pytest.fixture(name='entry')
async def mock_config_entry() -> MockConfigEntry:
    """Return the default mocked config entry."""
    return MockConfigEntry(domain=DOMAIN, data=DEFAULT_DATA, options=DEFAULT_OPTIONS, unique_id=DEFAULT_DATA['server_id'])

@pytest.fixture
def mock_websocket() -> Generator[Any, None, None]:
    """Mock the PlexWebsocket class."""
    with patch('homeassistant.components.plex.PlexWebsocket', autospec=True) as ws:
        yield ws

@pytest.fixture
def mock_plex_calls(entry: MockConfigEntry, requests_mock: requests_mock.Mocker, children_20: str, children_30: str, children_200: str, children_300: str, empty_library: str, empty_payload: str, grandchildren_300: str, library: str, library_sections: str, library_movies_all: str, library_movies_collections: str, library_movies_metadata: str, library_movies_sort: str, library_music_all: str, library_music_collections: str, library_music_metadata: str, library_music_sort: str, library_tvshows_all: str, library_tvshows_collections: str, library_tvshows_metadata: str, library_tvshows_sort: str, media_1: str, media_30: str, media_100: str, media_200: str, playlists: str, playlist_