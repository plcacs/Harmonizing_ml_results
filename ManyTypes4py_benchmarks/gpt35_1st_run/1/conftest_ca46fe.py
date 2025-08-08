from typing import Any, Dict

def plex_server_url(entry: Any) -> str:
    """Return a protocol-less URL from a config entry."""
    return entry.data[PLEX_SERVER_CONFIG][CONF_URL].split(':', 1)[-1]

def mock_setup_entry() -> Generator:
    """Override async_setup_entry."""
    ...

def album_fixture() -> Generator:
    """Load album payload and return it."""
    ...

def artist_albums_fixture() -> Generator:
    """Load artist's albums payload and return it."""
    ...

# Add type annotations for the remaining fixtures as well

async def mock_config_entry() -> Any:
    """Return the default mocked config entry."""
    ...

def mock_websocket() -> Generator:
    """Mock the PlexWebsocket class."""
    ...

def mock_plex_calls(entry: Any, requests_mock: Any, children_20: Any, children_30: Any, children_200: Any, children_300: Any, empty_library: Any, empty_payload: Any, grandchildren_300: Any, library: Any, library_sections: Any, library_movies_all: Any, library_movies_collections: Any, library_movies_metadata: Any, library_movies_sort: Any, library_music_all: Any, library_music_collections: Any, library_music_metadata: Any, library_music_sort: Any, library_tvshows_all: Any, library_tvshows_collections: Any, library_tvshows_metadata: Any, library_tvshows_sort: Any, media_1: Any, media_30: Any, media_100: Any, media_200: Any, playlists: Any, playlist_500: Any, plextv_account: Any, plextv_resources: Any, plextv_shared_users: Any, plex_server_accounts: Any, plex_server_clients: Any, plex_server_default: Any, security_token: Any, update_check_nochange: Any) -> None:
    """Mock Plex API calls."""
    ...

async def setup_plex_server(hass: Any, entry: Any, livetv_sessions: Any, mock_websocket: Any, mock_plex_calls: Any, requests_mock: Any, empty_payload: Any, session_default: Any, session_live_tv: Any, session_photo: Any, session_plexweb: Any, session_transient: Any, session_unknown: Any, **kwargs: Dict[str, Any]) -> Any:
    """Set up and return a mocked Plex server instance."""
    ...

async def mock_plex_server(entry: Any, setup_plex_server: Any) -> Any:
    """Init from a config entry and return a mocked PlexServer instance."""
    ...
