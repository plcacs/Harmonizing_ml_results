    def __init__(self, hass: HomeAssistant, server_config: dict, known_server_id: str = None, options: dict = None, entry_id: str = None) -> None:
    def account(self) -> MyPlexAccount:
    def plextv_clients(self) -> List[PlexClient]:
    def connect(self) -> None:
    def async_refresh_entity(self, machine_identifier: str, device: PlexClient, session: PlexSession, source: str) -> None:
    async def async_update_session(self, payload: dict) -> None:
    def _fetch_platform_data(self) -> Tuple[List[PlexClient], List[PlexSession], List[PlexClient]]:
    async def _async_update_platforms(self) -> None:
    @property
    def plex_server(self) -> PlexServer:
    @property
    def has_token(self) -> bool:
    @property
    def accounts(self) -> Set[str]:
    @property
    def owner(self) -> str:
    @property
    def version(self) -> str:
    @property
    def friendly_name(self) -> str:
    @property
    def machine_identifier(self) -> str:
    @property
    def url_in_use(self) -> str:
    @property
    def option_ignore_new_shared_users(self) -> bool:
    @property
    def option_use_episode_art(self) -> bool:
    @property
    def option_monitored_users(self) -> Dict[str, Dict[str, bool]]:
    @property
    def option_ignore_plexweb_clients(self) -> bool:
    @property
    def library(self) -> PlexLibrary:
    def playlist(self, title: str) -> PlexPlaylist:
    def playlists(self) -> List[PlexPlaylist]:
    def create_playqueue(self, media: Union[PlexLibraryItem, PlexPlaylist], **kwargs) -> PlexPlayQueue:
    def create_station_playqueue(self, key: str) -> PlexPlayQueue:
    def get_playqueue(self, playqueue_id: str) -> PlexPlayQueue:
    def fetch_item(self, item: Union[int, str]) -> PlexLibraryItem:
    def lookup_media(self, media_type: str, **kwargs) -> Union[PlexLibraryItem, PlexPlaylist]:
    @property
    def sensor_attributes(self) -> Dict[str, str]:
    def set_plex_server(self, plex_server: PlexServer) -> None:
    def switch_user(self, username: str) -> PlexServer:
