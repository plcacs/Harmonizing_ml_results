async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class ForkedDaapdZone(MediaPlayerEntity):
    def __init__(self, api: ForkedDaapdAPI, output: Any, entry_id: str) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_toggle(self) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    async def async_turn_on(self) -> None:
        ...

    async def async_turn_off(self) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def state(self) -> MediaPlayerState:
        ...

    @property
    def volume_level(self) -> float:
        ...

    @property
    def is_volume_muted(self) -> bool:
        ...

    async def async_mute_volume(self, mute: bool) -> None:
        ...

    async def async_set_volume_level(self, volume: float) -> None:
        ...

    @property
    def supported_features(self) -> int:
        ...

class ForkedDaapdMaster(MediaPlayerEntity):
    def __init__(self, clientsession: Any, api: ForkedDaapdAPI, ip_address: str, config_entry: ConfigEntry) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def _update_callback(self, available: bool) -> None:
        ...

    @callback
    def update_options(self, options: dict) -> None:
        ...

    @callback
    def _update_player(self, player: dict, event: asyncio.Event) -> None:
        ...

    @callback
    def _update_queue(self, queue: dict, event: asyncio.Event) -> None:
        ...

    @callback
    def _update_outputs(self, outputs: list, event: Optional[asyncio.Event] = None) -> None:
        ...

    @callback
    def _update_database(self, pipes: list, playlists: list, event: asyncio.Event) -> None:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def available(self) -> bool:
        ...

    async def async_turn_on(self) -> None:
        ...

    async def async_turn_off(self) -> None:
        ...

    async def async_toggle(self) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def state(self) -> MediaPlayerState:
        ...

    @property
    def volume_level(self) -> float:
        ...

    @property
    def is_volume_muted(self) -> bool:
        ...

    @property
    def media_content_id(self) -> str:
        ...

    @property
    def media_content_type(self) -> str:
        ...

    @property
    def media_duration(self) -> float:
        ...

    @property
    def media_position(self) -> float:
        ...

    @property
    def media_position_updated_at(self) -> datetime:
        ...

    @property
    def media_title(self) -> str:
        ...

    @property
    def media_artist(self) -> str:
        ...

    @property
    def media_album_name(self) -> str:
        ...

    @property
    def media_album_artist(self) -> str:
        ...

    @property
    def media_track(self) -> int:
        ...

    @property
    def shuffle(self) -> bool:
        ...

    @property
    def supported_features(self) -> int:
        ...

    @property
    def source(self) -> str:
        ...

    @property
    def source_list(self) -> list:
        ...

    async def async_mute_volume(self, mute: bool) -> None:
        ...

    async def async_set_volume_level(self, volume: float) -> None:
        ...

    async def async_media_play(self) -> None:
        ...

    async def async_media_pause(self) -> None:
        ...

    async def async_media_stop(self) -> None:
        ...

    async def async_media_previous_track(self) -> None:
        ...

    async def async_media_next_track(self) -> None:
        ...

    async def async_media_seek(self, position: float) -> None:
        ...

    async def async_clear_playlist(self) -> None:
        ...

    async def async_set_shuffle(self, shuffle: bool) -> None:
        ...

    @property
    def media_image_url(self) -> str:
        ...

    async def _save_and_set_tts_volumes(self) -> None:
        ...

    async def _pause_and_wait_for_callback(self) -> None:
        ...

    async def async_play_media(self, media_type: str, media_id: str, **kwargs) -> None:
        ...

    async def _async_announce(self, media_id: str) -> None:
        ...

    async def async_select_source(self, source: str) -> None:
        ...

    def _use_pipe_control(self) -> str:
        ...

    async def _pipe_call(self, pipe_name: str, base_function_name: str) -> None:
        ...

    async def async_browse_media(self, media_content_type: Optional[str] = None, media_content_id: Optional[str] = None) -> BrowseMedia:
        ...

    async def async_get_browse_image(self, media_content_type: str, media_content_id: str, media_image_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]:
        ...
