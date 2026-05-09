async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class LgTVDevice(MediaPlayerEntity):
    _attr_assumed_state: bool
    _attr_device_class: MediaPlayerDeviceClass
    _attr_media_content_type: MediaType
    _attr_has_entity_name: bool
    _attr_name: str

    def __init__(self, client: LgNetCastClient, name: str, model: str, unique_id: str) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def send_command(self, command: LG_COMMAND) -> None:
        ...

    def update(self) -> None:
        ...

    @property
    def is_volume_muted(self) -> bool:
        ...

    @property
    def volume_level(self) -> float:
        ...

    @property
    def source(self) -> str:
        ...

    @property
    def source_list(self) -> list[str]:
        ...

    @property
    def media_content_id(self) -> str:
        ...

    @property
    def media_channel(self) -> str:
        ...

    @property
    def media_title(self) -> str:
        ...

    @property
    def supported_features(self) -> MediaPlayerEntityFeature:
        ...

    @property
    def media_image_url(self) -> str:
        ...

    def turn_off(self) -> None:
        ...

    async def async_turn_on(self) -> None:
        ...

    def volume_up(self) -> None:
        ...

    def volume_down(self) -> None:
        ...

    def set_volume_level(self, volume: float) -> None:
        ...

    def mute_volume(self, mute: bool) -> None:
        ...

    def select_source(self, source: str) -> None:
        ...

    def media_play(self) -> None:
        ...

    def media_pause(self) -> None:
        ...

    def media_stop(self) -> None:
        ...

    def media_next_track(self) -> None:
        ...

    def media_previous_track(self) -> None:
        ...

    def play_media(self, media_type: MediaType, media_id: str, **kwargs: Any) -> None:
        ...
