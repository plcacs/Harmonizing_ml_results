async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class VizioDevice(MediaPlayerEntity):
    def __init__(self, config_entry: ConfigEntry, device: VizioAsync, name: str, device_class: str, apps_coordinator: VizioAppsDataUpdateCoordinator) -> None:

    async def async_update(self) -> None:

    async def _async_send_update_options_signal(hass: HomeAssistant, config_entry: ConfigEntry) -> None:

    async def _async_update_options(self, config_entry: ConfigEntry) -> None:

    async def async_update_setting(self, setting_type: str, setting_name: str, new_value: str) -> None:

    async def async_added_to_hass(self) -> None:

    @property
    def source(self) -> str:

    @property
    def source_list(self) -> List[str]:

    @property
    def app_id(self) -> Optional[Dict[str, str]]:

    async def async_select_sound_mode(self, sound_mode: str) -> None:

    async def async_turn_on(self) -> None:

    async def async_turn_off(self) -> None:

    async def async_mute_volume(self, mute: bool) -> None:

    async def async_media_previous_track(self) -> None:

    async def async_media_next_track(self) -> None:

    async def async_select_source(self, source: str) -> None:

    async def async_volume_up(self) -> None:

    async def async_volume_down(self) -> None:

    async def async_set_volume_level(self, volume: float) -> None:

    async def async_media_play(self) -> None:

    async def async_media_pause(self) -> None:
