from typing import Any, cast

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class PS4Device(MediaPlayerEntity):
    def __init__(self, config: ConfigEntry, name: str, host: str, region: str, ps4: pyps4.Ps4Async, creds: str) -> None:

    def status_callback(self) -> None:

    def subscribe_to_protocol(self) -> None:

    def unsubscribe_to_protocol(self) -> None:

    def check_region(self) -> None:

    async def async_added_to_hass(self) -> None:

    async def async_update(self) -> None:

    def _parse_status(self) -> None:

    def _use_saved(self) -> bool:

    def idle(self) -> None:

    def state_standby(self) -> None:

    def state_unknown(self) -> None:

    def reset_title(self) -> None:

    async def async_get_title_data(self, title_id: str, name: str) -> None:

    def update_list(self) -> None:

    def get_source_list(self) -> None:

    def add_games(self, title_id: str, app_name: str, image: str, g_type: str, is_locked: bool = False) -> None:

    async def async_get_device_info(self, status: Any) -> None:

    async def async_will_remove_from_hass(self) -> None:

    @property
    def entity_picture(self) -> str:

    @property
    def media_image_url(self) -> str:

    async def async_turn_off(self) -> None:

    async def async_turn_on(self) -> None:

    async def async_toggle(self) -> None:

    async def async_media_pause(self) -> None:

    async def async_media_stop(self) -> None:

    async def async_select_source(self, source: str) -> None:

    async def async_send_command(self, command: str) -> None:

    async def async_send_remote_control(self, command: str) -> None:
