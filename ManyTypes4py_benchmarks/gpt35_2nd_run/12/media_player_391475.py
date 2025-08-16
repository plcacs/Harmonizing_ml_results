def __init__(self, config: ConfigEntry, name: str, host: str, region: str, ps4: pyps4.Ps4Async, creds: str) -> None:
async def async_added_to_hass(self) -> None:
async def async_update(self) -> None:
async def async_get_title_data(self, title_id: str, name: str) -> None:
async def async_get_device_info(self, status: dict[str, Any]) -> None:
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
