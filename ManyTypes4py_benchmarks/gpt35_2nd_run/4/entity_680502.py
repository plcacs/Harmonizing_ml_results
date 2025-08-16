    def __init__(self, hap: HomematicipHAP, device: Any, post: str = None, channel: int = None, is_multi_channel: bool = False) -> None:
    def device_info(self) -> DeviceInfo:
    async def async_added_to_hass(self) -> None:
    def _async_device_changed(self, *args, **kwargs) -> None:
    async def async_will_remove_from_hass(self) -> None:
    def async_remove_from_registries(self) -> None:
    def _async_device_removed(self, *args, **kwargs) -> None:
    def name(self) -> str:
    def available(self) -> bool:
    def unique_id(self) -> str:
    def icon(self) -> str:
    def extra_state_attributes(self) -> dict:
    def get_current_channel(self) -> FunctionalChannel:
