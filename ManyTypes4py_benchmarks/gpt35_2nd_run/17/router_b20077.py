    def get_api(password: str, host: str = None, username: str = None, port: int = None, ssl: bool = False) -> Netgear:
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
    async def async_setup(self) -> bool:
    async def async_get_attached_devices(self) -> Any:
    async def async_update_device_trackers(self, now: Any = None) -> bool:
    async def async_get_traffic_meter(self) -> Any:
    async def async_get_speed_test(self) -> Any:
    async def async_get_link_status(self) -> Any:
    async def async_allow_block_device(self, mac: str, allow_block: bool) -> None:
    async def async_get_utilization(self) -> Any:
    async def async_reboot(self) -> None:
    async def async_check_new_firmware(self) -> Any:
    async def async_update_new_firmware(self) -> None:
    @property
    def port(self) -> int:
    @property
    def ssl(self) -> bool:
