    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    async def _async_update_listener(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
    async def async_setup(self) -> None:
    async def async_stop(self, event: Any = None) -> None:
    async def async_manually_set_date_and_time(self) -> None:
    async def async_check_date_and_time(self) -> None:
    def _async_log_time_out_of_sync(self, cam_date_utc: dt.datetime, system_date: dt.datetime) -> None:
    async def async_get_device_info(self) -> DeviceInfo:
    async def async_get_capabilities(self) -> Capabilities:
    async def async_start_events(self) -> bool:
    async def async_get_profiles(self) -> List[Profile]:
    async def async_get_stream_uri(self, profile: Profile) -> str:
    async def async_perform_ptz(self, profile: Profile, distance: float, speed: float, move_mode: str, continuous_duration: float, preset: str, pan: str = None, tilt: str = None, zoom: str = None) -> None:
    async def async_run_aux_command(self, profile: Profile, cmd: Any) -> None:
    async def async_set_imaging_settings(self, profile: Profile, settings: Any) -> None:
    def get_device(hass: HomeAssistant, host: str, port: int, username: str, password: str) -> ONVIFCamera:
