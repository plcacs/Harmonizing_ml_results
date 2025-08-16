    def __init__(self, hass: HomeAssistant, driver: HomeDriver, name: str, entity_id: str, aid: int, config: dict) -> None:
    def _async_update_motion_state_event(self, event: Event) -> None:
    def _async_update_motion_state(self, old_state: State, new_state: State) -> None:
    def async_update_state(self, new_state: State) -> None:
    async def _async_get_stream_source(self) -> str:
    async def start_stream(self, session_info: dict, stream_config: dict) -> bool:
    async def _async_log_stderr_stream(self, stderr_reader: asyncio.StreamReader) -> None:
    async def _async_ffmpeg_watch(self, session_id: str) -> bool:
    def _async_stop_ffmpeg_watch(self, session_id: str) -> None:
    def async_stop(self) -> None:
    async def stop_stream(self, session_info: dict) -> None:
    async def reconfigure_stream(self, session_info: dict, stream_config: dict) -> bool:
    async def async_get_snapshot(self, image_size: dict) -> bytes:
