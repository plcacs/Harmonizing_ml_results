    def __init__(self, hass: HomeAssistant, config: dict, options: dict, config_entry_id: str = None) -> None:
    def get_aiohttp_session() -> aiohttp.ClientSession:
    async def async_init(self) -> None:
    async def _async_check_tcp_push(self, *_: Any) -> None:
    async def _async_check_onvif(self, *_: Any) -> None:
    async def _async_check_onvif_long_poll(self, *_: Any) -> None:
    async def update_states(self) -> None:
    async def disconnect(self) -> None:
    async def _async_start_long_polling(self, initial: bool = False) -> None:
    async def _async_stop_long_polling(self) -> None:
    async def stop(self, *_: Any) -> None:
    async def subscribe(self) -> None:
    async def renew(self) -> None:
    async def _renew(self, sub_type: SubType) -> None:
    def register_webhook(self) -> None:
    def unregister_webhook(self) -> None:
    async def _async_long_polling(self, *_: Any) -> None:
    async def _async_poll_all_motion(self, *_: Any) -> None:
    async def handle_webhook(self, hass: HomeAssistant, webhook_id: str, request: Request) -> None:
    async def _process_webhook_data(self, hass: HomeAssistant, webhook_id: str, data: bytes) -> None:
    def _signal_write_ha_state(self, channels: Any = None) -> None
