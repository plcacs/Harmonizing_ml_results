def services_to_strings(services: int, service_to_string: dict[int, str]) -> list[str]:
def _strings_to_services(strings: list[str], string_to_service: dict[str, int]) -> int:
def _update_state_attributes(self, payload: dict[str, Any]) -> None:
def _state_message_received(self, msg: ReceiveMessage) -> None:
def _prepare_subscribe_topics(self) -> None:
async def _subscribe_topics(self) -> None:
async def _async_publish_command(self, feature: VacuumEntityFeature) -> None:
async def async_start(self) -> None:
async def async_pause(self) -> None:
async def async_stop(self, **kwargs: Any) -> None:
async def async_return_to_base(self, **kwargs: Any) -> None:
async def async_clean_spot(self, **kwargs: Any) -> None:
async def async_locate(self, **kwargs: Any) -> None:
async def async_set_fan_speed(self, fan_speed: str, **kwargs: Any) -> None:
async def async_send_command(self, command: str, params: dict[str, Any] = None, **kwargs: Any) -> None:
