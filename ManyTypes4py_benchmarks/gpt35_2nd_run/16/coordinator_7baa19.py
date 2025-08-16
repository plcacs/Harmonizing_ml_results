    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, api_key: str, config: HERETravelTimeConfig) -> None:
    async def _async_update_data(self) -> HERETravelTimeData:
    def _parse_routing_response(self, response: dict) -> HERETravelTimeData:
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, api_key: str, config: HERETravelTimeConfig) -> None:
    async def _async_update_data(self) -> HERETravelTimeData | None:
    def _parse_transit_response(self, response: dict) -> HERETravelTimeData:
    def prepare_parameters(hass: HomeAssistant, config: HERETravelTimeConfig) -> tuple:
    def build_hass_attribution(sections: list) -> str:
    def next_datetime(simple_time: time) -> datetime:
