async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
async def async_set_lights_xy(hass: HomeAssistant, lights: list[str], x_val: float, y_val: float, brightness: int, transition: int) -> None:
async def async_set_lights_temp(hass: HomeAssistant, lights: list[str], kelvin: int, brightness: int, transition: int) -> None:
async def async_set_lights_rgb(hass: HomeAssistant, lights: list[str], rgb: tuple[int, int, int], transition: int) -> None:
async def async_update(call: ServiceCall = None) -> None:
class FluxSwitch(SwitchEntity, RestoreEntity):
    def __init__(self, name: str, hass: HomeAssistant, lights: list[str], start_time: datetime.time, stop_time: datetime.time, start_colortemp: int, sunset_colortemp: int, stop_colortemp: int, brightness: int, disable_brightness_adjust: bool, mode: str, interval: int, transition: int, unique_id: str) -> None:
    async def async_added_to_hass(self) -> None:
    async def async_will_remove_from_hass(self) -> None:
    async def async_turn_on(self, **kwargs) -> None:
    async def async_turn_off(self, **kwargs) -> None:
    async def async_flux_update(self, utcnow: datetime.datetime = None) -> None:
    def find_start_time(self, now: datetime.datetime) -> datetime.datetime:
    def find_stop_time(self, now: datetime.datetime) -> datetime.datetime:
