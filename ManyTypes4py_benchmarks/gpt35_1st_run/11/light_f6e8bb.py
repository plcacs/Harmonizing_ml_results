def create_light(item_class: LightEntity, coordinator: DataUpdateCoordinator, bridge: HueBridge, is_group: bool, rooms: dict, api: dict, item_id: str) -> LightEntity:
async def async_setup_entry(hass, config_entry, async_add_entities):
async def async_safe_fetch(bridge: HueBridge, fetch_method: callable) -> dict:
def async_update_items(bridge: HueBridge, api: dict, current: dict, async_add_entities: callable, create_item: callable, new_items_callback: callable):
def hue_brightness_to_hass(value: int) -> int:
def hass_to_hue_brightness(value: int) -> int:
class HueLight(CoordinatorEntity, LightEntity):
    def __init__(self, coordinator: DataUpdateCoordinator, bridge: HueBridge, is_group: bool, light: aiohue.Light, supported_color_modes: set, supported_features: LightEntityFeature, rooms: dict):
    @property
    def unique_id(self) -> str:
    @property
    def device_id(self) -> str:
    @property
    def name(self) -> str:
    @property
    def brightness(self) -> int:
    @property
    def color_mode(self) -> ColorMode:
    @property
    def _color_mode(self) -> str:
    @property
    def hs_color(self) -> tuple:
    @property
    def color_temp_kelvin(self) -> int:
    @property
    def max_color_temp_kelvin(self) -> int:
    @property
    def min_color_temp_kelvin(self) -> int:
    @property
    def is_on(self) -> bool:
    @property
    def available(self) -> bool:
    @property
    def effect(self) -> str:
    @property
    def effect_list(self) -> list:
    @property
    def device_info(self) -> DeviceInfo:
    async def async_turn_on(self, **kwargs):
    async def async_turn_off(self, **kwargs):
    @property
    def extra_state_attributes(self) -> dict:
