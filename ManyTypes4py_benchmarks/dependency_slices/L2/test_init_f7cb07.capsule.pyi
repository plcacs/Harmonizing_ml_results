from typing import Any

# === Internal dependency: homeassistant.components.light ===
class LightEntityFeature(IntFlag):
    EFFECT: int
    FLASH: int
    TRANSITION: int
class ColorMode(StrEnum):
    UNKNOWN: str
    ONOFF: str
    BRIGHTNESS: str
    COLOR_TEMP: str
    HS: str
    XY: str
    RGB: str
    RGBW: str
    RGBWW: str
    WHITE: str
def filter_supported_color_modes(color_modes: Iterable[ColorMode]) -> set[ColorMode]: ...
def valid_supported_color_modes(color_modes: Iterable[ColorMode | str]) -> set[ColorMode | str]: ...
def is_on(hass: HomeAssistant, entity_id: str) -> bool: ...
class Profile:
    ...
class Profiles: ...
class LightEntity(ToggleEntity):
    def _light_internal_color_mode(self) -> str: ...
    def _light_internal_rgbw_color(self) -> tuple[int, int, int, int] | None: ...
    def color_temp_kelvin(self) -> int | None: ...
    def min_color_temp_kelvin(self) -> int: ...
    def max_color_temp_kelvin(self) -> int: ...
    def capability_attributes(self) -> dict[str, Any]: ...
    def state_attributes(self) -> dict[str, Any] | None: ...
    def _light_internal_supported_color_modes(self) -> set[ColorMode] | set[str]: ...
    def supported_features_compat(self) -> LightEntityFeature: ...
DOMAIN: str
SUPPORT_BRIGHTNESS: int
SUPPORT_COLOR_TEMP: int
SUPPORT_COLOR: int
ATTR_TRANSITION: str
ATTR_RGB_COLOR: str
ATTR_RGBW_COLOR: str
ATTR_RGBWW_COLOR: str
ATTR_XY_COLOR: str
ATTR_HS_COLOR: str
ATTR_COLOR_TEMP: str
ATTR_COLOR_TEMP_KELVIN: str
ATTR_COLOR_NAME: str
ATTR_BRIGHTNESS: str
ATTR_BRIGHTNESS_PCT: str
ATTR_PROFILE: str
ATTR_FLASH: str
ATTR_EFFECT: str
EFFECT_OFF: str

# === Internal dependency: homeassistant.const ===
ENTITY_MATCH_ALL: Final
CONF_PLATFORM: Final
STATE_ON: Final
STATE_OFF: Final
ATTR_ENTITY_ID: Final
SERVICE_TURN_ON: Final
SERVICE_TURN_OFF: Final
SERVICE_TOGGLE: Final

# === Internal dependency: homeassistant.core ===
class Context: ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...
class Unauthorized(HomeAssistantError): ...

# === Internal dependency: homeassistant.setup ===
async def async_setup_component(hass: core.HomeAssistant, domain: str, config: ConfigType) -> bool: ...

# === Internal dependency: homeassistant.util.color ===
class RGBColor(NamedTuple): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.common ===
def async_mock_service(hass: HomeAssistant, domain: str, service: str, schema: vol.Schema | None = ..., response: ServiceResponse = ..., supports_response: SupportsResponse | None = ..., raise_exception: Exception | None = ...) -> list[ServiceCall]: ...
class MockEntityPlatform(entity_platform.EntityPlatform):
    def __init__(self, hass: HomeAssistant, logger = ..., domain = ..., platform_name = ..., platform = ..., scan_interval = ..., entity_namespace = ...) -> None: ...
def setup_test_component_platform(hass: HomeAssistant, domain: str, entities: Sequence[Entity], from_config_entry: bool = ..., built_in: bool = ...) -> MockPlatform: ...

# === Internal dependency: tests.components.light.common ===
class MockLight(MockToggleEntity, LightEntity):
    def __init__(self, name: str | None, state: Literal['on', 'off'] | None, supported_color_modes: set[ColorMode] | None = ...) -> None: ...

# === Third-party dependency: voluptuous ===
# Used symbols: Error, MultipleInvalid