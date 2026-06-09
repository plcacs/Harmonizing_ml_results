# === Internal dependency: homeassistant.components.light ===
class LightEntityFeature(IntFlag):
    EFFECT = 4
    FLASH = 8
    TRANSITION = 32
class ColorMode(StrEnum):
    UNKNOWN = 'unknown'
    ONOFF = 'onoff'
    BRIGHTNESS = 'brightness'
    COLOR_TEMP = 'color_temp'
    HS = 'hs'
    XY = 'xy'
    RGB = 'rgb'
    RGBW = 'rgbw'
    RGBWW = 'rgbww'
    WHITE = 'white'
def filter_supported_color_modes(color_modes): ...
def valid_supported_color_modes(color_modes): ...
def is_on(hass, entity_id): ...
class Profile:
    ...
class Profiles: ...
class LightEntity(ToggleEntity):
    def _light_internal_color_mode(self): ...
    def _light_internal_rgbw_color(self): ...
    def color_temp_kelvin(self): ...
    def min_color_temp_kelvin(self): ...
    def max_color_temp_kelvin(self): ...
    def capability_attributes(self): ...
    def state_attributes(self): ...
    def _light_internal_supported_color_modes(self): ...
    def supported_features_compat(self): ...
DOMAIN = 'light'
SUPPORT_BRIGHTNESS = 1
SUPPORT_COLOR_TEMP = 2
SUPPORT_COLOR = 16
ATTR_TRANSITION = 'transition'
ATTR_RGB_COLOR = 'rgb_color'
ATTR_RGBW_COLOR = 'rgbw_color'
ATTR_RGBWW_COLOR = 'rgbww_color'
ATTR_XY_COLOR = 'xy_color'
ATTR_HS_COLOR = 'hs_color'
ATTR_COLOR_TEMP = 'color_temp'
ATTR_COLOR_TEMP_KELVIN = 'color_temp_kelvin'
ATTR_COLOR_NAME = 'color_name'
ATTR_BRIGHTNESS = 'brightness'
ATTR_BRIGHTNESS_PCT = 'brightness_pct'
ATTR_PROFILE = 'profile'
ATTR_FLASH = 'flash'
ATTR_EFFECT = 'effect'
EFFECT_OFF = 'off'

# === Internal dependency: homeassistant.const ===
ENTITY_MATCH_ALL = 'all'
CONF_PLATFORM = 'platform'
STATE_ON = 'on'
STATE_OFF = 'off'
ATTR_ENTITY_ID = 'entity_id'
SERVICE_TURN_ON = 'turn_on'
SERVICE_TURN_OFF = 'turn_off'
SERVICE_TOGGLE = 'toggle'

# === Internal dependency: homeassistant.core ===
class Context: ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...
class Unauthorized(HomeAssistantError): ...

# === Internal dependency: homeassistant.setup ===
async def async_setup_component(hass, domain, config): ...

# === Internal dependency: homeassistant.util.color ===
class RGBColor(NamedTuple): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.common ===
def async_mock_service(hass, domain, service, schema=..., response=..., supports_response=..., raise_exception=...): ...
class MockEntityPlatform(entity_platform.EntityPlatform):
    def __init__(self, hass, logger=..., domain=..., platform_name=..., platform=..., scan_interval=..., entity_namespace=...): ...
def setup_test_component_platform(hass, domain, entities, from_config_entry=..., built_in=...): ...

# === Internal dependency: tests.components.light.common ===
class MockLight(MockToggleEntity, LightEntity):
    def __init__(self, name, state, supported_color_modes=...): ...

# === Third-party dependency: voluptuous ===
# Used symbols: Error, MultipleInvalid