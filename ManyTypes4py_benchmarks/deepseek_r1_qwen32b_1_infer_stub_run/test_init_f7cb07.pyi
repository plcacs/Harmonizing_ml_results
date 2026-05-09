"""Stub file for 'test_init_f7cb07' module."""

from homeassistant.components import light
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_BRIGHTNESS,
    ATTR_BRIGHTNESS_PCT,
    ATTR_COLOR_NAME,
    ATTR_COLOR_TEMP,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_EFFECT,
    ATTR_FLASH,
    ATTR_HS_COLOR,
    ATTR_PROFILE,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_TRANSITION,
    ATTR_XY_COLOR,
    CONF_PLATFORM,
    ENTITY_MATCH_ALL,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    SERVICE_TOGGLE,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import (
    HomeAssistant,
    Context,
    State,
)
from homeassistant.exceptions import (
    HomeAssistantError,
    Unauthorized,
)
from homeassistant.setup import async_setup_component
from homeassistant.util.color import (
    RGBColor,
    color_util,
)
from pytest import (
    fixture,
    mark,
    parametrize,
)
from unittest.mock import (
    MagicMock,
    patch,
)
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

class MockLight:
    """Mock Light entity class."""
    def __init__(self, name: str, state: str) -> None:
        """Initialize mock light."""
        self.entity_id: str = f"light.{name}"
        self.state: str = state
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.supported_features: int = 0
        self.supported_color_modes: Optional[List[light.ColorMode]] = None
        self.color_mode: Optional[light.ColorMode] = None
        self.brightness: Optional[int] = None
        self.hs_color: Optional[Tuple[float, float]] = None
        self.rgb_color: Optional[Tuple[int, int, int]] = None
        self.rgbw_color: Optional[Tuple[int, int, int, int]] = None
        self.rgbww_color: Optional[Tuple[int, int, int, int, int]] = None
        self.xy_color: Optional[Tuple[float, float]] = None
        self.color_temp: Optional[int] = None
        self.color_temp_kelvin: Optional[int] = None
        self.effect: Optional[str] = None

    def last_call(self, method: str) -> Tuple[str, Dict[str, Any]]:
        """Return last call to method."""
        ...

class MockUser:
    """Mock User class."""
    def __init__(self, id: str) -> None:
        """Initialize mock user."""
        self.id: str = id
        self.mock_policy: Callable[[Dict[str, Any]], None] = lambda _: None

class Profile:
    """Light profile class."""
    def __init__(self, name: str, x: float, y: float, brightness: int, transition: Optional[float]) -> None:
        """Initialize profile."""
        self.name: str = name
        self.x: float = x
        self.y: float = y
        self.brightness: int = brightness
        self.transition: Optional[float] = transition
        self.hs_color: Optional[Tuple[float, float]] = None

@fixture
def mock_light_entities() -> List[MockLight]:
    """Fixture for mock light entities."""
    ...

@fixture
def mock_light_profiles() -> Dict[str, Profile]:
    """Fixture for mock light profiles."""
    ...

@mark.parametrize(('profile_name', 'last_call', 'expected_data'), [
    ('test', 'turn_on', {ATTR_HS_COLOR: (71.059, 100), ATTR_BRIGHTNESS: 100, ATTR_TRANSITION: 0}),
    ('color_no_brightness_no_transition', 'turn_on', {ATTR_HS_COLOR: (71.059, 100)}),
    ('no color', 'turn_on', {ATTR_BRIGHTNESS: 110, ATTR_TRANSITION: 0}),
    ('test_off', 'turn_off', {ATTR_TRANSITION: 0}),
    ('no brightness', 'turn_on', {ATTR_HS_COLOR: (71.059, 100)}),
    ('color_and_brightness', 'turn_on', {ATTR_HS_COLOR: (71.059, 100), ATTR_BRIGHTNESS: 120}),
    ('color_and_transition', 'turn_on', {ATTR_HS_COLOR: (71.059, 100), ATTR_TRANSITION: 4.2}),
    ('brightness_and_transition', 'turn_on', {ATTR_BRIGHTNESS: 130, ATTR_TRANSITION: 5.3}),
])
async def test_light_profiles(
    hass: HomeAssistant,
    mock_light_profiles: Dict[str, Profile],
    profile_name: str,
    expected_data: Dict[str, Any],
    last_call: str,
    mock_light_entities: List[MockLight],
) -> Awaitable[None]:
    """Test light profiles."""
    ...

@mark.parametrize(('extra_call_params', 'expected_params_state_was_off', 'expected_params_state_was_on'), [
    ({}, {ATTR_HS_COLOR: (50.353, 100), ATTR_BRIGHTNESS: 100, ATTR_TRANSITION: 3}, {ATTR_HS_COLOR: (50.353, 100), ATTR_BRIGHTNESS: 100, ATTR_TRANSITION: 3}),
    ({ATTR_BRIGHTNESS: 22}, {ATTR_HS_COLOR: (50.353, 100), ATTR_BRIGHTNESS: 22, ATTR_TRANSITION: 3}, {ATTR_BRIGHTNESS: 22, ATTR_TRANSITION: 3}),
    ({ATTR_TRANSITION: 22}, {ATTR_HS_COLOR: (50.353, 100), ATTR_BRIGHTNESS: 100, ATTR_TRANSITION: 22}, {ATTR_TRANSITION: 22}),
    ({ATTR_COLOR_TEMP: 600, ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_COLOR_TEMP: 600, ATTR_COLOR_TEMP_KELVIN: 1666, ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_COLOR_TEMP: 600, ATTR_COLOR_TEMP_KELVIN: 1666, ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}),
    ({ATTR_COLOR_TEMP_KELVIN: 6500, ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_COLOR_TEMP: 153, ATTR_COLOR_TEMP_KELVIN: 6500, ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_COLOR_TEMP: 153, ATTR_COLOR_TEMP_KELVIN: 6500, ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}),
    ({ATTR_HS_COLOR: [70, 80], ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_HS_COLOR: (70, 80), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_HS_COLOR: (70, 80), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}),
    ({ATTR_RGB_COLOR: [1, 2, 3], ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_RGB_COLOR: (1, 2, 3), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_RGB_COLOR: (1, 2, 3), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}),
    ({ATTR_RGBW_COLOR: [1, 2, 3, 4], ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_RGBW_COLOR: (1, 2, 3, 4), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_RGBW_COLOR: (1, 2, 3, 4), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}),
    ({ATTR_RGBWW_COLOR: [1, 2, 3, 4, 5], ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_RGBWW_COLOR: (1, 2, 3, 4, 5), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_RGBWW_COLOR: (1, 2, 3, 4, 5), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}),
    ({ATTR_XY_COLOR: [0.4448, 0.4066], ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_XY_COLOR: (0.4448, 0.4066), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_XY_COLOR: (0.4448, 0.4066), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}),
    ({ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_HS_COLOR: (50.353, 100), ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}, {ATTR_BRIGHTNESS: 11, ATTR_TRANSITION: 1}),
])
async def test_default_profiles_light(
    hass: HomeAssistant,
    mock_light_profiles: Dict[str, Profile],
    extra_call_params: Dict[str, Any],
    expected_params_state_was_off: Dict[str, Any],
    expected_params_state_was_on: Dict[str, Any],
    mock_light_entities: List[MockLight],
) -> Awaitable[None]:
    """Test default turn-on light profile for a specific light."""
    ...

@mark.parametrize('light_state', [STATE_ON, STATE_OFF])
async def test_light_backwards_compatibility_supported_color_modes(
    hass: HomeAssistant,
    light_state: str,
) -> Awaitable[None]:
    """Test supported_color_modes if not implemented by the entity."""
    ...

@mark.parametrize('entity_feature', list(light.LightEntityFeature))
def test_deprecated_support_light_constants_enums(
    caplog: Any,
    entity_feature: light.LightEntityFeature,
) -> None:
    """Test deprecated support light constants."""
    ...

@mark.parametrize('entity_feature', list(light.ColorMode))
def test_deprecated_color_mode_constants_enums(
    caplog: Any,
    entity_feature: light.ColorMode,
) -> None:
    """Test deprecated support light constants."""
    ...

@mark.parametrize(('constant_name', 'constant_value', 'constant_replacement'), [
    ('SUPPORT_BRIGHTNESS', 1, 'supported_color_modes'),
    ('SUPPORT_COLOR_TEMP', 2, 'supported_color_modes'),
    ('SUPPORT_COLOR', 16, 'supported_color_modes'),
    ('ATTR_COLOR_TEMP', 'color_temp', 'kelvin equivalent (ATTR_COLOR_TEMP_KELVIN)'),
    ('ATTR_KELVIN', 'kelvin', 'ATTR_COLOR_TEMP_KELVIN'),
    ('ATTR_MIN_MIREDS', 'min_mireds', 'kelvin equivalent (ATTR_MAX_COLOR_TEMP_KELVIN)'),
    ('ATTR_MAX_MIREDS', 'max_mireds', 'kelvin equivalent (ATTR_MIN_COLOR_TEMP_KELVIN)'),
])
def test_deprecated_light_constants(
    caplog: Any,
    constant_name: str,
    constant_value: Any,
    constant_replacement: str,
) -> None:
    """Test deprecated light constants."""
    ...

async def test_deprecated_turn_on_arguments(
    hass: HomeAssistant,
    caplog: Any,
) -> Awaitable[None]:
    """Test color temp conversion in service calls."""
    ...