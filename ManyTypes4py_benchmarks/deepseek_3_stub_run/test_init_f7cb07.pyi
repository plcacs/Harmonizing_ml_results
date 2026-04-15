from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any, Literal, Optional, Union
from unittest.mock import MagicMock

import pytest
import voluptuous as vol
from homeassistant import core
from homeassistant.components import light
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_PLATFORM,
    ENTITY_MATCH_ALL,
    SERVICE_TOGGLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers import frame
from homeassistant.setup import async_setup_component
from homeassistant.util import color as color_util

from .common import MockLight
from tests.common import (
    MockEntityPlatform,
    MockUser,
    async_mock_service,
    help_test_all,
    import_and_test_deprecated_constant,
    import_and_test_deprecated_constant_enum,
    setup_test_component_platform,
)

orig_Profiles: type = ...

async def test_methods(hass: HomeAssistant) -> None: ...

async def test_services(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    mock_light_entities: Sequence[MockLight],
) -> None: ...

@pytest.mark.parametrize(
    ("profile_name", "last_call", "expected_data"),
    [
        ("test", "turn_on", {light.ATTR_HS_COLOR: (71.059, 100), light.ATTR_BRIGHTNESS: 100, light.ATTR_TRANSITION: 0}),
        ("color_no_brightness_no_transition", "turn_on", {light.ATTR_HS_COLOR: (71.059, 100)}),
        ("no color", "turn_on", {light.ATTR_BRIGHTNESS: 110, light.ATTR_TRANSITION: 0}),
        ("test_off", "turn_off", {light.ATTR_TRANSITION: 0}),
        ("no brightness", "turn_on", {light.ATTR_HS_COLOR: (71.059, 100)}),
        ("color_and_brightness", "turn_on", {light.ATTR_HS_COLOR: (71.059, 100), light.ATTR_BRIGHTNESS: 120}),
        ("color_and_transition", "turn_on", {light.ATTR_HS_COLOR: (71.059, 100), light.ATTR_TRANSITION: 4.2}),
        ("brightness_and_transition", "turn_on", {light.ATTR_BRIGHTNESS: 130, light.ATTR_TRANSITION: 5.3}),
    ],
)
async def test_light_profiles(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    profile_name: str,
    expected_data: dict[str, Any],
    last_call: Literal["turn_on", "turn_off"],
    mock_light_entities: Sequence[MockLight],
) -> None: ...

async def test_default_profiles_group(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    mock_light_entities: Sequence[MockLight],
) -> None: ...

@pytest.mark.parametrize(
    ("extra_call_params", "expected_params_state_was_off", "expected_params_state_was_on"),
    [
        (
            {},
            {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 100, light.ATTR_TRANSITION: 3},
            {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 100, light.ATTR_TRANSITION: 3},
        ),
        (
            {light.ATTR_BRIGHTNESS: 22},
            {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 22, light.ATTR_TRANSITION: 3},
            {light.ATTR_BRIGHTNESS: 22, light.ATTR_TRANSITION: 3},
        ),
        (
            {light.ATTR_TRANSITION: 22},
            {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 100, light.ATTR_TRANSITION: 22},
            {light.ATTR_TRANSITION: 22},
        ),
        (
            {light.ATTR_COLOR_TEMP: 600, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_COLOR_TEMP: 600, light.ATTR_COLOR_TEMP_KELVIN: 1666, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_COLOR_TEMP: 600, light.ATTR_COLOR_TEMP_KELVIN: 1666, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
        ),
        (
            {light.ATTR_COLOR_TEMP_KELVIN: 6500, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_COLOR_TEMP: 153, light.ATTR_COLOR_TEMP_KELVIN: 6500, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_COLOR_TEMP: 153, light.ATTR_COLOR_TEMP_KELVIN: 6500, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
        ),
        (
            {light.ATTR_HS_COLOR: [70, 80], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_HS_COLOR: (70, 80), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_HS_COLOR: (70, 80), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
        ),
        (
            {light.ATTR_RGB_COLOR: [1, 2, 3], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_RGB_COLOR: (1, 2, 3), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_RGB_COLOR: (1, 2, 3), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
        ),
        (
            {light.ATTR_RGBW_COLOR: [1, 2, 3, 4], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_RGBW_COLOR: (1, 2, 3, 4), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_RGBW_COLOR: (1, 2, 3, 4), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
        ),
        (
            {light.ATTR_RGBWW_COLOR: [1, 2, 3, 4, 5], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_RGBWW_COLOR: (1, 2, 3, 4, 5), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_RGBWW_COLOR: (1, 2, 3, 4, 5), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
        ),
        (
            {light.ATTR_XY_COLOR: [0.4448, 0.4066], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_XY_COLOR: (0.4448, 0.4066), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_XY_COLOR: (0.4448, 0.4066), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
        ),
        (
            {light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
        ),
    ],
)
async def test_default_profiles_light(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    extra_call_params: dict[str, Any],
    expected_params_state_was_off: dict[str, Any],
    expected_params_state_was_on: dict[str, Any],
    mock_light_entities: Sequence[MockLight],
) -> None: ...

async def test_light_context(
    hass: HomeAssistant,
    hass_admin_user: MockUser,
    mock_light_entities: Sequence[MockLight],
) -> None: ...

async def test_light_turn_on_auth(
    hass: HomeAssistant,
    hass_read_only_user: MockUser,
    mock_light_entities: Sequence[MockLight],
) -> None: ...

async def test_light_brightness_step(hass: HomeAssistant) -> None: ...

@pytest.mark.usefixtures("enable_custom_integrations")
async def test_light_brightness_pct_conversion(
    hass: HomeAssistant,
    mock_light_entities: Sequence[MockLight],
) -> None: ...

async def test_profiles(hass: HomeAssistant) -> None: ...

@patch("os.path.isfile", MagicMock(side_effect=(True, False)))
async def test_profile_load_optional_hs_color(hass: HomeAssistant) -> None: ...

@pytest.mark.parametrize("light_state", [STATE_ON, STATE_OFF])
async def test_light_backwards_compatibility_supported_color_modes(
    hass: HomeAssistant,
    light_state: str,
) -> None: ...

async def test_light_backwards_compatibility_color_mode(hass: HomeAssistant) -> None: ...

async def test_light_service_call_rgbw(hass: HomeAssistant) -> None: ...

async def test_light_state_off(hass: HomeAssistant) -> None: ...

async def test_light_state_rgbw(hass: HomeAssistant) -> None: ...

async def test_light_state_rgbww(hass: HomeAssistant) -> None: ...

async def test_light_service_call_color_conversion(hass: HomeAssistant) -> None: ...

async def test_light_service_call_color_conversion_named_tuple(hass: HomeAssistant) -> None: ...

async def test_light_service_call_color_temp_emulation(hass: HomeAssistant) -> None: ...

async def test_light_service_call_color_temp_conversion(hass: HomeAssistant) -> None: ...

async def test_light_mired_color_temp_conversion(hass: HomeAssistant) -> None: ...

async def test_light_service_call_white_mode(hass: HomeAssistant) -> None: ...

async def test_light_state_color_conversion(hass: HomeAssistant) -> None: ...

async def test_services_filter_parameters(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    mock_light_entities: Sequence[MockLight],
) -> None: ...

def test_valid_supported_color_modes() -> None: ...

def test_filter_supported_color_modes() -> None: ...

def test_deprecated_supported_features_ints(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None: ...

@pytest.mark.parametrize(
    ("color_mode", "supported_color_modes", "warning_expected"),
    [
        (None, {light.ColorMode.ONOFF}, True),
        (light.ColorMode.ONOFF, {light.ColorMode.ONOFF}, False),
    ],
)
async def test_report_no_color_mode(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: Optional[light.ColorMode],
    supported_color_modes: Optional[set[light.ColorMode]],
    warning_expected: bool,
) -> None: ...

@pytest.mark.parametrize(
    ("color_mode", "supported_color_modes", "warning_expected"),
    [
        (light.ColorMode.ONOFF, None, True),
        (light.ColorMode.ONOFF, {light.ColorMode.ONOFF}, False),
    ],
)
async def test_report_no_color_modes(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: light.ColorMode,
    supported_color_modes: Optional[set[light.ColorMode]],
    warning_expected: bool,
) -> None: ...

@pytest.mark.parametrize(
    ("color_mode", "supported_color_modes", "effect", "warning_expected"),
    [
        (light.ColorMode.ONOFF, {light.ColorMode.ONOFF}, None, False),
        (light.ColorMode.ONOFF, {light.ColorMode.BRIGHTNESS}, None, True),
        (light.ColorMode.ONOFF, {light.ColorMode.BRIGHTNESS}, light.EFFECT_OFF, True),
        (light.ColorMode.ONOFF, {light.ColorMode.BRIGHTNESS}, "effect", False),
        (light.ColorMode.BRIGHTNESS, {light.ColorMode.BRIGHTNESS}, "effect", False),
        (light.ColorMode.BRIGHTNESS, {light.ColorMode.BRIGHTNESS}, None, False),
        (light.ColorMode.BRIGHTNESS, {light.ColorMode.HS}, None, True),
        (light.ColorMode.BRIGHTNESS, {light.ColorMode.HS}, light.EFFECT_OFF, True),
        (light.ColorMode.ONOFF, {light.ColorMode.HS}, None, True),
        (light.ColorMode.ONOFF, {light.ColorMode.HS}, light.EFFECT_OFF, True),
        (light.ColorMode.BRIGHTNESS, {light.ColorMode.HS}, "effect", False),
        (light.ColorMode.ONOFF, {light.ColorMode.HS}, "effect", False),
        (light.ColorMode.HS, {light.ColorMode.HS}, "effect", False),
        (light.ColorMode.HS, {light.ColorMode.BRIGHTNESS}, "effect", True),
    ],
)
async def test_report_invalid_color_mode(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: light.ColorMode,
    supported_color_modes: set[light.ColorMode],
    effect: Optional[str],
    warning_expected: bool,
) -> None: ...

@pytest.mark.parametrize(
    ("color_mode", "supported_color_modes", "platform_name", "warning_expected"),
    [
        (light.ColorMode.ONOFF, {light.ColorMode.ONOFF}, "test", False),
        (light.ColorMode.ONOFF, {light.ColorMode.ONOFF, light.ColorMode.BRIGHTNESS}, "test", True),
        (light.ColorMode.HS, {light.ColorMode.HS, light.ColorMode.BRIGHTNESS}, "test", True),
        (light.ColorMode.HS, {light.ColorMode.COLOR_TEMP, light.ColorMode.HS}, "test", False),
        (light.ColorMode.ONOFF, {light.ColorMode.ONOFF, light.ColorMode.BRIGHTNESS}, "philips_js", False),
    ],
)
def test_report_invalid_color_modes(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: light.ColorMode,
    supported_color_modes: set[light.ColorMode],
    platform_name: str,
    warning_expected: bool,
) -> None: ...

@pytest.mark.parametrize(
    ("attributes", "expected_warnings", "expected_values"),
    [
        (
            {
                "_attr_color_temp_kelvin": 4000,
                "_attr_min_color_temp_kelvin": 3000,
                "_attr_max_color_temp_kelvin": 5000,
            },
            {"current": False, "warmest": False, "coldest": False},
            (3000, 4000, 5000, 200, 250, 333, 153, None, 500),
        ),
        (
            {
                "_attr_color_temp": 350,
                "_attr_min_mireds": 300,
                "_attr_max_mireds": 400,
            },
            {"current": True, "warmest": True, "coldest": True},
            (2500, 2857, 3333, 300, 350, 400, 300, 350, 400),
        ),
        (
            {},
            {"current": False, "warmest": True, "coldest": True},
            (2000, None, 6535, 153, None, 500, 153, None, 500),
        ),
    ],
    ids=["with_kelvin", "with_mired_values", "with_mired_defaults"],
)
@patch.object(frame, "_REPORTED_INTEGRATIONS", set())
def test_missing_kelvin_property_warnings(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    attributes: dict[str, Any],
    expected_warnings: dict[str, bool],
    expected_values: tuple[
        Optional[int], Optional[int], Optional[int],
        Optional[int], Optional[int], Optional[int],
        int, Optional[int], int
    ],
) -> None: ...

@pytest.mark.parametrize("module", [light])
def test_all(module: ModuleType) -> None: ...

@pytest.mark.parametrize(
    ("constant_name", "constant_value", "constant