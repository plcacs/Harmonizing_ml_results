from typing import Literal, Optional, Set
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
from homeassistant.core import HomeAssistant
from homeassistant.helpers import frame
from homeassistant.util import color as color_util

async def test_methods(hass: HomeAssistant) -> None:
    # ...

async def test_services(hass: HomeAssistant, mock_light_profiles: dict, mock_light_entities: list) -> None:
    # ...

async def test_light_profiles(hass: HomeAssistant, mock_light_profiles: dict, profile_name: str, expected_data: dict, last_call: str, mock_light_entities: list) -> None:
    # ...

async def test_default_profiles_group(hass: HomeAssistant, mock_light_profiles: dict, mock_light_entities: list) -> None:
    # ...

async def test_default_profiles_light(hass: HomeAssistant, mock_light_profiles: dict, extra_call_params: dict, expected_params_state_was_off: dict, expected_params_state_was_on: dict, mock_light_entities: list) -> None:
    # ...

async def test_light_context(hass: HomeAssistant, hass_admin_user: "User", mock_light_entities: list) -> None:
    # ...

async def test_light_turn_on_auth(hass: HomeAssistant, hass_read_only_user: "User", mock_light_entities: list) -> None:
    # ...

async def test_light_brightness_step(hass: HomeAssistant) -> None:
    # ...

async def test_light_brightness_pct_conversion(hass: HomeAssistant, mock_light_entities: list) -> None:
    # ...

async def test_profiles(hass: HomeAssistant) -> None:
    # ...

async def test_profile_load_optional_hs_color(hass: HomeAssistant) -> None:
    # ...

async def test_light_backwards_compatibility_supported_color_modes(hass: HomeAssistant, light_state: str) -> None:
    # ...

async def test_light_backwards_compatibility_color_mode(hass: HomeAssistant) -> None:
    # ...

async def test_light_service_call_rgbw(hass: HomeAssistant) -> None:
    # ...

async def test_light_state_off(hass: HomeAssistant) -> None:
    # ...

async def test_light_state_rgbw(hass: HomeAssistant) -> None:
    # ...

async def test_light_state_rgbww(hass: HomeAssistant) -> None:
    # ...

async def test_light_service_call_color_conversion(hass: HomeAssistant) -> None:
    # ...

async def test_light_service_call_color_conversion_named_tuple(hass: HomeAssistant) -> None:
    # ...

async def test_light_service_call_color_temp_emulation(hass: HomeAssistant) -> None:
    # ...

async def test_light_service_call_color_temp_conversion(hass: HomeAssistant) -> None:
    # ...

async def test_light_mired_color_temp_conversion(hass: HomeAssistant) -> None:
    # ...

async def test_light_service_call_white_mode(hass: HomeAssistant) -> None:
    # ...

async def test_light_state_color_conversion(hass: HomeAssistant) -> None:
    # ...

async def test_services_filter_parameters(hass: HomeAssistant, mock_light_profiles: dict, mock_light_entities: list) -> None:
    # ...

def test_valid_supported_color_modes() -> None:
    # ...

def test_filter_supported_color_modes() -> None:
    # ...

def test_deprecated_supported_features_ints(hass: HomeAssistant, caplog: "pytest.LogCaptureFixture") -> None:
    # ...

@pytest.mark.parametrize(('color_mode', 'supported_color_modes', 'warning_expected'), [  # ...
@pytest.mark.parametrize(('color_mode', 'supported_color_modes', 'effect', 'warning_expected'), [  # ...

@pytest.mark.parametrize('module', [light])
def test_all(module: "ModuleType") -> None:
    # ...

@pytest.mark.parametrize(('constant_name', 'constant_value', 'constant_replacement'), [  # ...
@pytest.mark.parametrize('entity_feature', list(light.LightEntityFeature))
def test_deprecated_support_light_constants_enums(caplog: "pytest.LogCaptureFixture", entity_feature: "LightEntityFeature") -> None:
    # ...

@pytest.mark.parametrize('entity_feature', list(light.ColorMode))
def test_deprecated_color_mode_constants_enums(caplog: "pytest.LogCaptureFixture", entity_feature: "ColorMode") -> None:
    # ...

async def test_deprecated_turn_on_arguments(hass: HomeAssistant, caplog: "pytest.LogCaptureFixture") -> None:
    # ...
