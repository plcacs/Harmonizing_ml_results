from types import ModuleType
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest

from homeassistant.components import light
from homeassistant.core import HomeAssistant

from .common import MockLight

orig_Profiles: type

async def test_methods(hass: HomeAssistant) -> None: ...

async def test_services(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    mock_light_entities: list[MockLight],
) -> None: ...

async def test_light_profiles(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    profile_name: str,
    expected_data: dict[str, Any],
    last_call: str,
    mock_light_entities: list[MockLight],
) -> None: ...

async def test_default_profiles_group(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    mock_light_entities: list[MockLight],
) -> None: ...

async def test_default_profiles_light(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    extra_call_params: dict[str, Any],
    expected_params_state_was_off: dict[str, Any],
    expected_params_state_was_on: dict[str, Any],
    mock_light_entities: list[MockLight],
) -> None: ...

async def test_light_context(
    hass: HomeAssistant,
    hass_admin_user: Any,
    mock_light_entities: list[MockLight],
) -> None: ...

async def test_light_turn_on_auth(
    hass: HomeAssistant,
    hass_read_only_user: Any,
    mock_light_entities: list[MockLight],
) -> None: ...

async def test_light_brightness_step(hass: HomeAssistant) -> None: ...

async def test_light_brightness_pct_conversion(
    hass: HomeAssistant,
    mock_light_entities: list[MockLight],
) -> None: ...

async def test_profiles(hass: HomeAssistant) -> None: ...

async def test_profile_load_optional_hs_color(hass: HomeAssistant) -> None: ...

async def test_light_backwards_compatibility_supported_color_modes(
    hass: HomeAssistant,
    light_state: str,
) -> None: ...

async def test_light_backwards_compatibility_color_mode(
    hass: HomeAssistant,
) -> None: ...

async def test_light_service_call_rgbw(hass: HomeAssistant) -> None: ...

async def test_light_state_off(hass: HomeAssistant) -> None: ...

async def test_light_state_rgbw(hass: HomeAssistant) -> None: ...

async def test_light_state_rgbww(hass: HomeAssistant) -> None: ...

async def test_light_service_call_color_conversion(hass: HomeAssistant) -> None: ...

async def test_light_service_call_color_conversion_named_tuple(
    hass: HomeAssistant,
) -> None: ...

async def test_light_service_call_color_temp_emulation(
    hass: HomeAssistant,
) -> None: ...

async def test_light_service_call_color_temp_conversion(
    hass: HomeAssistant,
) -> None: ...

async def test_light_mired_color_temp_conversion(hass: HomeAssistant) -> None: ...

async def test_light_service_call_white_mode(hass: HomeAssistant) -> None: ...

async def test_light_state_color_conversion(hass: HomeAssistant) -> None: ...

async def test_services_filter_parameters(
    hass: HomeAssistant,
    mock_light_profiles: dict[str, light.Profile],
    mock_light_entities: list[MockLight],
) -> None: ...

def test_valid_supported_color_modes() -> None: ...

def test_filter_supported_color_modes() -> None: ...

def test_deprecated_supported_features_ints(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None: ...

async def test_report_no_color_mode(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: light.ColorMode | None,
    supported_color_modes: set[light.ColorMode],
    warning_expected: bool,
) -> None: ...

async def test_report_no_color_modes(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: light.ColorMode,
    supported_color_modes: set[light.ColorMode] | None,
    warning_expected: bool,
) -> None: ...

async def test_report_invalid_color_mode(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: light.ColorMode,
    supported_color_modes: set[light.ColorMode],
    effect: str | None,
    warning_expected: bool,
) -> None: ...

def test_report_invalid_color_modes(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: light.ColorMode,
    supported_color_modes: set[light.ColorMode],
    platform_name: str,
    warning_expected: bool,
) -> None: ...

def test_missing_kelvin_property_warnings(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    attributes: dict[str, Any],
    expected_warnings: dict[str, bool],
    expected_values: tuple[int | None, ...],
) -> None: ...

def test_all(module: ModuleType) -> None: ...

def test_deprecated_light_constants(
    caplog: pytest.LogCaptureFixture,
    constant_name: str,
    constant_value: int | str,
    constant_replacement: str,
) -> None: ...

def test_deprecated_support_light_constants_enums(
    caplog: pytest.LogCaptureFixture,
    entity_feature: light.LightEntityFeature,
) -> None: ...

def test_deprecated_color_mode_constants_enums(
    caplog: pytest.LogCaptureFixture,
    entity_feature: light.ColorMode,
) -> None: ...

async def test_deprecated_turn_on_arguments(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None: ...