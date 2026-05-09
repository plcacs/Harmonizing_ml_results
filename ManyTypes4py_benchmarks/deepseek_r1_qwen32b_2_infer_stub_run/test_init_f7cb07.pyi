"""The tests for the Light component."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, Mock

from homeassistant import core
from homeassistant.components import light
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers import frame
from homeassistant.util.color import RGBColor

__all__ = [
    "test_methods",
    "test_services",
    "test_light_profiles",
    "test_default_profiles_light",
    "test_light_context",
    "test_light_turn_on_auth",
    "test_light_brightness_step",
    "test_light_brightness_pct_conversion",
    "test_profiles",
    "test_profile_load_optional_hs_color",
    "test_light_backwards_compatibility_supported_color_modes",
    "test_light_backwards_compatibility_color_mode",
    "test_light_service_call_rgbw",
    "test_light_state_off",
    "test_light_state_rgbw",
    "test_light_state_rgbww",
    "test_light_service_call_color_conversion",
    "test_light_service_call_color_conversion_named_tuple",
    "test_light_service_call_color_temp_emulation",
    "test_light_service_call_color_temp_conversion",
    "test_light_mired_color_temp_conversion",
    "test_light_service_call_white_mode",
    "test_light_state_color_conversion",
    "test_services_filter_parameters",
    "test_report_no_color_mode",
    "test_report_no_color_modes",
    "test_report_invalid_color_mode",
    "test_report_invalid_color_modes",
    "test_missing_kelvin_property_warnings",
    "test_deprecated_turn_on_arguments",
]

orig_Profiles: Type[light.Profiles] = light.Profiles

async def test_methods(hass: HomeAssistant) -> Coroutine[Any, Any, None]:
    ...

async def test_services(
    hass: HomeAssistant,
    mock_light_profiles: Mock,
    mock_light_entities: Iterable[Mock],
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_profiles(
    hass: HomeAssistant,
    mock_light_profiles: Mock,
    profile_name: str,
    expected_data: Dict[str, Any],
    last_call: str,
    mock_light_entities: Iterable[Mock],
) -> Coroutine[Any, Any, None]:
    ...

async def test_default_profiles_light(
    hass: HomeAssistant,
    mock_light_profiles: Mock,
    extra_call_params: Dict[str, Any],
    expected_params_state_was_off: Dict[str, Any],
    expected_params_state_was_on: Dict[str, Any],
    mock_light_entities: Iterable[Mock],
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_context(
    hass: HomeAssistant,
    hass_admin_user: core.User,
    mock_light_entities: Iterable[Mock],
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_turn_on_auth(
    hass: HomeAssistant,
    hass_read_only_user: core.User,
    mock_light_entities: Iterable[Mock],
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_brightness_step(hass: HomeAssistant) -> Coroutine[Any, Any, None]:
    ...

async def test_light_brightness_pct_conversion(
    hass: HomeAssistant,
    mock_light_entities: Iterable[Mock],
) -> Coroutine[Any, Any, None]:
    ...

async def test_profiles(hass: HomeAssistant) -> Coroutine[Any, Any, None]:
    ...

async def test_profile_load_optional_hs_color(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_backwards_compatibility_supported_color_modes(
    hass: HomeAssistant,
    light_state: str = "on",
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_backwards_compatibility_color_mode(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_service_call_rgbw(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_state_off(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_state_rgbw(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_state_rgbww(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_service_call_color_conversion(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_service_call_color_conversion_named_tuple(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_service_call_color_temp_emulation(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_service_call_color_temp_conversion(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_mired_color_temp_conversion(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_service_call_white_mode(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_light_state_color_conversion(
    hass: HomeAssistant,
) -> Coroutine[Any, Any, None]:
    ...

async def test_services_filter_parameters(
    hass: HomeAssistant,
    mock_light_profiles: Mock,
    mock_light_entities: Iterable[Mock],
) -> Coroutine[Any, Any, None]:
    ...

async def test_report_no_color_mode(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> Coroutine[Any, Any, None]:
    ...

async def test_report_no_color_modes(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> Coroutine[Any, Any, None]:
    ...

async def test_report_invalid_color_mode(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> Coroutine[Any, Any, None]:
    ...

async def test_report_invalid_color_modes(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> Coroutine[Any, Any, None]:
    ...

async def test_missing_kelvin_property_warnings(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> Coroutine[Any, Any, None]:
    ...

async def test_deprecated_turn_on_arguments(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> Coroutine[Any, Any, None]:
    ...