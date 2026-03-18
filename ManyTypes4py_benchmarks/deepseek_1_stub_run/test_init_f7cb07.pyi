```python
from typing import Any
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

from homeassistant.components import light
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.const import CONF_PLATFORM
from homeassistant.const import ENTITY_MATCH_ALL
from homeassistant.const import SERVICE_TOGGLE
from homeassistant.const import SERVICE_TURN_OFF
from homeassistant.const import SERVICE_TURN_ON
from homeassistant.const import STATE_OFF
from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.exceptions import Unauthorized
from homeassistant.helpers import frame
from homeassistant.setup import async_setup_component
from homeassistant.util import color as color_util
from tests.common import MockEntityPlatform
from tests.common import MockUser
from tests.common import async_mock_service
from tests.common import help_test_all
from tests.common import import_and_test_deprecated_constant
from tests.common import import_and_test_deprecated_constant_enum
from tests.common import setup_test_component_platform
from .common import MockLight

orig_Profiles: Any = ...

async def test_methods(hass: HomeAssistant) -> None: ...

async def test_services(
    hass: HomeAssistant,
    mock_light_profiles: Any,
    mock_light_entities: Any
) -> None: ...

async def test_light_profiles(
    hass: HomeAssistant,
    mock_light_profiles: Any,
    profile_name: str,
    expected_data: Any,
    last_call: str,
    mock_light_entities: Any
) -> None: ...

async def test_default_profiles_group(
    hass: HomeAssistant,
    mock_light_profiles: Any,
    mock_light_entities: Any
) -> None: ...

async def test_default_profiles_light(
    hass: HomeAssistant,
    mock_light_profiles: Any,
    extra_call_params: Any,
    expected_params_state_was_off: Any,
    expected_params_state_was_on: Any,
    mock_light_entities: Any
) -> None: ...

async def test_light_context(
    hass: HomeAssistant,
    hass_admin_user: MockUser,
    mock_light_entities: Any
) -> None: ...

async def test_light_turn_on_auth(
    hass: HomeAssistant,
    hass_read_only_user: MockUser,
    mock_light_entities: Any
) -> None: ...

async def test_light_brightness_step(hass: HomeAssistant) -> None: ...

async def test_light_brightness_pct_conversion(
    hass: HomeAssistant,
    mock_light_entities: Any
) -> None: ...

async def test_profiles(hass: HomeAssistant) -> None: ...

async def test_profile_load_optional_hs_color(hass: HomeAssistant) -> None: ...

async def test_light_backwards_compatibility_supported_color_modes(
    hass: HomeAssistant,
    light_state: Union[Literal["on"], Literal["off"]]
) -> None: ...

async def test_light_backwards_compatibility_color_mode(
    hass: HomeAssistant
) -> None: ...

async def test_light_service_call_rgbw(hass: HomeAssistant) -> None: ...

async def test_light_state_off(hass: HomeAssistant) -> None: ...

async def test_light_state_rgbw(hass: HomeAssistant) -> None: ...

async def test_light_state_rgbww(hass: HomeAssistant) -> None: ...

async def test_light_service_call_color_conversion(
    hass: HomeAssistant
) -> None: ...

async def test_light_service_call_color_conversion_named_tuple(
    hass: HomeAssistant
) -> None: ...

async def test_light_service_call_color_temp_emulation(
    hass: HomeAssistant
) -> None: ...

async def test_light_service_call_color_temp_conversion(
    hass: HomeAssistant
) -> None: ...

async def test_light_mired_color_temp_conversion(
    hass: HomeAssistant
) -> None: ...

async def test_light_service_call_white_mode(
    hass: HomeAssistant
) -> None: ...

async def test_light_state_color_conversion(
    hass: HomeAssistant
) -> None: ...

async def test_services_filter_parameters(
    hass: HomeAssistant,
    mock_light_profiles: Any,
    mock_light_entities: Any
) -> None: ...

def test_valid_supported_color_modes() -> None: ...

def test_filter_supported_color_modes() -> None: ...

def test_deprecated_supported_features_ints(
    hass: HomeAssistant,
    caplog: Any
) -> None: ...

async def test_report_no_color_mode(
    hass: HomeAssistant,
    caplog: Any,
    color_mode: Optional[light.ColorMode],
    supported_color_modes: Optional[set[light.ColorMode]],
    warning_expected: bool
) -> None: ...

async def test_report_no_color_modes(
    hass: HomeAssistant,
    caplog: Any,
    color_mode: Optional[light.ColorMode],
    supported_color_modes: Optional[set[light.ColorMode]],
    warning_expected: bool
) -> None: ...

async def test_report_invalid_color_mode(
    hass: HomeAssistant,
    caplog: Any,
    color_mode: light.ColorMode,
    supported_color_modes: set[light.ColorMode],
    effect: Optional[str],
    warning_expected: bool
) -> None: ...

def test_report_invalid_color_modes(
    hass: HomeAssistant,
    caplog: Any,
    color_mode: light.ColorMode,
    supported_color_modes: set[light.ColorMode],
    platform_name: str,
    warning_expected: bool
) -> None: ...

def test_missing_kelvin_property_warnings(
    hass: HomeAssistant,
    caplog: Any,
    attributes: dict[str, Any],
    expected_warnings: dict[str, bool],
    expected_values: Tuple[Any, ...]
) -> None: ...

def test_all(module: Any) -> None: ...

def test_deprecated_light_constants(
    caplog: Any,
    constant_name: str,
    constant_value: Any,
    constant_replacement: str
) -> None: ...

def test_deprecated_support_light_constants_enums(
    caplog: Any,
    entity_feature: light.LightEntityFeature
) -> None: ...

def test_deprecated_color_mode_constants_enums(
    caplog: Any,
    entity_feature: light.ColorMode
) -> None: ...

async def test_deprecated_turn_on_arguments(
    hass: HomeAssistant,
    caplog: Any
) -> None: ...
```