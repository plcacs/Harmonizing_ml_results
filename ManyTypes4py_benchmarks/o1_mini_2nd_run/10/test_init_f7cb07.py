"""The tests for the Light component."""
from types import ModuleType
from typing import Literal, Any, Dict, Tuple, Optional, List, Set
from unittest.mock import MagicMock, mock_open, patch
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
from homeassistant.core import HomeAssistant, Context
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
orig_Profiles = light.Profiles


async def test_methods(hass: HomeAssistant) -> None:
    """Test if methods call the services as expected."""
    hass.states.async_set("light.test", STATE_ON)
    assert light.is_on(hass, "light.test")
    hass.states.async_set("light.test", STATE_OFF)
    assert not light.is_on(hass, "light.test")
    turn_on_calls = async_mock_service(
        hass, light.DOMAIN, SERVICE_TURN_ON
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: "entity_id_val",
            light.ATTR_TRANSITION: "transition_val",
            light.ATTR_BRIGHTNESS: "brightness_val",
            light.ATTR_RGB_COLOR: "rgb_color_val",
            light.ATTR_XY_COLOR: "xy_color_val",
            light.ATTR_PROFILE: "profile_val",
            light.ATTR_COLOR_NAME: "color_name_val",
        },
        blocking=True,
    )
    assert len(turn_on_calls) == 1
    call = turn_on_calls[-1]
    assert call.domain == light.DOMAIN
    assert call.service == SERVICE_TURN_ON
    assert call.data.get(ATTR_ENTITY_ID) == "entity_id_val"
    assert call.data.get(light.ATTR_TRANSITION) == "transition_val"
    assert call.data.get(light.ATTR_BRIGHTNESS) == "brightness_val"
    assert call.data.get(light.ATTR_RGB_COLOR) == "rgb_color_val"
    assert call.data.get(light.ATTR_XY_COLOR) == "xy_color_val"
    assert call.data.get(light.ATTR_PROFILE) == "profile_val"
    assert call.data.get(light.ATTR_COLOR_NAME) == "color_name_val"
    turn_off_calls = async_mock_service(
        hass, light.DOMAIN, SERVICE_TURN_OFF
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_OFF,
        {
            ATTR_ENTITY_ID: "entity_id_val",
            light.ATTR_TRANSITION: "transition_val",
        },
        blocking=True,
    )
    assert len(turn_off_calls) == 1
    call = turn_off_calls[-1]
    assert call.domain == light.DOMAIN
    assert call.service == SERVICE_TURN_OFF
    assert call.data[ATTR_ENTITY_ID] == "entity_id_val"
    assert call.data[light.ATTR_TRANSITION] == "transition_val"
    toggle_calls = async_mock_service(
        hass, light.DOMAIN, SERVICE_TOGGLE
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {
            ATTR_ENTITY_ID: "entity_id_val",
            light.ATTR_TRANSITION: "transition_val",
        },
        blocking=True,
    )
    assert len(toggle_calls) == 1
    call = toggle_calls[-1]
    assert call.domain == light.DOMAIN
    assert call.service == SERVICE_TOGGLE
    assert call.data[ATTR_ENTITY_ID] == "entity_id_val"
    assert call.data[light.ATTR_TRANSITION] == "transition_val"


async def test_services(
    hass: HomeAssistant,
    mock_light_profiles: Dict[str, light.Profile],
    mock_light_entities: List[MockLight],
) -> None:
    """Test the provided services."""
    setup_test_component_platform(
        hass, light.DOMAIN, mock_light_entities
    )
    assert await async_setup_component(
        hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}}
    )
    await hass.async_block_till_done()
    ent1, ent2, ent3 = mock_light_entities
    ent1.supported_color_modes = [light.ColorMode.HS]
    ent3.supported_color_modes = [light.ColorMode.HS]
    ent1.supported_features = light.LightEntityFeature.TRANSITION
    ent2.supported_features = (
        light.SUPPORT_COLOR
        | light.LightEntityFeature.EFFECT
        | light.LightEntityFeature.TRANSITION
    )
    ent2.supported_color_modes = None
    ent2.color_mode = None
    ent3.supported_features = (
        light.LightEntityFeature.FLASH
        | light.LightEntityFeature.TRANSITION
    )
    assert light.is_on(hass, ent1.entity_id)
    assert not light.is_on(hass, ent2.entity_id)
    assert not light.is_on(hass, ent3.entity_id)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_OFF,
        {ATTR_ENTITY_ID: ent1.entity_id},
        blocking=True,
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ent2.entity_id},
        blocking=True,
    )
    assert not light.is_on(hass, ent1.entity_id)
    assert light.is_on(hass, ent2.entity_id)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True,
    )
    assert light.is_on(hass, ent1.entity_id)
    assert light.is_on(hass, ent2.entity_id)
    assert light.is_on(hass, ent3.entity_id)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_OFF,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True,
    )
    assert not light.is_on(hass, ent1.entity_id)
    assert not light.is_on(hass, ent2.entity_id)
    assert not light.is_on(hass, ent3.entity_id)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True,
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ENTITY_MATCH_ALL,
            light.ATTR_BRIGHTNESS: 0,
        },
        blocking=True,
    )
    assert not light.is_on(hass, ent1.entity_id)
    assert not light.is_on(hass, ent2.entity_id)
    assert not light.is_on(hass, ent3.entity_id)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True,
    )
    assert light.is_on(hass, ent1.entity_id)
    assert light.is_on(hass, ent2.entity_id)
    assert light.is_on(hass, ent3.entity_id)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True,
    )
    assert not light.is_on(hass, ent1.entity_id)
    assert not light.is_on(hass, ent2.entity_id)
    assert not light.is_on(hass, ent3.entity_id)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_TRANSITION: 10,
            light.ATTR_BRIGHTNESS: 20,
            light.ATTR_COLOR_NAME: "blue",
        },
        blocking=True,
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent2.entity_id,
            light.ATTR_EFFECT: "fun_effect",
            light.ATTR_RGB_COLOR: (255, 255, 255),
        },
        blocking=True,
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent3.entity_id,
            light.ATTR_FLASH: "short",
            light.ATTR_XY_COLOR: (0.4, 0.6),
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {
        light.ATTR_TRANSITION: 10,
        light.ATTR_BRIGHTNESS: 20,
        light.ATTR_HS_COLOR: (240, 100),
    }
    _, data = ent2.last_call("turn_on")
    assert data == {
        light.ATTR_EFFECT: "fun_effect",
        light.ATTR_HS_COLOR: (0, 0),
    }
    _, data = ent3.last_call("turn_on")
    assert data == {
        light.ATTR_FLASH: "short",
        light.ATTR_HS_COLOR: (71.059, 100),
    }
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_TRANSITION: 10,
            light.ATTR_BRIGHTNESS: 0,
            light.ATTR_COLOR_NAME: "blue",
        },
        blocking=True,
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent2.entity_id,
            light.ATTR_BRIGHTNESS: 0,
            light.ATTR_RGB_COLOR: (255, 255, 255),
        },
        blocking=True,
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent3.entity_id,
            light.ATTR_BRIGHTNESS: 0,
            light.ATTR_XY_COLOR: (0.4, 0.6),
        },
        blocking=True,
    )
    assert not light.is_on(hass, ent1.entity_id)
    assert not light.is_on(hass, ent2.entity_id)
    assert not light.is_on(hass, ent3.entity_id)
    _, data = ent1.last_call("turn_off")
    assert data == {light.ATTR_TRANSITION: 10}
    _, data = ent2.last_call("turn_off")
    assert data == {}
    _, data = ent3.last_call("turn_off")
    assert data == {}
    profile = light.Profile("relax", 0.513, 0.413, 144, 0)
    mock_light_profiles[profile.name] = profile
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_PROFILE: profile.name,
        },
        blocking=True,
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent2.entity_id,
            light.ATTR_PROFILE: profile.name,
            light.ATTR_BRIGHTNESS: 100,
            light.ATTR_TRANSITION: 1,
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {
        light.ATTR_BRIGHTNESS: profile.brightness,
        light.ATTR_HS_COLOR: profile.hs_color,
        light.ATTR_TRANSITION: profile.transition,
    }
    _, data = ent2.last_call("turn_on")
    assert data == {
        light.ATTR_BRIGHTNESS: 100,
        light.ATTR_HS_COLOR: profile.hs_color,
        light.ATTR_TRANSITION: 1,
    }
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {
            ATTR_ENTITY_ID: ent3.entity_id,
            light.ATTR_PROFILE: profile.name,
            light.ATTR_BRIGHTNESS_PCT: 100,
        },
        blocking=True,
    )
    _, data = ent3.last_call("turn_on")
    assert data == {
        light.ATTR_BRIGHTNESS: 255,
        light.ATTR_HS_COLOR: profile.hs_color,
        light.ATTR_TRANSITION: profile.transition,
    }
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {
            ATTR_ENTITY_ID: ent3.entity_id,
            light.ATTR_TRANSITION: 4,
        },
        blocking=True,
    )
    _, data = ent3.last_call("turn_off")
    assert data == {light.ATTR_TRANSITION: 4}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True,
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_PROFILE: -1,
        },
        blocking=True,
    )
    with pytest.raises(vol.MultipleInvalid):
        await hass.services.async_call(
            light.DOMAIN,
            SERVICE_TURN_ON,
            {
                ATTR_ENTITY_ID: ent2.entity_id,
                light.ATTR_XY_COLOR: ["bla-di-bla", 5],
            },
            blocking=True,
        )
    with pytest.raises(vol.MultipleInvalid):
        await hass.services.async_call(
            light.DOMAIN,
            SERVICE_TURN_ON,
            {
                ATTR_ENTITY_ID: ent3.entity_id,
                light.ATTR_RGB_COLOR: [255, None, 2],
            },
            blocking=True,
        )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    _, data = ent2.last_call("turn_on")
    assert data == {}
    _, data = ent3.last_call("turn_on")
    assert data == {}
    with pytest.raises(vol.MultipleInvalid):
        await hass.services.async_call(
            light.DOMAIN,
            SERVICE_TURN_ON,
            {
                ATTR_ENTITY_ID: ent1.entity_id,
                light.ATTR_PROFILE: profile.name,
                light.ATTR_BRIGHTNESS: "bright",
            },
            blocking=True,
        )
    with pytest.raises(vol.MultipleInvalid):
        await hass.services.async_call(
            light.DOMAIN,
            SERVICE_TURN_ON,
            {
                ATTR_ENTITY_ID: ent1.entity_id,
                light.ATTR_RGB_COLOR: "yellowish",
            },
            blocking=True,
        )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    _, data = ent2.last_call("turn_on")
    assert data == {}


@pytest.mark.parametrize(
    (
        "profile_name",
        "last_call",
        "expected_data",
    ),
    [
        (
            "test",
            "turn_on",
            {
                light.ATTR_HS_COLOR: (71.059, 100),
                light.ATTR_BRIGHTNESS: 100,
                light.ATTR_TRANSITION: 0,
            },
        ),
        (
            "color_no_brightness_no_transition",
            "turn_on",
            {light.ATTR_HS_COLOR: (71.059, 100)},
        ),
        (
            "no color",
            "turn_on",
            {light.ATTR_BRIGHTNESS: 110, light.ATTR_TRANSITION: 0},
        ),
        (
            "test_off",
            "turn_off",
            {light.ATTR_TRANSITION: 0},
        ),
        (
            "no brightness",
            "turn_on",
            {light.ATTR_HS_COLOR: (71.059, 100)},
        ),
        (
            "color_and_brightness",
            "turn_on",
            {
                light.ATTR_HS_COLOR: (71.059, 100),
                light.ATTR_BRIGHTNESS: 120,
            },
        ),
        (
            "color_and_transition",
            "turn_on",
            {
                light.ATTR_HS_COLOR: (71.059, 100),
                light.ATTR_TRANSITION: 4.2,
            },
        ),
        (
            "brightness_and_transition",
            "turn_on",
            {
                light.ATTR_BRIGHTNESS: 130,
                light.ATTR_TRANSITION: 5.3,
            },
        ),
    ],
)
async def test_light_profiles(
    hass: HomeAssistant,
    mock_light_profiles: Dict[str, light.Profile],
    profile_name: str,
    expected_data: Dict[str, Any],
    last_call: str,
    mock_light_entities: List[MockLight],
) -> None:
    """Test light profiles."""
    setup_test_component_platform(
        hass, light.DOMAIN, mock_light_entities
    )
    profile_mock_data: Dict[str, Tuple[Any, ...]] = {
        "test": (0.4, 0.6, 100, 0),
        "color_no_brightness_no_transition": (0.4, 0.6, None, None),
        "no color": (None, None, 110, 0),
        "test_off": (0, 0, 0, 0),
        "no brightness": (0.4, 0.6, None),
        "color_and_brightness": (0.4, 0.6, 120),
        "color_and_transition": (0.4, 0.6, None, 4.2),
        "brightness_and_transition": (None, None, 130, 5.3),
    }
    for name, data in profile_mock_data.items():
        mock_light_profiles[name] = light.Profile(*(name, *data))
    assert await async_setup_component(
        hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}}
    )
    await hass.async_block_till_done()
    ent1, _, _ = mock_light_entities
    ent1.supported_color_modes = [light.ColorMode.HS]
    ent1.supported_features = light.LightEntityFeature.TRANSITION
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_PROFILE: profile_name,
        },
        blocking=True,
    )
    _, data = ent1.last_call(last_call)
    if last_call == "turn_on":
        assert light.is_on(hass, ent1.entity_id)
    else:
        assert not light.is_on(hass, ent1.entity_id)
    assert data == expected_data


async def test_default_profiles_group(
    hass: HomeAssistant,
    mock_light_profiles: Dict[str, light.Profile],
    mock_light_entities: List[MockLight],
) -> None:
    """Test default turn-on light profile for all lights."""
    setup_test_component_platform(
        hass, light.DOMAIN, mock_light_entities
    )
    assert await async_setup_component(
        hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}}
    )
    await hass.async_block_till_done()
    profile = light.Profile("group.all_lights.default", 0.4, 0.6, 99, 2)
    mock_light_profiles[profile.name] = profile
    ent, _, _ = mock_light_entities
    ent.supported_color_modes = [light.ColorMode.HS]
    ent.supported_features = light.LightEntityFeature.TRANSITION
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ent.entity_id},
        blocking=True,
    )
    _, data = ent.last_call("turn_on")
    assert data == {
        light.ATTR_HS_COLOR: (71.059, 100),
        light.ATTR_BRIGHTNESS: 99,
        light.ATTR_TRANSITION: 2,
    }


@pytest.mark.parametrize(
    (
        "extra_call_params",
        "expected_params_state_was_off",
        "expected_params_state_was_on",
    ),
    [
        (
            {},
            {
                light.ATTR_HS_COLOR: (50.353, 100),
                light.ATTR_BRIGHTNESS: 100,
                light.ATTR_TRANSITION: 3,
            },
            {
                light.ATTR_HS_COLOR: (50.353, 100),
                light.ATTR_BRIGHTNESS: 100,
                light.ATTR_TRANSITION: 3,
            },
        ),
        (
            {light.ATTR_BRIGHTNESS: 22},
            {
                light.ATTR_HS_COLOR: (50.353, 100),
                light.ATTR_BRIGHTNESS: 22,
                light.ATTR_TRANSITION: 3,
            },
            {
                light.ATTR_BRIGHTNESS: 22,
                light.ATTR_TRANSITION: 3,
            },
        ),
        (
            {light.ATTR_TRANSITION: 22},
            {
                light.ATTR_HS_COLOR: (50.353, 100),
                light.ATTR_BRIGHTNESS: 100,
                light.ATTR_TRANSITION: 22,
            },
            {light.ATTR_TRANSITION: 22},
        ),
        (
            {light.ATTR_COLOR_TEMP: 600, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1},
            {
                light.ATTR_COLOR_TEMP: 600,
                light.ATTR_COLOR_TEMP_KELVIN: 1666,
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_COLOR_TEMP: 600,
                light.ATTR_COLOR_TEMP_KELVIN: 1666,
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
        ),
        (
            {
                light.ATTR_COLOR_TEMP_KELVIN: 6500,
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_COLOR_TEMP: 153,
                light.ATTR_COLOR_TEMP_KELVIN: 6500,
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_COLOR_TEMP: 153,
                light.ATTR_COLOR_TEMP_KELVIN: 6500,
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
        ),
        (
            {
                light.ATTR_HS_COLOR: [70, 80],
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_HS_COLOR: (70, 80),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_HS_COLOR: (70, 80),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
        ),
        (
            {
                light.ATTR_RGB_COLOR: [1, 2, 3],
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_RGB_COLOR: (1, 2, 3),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_RGB_COLOR: (1, 2, 3),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
        ),
        (
            {
                light.ATTR_RGBW_COLOR: [1, 2, 3, 4],
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_RGBW_COLOR: (1, 2, 3, 4),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_RGBW_COLOR: (1, 2, 3, 4),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
        ),
        (
            {
                light.ATTR_RGBWW_COLOR: [1, 2, 3, 4, 5],
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_RGBWW_COLOR: (1, 2, 3, 4, 5),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_RGBWW_COLOR: (1, 2, 3, 4, 5),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
        ),
        (
            {
                light.ATTR_XY_COLOR: [0.4448, 0.4066],
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_XY_COLOR: (0.4448, 0.4066),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_XY_COLOR: (0.4448, 0.4066),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
        ),
        (
            {
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_HS_COLOR: (50.353, 100),
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
            {
                light.ATTR_BRIGHTNESS: 11,
                light.ATTR_TRANSITION: 1,
            },
        ),
    ],
)
async def test_default_profiles_light(
    hass: HomeAssistant,
    mock_light_profiles: Dict[str, light.Profile],
    extra_call_params: Dict[str, Any],
    expected_params_state_was_off: Dict[str, Any],
    expected_params_state_was_on: Dict[str, Any],
    mock_light_entities: List[MockLight],
) -> None:
    """Test default turn-on light profile for a specific light."""
    setup_test_component_platform(
        hass, light.DOMAIN, mock_light_entities
    )
    assert await async_setup_component(
        hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}}
    )
    await hass.async_block_till_done()
    profile = light.Profile("group.all_lights.default", 0.3, 0.5, 200, 0)
    mock_light_profiles[profile.name] = profile
    profile = light.Profile("light.ceiling_2.default", 0.6, 0.6, 100, 3)
    mock_light_profiles[profile.name] = profile
    dev: MockLight = next(
        filter(
            lambda x: x.entity_id == "light.ceiling_2",
            mock_light_entities,
        )
    )
    dev.supported_color_modes = {
        light.ColorMode.COLOR_TEMP,
        light.ColorMode.HS,
        light.ColorMode.RGB,
        light.ColorMode.RGBW,
        light.ColorMode.RGBWW,
        light.ColorMode.XY,
    }
    dev.supported_features = light.LightEntityFeature.TRANSITION
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {**{ATTR_ENTITY_ID: dev.entity_id}, **extra_call_params},
        blocking=True,
    )
    _, data = dev.last_call("turn_on")
    assert data == expected_params_state_was_off
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {**{ATTR_ENTITY_ID: dev.entity_id}, **extra_call_params},
        blocking=True,
    )
    _, data = dev.last_call("turn_on")
    assert data == expected_params_state_was_on
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_OFF,
        {ATTR_ENTITY_ID: dev.entity_id},
        blocking=True,
    )
    _, data = dev.last_call("turn_off")
    assert data == {light.ATTR_TRANSITION: 3}


async def test_light_context(
    hass: HomeAssistant, hass_admin_user: MockUser, mock_light_entities: List[MockLight]
) -> None:
    """Test that light context works."""
    setup_test_component_platform(
        hass, light.DOMAIN, mock_light_entities
    )
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get("light.ceiling")
    assert state is not None
    await hass.services.async_call(
        "light",
        "toggle",
        {"entity_id": state.entity_id},
        blocking=True,
        context=core.Context(user_id=hass_admin_user.id),
    )
    state2 = hass.states.get("light.ceiling")
    assert state2 is not None
    assert state.state != state2.state
    assert state2.context.user_id == hass_admin_user.id


async def test_light_turn_on_auth(
    hass: HomeAssistant,
    hass_read_only_user: MockUser,
    mock_light_entities: List[MockLight],
) -> None:
    """Test that light context works."""
    setup_test_component_platform(
        hass, light.DOMAIN, mock_light_entities
    )
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get("light.ceiling")
    assert state is not None
    hass_read_only_user.mock_policy({})
    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            "light",
            "turn_on",
            {"entity_id": state.entity_id},
            blocking=True,
            context=core.Context(user_id=hass_read_only_user.id),
        )


async def test_light_brightness_step(hass: HomeAssistant) -> None:
    """Test that light context works."""
    entities: List[MockLight] = [
        MockLight("Test_0", STATE_ON),
        MockLight("Test_1", STATE_ON),
    ]
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    entity0: MockLight = entities[0]
    entity0.supported_features = light.SUPPORT_BRIGHTNESS
    entity0.supported_color_modes = None
    entity0.color_mode = None
    entity0.brightness = 100
    entity1: MockLight = entities[1]
    entity1.supported_features = light.SUPPORT_BRIGHTNESS
    entity1.supported_color_modes = None
    entity1.color_mode = None
    entity1.brightness = 50
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state is not None
    assert state.attributes["brightness"] == 100
    state = hass.states.get(entity1.entity_id)
    assert state is not None
    assert state.attributes["brightness"] == 50
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity0.entity_id, entity1.entity_id],
            "brightness_step": -10,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data["brightness"] == 90
    _, data = entity1.last_call("turn_on")
    assert data["brightness"] == 40
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity0.entity_id, entity1.entity_id],
            "brightness_step_pct": 10,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data["brightness"] == 116
    _, data = entity1.last_call("turn_on")
    assert data["brightness"] == 66
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": entity0.entity_id,
            "brightness_step": -126,
        },
        blocking=True,
    )
    assert entity0.state == "off"


@pytest.mark.usefixtures("enable_custom_integrations")
async def test_light_brightness_pct_conversion(
    hass: HomeAssistant, mock_light_entities: List[MockLight]
) -> None:
    """Test that light brightness percent conversion."""
    setup_test_component_platform(
        hass, light.DOMAIN, mock_light_entities
    )
    entity: MockLight = mock_light_entities[0]
    entity.supported_features = light.SUPPORT_BRIGHTNESS
    entity.supported_color_modes = None
    entity.color_mode = None
    entity.brightness = 100
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity.entity_id)
    assert state is not None
    assert state.attributes["brightness"] == 100
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": entity.entity_id, "brightness_pct": 1},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 3
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": entity.entity_id, "brightness_pct": 2},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 5
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": entity.entity_id, "brightness_pct": 50},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 128
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": entity.entity_id, "brightness_pct": 99},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 252
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": entity.entity_id, "brightness_pct": 100},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 255


async def test_profiles(hass: HomeAssistant) -> None:
    """Test profiles loading."""
    profiles: light.Profiles = orig_Profiles(hass)
    await profiles.async_initialize()
    assert profiles.data == {
        "concentrate": light.Profile("concentrate", 0.5119, 0.4147, 219, None),
        "energize": light.Profile("energize", 0.368, 0.3686, 203, None),
        "reading": light.Profile("reading", 0.4448, 0.4066, 240, None),
        "relax": light.Profile("relax", 0.5119, 0.4147, 144, None),
    }
    assert profiles.data["concentrate"].hs_color == (35.932, 69.412)
    assert profiles.data["energize"].hs_color == (43.333, 21.176)
    assert profiles.data["reading"].hs_color == (38.88, 49.02)
    assert profiles.data["relax"].hs_color == (35.932, 69.412)


@patch("os.path.isfile", MagicMock(side_effect=(True, False)))
async def test_profile_load_optional_hs_color(
    hass: HomeAssistant,
) -> None:
    """Test profile loading with profiles containing no xy color."""
    csv_file: str = (
        "the first line is skipped\n"
        "no_color,,,100,1\n"
        "no_color_no_transition,,,110\n"
        "color,0.5119,0.4147,120,2\n"
        "color_no_transition,0.4448,0.4066,130\n"
        "color_and_brightness,0.4448,0.4066,170,\n"
        "only_brightness,,,140\n"
        "only_transition,,,,150\n"
        "transition_float,,,,1.6\n"
        "invalid_profile_1,\n"
        "invalid_color_2,,0.1,1,2\n"
        "invalid_color_3,,0.1,1\n"
        "invalid_color_4,0.1,,1,3\n"
        "invalid_color_5,0.1,,1\n"
        "invalid_brightness,0,0,256,4\n"
        "invalid_brightness_2,0,0,256\n"
        "invalid_no_brightness_no_color_no_transition,,,\n"
    )
    profiles: light.Profiles = orig_Profiles(hass)
    with patch("builtins.open", mock_open(read_data=csv_file)):
        await profiles.async_initialize()
        await hass.async_block_till_done()
    assert profiles.data["no_color"].hs_color is None
    assert profiles.data["no_color"].brightness == 100
    assert profiles.data["no_color"].transition == 1
    assert profiles.data["no_color_no_transition"].hs_color is None
    assert profiles.data["no_color_no_transition"].brightness == 110
    assert profiles.data["no_color_no_transition"].transition is None
    assert profiles.data["color"].hs_color == (35.932, 69.412)
    assert profiles.data["color"].brightness == 120
    assert profiles.data["color"].transition == 2
    assert profiles.data["color_no_transition"].hs_color == (38.88, 49.02)
    assert profiles.data["color_no_transition"].brightness == 130
    assert profiles.data["color_no_transition"].transition is None
    assert profiles.data["color_and_brightness"].hs_color == (38.88, 49.02)
    assert profiles.data["color_and_brightness"].brightness == 170
    assert profiles.data["color_and_brightness"].transition is None
    assert profiles.data["only_brightness"].hs_color is None
    assert profiles.data["only_brightness"].brightness == 140
    assert profiles.data["only_brightness"].transition is None
    assert profiles.data["only_transition"].hs_color is None
    assert profiles.data["only_transition"].brightness is None
    assert profiles.data["only_transition"].transition == 150
    assert profiles.data["transition_float"].hs_color is None
    assert profiles.data["transition_float"].brightness is None
    assert profiles.data["transition_float"].transition == 1.6
    for invalid_profile_name in (
        "invalid_profile_1",
        "invalid_color_2",
        "invalid_color_3",
        "invalid_color_4",
        "invalid_color_5",
        "invalid_brightness",
        "invalid_brightness_2",
        "invalid_no_brightness_no_color_no_transition",
    ):
        assert invalid_profile_name not in profiles.data


@pytest.mark.parametrize(
    ("light_state",),
    [
        (STATE_ON,),
        (STATE_OFF,),
    ],
)
async def test_light_backwards_compatibility_supported_color_modes(
    hass: HomeAssistant, light_state: str
) -> None:
    """Test supported_color_modes if not implemented by the entity."""
    entities: List[MockLight] = [
        MockLight("Test_0", light_state),
        MockLight("Test_1", light_state),
        MockLight("Test_2", light_state),
        MockLight("Test_3", light_state),
        MockLight("Test_4", light_state),
    ]
    entity0: MockLight = entities[0]
    entity1: MockLight = entities[1]
    entity1.supported_features = light.SUPPORT_BRIGHTNESS
    entity1.supported_color_modes = None
    entity1.color_mode = None
    entity2: MockLight = entities[2]
    entity2.supported_features = light.SUPPORT_BRIGHTNESS | light.SUPPORT_COLOR_TEMP
    entity2.supported_color_modes = None
    entity2.color_mode = None
    entity3: MockLight = entities[3]
    entity3.supported_features = light.SUPPORT_BRIGHTNESS | light.SUPPORT_COLOR
    entity3.supported_color_modes = None
    entity3.color_mode = None
    entity4: MockLight = entities[4]
    entity4.supported_features = (
        light.SUPPORT_BRIGHTNESS
        | light.SUPPORT_COLOR
        | light.SUPPORT_COLOR_TEMP
    )
    entity4.supported_color_modes = None
    entity4.color_mode = None
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state is not None
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.ONOFF
    ]
    if light_state == STATE_OFF:
        assert state.attributes["color_mode"] is None
    else:
        assert state.attributes["color_mode"] == light.ColorMode.ONOFF
    state = hass.states.get(entity1.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.BRIGHTNESS
    ]
    if light_state == STATE_OFF:
        assert state.attributes["color_mode"] is None
    else:
        assert state.attributes["color_mode"] == light.ColorMode.UNKNOWN
    state = hass.states.get(entity2.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.COLOR_TEMP
    ]
    if light_state == STATE_OFF:
        assert state.attributes["color_mode"] is None
    else:
        assert state.attributes["color_mode"] == light.ColorMode.UNKNOWN
    state = hass.states.get(entity3.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.HS]
    if light_state == STATE_OFF:
        assert state.attributes["color_mode"] is None
    else:
        assert state.attributes["color_mode"] == light.ColorMode.UNKNOWN
    state = hass.states.get(entity4.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.COLOR_TEMP,
        light.ColorMode.HS,
    ]
    if light_state == STATE_OFF:
        assert state.attributes["color_mode"] is None
    else:
        assert state.attributes["color_mode"] == light.ColorMode.UNKNOWN


async def test_light_backwards_compatibility_color_mode(
    hass: HomeAssistant,
) -> None:
    """Test color_mode if not implemented by the entity."""
    entities: List[MockLight] = [
        MockLight("Test_0", STATE_ON),
        MockLight("Test_1", STATE_ON),
        MockLight("Test_2", STATE_ON),
        MockLight("Test_3", STATE_ON),
        MockLight("Test_4", STATE_ON),
    ]
    entity0: MockLight = entities[0]
    entity1: MockLight = entities[1]
    entity1.supported_features = light.SUPPORT_BRIGHTNESS
    entity1.supported_color_modes = None
    entity1.color_mode = None
    entity1.brightness = 100
    entity2: MockLight = entities[2]
    entity2.supported_features = (
        light.SUPPORT_BRIGHTNESS | light.SUPPORT_COLOR_TEMP
    )
    entity2.supported_color_modes = None
    entity2.color_mode = None
    entity2.color_temp_kelvin = 10000
    entity3: MockLight = entities[3]
    entity3.supported_features = light.SUPPORT_BRIGHTNESS | light.SUPPORT_COLOR
    entity3.supported_color_modes = None
    entity3.color_mode = None
    entity3.hs_color = (240, 100)
    entity4: MockLight = entities[4]
    entity4.supported_features = (
        light.SUPPORT_BRIGHTNESS
        | light.SUPPORT_COLOR
        | light.SUPPORT_COLOR_TEMP
    )
    entity4.supported_color_modes = None
    entity4.color_mode = None
    entity4.hs_color = (240, 100)
    entity4.color_temp_kelvin = 10000
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.ONOFF
    ]
    assert state.attributes["color_mode"] == light.ColorMode.ONOFF
    state = hass.states.get(entity1.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.BRIGHTNESS
    ]
    assert state.attributes["color_mode"] == light.ColorMode.BRIGHTNESS
    state = hass.states.get(entity2.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.COLOR_TEMP
    ]
    assert state.attributes["color_mode"] == light.ColorMode.COLOR_TEMP
    assert state.attributes["rgb_color"] == (202, 218, 255)
    assert state.attributes["hs_color"] == (221.575, 20.9)
    assert state.attributes["xy_color"] == (0.278, 0.287)
    state = hass.states.get(entity3.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.HS]
    assert state.attributes["color_mode"] == light.ColorMode.HS
    state = hass.states.get(entity4.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.COLOR_TEMP,
        light.ColorMode.HS,
    ]
    assert state.attributes["color_mode"] == light.ColorMode.HS


async def test_light_service_call_rgbw(hass: HomeAssistant) -> None:
    """Test rgbw functionality in service calls."""
    entity0: MockLight = MockLight("Test_rgbw", STATE_ON)
    entity0.supported_color_modes = {light.ColorMode.RGBW}
    setup_test_component_platform(
        hass, light.DOMAIN, [entity0]
    )
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state is not None
    assert state.attributes["supported_color_modes"] == [light.ColorMode.RGBW]
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity0.entity_id, entity0.entity_id],
            "brightness_pct": 100,
            "rgbw_color": (10, 20, 30, 40),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbw_color": (10, 20, 30, 40),
    }


async def test_light_state_off(
    hass: HomeAssistant,
) -> None:
    """Test rgbw color conversion in state updates."""
    entities: List[MockLight] = [
        MockLight("Test_onoff", STATE_OFF),
        MockLight("Test_brightness", STATE_OFF),
        MockLight("Test_ct", STATE_OFF),
        MockLight("Test_rgbw", STATE_OFF),
    ]
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    entity0: MockLight = entities[0]
    entity0.supported_color_modes = {light.ColorMode.ONOFF}
    entity1: MockLight = entities[1]
    entity1.supported_color_modes = {light.ColorMode.BRIGHTNESS}
    entity2: MockLight = entities[2]
    entity2.supported_color_modes = {light.ColorMode.COLOR_TEMP}
    entity3: MockLight = entities[3]
    entity3.supported_color_modes = {light.ColorMode.RGBW}
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state is not None
    assert state.attributes == {
        "color_mode": None,
        "friendly_name": "Test_onoff",
        "supported_color_modes": [light.ColorMode.ONOFF],
        "supported_features": 0,
    }
    state = hass.states.get(entity1.entity_id)
    assert state is not None
    assert state.attributes == {
        "color_mode": None,
        "friendly_name": "Test_brightness",
        "supported_color_modes": [light.ColorMode.BRIGHTNESS],
        "supported_features": 0,
        "brightness": None,
    }
    state = hass.states.get(entity2.entity_id)
    assert state is not None
    assert state.attributes == {
        "color_mode": None,
        "friendly_name": "Test_ct",
        "supported_color_modes": [light.ColorMode.COLOR_TEMP],
        "supported_features": 0,
        "brightness": None,
        "color_temp": None,
        "color_temp_kelvin": None,
        "hs_color": None,
        "rgb_color": None,
        "xy_color": None,
        "max_color_temp_kelvin": 6535,
        "max_mireds": 500,
        "min_color_temp_kelvin": 2000,
        "min_mireds": 153,
    }
    state = hass.states.get(entity3.entity_id)
    assert state is not None
    assert state.attributes == {
        "color_mode": None,
        "friendly_name": "Test_rgbw",
        "supported_color_modes": [light.ColorMode.RGBW],
        "supported_features": 0,
        "brightness": None,
        "rgbw_color": None,
        "hs_color": None,
        "rgb_color": None,
        "xy_color": None,
    }


async def test_light_state_rgbw(
    hass: HomeAssistant,
) -> None:
    """Test rgbw color conversion in state updates."""
    entity0: MockLight = MockLight("Test_rgbw", STATE_ON)
    setup_test_component_platform(
        hass, light.DOMAIN, [entity0]
    )
    entity0.brightness = 255
    entity0.supported_color_modes = {light.ColorMode.RGBW}
    entity0.color_mode = light.ColorMode.RGBW
    entity0.hs_color = "Invalid"
    entity0.rgb_color = "Invalid"
    entity0.rgbw_color = (1, 2, 3, 4)
    entity0.rgbww_color = "Invalid"
    entity0.xy_color = "Invalid"
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes == {
        "color_mode": light.ColorMode.RGBW,
        "friendly_name": "Test_rgbw",
        "supported_color_modes": [light.ColorMode.RGBW],
        "supported_features": 0,
        "hs_color": (240.0, 25.0),
        "rgb_color": (3, 3, 4),
        "rgbw_color": (1, 2, 3, 4),
        "xy_color": (0.301, 0.295),
        "brightness": 255,
    }


async def test_light_state_rgbww(
    hass: HomeAssistant,
) -> None:
    """Test rgbww color conversion in state updates."""
    entity0: MockLight = MockLight("Test_rgbww", STATE_ON)
    setup_test_component_platform(
        hass, light.DOMAIN, [entity0]
    )
    entity0.supported_color_modes = {light.ColorMode.RGBWW}
    entity0.color_mode = light.ColorMode.RGBWW
    entity0.hs_color = "Invalid"
    entity0.rgb_color = "Invalid"
    entity0.rgbw_color = "Invalid"
    entity0.rgbww_color = (1, 2, 3, 4, 5)
    entity0.xy_color = "Invalid"
    entity0.brightness = 255
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes == {
        "color_mode": light.ColorMode.RGBWW,
        "friendly_name": "Test_rgbww",
        "supported_color_modes": [light.ColorMode.RGBWW],
        "supported_features": 0,
        "hs_color": (60.0, 20.0),
        "rgb_color": (5, 5, 4),
        "rgbww_color": (1, 2, 3, 4, 5),
        "xy_color": (0.339, 0.354),
        "brightness": 255,
    }


async def test_light_service_call_color_conversion(
    hass: HomeAssistant,
) -> None:
    """Test color conversion in service calls."""
    entities: List[MockLight] = [
        MockLight("Test_hs", STATE_ON),
        MockLight("Test_rgb", STATE_ON),
        MockLight("Test_xy", STATE_ON),
        MockLight("Test_all", STATE_ON),
        MockLight("Test_legacy", STATE_ON),
        MockLight("Test_rgbw", STATE_ON),
        MockLight("Test_rgbww", STATE_ON),
        MockLight("Test_temperature", STATE_ON),
    ]
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    entity0: MockLight = entities[0]
    entity0.supported_color_modes = {light.ColorMode.HS}
    entity1: MockLight = entities[1]
    entity1.supported_color_modes = {light.ColorMode.RGB}
    entity2: MockLight = entities[2]
    entity2.supported_color_modes = {light.ColorMode.XY}
    entity3: MockLight = entities[3]
    entity3.supported_color_modes = {
        light.ColorMode.HS,
        light.ColorMode.RGB,
        light.ColorMode.XY,
    }
    entity4: MockLight = entities[4]
    entity4.supported_features = light.SUPPORT_COLOR
    entity4.supported_color_modes = None
    entity4.color_mode = None
    entity5: MockLight = entities[5]
    entity5.supported_color_modes = {light.ColorMode.RGBW}
    entity6: MockLight = entities[6]
    entity6.supported_color_modes = {light.ColorMode.RGBWW}
    entity7: MockLight = entities[7]
    entity7.supported_color_modes = {light.ColorMode.COLOR_TEMP}
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.HS]
    state = hass.states.get(entity1.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.RGB]
    state = hass.states.get(entity2.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.XY]
    state = hass.states.get(entity3.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.HS,
        light.ColorMode.RGB,
        light.ColorMode.XY,
    ]
    state = hass.states.get(entity4.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.HS]
    state = hass.states.get(entity5.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.RGBW]
    state = hass.states.get(entity6.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.RGBWW]
    state = hass.states.get(entity7.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.COLOR_TEMP]
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 100,
            "hs_color": (240, 100),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "hs_color": (240.0, 100.0),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgb_color": (0, 0, 255),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "xy_color": (0.136, 0.04),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "hs_color": (240.0, 100.0),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "hs_color": (240.0, 100.0),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbw_color": (0, 0, 255, 0),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbww_color": (0, 0, 255, 0, 0),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "color_temp_kelvin": 1739,
        "color_temp": 575,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 100,
            "hs_color": (240, 0),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "hs_color": (240.0, 0.0),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgb_color": (255, 255, 255),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "xy_color": (0.323, 0.329),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgb_color": (255, 255, 255),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "hs_color": (240.0, 0.0),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbw_color": (0, 0, 0, 255),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbww_color": (0, 76, 141, 255, 255),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "color_temp_kelvin": 5962,
        "color_temp": 167,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 50,
            "rgb_color": (128, 0, 0),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 100.0),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (128, 0, 0),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.701, 0.299),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (128, 0, 0),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 100.0),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbw_color": (128, 0, 0, 0),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (128, 0, 0, 0, 0),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp_kelvin": 6279,
        "color_temp": 159,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 50,
            "rgb_color": (255, 255, 255),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 0.0),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (255, 255, 255),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.323, 0.329),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (255, 255, 255),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 0.0),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbw_color": (0, 0, 0, 255),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (0, 76, 141, 255, 255),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp_kelvin": 5962,
        "color_temp": 167,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 50,
            "xy_color": (0.1, 0.8),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (125.176, 100.0),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (0, 255, 22),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.1, 0.8),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.1, 0.8),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (125.176, 100.0),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbw_color": (0, 255, 22, 0),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (0, 75, 140, 255, 255),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp_kelvin": 8645,
        "color_temp": 115,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 50,
            "xy_color": (0.323, 0.329),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 0.392),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (255, 254, 254),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.323, 0.329),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.323, 0.329),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 0.392),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbw_color": (1, 0, 0, 255),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (0, 76, 141, 255, 255),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp_kelvin": 5962,
        "color_temp": 167,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 50,
            "rgbw_color": (128, 0, 0, 64),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 66.406),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (128, 43, 43),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.592, 0.308),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (128, 43, 43),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 66.406),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbw_color": (128, 0, 0, 64),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (128, 0, 30, 117, 117),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp_kelvin": 3845,
        "color_temp": 260,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 50,
            "rgbw_color": (255, 255, 255, 255),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 0.0),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (255, 255, 255),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.396, 0.359),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (255, 255, 255),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (0.0, 0.0),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbw_color": (96, 44, 0, 255),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (255, 255, 255, 255, 255),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp_kelvin": 3451,
        "color_temp": 289,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 50,
            "rgbww_color": (128, 0, 0, 64, 32),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (4.118, 79.688),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (128, 33, 26),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.639, 0.312),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (128, 33, 26),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (4.118, 79.688),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbw_color": (128, 9, 0, 33),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (0, 76, 141, 255, 255),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp_kelvin": 3845,
        "color_temp": 260,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
                entity7.entity_id,
            ],
            "brightness_pct": 50,
            "rgbww_color": (255, 255, 255, 255, 255),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (27.429, 27.451),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (255, 217, 185),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "xy_color": (0.396, 0.359),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgb_color": (255, 217, 185),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "hs_color": (27.429, 27.451),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbw_color": (96, 44, 0, 255),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (255, 255, 255, 255, 255),
    }
    _, data = entity7.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp_kelvin": 3451,
        "color_temp": 289,
    }


async def test_light_service_call_color_conversion_named_tuple(
    hass: HomeAssistant,
) -> None:
    """Test a named tuple (RGBColor) is handled correctly."""
    entities: List[MockLight] = [
        MockLight("Test_hs", STATE_ON),
        MockLight("Test_rgb", STATE_ON),
        MockLight("Test_xy", STATE_ON),
        MockLight("Test_all", STATE_ON),
        MockLight("Test_legacy", STATE_ON),
        MockLight("Test_rgbw", STATE_ON),
        MockLight("Test_rgbww", STATE_ON),
    ]
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    entity0: MockLight = entities[0]
    entity0.supported_color_modes = {light.ColorMode.HS}
    entity1: MockLight = entities[1]
    entity1.supported_color_modes = {light.ColorMode.RGB}
    entity2: MockLight = entities[2]
    entity2.supported_color_modes = {light.ColorMode.XY}
    entity3: MockLight = entities[3]
    entity3.supported_color_modes = {
        light.ColorMode.HS,
        light.ColorMode.RGB,
        light.ColorMode.XY,
    }
    entity4: MockLight = entities[4]
    entity4.supported_features = light.SUPPORT_COLOR
    entity4.supported_color_modes = None
    entity4.color_mode = None
    entity5: MockLight = entities[5]
    entity5.supported_color_modes = {light.ColorMode.RGBW}
    entity6: MockLight = entities[6]
    entity6.supported_color_modes = {light.ColorMode.RGBWW}
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
                entity3.entity_id,
                entity4.entity_id,
                entity5.entity_id,
                entity6.entity_id,
            ],
            "brightness_pct": 25,
            "rgb_color": color_util.RGBColor(128, 0, 0),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 64,
        "hs_color": (0.0, 100.0),
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 64,
        "rgb_color": (128, 0, 0),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 64,
        "xy_color": (0.701, 0.299),
    }
    _, data = entity3.last_call("turn_on")
    assert data == {
        "brightness": 64,
        "rgb_color": (128, 0, 0),
    }
    _, data = entity4.last_call("turn_on")
    assert data == {
        "brightness": 64,
        "hs_color": (0.0, 100.0),
    }
    _, data = entity5.last_call("turn_on")
    assert data == {
        "brightness": 64,
        "rgbw_color": (128, 0, 0, 0),
    }
    _, data = entity6.last_call("turn_on")
    assert data == {
        "brightness": 64,
        "rgbww_color": (128, 0, 0, 0, 0),
    }


async def test_light_service_call_color_temp_emulation(
    hass: HomeAssistant,
) -> None:
    """Test color conversion in service calls."""
    entities: List[MockLight] = [
        MockLight("Test_hs_ct", STATE_ON),
        MockLight("Test_hs", STATE_ON),
        MockLight("Test_hs_white", STATE_ON),
    ]
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    entity0: MockLight = entities[0]
    entity0.supported_color_modes = {light.ColorMode.COLOR_TEMP, light.ColorMode.HS}
    entity1: MockLight = entities[1]
    entity1.supported_color_modes = {light.ColorMode.HS}
    entity2: MockLight = entities[2]
    entity2.supported_color_modes = {light.ColorMode.HS, light.ColorMode.WHITE}
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.COLOR_TEMP,
        light.ColorMode.HS,
    ]
    state = hass.states.get(entity1.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.HS]
    state = hass.states.get(entity2.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.HS,
        light.ColorMode.WHITE,
    ]
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
                entity2.entity_id,
            ],
            "brightness_pct": 100,
            "color_temp": 200,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "color_temp": 200,
        "color_temp_kelvin": 5000,
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "hs_color": (27.001, 19.243),
    }
    _, data = entity2.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "hs_color": (27.001, 19.243),
    }


async def test_light_service_call_color_temp_conversion(
    hass: HomeAssistant,
) -> None:
    """Test color temp conversion in service calls."""
    entities: List[MockLight] = [
        MockLight("Test_rgbww_ct", STATE_ON),
        MockLight("Test_rgbww", STATE_ON),
    ]
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    entity0: MockLight = entities[0]
    entity0.supported_color_modes = {
        light.ColorMode.COLOR_TEMP,
        light.ColorMode.RGBWW,
    }
    entity1: MockLight = entities[1]
    entity1.supported_color_modes = {light.ColorMode.RGBWW}
    assert entity1.min_mireds == 153
    assert entity1.max_mireds == 500
    assert entity1.min_color_temp_kelvin == 2000
    assert entity1.max_color_temp_kelvin == 6535
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.COLOR_TEMP,
        light.ColorMode.RGBWW,
    ]
    assert state.attributes["min_mireds"] == 153
    assert state.attributes["max_mireds"] == 500
    assert state.attributes["min_color_temp_kelvin"] == 2000
    assert state.attributes["max_color_temp_kelvin"] == 6535
    state = hass.states.get(entity1.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.RGBWW]
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
            ],
            "brightness_pct": 100,
            "color_temp": 153,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "color_temp": 153,
        "color_temp_kelvin": 6535,
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbww_color": (0, 0, 0, 255, 0),
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
            ],
            "brightness_pct": 50,
            "color_temp": 500,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp": 500,
        "color_temp_kelvin": 2000,
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "rgbww_color": (0, 0, 0, 0, 128),
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
            ],
            "brightness_pct": 100,
            "color_temp": 327,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "color_temp": 327,
        "color_temp_kelvin": 3058,
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbww_color": (0, 0, 0, 127, 128),
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
            ],
            "brightness_pct": 100,
            "color_temp": 240,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "color_temp": 240,
        "color_temp_kelvin": 4166,
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbww_color": (0, 0, 0, 191, 64),
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
                entity1.entity_id,
            ],
            "brightness_pct": 100,
            "color_temp": 410,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "color_temp": 410,
        "color_temp_kelvin": 2439,
    }
    _, data = entity1.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "rgbww_color": (0, 0, 0, 66, 189),
    }


async def test_light_mired_color_temp_conversion(
    hass: HomeAssistant,
) -> None:
    """Test color temp conversion from K to legacy mired."""
    entities: List[MockLight] = [
        MockLight("Test_rgbww_ct", STATE_ON),
        MockLight("Test_rgbww", STATE_ON),
    ]
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    entity0: MockLight = entities[0]
    entity0.supported_color_modes = {light.ColorMode.COLOR_TEMP}
    entity0._attr_min_color_temp_kelvin = 1800
    entity0._attr_max_color_temp_kelvin = 6700
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.COLOR_TEMP
    ]
    assert state.attributes["min_mireds"] == 149
    assert state.attributes["max_mireds"] == 555
    assert state.attributes["min_color_temp_kelvin"] == 1800
    assert state.attributes["max_color_temp_kelvin"] == 6700
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
            ],
            "brightness_pct": 100,
            "color_temp_kelvin": 3500,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "color_temp": 285,
        "color_temp_kelvin": 3500,
    }
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [
                entity0.entity_id,
            ],
            "brightness_pct": 50,
            "color_temp_kelvin": 5000,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 128,
        "color_temp": 285,
        "color_temp_kelvin": 5000,
    }


async def test_light_service_call_white_mode(
    hass: HomeAssistant,
) -> None:
    """Test color_mode white in service calls."""
    entity0: MockLight = MockLight("Test_white", STATE_ON)
    entity0.supported_color_modes = {light.ColorMode.HS, light.ColorMode.WHITE}
    setup_test_component_platform(
        hass, light.DOMAIN, [entity0]
    )
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.HS,
        light.ColorMode.WHITE,
    ]
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity0.entity_id],
            "brightness_pct": 100,
            "hs_color": (240, 100),
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {
        "brightness": 255,
        "hs_color": (240.0, 100.0),
    }
    entity0.calls = []
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": [entity0.entity_id], "white": 50},
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {"white": 50}
    entity0.calls = []
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": [entity0.entity_id], "white": 0},
        blocking=True,
    )
    _, data = entity0.last_call("turn_off")
    assert data == {}
    entity0.calls = []
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity0.entity_id],
            "brightness_pct": 100,
            "white": 50,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {"white": 255}
    entity0.calls = []
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": [entity0.entity_id], "brightness": 100, "white": 0},
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {"white": 100}
    entity0.calls = []
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity0.entity_id],
            "brightness_pct": 0,
            "white": 50,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_off")
    assert data == {}
    entity0.calls = []
    await hass.services.async_call(
        "light",
        "turn_on",
        {"entity_id": [entity0.entity_id], "white": True},
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {"white": 100}
    entity0.calls = []
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity0.entity_id],
            "brightness_pct": 50,
            "white": True,
        },
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data == {"white": 128}


async def test_light_state_color_conversion(
    hass: HomeAssistant,
) -> None:
    """Test color conversion in state updates."""
    entities: List[MockLight] = [
        MockLight("Test_hs", STATE_ON),
        MockLight("Test_rgb", STATE_ON),
        MockLight("Test_xy", STATE_ON),
        MockLight("Test_legacy", STATE_ON),
    ]
    setup_test_component_platform(
        hass, light.DOMAIN, entities
    )
    entity0: MockLight = entities[0]
    entity0.supported_color_modes = {light.ColorMode.HS}
    entity0.color_mode = light.ColorMode.HS
    entity0.hs_color = (240, 100)
    entity0.rgb_color = "Invalid"
    entity0.xy_color = "Invalid"
    entity1: MockLight = entities[1]
    entity1.supported_color_modes = {light.ColorMode.RGB}
    entity1.color_mode = light.ColorMode.RGB
    entity1.hs_color = "Invalid"
    entity1.rgb_color = (128, 0, 0)
    entity1.xy_color = "Invalid"
    entity2: MockLight = entities[2]
    entity2.supported_color_modes = {light.ColorMode.XY}
    entity2.color_mode = light.ColorMode.XY
    entity2.hs_color = "Invalid"
    entity2.rgb_color = "Invalid"
    entity2.xy_color = (0.1, 0.8)
    entity3: MockLight = entities[3]
    entity3.hs_color = (240, 100)
    entity3.supported_features = light.SUPPORT_COLOR
    entity3.supported_color_modes = None
    entity3.color_mode = None
    assert await async_setup_component(
        hass, "light", {"light": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes["color_mode"] == light.ColorMode.HS
    assert state.attributes["hs_color"] == (240, 100)
    assert state.attributes["rgb_color"] == (0, 0, 255)
    assert state.attributes["xy_color"] == (0.136, 0.04)
    state = hass.states.get(entity1.entity_id)
    assert state.attributes["color_mode"] == light.ColorMode.RGB
    assert state.attributes["hs_color"] == (0.0, 100.0)
    assert state.attributes["rgb_color"] == (128, 0, 0)
    assert state.attributes["xy_color"] == (0.701, 0.299)
    state = hass.states.get(entity2.entity_id)
    assert state.attributes["color_mode"] == light.ColorMode.XY
    assert state.attributes["hs_color"] == (125.176, 100.0)
    assert state.attributes["rgb_color"] == (0, 255, 22)
    assert state.attributes["xy_color"] == (0.1, 0.8)
    state = hass.states.get(entity3.entity_id)
    assert state.attributes["color_mode"] == light.ColorMode.HS
    assert state.attributes["hs_color"] == (240, 100)
    assert state.attributes["rgb_color"] == (0, 0, 255)
    assert state.attributes["xy_color"] == (0.136, 0.04)


async def test_services_filter_parameters(
    hass: HomeAssistant,
    mock_light_profiles: Dict[str, light.Profile],
    mock_light_entities: List[MockLight],
) -> None:
    """Test turn_on and turn_off filters unsupported parameters."""
    setup_test_component_platform(
        hass, light.DOMAIN, mock_light_entities
    )
    assert await async_setup_component(
        hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}}
    )
    await hass.async_block_till_done()
    ent1: MockLight = mock_light_entities[0]
    await hass.services.async_call(
        light.DOMAIN, SERVICE_TURN_ON, {ATTR_ENTITY_ID: ENTITY_MATCH_ALL}, blocking=True
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL, light.ATTR_BRIGHTNESS: 0},
        blocking=True,
    )
    assert not light.is_on(hass, ent1.entity_id)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_BRIGHTNESS: 0,
            light.ATTR_EFFECT: "fun_effect",
            light.ATTR_FLASH: "short",
            light.ATTR_TRANSITION: 10,
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_COLOR_TEMP: 153,
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_HS_COLOR: (0, 0),
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_RGB_COLOR: (0, 0, 0),
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_RGBW_COLOR: (0, 0, 0, 0),
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_RGBWW_COLOR: (0, 0, 0, 0, 0),
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_XY_COLOR: (0, 0),
        },
        blocking=True,
    )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_BRIGHTNESS: 0,
            light.ATTR_EFFECT: "fun_effect",
            light.ATTR_FLASH: "short",
            light.ATTR_TRANSITION: 10,
        },
        blocking=True,
    )
    assert not light.is_on(hass, ent1.entity_id)
    _, data = ent1.last_call("turn_off")
    assert data == {}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_OFF,
        {
            ATTR_ENTITY_ID: ent1.entity_id,
            light.ATTR_FLASH: "short",
            light.ATTR_TRANSITION: 10,
        },
        blocking=True,
    )
    assert not light.is_on(hass, ent1.entity_id)
    _, data = ent1.last_call("turn_off")
    assert data == {}


def test_valid_supported_color_modes() -> None:
    """Test valid_supported_color_modes."""
    supported: Set[light.ColorMode] = {light.ColorMode.HS}
    assert light.valid_supported_color_modes(supported) == supported
    supported = set()
    with pytest.raises(vol.Error):
        light.valid_supported_color_modes(supported)
    supported = {light.ColorMode.WHITE}
    with pytest.raises(vol.Error):
        light.valid_supported_color_modes(supported)
    supported = {light.ColorMode.WHITE, light.ColorMode.COLOR_TEMP}
    with pytest.raises(vol.Error):
        light.valid_supported_color_modes(supported)
    supported = {light.ColorMode.WHITE, light.ColorMode.HS}
    assert light.valid_supported_color_modes(supported) == supported
    supported = {light.ColorMode.ONOFF}
    assert light.valid_supported_color_modes(supported) == supported
    supported = {light.ColorMode.ONOFF, light.ColorMode.COLOR_TEMP}
    with pytest.raises(vol.Error):
        light.valid_supported_color_modes(supported)
    supported = {light.ColorMode.BRIGHTNESS}
    assert light.valid_supported_color_modes(supported) == supported
    supported = {light.ColorMode.BRIGHTNESS, light.ColorMode.COLOR_TEMP}
    with pytest.raises(vol.Error):
        light.valid_supported_color_modes(supported)


def test_filter_supported_color_modes() -> None:
    """Test filter_supported_color_modes."""
    supported: Set[light.ColorMode] = {light.ColorMode.HS}
    assert light.filter_supported_color_modes(supported) == supported
    supported = set()
    with pytest.raises(HomeAssistantError):
        light.filter_supported_color_modes(supported)
    supported = {light.ColorMode.WHITE}
    with pytest.raises(HomeAssistantError):
        light.filter_supported_color_modes(supported)
    supported = {light.ColorMode.WHITE, light.ColorMode.COLOR_TEMP}
    with pytest.raises(HomeAssistantError):
        light.filter_supported_color_modes(supported)
    supported = {light.ColorMode.WHITE, light.ColorMode.HS}
    assert light.filter_supported_color_modes(supported) == supported
    supported = {light.ColorMode.ONOFF}
    assert light.filter_supported_color_modes(supported) == supported
    supported = {light.ColorMode.ONOFF, light.ColorMode.COLOR_TEMP}
    assert light.filter_supported_color_modes(supported) == {light.ColorMode.COLOR_TEMP}
    supported = {light.ColorMode.BRIGHTNESS}
    assert light.filter_supported_color_modes(supported) == supported
    supported = {light.ColorMode.BRIGHTNESS, light.ColorMode.COLOR_TEMP}
    assert light.filter_supported_color_modes(supported) == {light.ColorMode.COLOR_TEMP}
    supported = {light.ColorMode.ONOFF, light.ColorMode.BRIGHTNESS}
    assert light.filter_supported_color_modes(supported) == {light.ColorMode.BRIGHTNESS}


def test_deprecated_supported_features_ints(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """Test deprecated supported features ints."""

    class MockLightEntityEntity(light.LightEntity):
        """Mock LightEntity for testing."""

        @property
        def supported_features(self) -> int:
            """Return supported features."""
            return 1

    entity = MockLightEntityEntity()
    entity.hass = hass
    entity.platform = MockEntityPlatform(
        hass, domain="test", platform_name="test"
    )
    assert entity.supported_features_compat == light.LightEntityFeature(1)
    assert "MockLightEntityEntity" in caplog.text
    assert "is using deprecated supported features values" in caplog.text
    assert "Instead it should use" in caplog.text
    assert "LightEntityFeature" in caplog.text
    assert "and color modes" in caplog.text
    caplog.clear()
    assert entity.supported_features_compat == light.LightEntityFeature(1)
    assert "is using deprecated supported features values" not in caplog.text


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
    supported_color_modes: Set[light.ColorMode],
    warning_expected: bool,
) -> None:
    """Test a light setting no color mode."""

    class MockLightEntityEntity(light.LightEntity):
        """Mock LightEntity for testing."""

        _attr_color_mode: Optional[light.ColorMode] = color_mode
        _attr_is_on: bool = True
        _attr_supported_features: int = light.LightEntityFeature.EFFECT
        _attr_supported_color_modes: Optional[Set[light.ColorMode]] = supported_color_modes

    entity = MockLightEntityEntity()
    platform = MockEntityPlatform(
        hass, domain="test", platform_name="test"
    )
    await platform.async_add_entities([entity])
    entity._async_calculate_state()
    expected_warning = "does not report a color mode"
    assert (expected_warning in caplog.text) is warning_expected


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
    color_mode: Optional[light.ColorMode],
    supported_color_modes: Optional[Set[light.ColorMode]],
    warning_expected: bool,
) -> None:
    """Test a light setting no color mode."""

    class MockLightEntityEntity(light.LightEntity):
        """Mock LightEntity for testing."""

        _attr_color_mode: Optional[light.ColorMode] = color_mode
        _attr_is_on: bool = True
        _attr_supported_features: int = light.LightEntityFeature.EFFECT
        _attr_supported_color_modes: Optional[Set[light.ColorMode]] = supported_color_modes

    entity = MockLightEntityEntity()
    platform = MockEntityPlatform(
        hass, domain="test", platform_name="test"
    )
    await platform.async_add_entities([entity])
    entity._async_calculate_state()
    expected_warning = "does not set supported color modes"
    assert (expected_warning in caplog.text) is warning_expected


@pytest.mark.parametrize(
    (
        "color_mode",
        "supported_color_modes",
        "effect",
        "warning_expected",
    ),
    [
        (light.ColorMode.ONOFF, {light.ColorMode.ONOFF}, None, False),
        (
            light.ColorMode.ONOFF,
            {light.ColorMode.BRIGHTNESS},
            None,
            True,
        ),
        (
            light.ColorMode.ONOFF,
            {light.ColorMode.BRIGHTNESS},
            light.EFFECT_OFF,
            True,
        ),
        (
            light.ColorMode.ONOFF,
            {light.ColorMode.BRIGHTNESS},
            "effect",
            False,
        ),
        (
            light.ColorMode.BRIGHTNESS,
            {light.ColorMode.BRIGHTNESS},
            "effect",
            False,
        ),
        (
            light.ColorMode.BRIGHTNESS,
            {light.ColorMode.BRIGHTNESS},
            None,
            False,
        ),
        (
            light.ColorMode.BRIGHTNESS,
            {light.ColorMode.HS},
            None,
            True,
        ),
        (
            light.ColorMode.BRIGHTNESS,
            {light.ColorMode.HS},
            light.EFFECT_OFF,
            True,
        ),
        (
            light.ColorMode.ONOFF,
            {light.ColorMode.HS},
            None,
            True,
        ),
        (
            light.ColorMode.ONOFF,
            {light.ColorMode.HS},
            light.EFFECT_OFF,
            True,
        ),
        (
            light.ColorMode.BRIGHTNESS,
            {light.ColorMode.HS},
            "effect",
            False,
        ),
        (
            light.ColorMode.ONOFF,
            {light.ColorMode.HS},
            "effect",
            False,
        ),
        (
            light.ColorMode.HS,
            {light.ColorMode.HS},
            "effect",
            False,
        ),
        (
            light.ColorMode.HS,
            {light.ColorMode.BRIGHTNESS},
            "effect",
            True,
        ),
    ],
)
async def test_report_invalid_color_mode(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: Optional[light.ColorMode],
    supported_color_modes: Set[light.ColorMode],
    effect: Optional[str],
    warning_expected: bool,
) -> None:
    """Test a light setting an invalid color mode."""

    class MockLightEntityEntity(light.LightEntity):
        """Mock LightEntity for testing."""

        _attr_color_mode: Optional[light.ColorMode] = color_mode
        _attr_effect: Optional[str] = effect
        _attr_is_on: bool = True
        _attr_supported_features: int = light.LightEntityFeature.EFFECT
        _attr_supported_color_modes: Set[light.ColorMode] = supported_color_modes

    entity = MockLightEntityEntity()
    platform = MockEntityPlatform(
        hass, domain="test", platform_name="test"
    )
    await platform.async_add_entities([entity])
    entity._async_calculate_state()
    expected_warning = f"set to unsupported color mode {color_mode}"
    assert (expected_warning in caplog.text) is warning_expected


@pytest.mark.parametrize(
    (
        "color_mode",
        "supported_color_modes",
        "platform_name",
        "warning_expected",
    ),
    [
        (light.ColorMode.ONOFF, {light.ColorMode.ONOFF}, "test", False),
        (
            light.ColorMode.ONOFF,
            {light.ColorMode.ONOFF, light.ColorMode.BRIGHTNESS},
            "test",
            True,
        ),
        (
            light.ColorMode.HS,
            {light.ColorMode.HS, light.ColorMode.BRIGHTNESS},
            "test",
            True,
        ),
        (
            light.ColorMode.HS,
            {light.ColorMode.COLOR_TEMP, light.ColorMode.HS},
            "test",
            False,
        ),
        (
            light.ColorMode.ONOFF,
            {light.ColorMode.ONOFF, light.ColorMode.BRIGHTNESS},
            "philips_js",
            False,
        ),
    ],
)
def test_report_invalid_color_modes(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    color_mode: Optional[light.ColorMode],
    supported_color_modes: Set[light.ColorMode],
    platform_name: str,
    warning_expected: bool,
) -> None:
    """Test a light setting an invalid color mode."""

    class MockLightEntityEntity(light.LightEntity):
        """Mock LightEntity for testing."""

        _attr_color_mode: Optional[light.ColorMode] = color_mode
        _attr_is_on: bool = True
        _attr_supported_features: int = light.LightEntityFeature.EFFECT
        _attr_supported_color_modes: Set[light.ColorMode] = supported_color_modes
        platform: MockEntityPlatform = MockEntityPlatform(
            hass, platform_name=platform_name
        )

    entity = MockLightEntityEntity()
    entity._async_calculate_state()
    expected_warning = "sets invalid supported color modes"
    assert (expected_warning in caplog.text) is warning_expected


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
async def test_missing_kelvin_property_warnings(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    attributes: Dict[str, Any],
    expected_warnings: Dict[str, bool],
    expected_values: Tuple[Any, ...],
) -> None:
    """Test missing kelvin properties."""

    class MockLightEntityEntity(light.LightEntity):
        """Mock LightEntity for testing."""

        _attr_color_mode: Optional[light.ColorMode] = light.ColorMode.COLOR_TEMP
        _attr_is_on: bool = True
        _attr_supported_features: int = light.LightEntityFeature.EFFECT
        _attr_supported_color_modes: Set[light.ColorMode] = {light.ColorMode.COLOR_TEMP}

    entity = MockLightEntityEntity()
    for k, v in attributes.items():
        setattr(entity, k, v)
    state = entity._async_calculate_state()
    for warning, expected in expected_warnings.items():
        assert (
            f"is using mireds for {warning} light color temperature" in caplog.text
        ) is expected, f"Expected {expected} for '{warning}'"
    (
        min_color_temp_kelvin,
        color_temp_kelvin,
        max_color_temp_kelvin,
        min_mireds,
        color_temp,
        max_mireds,
        entity_min_mireds,
        entity_color_temp,
        entity_max_mireds,
    ) = expected_values
    assert (
        state.attributes[light.ATTR_MIN_COLOR_TEMP_KELVIN]
        == min_color_temp_kelvin
    )
    assert (
        state.attributes[light.ATTR_COLOR_TEMP_KELVIN] == color_temp_kelvin
    )
    assert (
        state.attributes[light.ATTR_MAX_COLOR_TEMP_KELVIN]
        == max_color_temp_kelvin
    )
    assert state.attributes[light.ATTR_MIN_MIREDS] == min_mireds
    assert state.attributes[light.ATTR_COLOR_TEMP] == color_temp
    assert state.attributes[light.ATTR_MAX_MIREDS] == max_mireds
    assert entity.min_mireds == entity_min_mireds
    assert entity.color_temp == entity_color_temp
    assert entity.max_mireds == entity_max_mireds


@pytest.mark.parametrize("module", [light])
def test_all(module: ModuleType) -> None:
    """Test module.__all__ is correctly set."""
    help_test_all(module)


@pytest.mark.parametrize(
    (
        "constant_name",
        "constant_value",
        "constant_replacement",
    ),
    [
        (
            "SUPPORT_BRIGHTNESS",
            1,
            "supported_color_modes",
        ),
        (
            "SUPPORT_COLOR_TEMP",
            2,
            "supported_color_modes",
        ),
        (
            "SUPPORT_COLOR",
            16,
            "supported_color_modes",
        ),
        (
            "ATTR_COLOR_TEMP",
            "color_temp",
            "kelvin equivalent (ATTR_COLOR_TEMP_KELVIN)",
        ),
        (
            "ATTR_KELVIN",
            "kelvin",
            "ATTR_COLOR_TEMP_KELVIN",
        ),
        (
            "ATTR_MIN_MIREDS",
            "min_mireds",
            "kelvin equivalent (ATTR_MAX_COLOR_TEMP_KELVIN)",
        ),
        (
            "ATTR_MAX_MIREDS",
            "max_mireds",
            "kelvin equivalent (ATTR_MIN_COLOR_TEMP_KELVIN)",
        ),
    ],
)
def test_deprecated_light_constants(
    caplog: pytest.LogCaptureFixture,
    constant_name: str,
    constant_value: Any,
    constant_replacement: str,
) -> None:
    """Test deprecated light constants."""
    import_and_test_deprecated_constant(
        caplog,
        light,
        constant_name,
        constant_replacement,
        constant_value,
        "2026.1",
    )


@pytest.mark.parametrize("entity_feature", list(light.LightEntityFeature))
def test_deprecated_support_light_constants_enums(
    caplog: pytest.LogCaptureFixture, entity_feature: light.LightEntityFeature
) -> None:
    """Test deprecated support light constants."""
    import_and_test_deprecated_constant_enum(
        caplog, light, entity_feature, "SUPPORT_", "2026.1"
    )


@pytest.mark.parametrize("entity_feature", list(light.ColorMode))
def test_deprecated_color_mode_constants_enums(
    caplog: pytest.LogCaptureFixture, entity_feature: light.ColorMode
) -> None:
    """Test deprecated support light constants."""
    import_and_test_deprecated_constant_enum(
        caplog, light, entity_feature, "COLOR_MODE_", "2026.1"
    )


async def test_deprecated_turn_on_arguments(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test color temp conversion in service calls."""
    entity: MockLight = MockLight("Test_ct", STATE_ON, {light.ColorMode.COLOR_TEMP})
    setup_test_component_platform(
        hass, light.DOMAIN, [entity]
    )
    assert await async_setup_component(
        hass, light.DOMAIN, {light.DOMAIN: {"platform": "test"}}
    )
    await hass.async_block_till_done()
    state = hass.states.get(entity.entity_id)
    assert state.attributes["supported_color_modes"] == [
        light.ColorMode.COLOR_TEMP
    ]
    caplog.clear()
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity.entity_id],
            "color_temp": 200,
        },
        blocking=True,
    )
    assert "Got `color_temp` argument in `turn_on` service" in caplog.text
    _, data = entity.last_call("turn_on")
    assert data == {
        "color_temp": 200,
        "color_temp_kelvin": 5000,
    }
    caplog.clear()
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity.entity_id],
            "kelvin": 5000,
        },
        blocking=True,
    )
    assert "Got `kelvin` argument in `turn_on` service" in caplog.text
    _, data = entity.last_call("turn_on")
    assert data == {
        "color_temp": 200,
        "color_temp_kelvin": 5000,
    }
    caplog.clear()
    await hass.services.async_call(
        "light",
        "turn_on",
        {
            "entity_id": [entity.entity_id],
            "color_temp_kelvin": 5000,
        },
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data == {
        "color_temp": 200,
        "color_temp_kelvin": 5000,
    }
    assert "argument in `turn_on` service" not in caplog.text
