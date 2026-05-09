"""The tests for the Light component."""
from types import ModuleType
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
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
from homeassistant.core import HomeAssistant
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


@pytest.mark.asyncio
async def test_methods(hass: HomeAssistant) -> None:
    """Test if methods call the services as expected."""
    hass.states.async_set("light.test", STATE_ON)
    assert light.is_on(hass, "light.test")
    hass.states.async_set("light.test", STATE_OFF)
    assert not light.is_on(hass, "light.test")
    turn_on_calls = async_mock_service(hass, light.DOMAIN, SERVICE_TURN_ON)
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
    turn_off_calls = async_mock_service(hass, light.DOMAIN, SERVICE_TURN_OFF)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_OFF,
        {ATTR_ENTITY_ID: "entity_id_val", light.ATTR_TRANSITION: "transition_val"},
        blocking=True,
    )
    assert len(turn_off_calls) == 1
    call = turn_off_calls[-1]
    assert call.domain == light.DOMAIN
    assert call.service == SERVICE_TURN_OFF
    assert call.data[ATTR_ENTITY_ID] == "entity_id_val"
    assert call.data[light.ATTR_TRANSITION] == "transition_val"
    toggle_calls = async_mock_service(hass, light.DOMAIN, SERVICE_TOGGLE)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {ATTR_ENTITY_ID: "entity_id_val", light.ATTR_TRANSITION: "transition_val"},
        blocking=True,
    )
    assert len(toggle_calls) == 1
    call = toggle_calls[-1]
    assert call.domain == light.DOMAIN
    assert call.service == SERVICE_TOGGLE
    assert call.data[ATTR_ENTITY_ID] == "entity_id_val"
    assert call.data[light.ATTR_TRANSITION] == "transition_val"


@pytest.mark.asyncio
async def test_services(
    hass: HomeAssistant,
    mock_light_profiles: Dict,
    mock_light_entities: List[MockLight],
) -> None:
    """Test the provided services."""
    setup_test_component_platform(hass, light.DOMAIN, mock_light_entities)
    assert await async_setup_component(hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}})
    await hass.async_block_till_done()
    ent1, ent2, ent3 = mock_light_entities
    ent1.supported_color_modes = [light.ColorMode.HS]
    ent3.supported_color_modes = [light.ColorMode.HS]
    ent1.supported_features = light.LightEntityFeature.TRANSITION
    ent2.supported_features = light.LightEntityFeature.EFFECT | light.LightEntityFeature.TRANSITION
    ent2.supported_color_modes = None
    ent2.color_mode = None
    ent3.supported_features = light.LightEntityFeature.FLASH | light.LightEntityFeature.TRANSITION
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
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL, light.ATTR_BRIGHTNESS: 0},
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
    assert data == {light.ATTR_TRANSITION: 10, light.ATTR_BRIGHTNESS: 20, light.ATTR_HS_COLOR: (240, 100)}
    _, data = ent2.last_call("turn_on")
    assert data == {light.ATTR_EFFECT: "fun_effect", light.ATTR_HS_COLOR: (0, 0)}
    _, data = ent3.last_call("turn_on")
    assert data == {light.ATTR_FLASH: "short", light.ATTR_HS_COLOR: (71.059, 100)}
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
        {ATTR_ENTITY_ID: ent1.entity_id, light.ATTR_PROFILE: profile.name},
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
    assert data == {light.ATTR_BRIGHTNESS: profile.brightness, light.ATTR_HS_COLOR: profile.hs_color, light.ATTR_TRANSITION: profile.transition}
    _, data = ent2.last_call("turn_on")
    assert data == {light.ATTR_BRIGHTNESS: 100, light.ATTR_HS_COLOR: profile.hs_color, light.ATTR_TRANSITION: 1}
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
    assert data == {light.ATTR_BRIGHTNESS: 255, light.ATTR_HS_COLOR: profile.hs_color, light.ATTR_TRANSITION: profile.transition}
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {ATTR_ENTITY_ID: ent3.entity_id, light.ATTR_TRANSITION: 4},
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
        {ATTR_ENTITY_ID: ent1.entity_id, light.ATTR_PROFILE: -1},
        blocking=True,
    )
    with pytest.raises(vol.MultipleInvalid):
        await hass.services.async_call(
            light.DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: ent2.entity_id, light.ATTR_XY_COLOR: ["bla-di-bla", 5]},
            blocking=True,
        )
    with pytest.raises(vol.MultipleInvalid):
        await hass.services.async_call(
            light.DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: ent3.entity_id, light.ATTR_RGB_COLOR: [255, None, 2]},
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
            {ATTR_ENTITY_ID: ent1.entity_id, light.ATTR_PROFILE: profile.name, light.ATTR_BRIGHTNESS: "bright"},
            blocking=True,
        )
    with pytest.raises(vol.MultipleInvalid):
        await hass.services.async_call(
            light.DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: ent1.entity_id, light.ATTR_RGB_COLOR: "yellowish"},
            blocking=True,
        )
    _, data = ent1.last_call("turn_on")
    assert data == {}
    _, data = ent2.last_call("turn_on")
    assert data == {}


@pytest.mark.asyncio
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
    ids=[
        "test",
        "color_no_brightness_no_transition",
        "no_color",
        "test_off",
        "no_brightness",
        "color_and_brightness",
        "color_and_transition",
        "brightness_and_transition",
    ],
)
async def test_light_profiles(
    hass: HomeAssistant,
    mock_light_profiles: Dict,
    profile_name: str,
    expected_data: Dict,
    last_call: str,
    mock_light_entities: List[MockLight],
) -> None:
    """Test light profiles."""
    setup_test_component_platform(hass, light.DOMAIN, mock_light_entities)
    profile_mock_data = {
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
    assert await async_setup_component(hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}})
    await hass.async_block_till_done()
    ent1, _, _ = mock_light_entities
    ent1.supported_color_modes = [light.ColorMode.HS]
    ent1.supported_features = light.LightEntityFeature.TRANSITION
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ent1.entity_id, light.ATTR_PROFILE: profile_name},
        blocking=True,
    )
    _, data = ent1.last_call(last_call)
    if last_call == "turn_on":
        assert light.is_on(hass, ent1.entity_id)
    else:
        assert not light.is_on(hass, ent1.entity_id)
    assert data == expected_data


@pytest.mark.asyncio
async def test_default_profiles_group(
    hass: HomeAssistant,
    mock_light_profiles: Dict,
    mock_light_entities: List[MockLight],
) -> None:
    """Test default turn-on light profile for all lights."""
    setup_test_component_platform(hass, light.DOMAIN, mock_light_entities)
    assert await async_setup_component(hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}})
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
    assert data == {light.ATTR_HS_COLOR: (71.059, 100), light.ATTR_BRIGHTNESS: 99, light.ATTR_TRANSITION: 2}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_call_params", "expected_params_state_was_off", "expected_params_state_was_on"),
    [
        ({}, {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 100, light.ATTR_TRANSITION: 3}, {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 100, light.ATTR_TRANSITION: 3}),
        ({light.ATTR_BRIGHTNESS: 22}, {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 22, light.ATTR_TRANSITION: 3}, {light.ATTR_BRIGHTNESS: 22, light.ATTR_TRANSITION: 3}),
        ({light.ATTR_TRANSITION: 22}, {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 100, light.ATTR_TRANSITION: 22}, {light.ATTR_TRANSITION: 22}),
        ({light.ATTR_COLOR_TEMP: 600, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_COLOR_TEMP: 600, light.ATTR_COLOR_TEMP_KELVIN: 1666, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_COLOR_TEMP: 600, light.ATTR_COLOR_TEMP_KELVIN: 1666, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}),
        ({light.ATTR_COLOR_TEMP_KELVIN: 6500, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_COLOR_TEMP: 153, light.ATTR_COLOR_TEMP_KELVIN: 6500, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_COLOR_TEMP: 153, light.ATTR_COLOR_TEMP_KELVIN: 6500, light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}),
        ({light.ATTR_HS_COLOR: [70, 80], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_HS_COLOR: (70, 80), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_HS_COLOR: (70, 80), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}),
        ({light.ATTR_RGB_COLOR: [1, 2, 3], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_RGB_COLOR: (1, 2, 3), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_RGB_COLOR: (1, 2, 3), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}),
        ({light.ATTR_RGBW_COLOR: [1, 2, 3, 4], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_RGBW_COLOR: (1, 2, 3, 4), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_RGBW_COLOR: (1, 2, 3, 4), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}),
        ({light.ATTR_RGBWW_COLOR: [1, 2, 3, 4, 5], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_RGBWW_COLOR: (1, 2, 3, 4, 5), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_RGBWW_COLOR: (1, 2, 3, 4, 5), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}),
        ({light.ATTR_XY_COLOR: [0.4448, 0.4066], light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_XY_COLOR: (0.4448, 0.4066), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_XY_COLOR: (0.4448, 0.4066), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}),
        ({light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_HS_COLOR: (50.353, 100), light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}, {light.ATTR_BRIGHTNESS: 11, light.ATTR_TRANSITION: 1}),
    ],
    ids=[
        "no_params",
        "brightness_22",
        "transition_22",
        "color_temp_600",
        "color_temp_kelvin_6500",
        "hs_color_70_80",
        "rgb_color_1_2_3",
        "rgbw_color_1_2_3_4",
        "rgbww_color_1_2_3_4_5",
        "xy_color_0.4448_0.4066",
        "brightness_11_transition_1",
    ],
)
async def test_default_profiles_light(
    hass: HomeAssistant,
    mock_light_profiles: Dict,
    extra_call_params: Dict,
    expected_params_state_was_off: Dict,
    expected_params_state_was_on: Dict,
    mock_light_entities: List[MockLight],
) -> None:
    """Test default turn-on light profile for a specific light."""
    setup_test_component_platform(hass, light.DOMAIN, mock_light_entities)
    assert await async_setup_component(hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: "test"}})
    await hass.async_block_till_done()
    profile = light.Profile("group.all_lights.default", 0.3, 0.5, 200, 0)
    mock_light_profiles[profile.name] = profile
    profile = light.Profile("light.ceiling_2.default", 0.6, 0.6, 100, 3)
    mock_light_profiles[profile.name] = profile
    dev = next(filter(lambda x: x.entity_id == "light.ceiling_2", mock_light_entities))
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
        {ATTR_ENTITY_ID: dev.entity_id, **extra_call_params},
        blocking=True,
    )
    _, data = dev.last_call("turn_on")
    assert data == expected_params_state_was_off
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: dev.entity_id, **extra_call_params},
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


@pytest.mark.asyncio
async def test_light_context(
    hass: HomeAssistant,
    hass_admin_user: MockUser,
    mock_light_entities: List[MockLight],
) -> None:
    """Test that light context works."""
    setup_test_component_platform(hass, light.DOMAIN, mock_light_entities)
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    await hass.async_block_till_done()
    state = hass.states.get("light.ceiling")
    assert state is not None
    await hass.services.async_call(
        "light",
        "toggle",
        {ATTR_ENTITY_ID: state.entity_id},
        blocking=True,
        context=core.Context(user_id=hass_admin_user.id),
    )
    state2 = hass.states.get("light.ceiling")
    assert state2 is not None
    assert state.state != state2.state
    assert state2.context.user_id == hass_admin_user.id


@pytest.mark.asyncio
async def test_light_turn_on_auth(
    hass: HomeAssistant,
    hass_read_only_user: MockUser,
    mock_light_entities: List[MockLight],
) -> None:
    """Test that light context works."""
    setup_test_component_platform(hass, light.DOMAIN, mock_light_entities)
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    await hass.async_block_till_done()
    state = hass.states.get("light.ceiling")
    assert state is not None
    hass_read_only_user.mock_policy({})
    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            "light",
            "turn_on",
            {ATTR_ENTITY_ID: state.entity_id},
            blocking=True,
            context=core.Context(user_id=hass_read_only_user.id),
        )


@pytest.mark.asyncio
async def test_light_brightness_step(hass: HomeAssistant) -> None:
    """Test that light context works."""
    entities = [MockLight("Test_0", STATE_ON), MockLight("Test_1", STATE_ON)]
    setup_test_component_platform(hass, light.DOMAIN, entities)
    entity0 = entities[0]
    entity0.supported_features = light.SUPPORT_BRIGHTNESS
    entity0.supported_color_modes = None
    entity0.color_mode = None
    entity0.brightness = 100
    entity1 = entities[1]
    entity1.supported_features = light.SUPPORT_BRIGHTNESS
    entity1.supported_color_modes = None
    entity1.color_mode = None
    entity1.brightness = 50
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
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
        {ATTR_ENTITY_ID: [entity0.entity_id, entity1.entity_id], "brightness_step": -10},
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data["brightness"] == 90
    _, data = entity1.last_call("turn_on")
    assert data["brightness"] == 40
    await hass.services.async_call(
        "light",
        "turn_on",
        {ATTR_ENTITY_ID: [entity0.entity_id, entity1.entity_id], "brightness_step_pct": 10},
        blocking=True,
    )
    _, data = entity0.last_call("turn_on")
    assert data["brightness"] == 116
    _, data = entity1.last_call("turn_on")
    assert data["brightness"] == 66
    await hass.services.async_call(
        "light",
        "turn_on",
        {ATTR_ENTITY_ID: entity0.entity_id, "brightness_step": -126},
        blocking=True,
    )
    assert entity0.state == "off"


@pytest.mark.asyncio
@pytest.mark.usefixtures("enable_custom_integrations")
async def test_light_brightness_pct_conversion(
    hass: HomeAssistant,
    mock_light_entities: List[MockLight],
) -> None:
    """Test that light brightness percent conversion."""
    setup_test_component_platform(hass, light.DOMAIN, mock_light_entities)
    entity = mock_light_entities[0]
    entity.supported_features = light.SUPPORT_BRIGHTNESS
    entity.supported_color_modes = None
    entity.color_mode = None
    entity.brightness = 100
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    await hass.async_block_till_done()
    state = hass.states.get(entity.entity_id)
    assert state is not None
    assert state.attributes["brightness"] == 100
    await hass.services.async_call(
        "light",
        "turn_on",
        {ATTR_ENTITY_ID: entity.entity_id, "brightness_pct": 1},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 3
    await hass.services.async_call(
        "light",
        "turn_on",
        {ATTR_ENTITY_ID: entity.entity_id, "brightness_pct": 2},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 5
    await hass.services.async_call(
        "light",
        "turn_on",
        {ATTR_ENTITY_ID: entity.entity_id, "brightness_pct": 50},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 128
    await hass.services.async_call(
        "light",
        "turn_on",
        {ATTR_ENTITY_ID: entity.entity_id, "brightness_pct": 99},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 252
    await hass.services.async_call(
        "light",
        "turn_on",
        {ATTR_ENTITY_ID: entity.entity_id, "brightness_pct": 100},
        blocking=True,
    )
    _, data = entity.last_call("turn_on")
    assert data["brightness"] == 255


@pytest.mark.asyncio
async def test_profiles(hass: HomeAssistant) -> None:
    """Test profiles loading."""
    profiles = orig_Profiles(hass)
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


@pytest.mark.asyncio
@patch("os.path.isfile", MagicMock(side_effect=(True, False)))
async def test_profile_load_optional_hs_color(hass: HomeAssistant) -> None:
    """Test profile loading with profiles containing no xy color."""
    csv_file = (
        "the first line is skipped\nno_color,,,100,1\nno_color_no_transition,,,110\ncolor,0.5119,0.4147,120,2\ncolor_no_transition,0.4448,0.4066,130\ncolor_and_brightness,0.4448,0.4066,170,\nonly_brightness,,,140\nonly_transition,,,,150\ntransition_float,,,,1.6\ninvalid_profile_1,\ninvalid_color_2,,0.1,1,2\ninvalid_color_3,,0.1,1\ninvalid_color_4,0.1,,1,3\ninvalid_color_5,0.1,,1\ninvalid_brightness,0,0,256,4\ninvalid_brightness_2,0,0,256\ninvalid_no_brightness_no_color_no_transition,,,\n"
    )
    profiles = orig_Profiles(hass)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("light_state", [STATE_ON, STATE_OFF])
async def test_light_backwards_compatibility_supported_color_modes(
    hass: HomeAssistant, light_state: str
) -> None:
    """Test supported_color_modes if not implemented by the entity."""
    entities = [
        MockLight("Test_0", light_state),
        MockLight("Test_1", light_state),
        MockLight("Test_2", light_state),
        MockLight("Test_3", light_state),
        MockLight("Test_4", light_state),
    ]
    entity0 = entities[0]
    entity1 = entities[1]
    entity1.supported_features = light.SUPPORT_BRIGHTNESS
    entity1.supported_color_modes = None
    entity1.color_mode = None
    entity2 = entities[2]
    entity2.supported_features = light.SUPPORT_BRIGHTNESS | light.SUPPORT_COLOR_TEMP
    entity2.supported_color_modes = None
    entity2.color_mode = None
    entity3 = entities[3]
    entity3.supported_features = light.SUPPORT_BRIGHTNESS | light.SUPPORT_COLOR
    entity3.supported_color_modes = None
    entity3.color_mode = None
    entity4 = entities[4]
    entity4.supported_features = light.SUPPORT_BRIGHTNESS | light.SUPPORT_COLOR | light.SUPPORT_COLOR_TEMP
    entity4.supported_color_modes = None
    entity4.color_mode = None
    setup_test_component_platform(hass, light.DOMAIN, entities)
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    await hass.async_block_till_done()
    state = hass.states.get(entity0.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.ONOFF]
    if light_state == STATE_OFF:
        assert state.attributes["color_mode"] is None
    else:
        assert state.attributes["color_mode"] == light.ColorMode.ONOFF
    state = hass.states.get(entity1.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.BRIGHTNESS]
    if light_state == STATE_OFF:
        assert state.attributes["color_mode"] is None
    else:
        assert state.attributes["color_mode"] == light.ColorMode.UNKNOWN
    state = hass.states.get(entity2.entity_id)
    assert state.attributes["supported_color_modes"] == [light.ColorMode.COLOR_TEMP]
    if light_state == STATE_OFF:
        assert state.attributes["color_mode"] is None
    else:
        assert state.attributes["color_mode"] == light.ColorMode.UNKNOWN