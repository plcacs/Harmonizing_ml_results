"""The tests for the Light component."""
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
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

orig_Profiles = light.Profiles

async def test_methods(hass: HomeAssistant) -> None:
    """Test if methods call the services as expected."""
    hass.states.async_set('light.test', STATE_ON)
    assert light.is_on(hass, 'light.test')
    hass.states.async_set('light.test', STATE_OFF)
    assert not light.is_on(hass, 'light.test')
    
    turn_on_calls = async_mock_service(hass, light.DOMAIN, SERVICE_TURN_ON)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: 'entity_id_val',
            light.ATTR_TRANSITION: 'transition_val',
            light.ATTR_BRIGHTNESS: 'brightness_val',
            light.ATTR_RGB_COLOR: 'rgb_color_val',
            light.ATTR_XY_COLOR: 'xy_color_val',
            light.ATTR_PROFILE: 'profile_val',
            light.ATTR_COLOR_NAME: 'color_name_val'
        },
        blocking=True
    )
    assert len(turn_on_calls) == 1
    call = turn_on_calls[-1]
    assert call.domain == light.DOMAIN
    assert call.service == SERVICE_TURN_ON
    assert call.data.get(ATTR_ENTITY_ID) == 'entity_id_val'
    assert call.data.get(light.ATTR_TRANSITION) == 'transition_val'
    assert call.data.get(light.ATTR_BRIGHTNESS) == 'brightness_val'
    assert call.data.get(light.ATTR_RGB_COLOR) == 'rgb_color_val'
    assert call.data.get(light.ATTR_XY_COLOR) == 'xy_color_val'
    assert call.data.get(light.ATTR_PROFILE) == 'profile_val'
    assert call.data.get(light.ATTR_COLOR_NAME) == 'color_name_val'

    turn_off_calls = async_mock_service(hass, light.DOMAIN, SERVICE_TURN_OFF)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_OFF,
        {
            ATTR_ENTITY_ID: 'entity_id_val',
            light.ATTR_TRANSITION: 'transition_val'
        },
        blocking=True
    )
    assert len(turn_off_calls) == 1
    call = turn_off_calls[-1]
    assert call.domain == light.DOMAIN
    assert call.service == SERVICE_TURN_OFF
    assert call.data[ATTR_ENTITY_ID] == 'entity_id_val'
    assert call.data[light.ATTR_TRANSITION] == 'transition_val'

    toggle_calls = async_mock_service(hass, light.DOMAIN, SERVICE_TOGGLE)
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {
            ATTR_ENTITY_ID: 'entity_id_val',
            light.ATTR_TRANSITION: 'transition_val'
        },
        blocking=True
    )
    assert len(toggle_calls) == 1
    call = toggle_calls[-1]
    assert call.domain == light.DOMAIN
    assert call.service == SERVICE_TOGGLE
    assert call.data[ATTR_ENTITY_ID] == 'entity_id_val'
    assert call.data[light.ATTR_TRANSITION] == 'transition_val'

async def test_services(
    hass: HomeAssistant,
    mock_light_profiles: Dict[str, light.Profile],
    mock_light_entities: List[MockLight]
) -> None:
    """Test the provided services."""
    setup_test_component_platform(hass, light.DOMAIN, mock_light_entities)
    assert await async_setup_component(hass, light.DOMAIN, {light.DOMAIN: {CONF_PLATFORM: 'test'}})
    await hass.async_block_till_done()
    
    ent1, ent2, ent3 = mock_light_entities
    ent1.supported_color_modes = [light.ColorMode.HS]
    ent3.supported_color_modes = [light.ColorMode.HS]
    ent1.supported_features = light.LightEntityFeature.TRANSITION
    ent2.supported_features = light.SUPPORT_COLOR | light.LightEntityFeature.EFFECT | light.LightEntityFeature.TRANSITION
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
        blocking=True
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ent2.entity_id},
        blocking=True
    )
    
    assert not light.is_on(hass, ent1.entity_id)
    assert light.is_on(hass, ent2.entity_id)
    
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True
    )
    assert light.is_on(hass, ent1.entity_id)
    assert light.is_on(hass, ent2.entity_id)
    assert light.is_on(hass, ent3.entity_id)
    
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_OFF,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True
    )
    assert not light.is_on(hass, ent1.entity_id)
    assert not light.is_on(hass, ent2.entity_id)
    assert not light.is_on(hass, ent3.entity_id)
    
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL, light.ATTR_BRIGHTNESS: 0},
        blocking=True
    )
    assert not light.is_on(hass, ent1.entity_id)
    assert not light.is_on(hass, ent2.entity_id)
    assert not light.is_on(hass, ent3.entity_id)
    
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True
    )
    assert light.is_on(hass, ent1.entity_id)
    assert light.is_on(hass, ent2.entity_id)
    assert light.is_on(hass, ent3.entity_id)
    
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TOGGLE,
        {ATTR_ENTITY_ID: ENTITY_MATCH_ALL},
        blocking=True
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
            light.ATTR_COLOR_NAME: 'blue'
        },
        blocking=True
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent2.entity_id,
            light.ATTR_EFFECT: 'fun_effect',
            light.ATTR_RGB_COLOR: (255, 255, 255)
        },
        blocking=True
    )
    await hass.services.async_call(
        light.DOMAIN,
        SERVICE_TURN_ON,
        {
            ATTR_ENTITY_ID: ent3.entity_id,
            light.ATTR_FLASH: 'short',
            light.ATTR_XY_COLOR: (0.4, 0.6)
        },
        blocking=True
    )
    
    _, data = ent1.last_call('turn_on')
    assert data == {
        light.ATTR_TRANSITION: 10,
        light.ATTR_BRIGHTNESS: 20,
        light.ATTR_HS_COLOR: (240, 100)
    }
    
    _, data = ent2.last_call('turn_on')
    assert data == {
        light.ATTR_EFFECT: 'fun_effect',
        light.ATTR_HS_COLOR: (0, 0)
    }
    
    _, data = ent3.last_call('turn_on')
    assert data == {
        light.ATTR_FLASH: 'short',
        light.ATTR_HS_COLOR: (71.059, 100)
    }

# ... (rest of the test functions with similar type annotations)
