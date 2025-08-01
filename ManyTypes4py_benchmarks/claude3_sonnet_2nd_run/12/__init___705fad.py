"""Tests for the Canary integration."""
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, PropertyMock, patch

from canary.model import SensorType
from homeassistant.components.canary.const import CONF_FFMPEG_ARGUMENTS, DEFAULT_FFMPEG_ARGUMENTS, DEFAULT_TIMEOUT, DOMAIN
from homeassistant.const import CONF_PASSWORD, CONF_TIMEOUT, CONF_USERNAME
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry

ENTRY_CONFIG: Dict[str, str] = {CONF_PASSWORD: 'test-password', CONF_USERNAME: 'test-username'}
ENTRY_OPTIONS: Dict[str, Union[str, int]] = {CONF_FFMPEG_ARGUMENTS: DEFAULT_FFMPEG_ARGUMENTS, CONF_TIMEOUT: DEFAULT_TIMEOUT}
USER_INPUT: Dict[str, str] = {CONF_PASSWORD: 'test-password', CONF_USERNAME: 'test-username'}
YAML_CONFIG: Dict[str, Union[str, int]] = {CONF_PASSWORD: 'test-password', CONF_USERNAME: 'test-username', CONF_TIMEOUT: 5}


def _patch_async_setup(return_value: bool = True) -> Any:
    return patch('homeassistant.components.canary.async_setup', return_value=return_value)


def _patch_async_setup_entry(return_value: bool = True) -> Any:
    return patch('homeassistant.components.canary.async_setup_entry', return_value=return_value)


async def init_integration(hass: HomeAssistant, *, skip_entry_setup: bool = False) -> MockConfigEntry:
    """Set up the Canary integration in Home Assistant."""
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_CONFIG, options=ENTRY_OPTIONS)
    entry.add_to_hass(hass)
    if not skip_entry_setup:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


def mock_device(device_id: str, name: str, is_online: bool = True, device_type_name: Optional[str] = None) -> MagicMock:
    """Mock Canary Device class."""
    device = MagicMock()
    type(device).device_id = PropertyMock(return_value=device_id)
    type(device).name = PropertyMock(return_value=name)
    type(device).is_online = PropertyMock(return_value=is_online)
    type(device).device_type = PropertyMock(return_value={'id': 1, 'name': device_type_name})
    return device


def mock_location(
    location_id: str, 
    name: str, 
    is_celsius: bool = True, 
    devices: Optional[List[MagicMock]] = None, 
    mode: Optional[MagicMock] = None, 
    is_private: bool = False
) -> MagicMock:
    """Mock Canary Location class."""
    location = MagicMock()
    type(location).location_id = PropertyMock(return_value=location_id)
    type(location).name = PropertyMock(return_value=name)
    type(location).is_celsius = PropertyMock(return_value=is_celsius)
    type(location).is_private = PropertyMock(return_value=is_private)
    type(location).devices = PropertyMock(return_value=devices or [])
    type(location).mode = PropertyMock(return_value=mode)
    return location


def mock_mode(mode_id: str, name: str) -> MagicMock:
    """Mock Canary Mode class."""
    mode = MagicMock()
    type(mode).mode_id = PropertyMock(return_value=mode_id)
    type(mode).name = PropertyMock(return_value=name)
    type(mode).resource_url = PropertyMock(return_value=f'/v1/modes/{mode_id}')
    return mode


def mock_reading(sensor_type: int, sensor_value: Any) -> MagicMock:
    """Mock Canary Reading class."""
    reading = MagicMock()
    type(reading).sensor_type = SensorType(sensor_type)
    type(reading).value = PropertyMock(return_value=sensor_value)
    return reading
