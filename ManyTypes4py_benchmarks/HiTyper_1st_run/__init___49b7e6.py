"""Tests for the Shelly integration."""
from collections.abc import Mapping
from copy import deepcopy
from datetime import timedelta
from typing import Any
from unittest.mock import Mock
from aioshelly.const import MODEL_25
from freezegun.api import FrozenDateTimeFactory
import pytest
from homeassistant.components.shelly.const import CONF_GEN, CONF_SLEEP_PERIOD, DOMAIN, REST_SENSORS_UPDATE_INTERVAL, RPC_SENSORS_POLLING_INTERVAL
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceEntry, DeviceRegistry, format_mac
from tests.common import MockConfigEntry, async_fire_time_changed
MOCK_MAC = '123456789ABC'

async def init_integration(hass, gen, model=MODEL_25, sleep_period=0, options=None, skip_setup=False):
    """Set up the Shelly integration in Home Assistant."""
    data = {CONF_HOST: '192.168.1.37', CONF_SLEEP_PERIOD: sleep_period, 'model': model}
    if gen is not None:
        data[CONF_GEN] = gen
    entry = MockConfigEntry(domain=DOMAIN, data=data, unique_id=MOCK_MAC, options=options)
    entry.add_to_hass(hass)
    if not skip_setup:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry

def mutate_rpc_device_status(monkeypatch: Union[str, dict[str, int], typing.Mapping], mock_rpc_device: Union[str, dict], top_level_key: Union[dict, dict[str, dict[str, typing.Any]], str], key: Union[dict, dict[str, dict[str, typing.Any]], str], value: Union[dict, dict[str, dict[str, typing.Any]], str]) -> None:
    """Mutate status for rpc device."""
    new_status = deepcopy(mock_rpc_device.status)
    new_status[top_level_key][key] = value
    monkeypatch.setattr(mock_rpc_device, 'status', new_status)

def inject_rpc_device_event(monkeypatch: dict[str, typing.Any], mock_rpc_device: Union[dict[str, typing.Any], unittesmock.Mock], event: dict[str, typing.Any]) -> None:
    """Inject event for rpc device."""
    monkeypatch.setattr(mock_rpc_device, 'event', event)
    mock_rpc_device.mock_event()

async def mock_rest_update(hass, freezer, seconds=REST_SENSORS_UPDATE_INTERVAL):
    """Move time to create REST sensors update event."""
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()

async def mock_polling_rpc_update(hass, freezer):
    """Move time to create polling RPC sensors update event."""
    freezer.tick(timedelta(seconds=RPC_SENSORS_POLLING_INTERVAL))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()

def register_entity(hass: Union[homeassistancore.HomeAssistant, homeassistanconfig_entries.ConfigEntry, str], domain: Union[homeassistancore.HomeAssistant, str], object_id: Union[homeassistancore.HomeAssistant, str], unique_id: Union[homeassistancore.HomeAssistant, str], config_entry: Union[None, homeassistancore.HomeAssistant, str]=None, capabilities: Union[None, homeassistancore.HomeAssistant, str]=None, device_id: Union[None, homeassistancore.HomeAssistant, str]=None) -> typing.Text:
    """Register enabled entity, return entity_id."""
    entity_registry = er.async_get(hass)
    entity_registry.async_get_or_create(domain, DOMAIN, f'{MOCK_MAC}-{unique_id}', suggested_object_id=object_id, disabled_by=None, config_entry=config_entry, capabilities=capabilities, device_id=device_id)
    return f'{domain}.{object_id}'

def get_entity(hass: Union[homeassistancore.HomeAssistant, str], domain: Union[str, homeassistancore.HomeAssistant], unique_id: Union[str, homeassistancore.HomeAssistant]):
    """Get Shelly entity."""
    entity_registry = er.async_get(hass)
    return entity_registry.async_get_entity_id(domain, DOMAIN, f'{MOCK_MAC}-{unique_id}')

def get_entity_state(hass: Union[homeassistancore.HomeAssistant, str], entity_id: Union[homeassistancore.HomeAssistant, str]):
    """Return entity state."""
    entity = hass.states.get(entity_id)
    assert entity
    return entity.state

def get_entity_attribute(hass: Union[str, homeassistancore.HomeAssistant, list[dict]], entity_id: Union[str, homeassistancore.HomeAssistant, list[dict]], attribute: Union[str, dict, int]):
    """Return entity attribute."""
    entity = hass.states.get(entity_id)
    assert entity
    return entity.attributes[attribute]

def register_device(device_registry: Union[str, dict, dict[str, typing.Any]], config_entry: Union[str, dict, dict[str, typing.Any]]):
    """Register Shelly device."""
    return device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(CONNECTION_NETWORK_MAC, format_mac(MOCK_MAC))})