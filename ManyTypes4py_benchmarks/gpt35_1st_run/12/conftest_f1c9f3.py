from __future__ import annotations
import asyncio
from asyncio import Event
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from aioesphomeapi import APIClient, APIVersion, BluetoothProxyFeature, DeviceInfo, EntityInfo, EntityState, HomeassistantServiceCall, ReconnectLogic, UserService, VoiceAssistantAnnounceFinished, VoiceAssistantAudioSettings, VoiceAssistantFeature
import pytest
from zeroconf import Zeroconf
from homeassistant.components.esphome import dashboard
from homeassistant.components.esphome.const import CONF_ALLOW_SERVICE_CALLS, CONF_DEVICE_NAME, CONF_NOISE_PSK, DEFAULT_NEW_CONFIG_ALLOW_ALLOW_SERVICE_CALLS, DOMAIN
from homeassistant.const import CONF_HOST, CONF_PASSWORD, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from . import DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_SLUG
from tests.common import MockConfigEntry
_ONE_SECOND: int = 16000 * 2

def mock_bluetooth(enable_bluetooth: Any) -> None:
    """Auto mock bluetooth."""

def esphome_mock_async_zeroconf(mock_async_zeroconf: Any) -> None:
    """Auto mock zeroconf."""

async def load_homeassistant(hass: HomeAssistant) -> None:
    """Load the homeassistant integration."""
    assert await async_setup_component(hass, 'homeassistant', {})

def mock_tts(mock_tts_cache_dir: Any) -> None:
    """Auto mock the tts cache."""

def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return the default mocked config entry."""
    ...

class BaseMockReconnectLogic(ReconnectLogic):
    """Mock ReconnectLogic."""
    ...

def mock_device_info() -> DeviceInfo:
    """Return the default mocked device info."""
    ...

async def init_integration(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> MockConfigEntry:
    """Set up the ESPHome integration for testing."""
    ...

def mock_client(mock_device_info: DeviceInfo) -> APIClient:
    """Mock APIClient."""
    ...

async def mock_dashboard(hass: HomeAssistant) -> dict:
    """Mock dashboard."""
    ...

class MockESPHomeDevice:
    """Mock an esphome device."""
    ...

async def _mock_generic_device_entry(hass: HomeAssistant, mock_client: APIClient, mock_device_info: DeviceInfo, mock_list_entities_services: tuple, states: list, entry: MockConfigEntry = None, hass_storage: dict = None) -> MockESPHomeDevice:
    ...

async def mock_voice_assistant_entry(hass: HomeAssistant, mock_client: APIClient) -> Callable:
    """Set up an ESPHome entry with voice assistant."""
    ...

async def mock_voice_assistant_v1_entry(mock_voice_assistant_entry: Callable) -> MockConfigEntry:
    """Set up an ESPHome entry with voice assistant."""
    ...

async def mock_voice_assistant_v2_entry(mock_voice_assistant_entry: Callable) -> MockConfigEntry:
    """Set up an ESPHome entry with voice assistant."""
    ...

async def mock_voice_assistant_api_entry(mock_voice_assistant_entry: Callable) -> MockConfigEntry:
    """Set up an ESPHome entry with voice assistant."""
    ...

async def mock_bluetooth_entry(hass: HomeAssistant, mock_client: APIClient) -> Callable:
    """Set up an ESPHome entry with bluetooth."""
    ...

async def mock_bluetooth_entry_with_raw_adv(mock_bluetooth_entry: Callable) -> MockConfigEntry:
    """Set up an ESPHome entry with bluetooth and raw advertisements."""
    ...

async def mock_bluetooth_entry_with_legacy_adv(mock_bluetooth_entry: Callable) -> MockConfigEntry:
    """Set up an ESPHome entry with bluetooth with legacy advertisements."""
    ...

async def mock_generic_device_entry(hass: HomeAssistant, hass_storage: dict) -> Callable:
    """Set up an ESPHome entry and return the MockConfigEntry."""
    ...

async def mock_esphome_device(hass: HomeAssistant, hass_storage: dict) -> Callable:
    """Set up an ESPHome entry and return the MockESPHomeDevice."""
    ...
