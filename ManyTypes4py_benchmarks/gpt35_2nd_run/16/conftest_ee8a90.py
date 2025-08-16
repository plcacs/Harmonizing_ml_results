from collections.abc import AsyncGenerator, Generator
import time
from typing import Any, List
from unittest.mock import MagicMock, patch
from myuplink import Device, DevicePoint, System
import orjson
import pytest
from homeassistant.components.application_credentials import ClientCredential, async_import_client_credential
from homeassistant.components.myuplink.const import DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_entry_oauth2_flow
from homeassistant.setup import async_setup_component
from homeassistant.util.json import json_loads
from .const import CLIENT_ID, CLIENT_SECRET, UNIQUE_ID
from tests.common import MockConfigEntry, load_fixture

@pytest.fixture(name='expires_at')
def mock_expires_at() -> float:
    return time.time() + 3600

@pytest.fixture
def mock_config_entry(hass, expires_at: float) -> MockConfigEntry:
    ...

@pytest.fixture(autouse=True)
async def setup_credentials(hass: HomeAssistant) -> None:
    ...

@pytest.fixture(scope='package')
def load_device_file() -> str:
    ...

@pytest.fixture
def device_fixture(load_device_file: str) -> Device:
    ...

@pytest.fixture
def load_systems_jv_file(load_systems_file: str) -> Any:
    ...

@pytest.fixture(scope='package')
def load_systems_file() -> str:
    ...

@pytest.fixture
def system_fixture(load_systems_file: str) -> List[System]:
    ...

@pytest.fixture
def load_device_points_file() -> str:
    ...

@pytest.fixture
def load_device_points_jv_file(load_device_points_file: str) -> str:
    ...

@pytest.fixture
def device_points_fixture(load_device_points_jv_file: str) -> List[DevicePoint]:
    ...

@pytest.fixture
def mock_myuplink_client(load_device_file: str, device_fixture: Device, load_device_points_jv_file: str, device_points_fixture: List[DevicePoint], system_fixture: List[System], load_systems_jv_file: Any) -> MagicMock:
    ...

@pytest.fixture
async def init_integration(hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_myuplink_client: MagicMock) -> MockConfigEntry:
    ...

@pytest.fixture
def platforms() -> List[str]:
    ...

@pytest.fixture
async def setup_platform(hass: HomeAssistant, mock_config_entry: MockConfigEntry, platforms: List[str]) -> None:
    ...

@pytest.fixture
async def access_token(hass: HomeAssistant) -> str:
    ...
