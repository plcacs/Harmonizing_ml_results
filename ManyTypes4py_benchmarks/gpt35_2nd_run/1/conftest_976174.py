from collections.abc import Generator
from unittest.mock import AsyncMock, patch
import pytest
from velbusaio.channels import Blind, Button, ButtonCounter, Dimmer, LightSensor, Relay, SelectedProgram, SensorNumber, Temperature
from velbusaio.module import Module
from homeassistant.components.velbus import VelbusConfigEntry
from homeassistant.components.velbus.const import DOMAIN
from homeassistant.const import CONF_NAME, CONF_PORT
from homeassistant.core import HomeAssistant
from .const import PORT_TCP
from tests.common import MockConfigEntry

@pytest.fixture(name='controller')
def mock_controller(mock_button: AsyncMock, mock_relay: AsyncMock, mock_temperature: AsyncMock, mock_select: AsyncMock, mock_buttoncounter: AsyncMock, mock_sensornumber: AsyncMock, mock_lightsensor: AsyncMock, mock_dimmer: AsyncMock, mock_module_no_subdevices: AsyncMock, mock_module_subdevices: AsyncMock, mock_cover: AsyncMock, mock_cover_no_position: AsyncMock) -> Generator:
    ...

@pytest.fixture
def mock_module_no_subdevices(mock_relay: AsyncMock) -> AsyncMock:
    ...

@pytest.fixture
def mock_module_subdevices() -> AsyncMock:
    ...

@pytest.fixture
def mock_button() -> AsyncMock:
    ...

@pytest.fixture
def mock_temperature() -> AsyncMock:
    ...

@pytest.fixture
def mock_relay() -> AsyncMock:
    ...

@pytest.fixture
def mock_select() -> AsyncMock:
    ...

@pytest.fixture
def mock_buttoncounter() -> AsyncMock:
    ...

@pytest.fixture
def mock_sensornumber() -> AsyncMock:
    ...

@pytest.fixture
def mock_lightsensor() -> AsyncMock:
    ...

@pytest.fixture
def mock_dimmer() -> AsyncMock:
    ...

@pytest.fixture
def mock_cover() -> AsyncMock:
    ...

@pytest.fixture
def mock_cover_no_position() -> AsyncMock:
    ...

@pytest.fixture(name='config_entry')
async def mock_config_entry(hass: HomeAssistant, controller: AsyncMock) -> MockConfigEntry:
    ...
