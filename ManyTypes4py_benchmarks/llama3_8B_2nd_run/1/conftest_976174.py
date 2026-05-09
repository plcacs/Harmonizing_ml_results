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

@pytest.fixture(name: 'controller', 
                autouse: True) -> Generator[AsyncMock, None, None]
def mock_controller(mock_button, mock_relay, mock_temperature, mock_select, mock_buttoncounter, mock_sensornumber, mock_lightsensor, mock_dimmer, mock_module_no_subdevices, mock_module_subdevices, mock_cover, mock_cover_no_position) -> AsyncMock:
    """Mock a successful velbus controller."""
    with patch('homeassistant.components.velbus.Velbus', autospec=True) as controller, patch('homeassistant.components.velbus.config_flow.velbusaio.controller.Velbus', new=controller):
        cont = controller.return_value
        cont.get_all_binary_sensor.return_value = [mock_button]
        cont.get_all_button.return_value = [mock_button]
        cont.get_all_switch.return_value = [mock_relay]
        cont.get_all_climate.return_value = [mock_temperature]
        cont.get_all_select.return_value = [mock_select]
        cont.get_all_sensor.return_value = [mock_buttoncounter, mock_temperature, mock_sensornumber, mock_lightsensor]
        cont.get_all_light.return_value = [mock_dimmer]
        cont.get_all_led.return_value = [mock_button]
        cont.get_all_cover.return_value = [mock_cover, mock_cover_no_position]
        cont.get_modules.return_value = {1: mock_module_no_subdevices, 2: mock_module_no_subdevices, 3: mock_module_no_subdevices, 4: mock_module_no_subdevices, 99: mock_module_subdevices}
        cont.get_module.return_value = mock_module_subdevices
        yield controller

@pytest.fixture
def mock_module_no_subdevices(mock_relay: AsyncMock) -> Module:
    """Mock a velbus module."""
    module = AsyncMock(spec=Module)
    module.get_type_name.return_value = 'VMB4RYLD'
    module.get_addresses.return_value = [1, 2, 3, 4]
    module.get_name.return_value = 'BedRoom'
    module.get_sw_version.return_value = '1.0.0'
    module.is_loaded.return_value = True
    module.get_channels.return_value = {}
    return module

@pytest.fixture
def mock_module_subdevices() -> Module:
    """Mock a velbus module."""
    module = AsyncMock(spec=Module)
    module.get_type_name.return_value = 'VMB2BLE'
    module.get_addresses.return_value = [88]
    module.get_name.return_value = 'Kitchen'
    module.get_sw_version.return_value = '2.0.0'
    module.is_loaded.return_value = True
    module.get_channels.return_value = {}
    return module

@pytest.fixture
def mock_button() -> Button:
    """Mock a successful velbus channel."""
    channel = AsyncMock(spec=Button)
    channel.get_categories.return_value = ['binary_sensor', 'led', 'button']
    channel.get_name.return_value = 'ButtonOn'
    channel.get_module_address.return_value = 1
    channel.get_channel_number.return_value = 1
    channel.get_module_type_name.return_value = 'VMB4RYLD'
    channel.get_module_type.return_value = 99
    channel.get_full_name.return_value = 'Bedroom kid 1'
    channel.get_module_sw_version.return_value = '1.0.0'
    channel.get_module_serial.return_value = 'a1b2c3d4e5f6'
    channel.is_sub_device.return_value = False
    channel.is_closed.return_value = True
    channel.is_on.return_value = False
    return channel

# ... and so on for the other fixtures
