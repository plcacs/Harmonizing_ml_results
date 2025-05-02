"""Fixtures for ViCare integration tests."""
from __future__ import annotations
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch
import pytest
from PyViCare.PyViCareDeviceConfig import PyViCareDeviceConfig
from PyViCare.PyViCareService import ViCareDeviceAccessor, readFeature
from homeassistant.components.vicare.const import DOMAIN
from homeassistant.core import HomeAssistant
from . import ENTRY_CONFIG, MODULE, setup_integration
from tests.common import MockConfigEntry, load_json_object_fixture

@dataclass
class Fixture:
    """Fixture representation with the assigned roles and dummy data location."""

class MockPyViCare:
    """Mocked PyVicare class based on a json dump."""

    def __init__(self, fixtures):
        """Init a single device from json dump."""
        self.devices = []
        for idx, fixture in enumerate(fixtures):
            self.devices.append(PyViCareDeviceConfig(MockViCareService(f'installation{idx}', f'gateway{idx}', f'device{idx}', fixture), f'deviceId{idx}', f'model{idx}', 'online'))

class MockViCareService:
    """PyVicareService mock using a json dump."""

    def __init__(self, installation_id, gateway_id, device_id, fixture):
        """Initialize the mock from a json dump."""
        self._test_data = load_json_object_fixture(fixture.data_file)
        self.fetch_all_features = Mock(return_value=self._test_data)
        self.roles = fixture.roles
        self.accessor = ViCareDeviceAccessor(installation_id, gateway_id, device_id)

    def hasRoles(self, requested_roles):
        """Return true if requested roles are assigned."""
        return requested_roles and set(requested_roles).issubset(self.roles)

    def getProperty(self, property_name):
        """Read a property from json dump."""
        return readFeature(self._test_data['data'], property_name)

@pytest.fixture
def mock_config_entry():
    """Return the default mocked config entry."""
    return MockConfigEntry(domain=DOMAIN, unique_id='ViCare', entry_id='1234', data=ENTRY_CONFIG)

@pytest.fixture
async def mock_vicare_gas_boiler(hass, mock_config_entry):
    """Return a mocked ViCare API representing a single gas boiler device."""
    fixtures = [Fixture({'type:boiler'}, 'vicare/Vitodens300W.json')]
    with patch(f'{MODULE}.login', return_value=MockPyViCare(fixtures)):
        await setup_integration(hass, mock_config_entry)
        yield mock_config_entry

@pytest.fixture
async def mock_vicare_room_sensors(hass, mock_config_entry):
    """Return a mocked ViCare API representing multiple room sensor devices."""
    fixtures = [Fixture({'type:climateSensor'}, 'vicare/RoomSensor1.json'), Fixture({'type:climateSensor'}, 'vicare/RoomSensor2.json')]
    with patch(f'{MODULE}.login', return_value=MockPyViCare(fixtures)):
        await setup_integration(hass, mock_config_entry)
        yield mock_config_entry

@pytest.fixture
def mock_setup_entry():
    """Mock setting up a config entry."""
    with patch(f'{MODULE}.async_setup_entry', return_value=True) as mock_setup_entry:
        yield mock_setup_entry