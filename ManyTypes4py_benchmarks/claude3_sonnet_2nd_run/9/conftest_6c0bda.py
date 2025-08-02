"""Fixtures for ViCare integration tests."""
from __future__ import annotations
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
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
    roles: Set[str]
    data_file: str

class MockPyViCare:
    """Mocked PyVicare class based on a json dump."""

    def __init__(self, fixtures: List[Fixture]) -> None:
        """Init a single device from json dump."""
        self.devices: List[PyViCareDeviceConfig] = []
        for idx, fixture in enumerate(fixtures):
            self.devices.append(PyViCareDeviceConfig(MockViCareService(f'installation{idx}', f'gateway{idx}', f'device{idx}', fixture), f'deviceId{idx}', f'model{idx}', 'online'))

class MockViCareService:
    """PyVicareService mock using a json dump."""

    def __init__(self, installation_id: str, gateway_id: str, device_id: str, fixture: Fixture) -> None:
        """Initialize the mock from a json dump."""
        self._test_data: Dict[str, Any] = load_json_object_fixture(fixture.data_file)
        self.fetch_all_features: Mock = Mock(return_value=self._test_data)
        self.roles: Set[str] = fixture.roles
        self.accessor: ViCareDeviceAccessor = ViCareDeviceAccessor(installation_id, gateway_id, device_id)

    def hasRoles(self, requested_roles: Optional[List[str]]) -> bool:
        """Return true if requested roles are assigned."""
        return requested_roles is not None and set(requested_roles).issubset(self.roles)

    def getProperty(self, property_name: str) -> Any:
        """Read a property from json dump."""
        return readFeature(self._test_data['data'], property_name)

@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Return the default mocked config entry."""
    return MockConfigEntry(domain=DOMAIN, unique_id='ViCare', entry_id='1234', data=ENTRY_CONFIG)

@pytest.fixture
async def mock_vicare_gas_boiler(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> MockConfigEntry:
    """Return a mocked ViCare API representing a single gas boiler device."""
    fixtures: List[Fixture] = [Fixture({'type:boiler'}, 'vicare/Vitodens300W.json')]
    with patch(f'{MODULE}.login', return_value=MockPyViCare(fixtures)):
        await setup_integration(hass, mock_config_entry)
        yield mock_config_entry

@pytest.fixture
async def mock_vicare_room_sensors(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> MockConfigEntry:
    """Return a mocked ViCare API representing multiple room sensor devices."""
    fixtures: List[Fixture] = [Fixture({'type:climateSensor'}, 'vicare/RoomSensor1.json'), Fixture({'type:climateSensor'}, 'vicare/RoomSensor2.json')]
    with patch(f'{MODULE}.login', return_value=MockPyViCare(fixtures)):
        await setup_integration(hass, mock_config_entry)
        yield mock_config_entry

@pytest.fixture
def mock_setup_entry() -> Generator[Mock, None, None]:
    """Mock setting up a config entry."""
    with patch(f'{MODULE}.async_setup_entry', return_value=True) as mock_setup_entry:
        yield mock_setup_entry
