from collections.abc import Generator
from typing import Any
import pytest
from homeassistant.components.device_tracker import ATTR_HOST_NAME, ATTR_IP, ATTR_MAC, ATTR_SOURCE_TYPE, DOMAIN, SourceType
from homeassistant.components.device_tracker.config_entry import CONNECTED_DEVICE_REGISTERED, BaseTrackerEntity, ScannerEntity, TrackerEntity
from homeassistant.components.zone import ATTR_RADIUS
from homeassistant.config_entries import ConfigEntry, ConfigEntryState, ConfigFlow
from homeassistant.const import ATTR_BATTERY_LEVEL, ATTR_GPS_ACCURACY, ATTR_LATITUDE, ATTR_LONGITUDE, STATE_HOME, STATE_NOT_HOME, STATE_UNKNOWN, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from tests.common import MockConfigEntry, MockModule, MockPlatform, mock_config_flow, mock_integration, mock_platform

TEST_DOMAIN: str = 'test'
TEST_MAC_ADDRESS: str = '12:34:56:AB:CD:EF'

class MockFlow(ConfigFlow):
    """Test flow."""

@pytest.fixture(autouse=True)
def config_flow_fixture(hass: HomeAssistant) -> Generator:
    """Mock config flow."""
    ...

@pytest.fixture(autouse=True)
def mock_setup_integration(hass: HomeAssistant) -> Generator:
    """Fixture to set up a mock integration."""
    ...

@pytest.fixture(name='config_entry')
def config_entry_fixture(hass: HomeAssistant) -> MockConfigEntry:
    """Return the config entry used for the tests."""
    ...

async def create_mock_platform(hass: HomeAssistant, config_entry: ConfigEntry, entities: list[Entity]) -> ConfigEntry:
    ...

@pytest.fixture(name='entity_id')
def entity_id_fixture() -> str:
    """Return the entity_id of the entity for the test."""
    ...

class MockTrackerEntity(TrackerEntity):
    """Test tracker entity."""
    ...

@pytest.fixture(name='battery_level')
def battery_level_fixture() -> Any:
    """Return the battery level of the entity for the test."""
    ...

@pytest.fixture(name='location_name')
def location_name_fixture() -> Any:
    """Return the location_name of the entity for the test."""
    ...

@pytest.fixture(name='latitude')
def latitude_fixture() -> Any:
    """Return the latitude of the entity for the test."""
    ...

@pytest.fixture(name='longitude')
def longitude_fixture() -> Any:
    """Return the longitude of the entity for the test."""
    ...

@pytest.fixture(name='tracker_entity')
def tracker_entity_fixture(entity_id: str, battery_level: Any, location_name: Any, latitude: Any, longitude: Any) -> MockTrackerEntity:
    """Create a test tracker entity."""
    ...

class MockScannerEntity(ScannerEntity):
    """Test scanner entity."""
    ...

@pytest.fixture(name='ip_address')
def ip_address_fixture() -> Any:
    """Return the ip_address of the entity for the test."""
    ...

@pytest.fixture(name='mac_address')
def mac_address_fixture() -> Any:
    """Return the mac_address of the entity for the test."""
    ...

@pytest.fixture(name='hostname')
def hostname_fixture() -> Any:
    """Return the hostname of the entity for the test."""
    ...

@pytest.fixture(name='unique_id')
def unique_id_fixture() -> Any:
    """Return the unique_id of the entity for the test."""
    ...

@pytest.fixture(name='scanner_entity')
def scanner_entity_fixture(entity_id: str, ip_address: Any, mac_address: Any, hostname: Any, unique_id: Any) -> MockScannerEntity:
    """Create a test scanner entity."""
    ...

async def test_load_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry, entity_id: str, tracker_entity: MockTrackerEntity) -> None:
    ...

async def test_tracker_entity_state(hass: HomeAssistant, config_entry: ConfigEntry, entity_id: str, tracker_entity: MockTrackerEntity, expected_state: str, expected_attributes: dict[str, Any]) -> None:
    ...

async def test_scanner_entity_state(hass: HomeAssistant, config_entry: ConfigEntry, device_registry: dr.DeviceRegistry, entity_id: str, ip_address: Any, mac_address: Any, hostname: Any, scanner_entity: MockScannerEntity) -> None:
    ...

def test_tracker_entity() -> None:
    ...

def test_scanner_entity() -> None:
    ...

def test_base_tracker_entity() -> None:
    ...

async def test_register_mac(hass: HomeAssistant, config_entry: ConfigEntry, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry, scanner_entity: MockScannerEntity, entity_id: str, mac_address: str, unique_id: str) -> None:
    ...

async def test_register_mac_not_found(hass: HomeAssistant, config_entry: ConfigEntry, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry, scanner_entity: MockScannerEntity, entity_id: str, connections: set[tuple[str, str]], mac_address: str, unique_id: str) -> None:
    ...

async def test_register_mac_ignored(hass: HomeAssistant, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry, scanner_entity: MockScannerEntity, entity_id: str, mac_address: str, unique_id: str) -> None:
    ...

async def test_connected_device_registered(hass: HomeAssistant, config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    ...

async def test_entity_has_device_info(hass: HomeAssistant, config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    ...
