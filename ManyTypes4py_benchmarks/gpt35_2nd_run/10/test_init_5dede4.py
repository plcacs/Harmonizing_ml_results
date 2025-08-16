from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.config_entries import ConfigEntry, ConfigEntryState, ConfigFlow
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_SET_VALVE_POSITION, SERVICE_TOGGLE, STATE_UNAVAILABLE, Platform
from homeassistant.components.valve import DOMAIN, ValveDeviceClass, ValveEntity, ValveEntityDescription, ValveEntityFeature, ValveState
from tests.common import MockConfigEntry, MockModule, MockPlatform, mock_config_flow, mock_integration, mock_platform
from typing import List, Tuple

def config_flow_fixture(hass: HomeAssistant) -> None:
    ...

def mock_config_entry(hass: HomeAssistant) -> Tuple[ConfigEntry, List[ValveEntity]]:
    ...

async def test_valve_setup(hass: HomeAssistant, mock_config_entry: Tuple[ConfigEntry, List[ValveEntity]], snapshot: SnapshotAssertion) -> None:
    ...

async def test_services(hass: HomeAssistant, mock_config_entry: Tuple[ConfigEntry, List[ValveEntity]) -> None:
    ...

async def test_valve_device_class(hass: HomeAssistant) -> None:
    ...

async def test_valve_report_position(hass: HomeAssistant) -> None:
    ...

async def test_none_state(hass: HomeAssistant) -> None:
    ...

async def test_supported_features(hass: HomeAssistant) -> None:
    ...

def call_service(hass: HomeAssistant, service: str, ent: ValveEntity, position: int = None) -> None:
    ...

def set_valve_position(ent: ValveEntity, position: int) -> None:
    ...

def is_open(hass: HomeAssistant, ent: ValveEntity) -> bool:
    ...

def is_opening(hass: HomeAssistant, ent: ValveEntity) -> bool:
    ...

def is_closed(hass: HomeAssistant, ent: ValveEntity) -> bool:
    ...

def is_closing(hass: HomeAssistant, ent: ValveEntity) -> bool:
    ...
