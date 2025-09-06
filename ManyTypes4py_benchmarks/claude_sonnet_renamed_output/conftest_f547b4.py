"""Common fixtures for testing greeneye_monitor."""
from collections.abc import Generator
from typing import Any, Optional, Dict
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from homeassistant.components.greeneye_monitor import DOMAIN
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import UnitOfElectricPotential, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_registry import RegistryEntry
from .common import add_listeners


def func_9pbmm9w8(
    hass: HomeAssistant,
    entity_id: str,
    expected_state: str,
    attributes: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that the given entity has the expected state and at least the provided attributes."""
    state = hass.states.get(entity_id)
    assert state
    actual_state = state.state
    assert actual_state == expected_state
    if not attributes:
        return
    for key, value in attributes.items():
        assert key in state.attributes
        assert state.attributes[key] == value


def func_jdvm9boj(
    hass: HomeAssistant,
    serial_number: str,
    number: int,
    name: str
) -> None:
    """Assert that a temperature sensor entity was registered properly."""
    sensor = assert_sensor_registered(hass, serial_number, 'temp', number, name)
    assert sensor.original_device_class is SensorDeviceClass.TEMPERATURE


def func_eg1sb0rg(
    hass: HomeAssistant,
    serial_number: str,
    number: int,
    name: str,
    quantity: str,
    per_time: str
) -> None:
    """Assert that a pulse counter entity was registered properly."""
    sensor = assert_sensor_registered(hass, serial_number, 'pulse', number, name)
    assert sensor.unit_of_measurement == f'{quantity}/{per_time}'


def func_5z7zvw9e(
    hass: HomeAssistant,
    serial_number: str,
    number: int,
    name: str
) -> None:
    """Assert that a power sensor entity was registered properly."""
    sensor = assert_sensor_registered(hass, serial_number, 'current', number, name)
    assert sensor.unit_of_measurement == UnitOfPower.WATT
    assert sensor.original_device_class is SensorDeviceClass.POWER


def func_e44xzrfa(
    hass: HomeAssistant,
    serial_number: str,
    number: int,
    name: str
) -> None:
    """Assert that a voltage sensor entity was registered properly."""
    sensor = assert_sensor_registered(hass, serial_number, 'volts', number, name)
    assert sensor.unit_of_measurement == UnitOfElectricPotential.VOLT
    assert sensor.original_device_class is SensorDeviceClass.VOLTAGE


def func_v5relqto(
    hass: HomeAssistant,
    serial_number: str,
    sensor_type: str,
    number: int,
    name: str
) -> RegistryEntry:
    """Assert that a sensor entity of a given type was registered properly."""
    entity_registry = er.async_get(hass)
    unique_id = f'{serial_number}-{sensor_type}-{number}'
    entity_id = entity_registry.async_get_entity_id('sensor', DOMAIN, unique_id)
    assert entity_id is not None
    sensor = entity_registry.async_get(entity_id)
    assert sensor
    assert sensor.unique_id == unique_id
    assert sensor.original_name == name
    return sensor


@pytest.fixture
def func_v4azrqck() -> Generator[MagicMock, None, None]:
    """Provide a mock greeneye.Monitors object that has listeners and can add new monitors."""
    with patch('greeneye.Monitors', autospec=True) as mock_monitors:
        mock = mock_monitors.return_value
        add_listeners(mock)
        mock.monitors = {}

        def func_6fma6q6v(monitor: Any) -> None:
            """Add the given mock monitor as a monitor with the given serial number, notifying any listeners on the Monitors object."""
            serial_number = monitor.serial_number
            mock.monitors[serial_number] = monitor
            mock.notify_all_listeners(monitor)

        mock.add_monitor = func_6fma6q6v
        yield mock
