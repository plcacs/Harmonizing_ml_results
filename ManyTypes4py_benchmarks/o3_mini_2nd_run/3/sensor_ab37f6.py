from __future__ import annotations
from datetime import datetime, timedelta
import logging
from typing import Any, Callable, List, Optional, Set

from pyoctoprintapi import OctoprintJobInfo, OctoprintPrinterInfo
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import OctoprintDataUpdateCoordinator
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)
JOB_PRINTING_STATES: List[str] = ['Printing from SD', 'Printing']


def _is_printer_printing(printer: Optional[OctoprintPrinterInfo]) -> bool:
    return printer is not None and printer.state and printer.state.flags and printer.state.flags.printing


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the available OctoPrint binary sensors."""
    coordinator: OctoprintDataUpdateCoordinator = hass.data[DOMAIN][config_entry.entry_id]['coordinator']
    device_id: Optional[str] = config_entry.unique_id
    assert device_id is not None
    known_tools: Set[str] = set()

    @callback
    def async_add_tool_sensors() -> None:
        if not coordinator.data.get('printer'):
            return
        new_tools: List[OctoPrintTemperatureSensor] = []
        for tool in [tool for tool in coordinator.data['printer'].temperatures if tool.name not in known_tools]:
            assert device_id is not None
            known_tools.add(tool.name)
            new_tools.extend(
                (
                    OctoPrintTemperatureSensor(coordinator, tool.name, temp_type, device_id)
                    for temp_type in ('actual', 'target')
                )
            )
        async_add_entities(new_tools)

    config_entry.async_on_unload(coordinator.async_add_listener(async_add_tool_sensors))
    if coordinator.data.get('printer'):
        async_add_tool_sensors()
    entities: List[SensorEntity] = [
        OctoPrintStatusSensor(coordinator, device_id),
        OctoPrintJobPercentageSensor(coordinator, device_id),
        OctoPrintEstimatedFinishTimeSensor(coordinator, device_id),
        OctoPrintStartTimeSensor(coordinator, device_id),
    ]
    async_add_entities(entities)


class OctoPrintSensorBase(CoordinatorEntity[OctoprintDataUpdateCoordinator], SensorEntity):
    """Representation of an OctoPrint sensor."""

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, sensor_type: str, device_id: str) -> None:
        """Initialize a new OctoPrint sensor."""
        super().__init__(coordinator)
        self._device_id: str = device_id
        self._attr_name: str = f'OctoPrint {sensor_type}'
        self._attr_unique_id: str = f'{sensor_type}-{device_id}'
        self._attr_device_info = coordinator.device_info


class OctoPrintStatusSensor(OctoPrintSensorBase):
    """Representation of an OctoPrint sensor."""
    _attr_icon: str = 'mdi:printer-3d'

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str) -> None:
        """Initialize a new OctoPrint sensor."""
        super().__init__(coordinator, 'Current State', device_id)

    @property
    def native_value(self) -> Optional[str]:
        """Return sensor state."""
        printer: Optional[OctoprintPrinterInfo] = self.coordinator.data.get('printer')
        if not printer:
            return None
        return printer.state.text

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.last_update_success and bool(self.coordinator.data.get('printer'))


class OctoPrintJobPercentageSensor(OctoPrintSensorBase):
    """Representation of an OctoPrint sensor."""
    _attr_native_unit_of_measurement: str = PERCENTAGE
    _attr_icon: str = 'mdi:file-percent'

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str) -> None:
        """Initialize a new OctoPrint sensor."""
        super().__init__(coordinator, 'Job Percentage', device_id)

    @property
    def native_value(self) -> Optional[float]:
        """Return sensor state."""
        job: Optional[OctoprintJobInfo] = self.coordinator.data.get('job')
        if not job:
            return None
        if not (state := job.progress.completion):
            return 0.0
        return round(state, 2)


class OctoPrintEstimatedFinishTimeSensor(OctoPrintSensorBase):
    """Representation of an OctoPrint sensor."""
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TIMESTAMP

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str) -> None:
        """Initialize a new OctoPrint sensor."""
        super().__init__(coordinator, 'Estimated Finish Time', device_id)

    @property
    def native_value(self) -> Optional[datetime]:
        """Return sensor state."""
        job: Optional[OctoprintJobInfo] = self.coordinator.data.get('job')
        printer: Optional[OctoprintPrinterInfo] = self.coordinator.data.get('printer')
        if not job or not job.progress.print_time_left or (not _is_printer_printing(printer)):
            return None
        read_time: datetime = self.coordinator.data['last_read_time']
        return (read_time + timedelta(seconds=job.progress.print_time_left)).replace(second=0)


class OctoPrintStartTimeSensor(OctoPrintSensorBase):
    """Representation of an OctoPrint sensor."""
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TIMESTAMP

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str) -> None:
        """Initialize a new OctoPrint sensor."""
        super().__init__(coordinator, 'Start Time', device_id)

    @property
    def native_value(self) -> Optional[datetime]:
        """Return sensor state."""
        job: Optional[OctoprintJobInfo] = self.coordinator.data.get('job')
        printer: Optional[OctoprintPrinterInfo] = self.coordinator.data.get('printer')
        if not job or not job.progress.print_time or (not _is_printer_printing(printer)):
            return None
        read_time: datetime = self.coordinator.data['last_read_time']
        return (read_time - timedelta(seconds=job.progress.print_time)).replace(second=0)


class OctoPrintTemperatureSensor(OctoPrintSensorBase):
    """Representation of an OctoPrint sensor."""
    _attr_native_unit_of_measurement: str = UnitOfTemperature.CELSIUS
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TEMPERATURE
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        coordinator: OctoprintDataUpdateCoordinator,
        tool: str,
        temp_type: str,
        device_id: str,
    ) -> None:
        """Initialize a new OctoPrint sensor."""
        super().__init__(coordinator, f'{temp_type} {tool} temp', device_id)
        self._temp_type: str = temp_type
        self._api_tool: str = tool

    @property
    def native_value(self) -> Optional[float]:
        """Return sensor state."""
        printer: Optional[OctoprintPrinterInfo] = self.coordinator.data.get('printer')
        if not printer:
            return None
        for temp in printer.temperatures:
            if temp.name == self._api_tool:
                val: Optional[float] = temp.actual_temp if self._temp_type == 'actual' else temp.target_temp
                if val is None:
                    return None
                return round(val, 2)
        return None

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.last_update_success and bool(self.coordinator.data.get('printer'))