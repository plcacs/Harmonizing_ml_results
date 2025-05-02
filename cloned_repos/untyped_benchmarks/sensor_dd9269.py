"""Support for hunterdouglass_powerview sensors."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final
from aiopvapi.helpers.constants import ATTR_NAME
from aiopvapi.resources.shade import BaseShade
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import PERCENTAGE, SIGNAL_STRENGTH_DECIBELS, EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .coordinator import PowerviewShadeUpdateCoordinator
from .entity import ShadeEntity
from .model import PowerviewConfigEntry, PowerviewDeviceInfo

@dataclass(frozen=True)
class PowerviewSensorDescriptionMixin:
    """Mixin to describe a Sensor entity."""

@dataclass(frozen=True)
class PowerviewSensorDescription(SensorEntityDescription, PowerviewSensorDescriptionMixin):
    """Class to describe a Sensor entity."""
    entity_category = EntityCategory.DIAGNOSTIC
    state_class = SensorStateClass.MEASUREMENT

def get_signal_device_class(shade):
    """Get the signal value based on version of API."""
    return SensorDeviceClass.SIGNAL_STRENGTH if shade.api_version >= 3 else None

def get_signal_native_unit(shade):
    """Get the unit of measurement for signal based on version of API."""
    return SIGNAL_STRENGTH_DECIBELS if shade.api_version >= 3 else PERCENTAGE
SENSORS = [PowerviewSensorDescription(key='charge', device_class_fn=lambda shade: SensorDeviceClass.BATTERY, native_unit_fn=lambda shade: PERCENTAGE, native_value_fn=lambda shade: shade.get_battery_strength(), create_entity_fn=lambda shade: shade.is_battery_powered(), update_fn=lambda shade: shade.refresh_battery(suppress_timeout=True)), PowerviewSensorDescription(key='signal', translation_key='signal_strength', icon='mdi:signal', device_class_fn=get_signal_device_class, native_unit_fn=get_signal_native_unit, native_value_fn=lambda shade: shade.get_signal_strength(), create_entity_fn=lambda shade: shade.has_signal_strength(), update_fn=lambda shade: shade.refresh(suppress_timeout=True), entity_registry_enabled_default=False)]

async def async_setup_entry(hass, entry, async_add_entities):
    """Set up the hunter douglas sensor entities."""
    pv_entry = entry.runtime_data
    entities = []
    for shade in pv_entry.shade_data.values():
        room_name = getattr(pv_entry.room_data.get(shade.room_id), ATTR_NAME, '')
        entities.extend((PowerViewSensor(pv_entry.coordinator, pv_entry.device_info, room_name, shade, shade.name, description) for description in SENSORS if description.create_entity_fn(shade)))
    async_add_entities(entities)

class PowerViewSensor(ShadeEntity, SensorEntity):
    """Representation of an shade sensor."""

    def __init__(self, coordinator, device_info, room_name, shade, name, description):
        """Initialize the sensor entity."""
        super().__init__(coordinator, device_info, room_name, shade, name)
        self.entity_description = description
        self.entity_description = description
        self._attr_unique_id = f'{self._attr_unique_id}_{description.key}'

    @property
    def native_value(self):
        """Get the current value of the sensor."""
        return self.entity_description.native_value_fn(self._shade)

    @property
    def native_unit_of_measurement(self):
        """Return native unit of measurement of sensor."""
        return self.entity_description.native_unit_fn(self._shade)

    @property
    def device_class(self):
        """Return the class of this entity."""
        return self.entity_description.device_class_fn(self._shade)

    async def async_added_to_hass(self):
        """When entity is added to hass."""
        self.async_on_remove(self.coordinator.async_add_listener(self._async_update_shade_from_group))

    @callback
    def _async_update_shade_from_group(self):
        """Update with new data from the coordinator."""
        self._shade.raw_data = self.data.get_raw_data(self._shade.id)
        self.async_write_ha_state()

    async def async_update(self):
        """Refresh sensor entity."""
        async with self.coordinator.radio_operation_lock:
            await self.entity_description.update_fn(self._shade)
        self.async_write_ha_state()