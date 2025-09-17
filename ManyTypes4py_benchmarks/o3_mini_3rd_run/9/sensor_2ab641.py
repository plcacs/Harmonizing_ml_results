from __future__ import annotations
from typing import Optional
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.const import CONF_USERNAME, UnitOfBloodGlucoseConcentration
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.config_entries import ConfigEntry
from .const import DOMAIN
from .coordinator import DexcomConfigEntry, DexcomCoordinator

TRENDS: dict[int, str] = {
    1: 'rising_quickly',
    2: 'rising',
    3: 'rising_slightly',
    4: 'steady',
    5: 'falling_slightly',
    6: 'falling',
    7: 'falling_quickly'
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    coordinator: DexcomCoordinator = config_entry.runtime_data
    username: str = config_entry.data[CONF_USERNAME]
    async_add_entities([
        DexcomGlucoseTrendSensor(coordinator, username, config_entry.entry_id),
        DexcomGlucoseValueSensor(coordinator, username, config_entry.entry_id)
    ])


class DexcomSensorEntity(CoordinatorEntity[DexcomCoordinator], SensorEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator: DexcomCoordinator, username: str, entry_id: str, key: str) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f'{username}-{key}'
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry_id)}, name=username)


class DexcomGlucoseValueSensor(DexcomSensorEntity):
    _attr_device_class = SensorDeviceClass.BLOOD_GLUCOSE_CONCENTRATION
    _attr_native_unit_of_measurement = UnitOfBloodGlucoseConcentration.MILLIGRAMS_PER_DECILITER
    _attr_translation_key = 'glucose_value'

    def __init__(self, coordinator: DexcomCoordinator, username: str, entry_id: str) -> None:
        super().__init__(coordinator, username, entry_id, 'value')

    @property
    def native_value(self) -> Optional[float]:
        if self.coordinator.data:
            return self.coordinator.data.mg_dl
        return None


class DexcomGlucoseTrendSensor(DexcomSensorEntity):
    _attr_translation_key = 'glucose_trend'
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = list(TRENDS.values())

    def __init__(self, coordinator: DexcomCoordinator, username: str, entry_id: str) -> None:
        super().__init__(coordinator, username, entry_id, 'trend')

    @property
    def native_value(self) -> Optional[str]:
        if self.coordinator.data:
            return TRENDS.get(self.coordinator.data.trend)
        return None

    @property
    def available(self) -> bool:
        return super().available and (self.coordinator.data is None or self.coordinator.data.trend != 9)