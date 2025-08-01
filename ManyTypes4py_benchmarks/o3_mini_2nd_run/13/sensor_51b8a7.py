"""Support for sensors."""
from __future__ import annotations
from typing import Final, cast, Any, Generator
from aiocomelit import ComelitSerialBridgeObject, ComelitVedoZoneObject
from aiocomelit.const import ALARM_ZONES, BRIDGE, OTHER, AlarmZoneState
from homeassistant.config_entries import ConfigEntry
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription
from homeassistant.const import CONF_TYPE, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .coordinator import ComelitConfigEntry, ComelitSerialBridge, ComelitVedoSystem

SENSOR_BRIDGE_TYPES: Final = (
    SensorEntityDescription(
        key='power',
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
    ),
)

SENSOR_VEDO_TYPES: Final = (
    SensorEntityDescription(
        key='human_status',
        translation_key='zone_status',
        name=None,
        device_class=SensorDeviceClass.ENUM,
        options=[zone_state.value for zone_state in AlarmZoneState],
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Comelit sensors."""
    if config_entry.data.get(CONF_TYPE, BRIDGE) == BRIDGE:
        await async_setup_bridge_entry(hass, config_entry, async_add_entities)
    else:
        await async_setup_vedo_entry(hass, config_entry, async_add_entities)


async def async_setup_bridge_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Comelit Bridge sensors."""
    coordinator = cast(ComelitSerialBridge, config_entry.runtime_data)
    entities: list[ComelitBridgeSensorEntity] = []
    for device in coordinator.data[OTHER].values():
        # Generator comprehension is used here to create sensor entities
        entities.extend(
            (
                ComelitBridgeSensorEntity(coordinator, device, config_entry.entry_id, sensor_desc)
                for sensor_desc in SENSOR_BRIDGE_TYPES
            )
        )
    async_add_entities(entities)


async def async_setup_vedo_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Comelit VEDO sensors."""
    coordinator = cast(ComelitVedoSystem, config_entry.runtime_data)
    entities: list[ComelitVedoSensorEntity] = []
    for zone in coordinator.data[ALARM_ZONES].values():
        entities.extend(
            (
                ComelitVedoSensorEntity(coordinator, zone, config_entry.entry_id, sensor_desc)
                for sensor_desc in SENSOR_VEDO_TYPES
            )
        )
    async_add_entities(entities)


class ComelitBridgeSensorEntity(CoordinatorEntity[ComelitSerialBridge], SensorEntity):
    """Sensor device."""
    _attr_has_entity_name: Final[bool] = True
    _attr_name: None = None

    def __init__(
        self,
        coordinator: ComelitSerialBridge,
        device: ComelitSerialBridgeObject,
        config_entry_entry_id: str,
        description: SensorEntityDescription,
    ) -> None:
        """Init sensor entity."""
        self._api = coordinator.api
        self._device = device
        super().__init__(coordinator)
        self._attr_unique_id = f'{config_entry_entry_id}-{device.index}'
        self._attr_device_info = coordinator.platform_device_info(device, device.type)
        self.entity_description = description

    @property
    def native_value(self) -> StateType:
        """Sensor value."""
        return getattr(self.coordinator.data[OTHER][self._device.index], self.entity_description.key)


class ComelitVedoSensorEntity(CoordinatorEntity[ComelitVedoSystem], SensorEntity):
    """Sensor device."""
    _attr_has_entity_name: Final[bool] = True

    def __init__(
        self,
        coordinator: ComelitVedoSystem,
        zone: ComelitVedoZoneObject,
        config_entry_entry_id: str,
        description: SensorEntityDescription,
    ) -> None:
        """Init sensor entity."""
        self._api = coordinator.api
        self._zone = zone
        super().__init__(coordinator)
        self._attr_unique_id = f'{config_entry_entry_id}-{zone.index}'
        self._attr_device_info = coordinator.platform_device_info(zone, 'zone')
        self.entity_description = description

    @property
    def _zone_object(self) -> ComelitVedoZoneObject:
        """Zone object."""
        return self.coordinator.data[ALARM_ZONES][self._zone.index]

    @property
    def available(self) -> bool:
        """Sensor availability."""
        return self._zone_object.human_status != AlarmZoneState.UNAVAILABLE

    @property
    def native_value(self) -> StateType:
        """Sensor value."""
        status = self._zone_object.human_status
        if status == AlarmZoneState.UNKNOWN:
            return None
        return status.value
