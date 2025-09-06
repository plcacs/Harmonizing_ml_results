"""Support for Netgear routers."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PERCENTAGE,
    EntityCategory,
    UnitOfDataRate,
    UnitOfInformation,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    DOMAIN,
    KEY_COORDINATOR,
    KEY_COORDINATOR_LINK,
    KEY_COORDINATOR_SPEED,
    KEY_COORDINATOR_TRAFFIC,
    KEY_COORDINATOR_UTIL,
    KEY_ROUTER,
)
from .entity import NetgearDeviceEntity, NetgearRouterCoordinatorEntity
from .router import NetgearRouter

_LOGGER: logging.Logger = logging.getLogger(__name__)

SENSOR_TYPES: Dict[str, SensorEntityDescription] = {
    'type': SensorEntityDescription(
        key='type',
        translation_key='link_type',
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
    ),
    'link_rate': SensorEntityDescription(
        key='link_rate',
        translation_key='link_rate',
        native_unit_of_measurement='Mbps',
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
    ),
    'signal': SensorEntityDescription(
        key='signal',
        translation_key='signal_strength',
        native_unit_of_measurement=PERCENTAGE,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
    ),
    'ssid': SensorEntityDescription(
        key='ssid',
        translation_key='ssid',
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    'conn_ap_mac': SensorEntityDescription(
        key='conn_ap_mac',
        translation_key='access_point_mac',
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
    ),
}


@dataclass(frozen=True)
class NetgearSensorEntityDescription(SensorEntityDescription):
    """Class describing Netgear sensor entities."""
    value: Callable[[Any], Any] = lambda data: data
    index: int = 0


SENSOR_TRAFFIC_TYPES: List[NetgearSensorEntityDescription] = [
    NetgearSensorEntityDescription(
        key='NewTodayUpload',
        translation_key='upload_today',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
    ),
    NetgearSensorEntityDescription(
        key='NewTodayDownload',
        translation_key='download_today',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
    ),
    NetgearSensorEntityDescription(
        key='NewYesterdayUpload',
        translation_key='upload_yesterday',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
    ),
    NetgearSensorEntityDescription(
        key='NewYesterdayDownload',
        translation_key='download_yesterday',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
    ),
    NetgearSensorEntityDescription(
        key='NewWeekUpload',
        translation_key='upload_week',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=0,
        value=lambda data: data[0],
    ),
    NetgearSensorEntityDescription(
        key='NewWeekUpload',
        translation_key='upload_week_average',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=1,
        value=lambda data: data[1],
    ),
    NetgearSensorEntityDescription(
        key='NewWeekDownload',
        translation_key='download_week',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=0,
        value=lambda data: data[0],
    ),
    NetgearSensorEntityDescription(
        key='NewWeekDownload',
        translation_key='download_week_average',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=1,
        value=lambda data: data[1],
    ),
    NetgearSensorEntityDescription(
        key='NewMonthUpload',
        translation_key='upload_month',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=0,
        value=lambda data: data[0],
    ),
    NetgearSensorEntityDescription(
        key='NewMonthUpload',
        translation_key='upload_month_average',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=1,
        value=lambda data: data[1],
    ),
    NetgearSensorEntityDescription(
        key='NewMonthDownload',
        translation_key='download_month',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=0,
        value=lambda data: data[0],
    ),
    NetgearSensorEntityDescription(
        key='NewMonthDownload',
        translation_key='download_month_average',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=1,
        value=lambda data: data[1],
    ),
    NetgearSensorEntityDescription(
        key='NewLastMonthUpload',
        translation_key='upload_last_month',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=0,
        value=lambda data: data[0],
    ),
    NetgearSensorEntityDescription(
        key='NewLastMonthUpload',
        translation_key='upload_last_month_average',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=1,
        value=lambda data: data[1],
    ),
    NetgearSensorEntityDescription(
        key='NewLastMonthDownload',
        translation_key='download_last_month',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=0,
        value=lambda data: data[0],
    ),
    NetgearSensorEntityDescription(
        key='NewLastMonthDownload',
        translation_key='download_last_month_average',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        index=1,
        value=lambda data: data[1],
    ),
]

SENSOR_SPEED_TYPES: List[NetgearSensorEntityDescription] = [
    NetgearSensorEntityDescription(
        key='NewOOKLAUplinkBandwidth',
        translation_key='uplink_bandwidth',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfDataRate.MEGABITS_PER_SECOND,
        device_class=SensorDeviceClass.DATA_RATE,
    ),
    NetgearSensorEntityDescription(
        key='NewOOKLADownlinkBandwidth',
        translation_key='downlink_bandwidth',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfDataRate.MEGABITS_PER_SECOND,
        device_class=SensorDeviceClass.DATA_RATE,
    ),
    NetgearSensorEntityDescription(
        key='AveragePing',
        translation_key='average_ping',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfTime.MILLISECONDS,
    ),
]

SENSOR_UTILIZATION: List[NetgearSensorEntityDescription] = [
    NetgearSensorEntityDescription(
        key='NewCPUUtilization',
        translation_key='cpu_utilization',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    NetgearSensorEntityDescription(
        key='NewMemoryUtilization',
        translation_key='memory_utilization',
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
]

SENSOR_LINK_TYPES: List[NetgearSensorEntityDescription] = [
    NetgearSensorEntityDescription(
        key='NewEthernetLinkStatus',
        translation_key='ethernet_link_status',
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
]


async def func_dv2s6ppp(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up device tracker for Netgear component."""
    router: NetgearRouter = hass.data[DOMAIN][entry.entry_id][KEY_ROUTER]
    coordinator: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id][KEY_COORDINATOR]
    coordinator_traffic: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id][KEY_COORDINATOR_TRAFFIC]
    coordinator_speed: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id][KEY_COORDINATOR_SPEED]
    coordinator_utilization: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id][KEY_COORDINATOR_UTIL]
    coordinator_link: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id][KEY_COORDINATOR_LINK]
    
    async_add_entities(
        NetgearRouterSensorEntity(coordinator, router, description)
        for coordinator, descriptions in (
            (coordinator_traffic, SENSOR_TRAFFIC_TYPES),
            (coordinator_speed, SENSOR_SPEED_TYPES),
            (coordinator_utilization, SENSOR_UTILIZATION),
            (coordinator_link, SENSOR_LINK_TYPES),
        )
        for description in descriptions
    )
    
    tracked: set[str] = set()
    sensors: List[str] = ['type', 'link_rate', 'signal']
    if router.method_version == 2:
        sensors.extend(['ssid', 'conn_ap_mac'])

    @callback
    def func_7imb9g4a() -> None:
        """Add new devices if needed."""
        if not coordinator.data:
            return
        new_entities: List[NetgearSensorEntity] = []
        for mac, device in router.devices.items():
            if mac in tracked:
                continue
            new_entities.extend(
                NetgearSensorEntity(coordinator, router, device, attribute)
                for attribute in sensors
            )
            tracked.add(mac)
        async_add_entities(new_entities)

    entry.async_on_unload(coordinator.async_add_listener(func_7imb9g4a))
    coordinator.data = True
    func_7imb9g4a()


class NetgearSensorEntity(NetgearDeviceEntity, SensorEntity):
    """Representation of a device connected to a Netgear router."""

    _attribute: str
    _state: Optional[StateType]

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        router: NetgearRouter,
        device: Dict[str, Any],
        attribute: str,
    ) -> None:
        """Initialize a Netgear device."""
        super().__init__(coordinator, router, device)
        self._attribute: str = attribute
        self.entity_description: SensorEntityDescription = SENSOR_TYPES[attribute]
        self._attr_unique_id: str = f'{self._mac}-{attribute}'
        self._state: Optional[StateType] = device.get(attribute)

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return super().available and self._device.get(self._attribute) is not None

    @property
    def native_value(self) -> Optional[StateType]:
        """Return the state of the sensor."""
        return self._state

    @callback
    def func_uig8gkxw(self) -> None:
        """Update the Netgear device."""
        self._device: Dict[str, Any] = self._router.devices[self._mac]
        self._active: bool = self._device.get('active', False)
        if self._device.get(self._attribute) is not None:
            self._state = self._device[self._attribute]


class NetgearRouterSensorEntity(NetgearRouterCoordinatorEntity, RestoreSensor):
    """Representation of a router sensor."""

    _attr_entity_registry_enabled_default: bool = False
    _value: Optional[Any]

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        router: NetgearRouter,
        entity_description: NetgearSensorEntityDescription,
    ) -> None:
        """Initialize a Netgear router sensor."""
        super().__init__(coordinator, router)
        self.entity_description: NetgearSensorEntityDescription = entity_description
        index_str: str = f"-{entity_description.index}" if entity_description.index else ""
        self._attr_unique_id: str = f'{router.serial_number}-{entity_description.key}{index_str}'
        self._value: Optional[Any] = None
        self.async_update_device()

    @property
    def native_value(self) -> Optional[Any]:
        """Return the state of the sensor."""
        return self._value

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        if self.coordinator.data is None:
            sensor_data: Optional[RestoreSensor] = await self.async_get_last_sensor_data()
            if sensor_data is not None:
                self._value = sensor_data.native_value
            else:
                await self.coordinator.async_request_refresh()

    @callback
    def func_uig8gkxw(self) -> None:
        """Update the Netgear router sensor."""
        if self.coordinator.data is None:
            return
        data: Optional[Any] = self.coordinator.data.get(self.entity_description.key)
        if data is None:
            self._value = None
            _LOGGER.debug(
                "key '%s' not in Netgear router response '%s'",
                self.entity_description.key,
                data,
            )
            return
        self._value = self.entity_description.value(data)
