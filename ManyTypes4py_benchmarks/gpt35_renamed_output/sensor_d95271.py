from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import logging
from homeassistant.components.sensor import RestoreSensor, SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfDataRate, UnitOfInformation, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from .const import DOMAIN, KEY_COORDINATOR, KEY_COORDINATOR_LINK, KEY_COORDINATOR_SPEED, KEY_COORDINATOR_TRAFFIC, KEY_COORDINATOR_UTIL, KEY_ROUTER
from .entity import NetgearDeviceEntity, NetgearRouterCoordinatorEntity
from .router import NetgearRouter
_LOGGER: logging.Logger = logging.getLogger(__name__)
SENSOR_TYPES: dict[str, SensorEntityDescription] = {'type': SensorEntityDescription(key='type', translation_key='link_type', entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False), 'link_rate': SensorEntityDescription(key='link_rate', translation_key='link_rate', native_unit_of_measurement='Mbps', entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False), 'signal': SensorEntityDescription(key='signal', translation_key='signal_strength', native_unit_of_measurement=PERCENTAGE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False), 'ssid': SensorEntityDescription(key='ssid', translation_key='ssid', entity_category=EntityCategory.DIAGNOSTIC), 'conn_ap_mac': SensorEntityDescription(key='conn_ap_mac', translation_key='access_point_mac', entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False)}

@dataclass(frozen=True)
class NetgearSensorEntityDescription(SensorEntityDescription):
    """Class describing Netgear sensor entities."""
    value: Callable = lambda data: data
    index: int = 0

SENSOR_TRAFFIC_TYPES: list[NetgearSensorEntityDescription] = [NetgearSensorEntityDescription(key='NewTodayUpload', translation_key='upload_today', entity_category=EntityCategory.DIAGNOSTIC, native_unit_of_measurement=UnitOfInformation.MEGABYTES, device_class=SensorDeviceClass.DATA_SIZE), NetgearSensorEntityDescription(key='NewTodayDownload', translation_key='download_today', entity_category=EntityCategory.DIAGNOSTIC, native_unit_of_measurement=UnitOfInformation.MEGABYTES, device_class=SensorDeviceClass.DATA_SIZE), ...]

SENSOR_SPEED_TYPES: list[NetgearSensorEntityDescription] = [NetgearSensorEntityDescription(key='NewOOKLAUplinkBandwidth', translation_key='uplink_bandwidth', entity_category=EntityCategory.DIAGNOSTIC, native_unit_of_measurement=UnitOfDataRate.MEGABITS_PER_SECOND, device_class=SensorDeviceClass.DATA_RATE), NetgearSensorEntityDescription(key='NewOOKLADownlinkBandwidth', translation_key='downlink_bandwidth', entity_category=EntityCategory.DIAGNOSTIC, native_unit_of_measurement=UnitOfDataRate.MEGABITS_PER_SECOND, device_class=SensorDeviceClass.DATA_RATE), ...]

SENSOR_UTILIZATION: list[NetgearSensorEntityDescription] = [NetgearSensorEntityDescription(key='NewCPUUtilization', translation_key='cpu_utilization', entity_category=EntityCategory.DIAGNOSTIC, native_unit_of_measurement=PERCENTAGE, state_class=SensorStateClass.MEASUREMENT), NetgearSensorEntityDescription(key='NewMemoryUtilization', translation_key='memory_utilization', entity_category=EntityCategory.DIAGNOSTIC, native_unit_of_measurement=PERCENTAGE, state_class=SensorStateClass.MEASUREMENT)]

SENSOR_LINK_TYPES: list[NetgearSensorEntityDescription] = [NetgearSensorEntityDescription(key='NewEthernetLinkStatus', translation_key='ethernet_link_status', entity_category=EntityCategory.DIAGNOSTIC)]

async def func_dv2s6ppp(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class NetgearSensorEntity(NetgearDeviceEntity, SensorEntity):
    def __init__(self, coordinator: DataUpdateCoordinator, router: NetgearRouter, device: dict, attribute: str) -> None:
        ...

    @property
    def func_8g94pd1k(self) -> bool:
        ...

    @property
    def func_sgm3xaxh(self) -> StateType:
        ...

    @callback
    def func_uig8gkxw(self) -> None:
        ...

class NetgearRouterSensorEntity(NetgearRouterCoordinatorEntity, RestoreSensor):
    def __init__(self, coordinator: DataUpdateCoordinator, router: NetgearRouter, entity_description: NetgearSensorEntityDescription) -> None:
        ...

    @property
    def func_sgm3xaxh(self) -> StateType:
        ...

    async def func_gd0u4wmp(self) -> None:
        ...

    @callback
    def func_uig8gkxw(self) -> None:
        ...
