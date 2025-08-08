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

_LOGGER: logging.Logger
SENSOR_TYPES: dict[str, SensorEntityDescription]
SENSOR_TRAFFIC_TYPES: list[NetgearSensorEntityDescription]
SENSOR_SPEED_TYPES: list[NetgearSensorEntityDescription]
SENSOR_UTILIZATION: list[NetgearSensorEntityDescription]
SENSOR_LINK_TYPES: list[NetgearSensorEntityDescription]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

@dataclass(frozen=True)
class NetgearSensorEntityDescription(SensorEntityDescription):
    value: Callable
    index: int

class NetgearSensorEntity(NetgearDeviceEntity, SensorEntity):

    def __init__(self, coordinator: DataUpdateCoordinator, router: NetgearRouter, device: dict, attribute: str) -> None:

    @property
    def available(self) -> bool:

    @property
    def native_value(self) -> StateType:

    @callback
    def async_update_device(self) -> None:

class NetgearRouterSensorEntity(NetgearRouterCoordinatorEntity, RestoreSensor):

    def __init__(self, coordinator: DataUpdateCoordinator, router: NetgearRouter, entity_description: NetgearSensorEntityDescription) -> None:

    @property
    def native_value(self) -> StateType:

    async def async_added_to_hass(self) -> None:

    @callback
    def async_update_device(self) -> None:
