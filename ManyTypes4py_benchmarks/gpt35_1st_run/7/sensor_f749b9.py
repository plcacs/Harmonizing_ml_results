from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
import logging
from typing import TYPE_CHECKING, Any, Final, Self
import voluptuous as vol
from homeassistant.components.sensor import DEVICE_CLASS_UNITS, PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, RestoreSensor, SensorDeviceClass, SensorExtraStoredData, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_DEVICE_CLASS, ATTR_UNIT_OF_MEASUREMENT, CONF_METHOD, CONF_NAME, CONF_UNIQUE_ID, STATE_UNAVAILABLE, UnitOfTime
from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, EventStateReportedData, HomeAssistant, State, callback
from homeassistant.helpers import config_validation as cv, entity_registry as er
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_call_later, async_track_state_change_event, async_track_state_report_event
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import CONF_MAX_SUB_INTERVAL, CONF_ROUND_DIGITS, CONF_SOURCE_SENSOR, CONF_UNIT_OF_MEASUREMENT, CONF_UNIT_PREFIX, CONF_UNIT_TIME, INTEGRATION_METHODS, METHOD_LEFT, METHOD_RIGHT, METHOD_TRAPEZOIDAL

_LOGGER: logging.Logger

ATTR_SOURCE_ID: Final[str] = 'source'
UNIT_PREFIXES: Final[dict[str, int]] = {None: 1, 'k': 10 ** 3, 'M': 10 ** 6, 'G': 10 ** 9, 'T': 10 ** 12}
UNIT_TIME: Final[dict[UnitOfTime, int]] = {UnitOfTime.SECONDS: 1, UnitOfTime.MINUTES: 60, UnitOfTime.HOURS: 60 * 60, UnitOfTime.DAYS: 24 * 60 * 60}
DEVICE_CLASS_MAP: Final[dict[SensorDeviceClass, SensorDeviceClass]] = {SensorDeviceClass.POWER: SensorDeviceClass.ENERGY}
DEFAULT_ROUND: Final[int] = 3

class _IntegrationMethod(ABC):

    @staticmethod
    def from_name(method_name: str) -> _IntegrationMethod:
        return _NAME_TO_INTEGRATION_METHOD[method_name]()

    @abstractmethod
    def validate_states(self, left: Any, right: Any) -> Any:

    @abstractmethod
    def calculate_area_with_two_states(self, elapsed_time: Decimal, left: Any, right: Any) -> Decimal:

    def calculate_area_with_one_state(self, elapsed_time: Decimal, constant_state: Any) -> Decimal:

class _Trapezoidal(_IntegrationMethod):

    def calculate_area_with_two_states(self, elapsed_time: Decimal, left: Any, right: Any) -> Decimal:

    def validate_states(self, left: Any, right: Any) -> Any:

class _Left(_IntegrationMethod):

    def calculate_area_with_two_states(self, elapsed_time: Decimal, left: Any, right: Any) -> Decimal:

    def validate_states(self, left: Any, right: Any) -> Any:

class _Right(_IntegrationMethod):

    def calculate_area_with_two_states(self, elapsed_time: Decimal, left: Any, right: Any) -> Decimal:

    def validate_states(self, left: Any, right: Any) -> Any:

def _decimal_state(state: Any) -> Decimal:

_NAME_TO_INTEGRATION_METHOD: Final[dict[str, _IntegrationMethod]] = {METHOD_LEFT: _Left, METHOD_RIGHT: _Right, METHOD_TRAPEZOIDAL: _Trapezoidal}

class _IntegrationTrigger(Enum):
    StateEvent: Final[str] = 'state_event'
    TimeElapsed: Final[str] = 'time_elapsed'

@dataclass
class IntegrationSensorExtraStoredData(SensorExtraStoredData):

    def as_dict(self) -> dict[str, Any]:

    @classmethod
    def from_dict(cls, restored: dict[str, Any]) -> IntegrationSensorExtraStoredData:

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

class IntegrationSensor(RestoreSensor):

    def __init__(self, *, integration_method: str, name: str, round_digits: int, source_entity: str, unique_id: str, unit_prefix: str, unit_time: UnitOfTime, max_sub_interval: timedelta, device_info: DeviceInfo = None) -> None:

    async def async_added_to_hass(self) -> None:

    @callback
    def _integrate_on_state_change_with_max_sub_interval(self, event: Event) -> None:

    @callback
    def _integrate_on_state_report_with_max_sub_interval(self, event: Event) -> None:

    @callback
    def _integrate_on_state_update_with_max_sub_interval(self, old_last_reported: Any, old_state: Any, new_state: Any) -> None:

    @callback
    def _integrate_on_state_change_callback(self, event: Event) -> None:

    @callback
    def _integrate_on_state_report_callback(self, event: Event) -> None:

    def _integrate_on_state_change(self, old_last_reported: Any, old_state: Any, new_state: Any) -> None:

    def _schedule_max_sub_interval_exceeded_if_state_is_numeric(self, source_state: State) -> None:

    def _cancel_max_sub_interval_exceeded_callback(self) -> None:

    @property
    def native_value(self) -> Any:

    @property
    def native_unit_of_measurement(self) -> str:

    @property
    def extra_state_attributes(self) -> dict[str, Any]:

    @property
    def extra_restore_state_data(self) -> IntegrationSensorExtraStoredData:

    async def async_get_last_sensor_data(self) -> IntegrationSensorExtraStoredData:
