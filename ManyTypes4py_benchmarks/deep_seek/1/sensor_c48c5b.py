"""Utility meter from sensors providing raw data."""
from __future__ import annotations
from collections.abc import Mapping, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException, InvalidOperation
import logging
from typing import Any, Self, Optional, Union, Dict, List, Tuple, cast
from cronsim import CronSim
import voluptuous as vol
from homeassistant.components.sensor import ATTR_LAST_RESET, DEVICE_CLASS_UNITS, RestoreSensor, SensorDeviceClass, SensorExtraStoredData, SensorStateClass
from homeassistant.components.sensor.recorder import _suggest_report_issue
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_DEVICE_CLASS, ATTR_UNIT_OF_MEASUREMENT, CONF_NAME, CONF_UNIQUE_ID, EVENT_CORE_CONFIG_UPDATE, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State, callback
from homeassistant.helpers import entity_platform, entity_registry as er
from homeassistant.helpers.device import async_device_info_to_link_from_entity, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_point_in_time, async_track_state_change_event
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.template import is_number
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util, slugify
from homeassistant.util.enum import try_parse_enum
from .const import ATTR_NEXT_RESET, ATTR_VALUE, BIMONTHLY, CONF_CRON_PATTERN, CONF_METER, CONF_METER_DELTA_VALUES, CONF_METER_NET_CONSUMPTION, CONF_METER_OFFSET, CONF_METER_PERIODICALLY_RESETTING, CONF_METER_TYPE, CONF_SENSOR_ALWAYS_AVAILABLE, CONF_SOURCE_SENSOR, CONF_TARIFF, CONF_TARIFF_ENTITY, CONF_TARIFFS, DAILY, DATA_TARIFF_SENSORS, DATA_UTILITY, HOURLY, MONTHLY, QUARTER_HOURLY, QUARTERLY, SERVICE_CALIBRATE_METER, SIGNAL_RESET_METER, WEEKLY, YEARLY

PERIOD2CRON: Dict[str, str] = {
    QUARTER_HOURLY: '{minute}/15 * * * *',
    HOURLY: '{minute} * * * *',
    DAILY: '{minute} {hour} * * *',
    WEEKLY: '{minute} {hour} * * {day}',
    MONTHLY: '{minute} {hour} {day} * *',
    BIMONTHLY: '{minute} {hour} {day} */2 *',
    QUARTERLY: '{minute} {hour} {day} */3 *',
    YEARLY: '{minute} {hour} {day} 1/12 *'
}

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_SOURCE_ID: str = 'source'
ATTR_STATUS: str = 'status'
ATTR_PERIOD: str = 'meter_period'
ATTR_LAST_PERIOD: str = 'last_period'
ATTR_LAST_VALID_STATE: str = 'last_valid_state'
ATTR_TARIFF: str = 'tariff'
PRECISION: int = 3
PAUSED: str = 'paused'
COLLECTING: str = 'collecting'

def validate_is_number(value: Any) -> Any:
    """Validate value is a number."""
    if is_number(value):
        return value
    raise vol.Invalid('Value is not a number')

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Initialize Utility Meter config entry."""
    entry_id: str = config_entry.entry_id
    registry: er.EntityRegistry = er.async_get(hass)
    source_entity_id: str = er.async_validate_entity_id(registry, config_entry.options[CONF_SOURCE_SENSOR])
    device_info: Optional[DeviceInfo] = async_device_info_to_link_from_entity(hass, source_entity_id)
    cron_pattern: Optional[str] = None
    delta_values: bool = config_entry.options[CONF_METER_DELTA_VALUES]
    meter_offset: timedelta = timedelta(days=config_entry.options[CONF_METER_OFFSET])
    meter_type: Optional[str] = config_entry.options[CONF_METER_TYPE]
    if meter_type == 'none':
        meter_type = None
    name: str = config_entry.title
    net_consumption: bool = config_entry.options[CONF_METER_NET_CONSUMPTION]
    periodically_resetting: bool = config_entry.options[CONF_METER_PERIODICALLY_RESETTING]
    tariff_entity: Optional[str] = hass.data[DATA_UTILITY][entry_id][CONF_TARIFF_ENTITY]
    sensor_always_available: bool = config_entry.options.get(CONF_SENSOR_ALWAYS_AVAILABLE, False)
    meters: List[UtilityMeterSensor] = []
    tariffs: List[str] = config_entry.options[CONF_TARIFFS]
    
    if not tariffs:
        meter_sensor: UtilityMeterSensor = UtilityMeterSensor(
            cron_pattern=cron_pattern,
            delta_values=delta_values,
            meter_offset=meter_offset,
            meter_type=meter_type,
            name=name,
            net_consumption=net_consumption,
            parent_meter=entry_id,
            periodically_resetting=periodically_resetting,
            source_entity=source_entity_id,
            tariff_entity=tariff_entity,
            tariff=None,
            unique_id=entry_id,
            device_info=device_info,
            sensor_always_available=sensor_always_available
        )
        meters.append(meter_sensor)
        hass.data[DATA_UTILITY][entry_id][DATA_TARIFF_SENSORS].append(meter_sensor)
    else:
        for tariff in tariffs:
            meter_sensor = UtilityMeterSensor(
                cron_pattern=cron_pattern,
                delta_values=delta_values,
                meter_offset=meter_offset,
                meter_type=meter_type,
                name=f'{name} {tariff}',
                net_consumption=net_consumption,
                parent_meter=entry_id,
                periodically_resetting=periodically_resetting,
                source_entity=source_entity_id,
                tariff_entity=tariff_entity,
                tariff=tariff,
                unique_id=f'{entry_id}_{tariff}',
                device_info=device_info,
                sensor_always_available=sensor_always_available
            )
            meters.append(meter_sensor)
            hass.data[DATA_UTILITY][entry_id][DATA_TARIFF_SENSORS].append(meter_sensor)
    async_add_entities(meters)
    platform: entity_platform.EntityPlatform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_CALIBRATE_METER,
        {vol.Required(ATTR_VALUE): validate_is_number},
        'async_calibrate'
    )

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the utility meter sensor."""
    if discovery_info is None:
        _LOGGER.error("This platform is not available to configure from 'sensor:' in configuration.yaml")
        return
    meters: List[UtilityMeterSensor] = []
    for conf in discovery_info.values():
        meter: str = conf[CONF_METER]
        conf_meter_source: str = hass.data[DATA_UTILITY][meter][CONF_SOURCE_SENSOR]
        conf_meter_unique_id: Optional[str] = hass.data[DATA_UTILITY][meter].get(CONF_UNIQUE_ID)
        conf_sensor_tariff: Optional[str] = conf.get(CONF_TARIFF, 'single_tariff')
        conf_sensor_unique_id: Optional[str] = f'{conf_meter_unique_id}_{conf_sensor_tariff}' if conf_meter_unique_id else None
        conf_meter_name: str = hass.data[DATA_UTILITY][meter].get(CONF_NAME, meter)
        conf_sensor_tariff = conf.get(CONF_TARIFF)
        suggested_entity_id: Optional[str] = None
        if conf_sensor_tariff:
            conf_sensor_name: str = f'{conf_meter_name} {conf_sensor_tariff}'
            slug: str = slugify(f'{meter} {conf_sensor_tariff}')
            suggested_entity_id = f'sensor.{slug}'
        else:
            conf_sensor_name = conf_meter_name
        conf_meter_type: Optional[str] = hass.data[DATA_UTILITY][meter].get(CONF_METER_TYPE)
        conf_meter_offset: timedelta = hass.data[DATA_UTILITY][meter][CONF_METER_OFFSET]
        conf_meter_delta_values: bool = hass.data[DATA_UTILITY][meter][CONF_METER_DELTA_VALUES]
        conf_meter_net_consumption: bool = hass.data[DATA_UTILITY][meter][CONF_METER_NET_CONSUMPTION]
        conf_meter_periodically_resetting: bool = hass.data[DATA_UTILITY][meter][CONF_METER_PERIODICALLY_RESETTING]
        conf_meter_tariff_entity: Optional[str] = hass.data[DATA_UTILITY][meter].get(CONF_TARIFF_ENTITY)
        conf_cron_pattern: Optional[str] = hass.data[DATA_UTILITY][meter].get(CONF_CRON_PATTERN)
        conf_sensor_always_available: bool = hass.data[DATA_UTILITY][meter][CONF_SENSOR_ALWAYS_AVAILABLE]
        meter_sensor = UtilityMeterSensor(
            cron_pattern=conf_cron_pattern,
            delta_values=conf_meter_delta_values,
            meter_offset=conf_meter_offset,
            meter_type=conf_meter_type,
            name=conf_sensor_name,
            net_consumption=conf_meter_net_consumption,
            parent_meter=meter,
            periodically_resetting=conf_meter_periodically_resetting,
            source_entity=conf_meter_source,
            tariff_entity=conf_meter_tariff_entity,
            tariff=conf_sensor_tariff,
            unique_id=conf_sensor_unique_id,
            suggested_entity_id=suggested_entity_id,
            sensor_always_available=conf_sensor_always_available
        )
        meters.append(meter_sensor)
        hass.data[DATA_UTILITY][meter][DATA_TARIFF_SENSORS].append(meter_sensor)
    async_add_entities(meters)
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_CALIBRATE_METER,
        {vol.Required(ATTR_VALUE): validate_is_number},
        'async_calibrate'
    )

@dataclass
class UtilitySensorExtraStoredData(SensorExtraStoredData):
    """Object to hold extra stored data."""
    last_period: Decimal
    last_reset: Optional[datetime]
    last_valid_state: Optional[Decimal]
    status: str
    input_device_class: Optional[SensorDeviceClass]

    def as_dict(self) -> Dict[str, Any]:
        """Return a dict representation of the utility sensor data."""
        data: Dict[str, Any] = super().as_dict()
        data['last_period'] = str(self.last_period)
        if isinstance(self.last_reset, datetime):
            data['last_reset'] = self.last_reset.isoformat()
        data['last_valid_state'] = str(self.last_valid_state) if self.last_valid_state else None
        data['status'] = self.status
        data['input_device_class'] = str(self.input_device_class)
        return data

    @classmethod
    def from_dict(cls, restored: Dict[str, Any]) -> Optional[UtilitySensorExtraStoredData]:
        """Initialize a stored sensor state from a dict."""
        extra: Optional[SensorExtraStoredData] = SensorExtraStoredData.from_dict(restored)
        if extra is None:
            return None
        try:
            last_period: Decimal = Decimal(restored['last_period'])
            last_reset: Optional[datetime] = dt_util.parse_datetime(restored['last_reset'])
            last_valid_state: Optional[Decimal] = Decimal(restored['last_valid_state']) if restored.get('last_valid_state') else None
            status: str = restored['status']
            input_device_class: Optional[SensorDeviceClass] = try_parse_enum(SensorDeviceClass, restored.get('input_device_class'))
        except KeyError:
            return None
        except InvalidOperation:
            return None
        return cls(
            extra.native_value,
            extra.native_unit_of_measurement,
            last_period,
            last_reset,
            last_valid_state,
            status,
            input_device_class
        )

class UtilityMeterSensor(RestoreSensor):
    """Representation of an utility meter sensor."""
    _attr_translation_key: str = 'utility_meter'
    _attr_should_poll: bool = False
    _unrecorded_attributes: frozenset[str] = frozenset({ATTR_NEXT_RESET})

    def __init__(
        self,
        *,
        cron_pattern: Optional[str],
        delta_values: bool,
        meter_offset: timedelta,
        meter_type: Optional[str],
        name: str,
        net_consumption: bool,
        parent_meter: str,
        periodically_resetting: bool,
        source_entity: str,
        tariff_entity: Optional[str],
        tariff: Optional[str],
        unique_id: Optional[str],
        sensor_always_available: bool,
        suggested_entity_id: Optional[str] = None,
        device_info: Optional[DeviceInfo] = None
    ) -> None:
        """Initialize the Utility Meter sensor."""
        self._attr_unique_id: Optional[str] = unique_id
        self._attr_device_info: Optional[DeviceInfo] = device_info
        self.entity_id: Optional[str] = suggested_entity_id
        self._parent_meter: str = parent_meter
        self._sensor_source_id: str = source_entity
        self._last_period: Decimal = Decimal(0)
        self._last_reset: datetime = dt_util.utcnow()
        self._last_valid_state: Optional[Decimal] = None
        self._collecting: Optional[Callable[[], None]] = None
        self._attr_name: str = name
        self._input_device_class: Optional[SensorDeviceClass] = None
        self._attr_native_unit_of_measurement: Optional[str] = None
        self._period: Optional[str] = meter_type
        if meter_type is not None:
            self._cron_pattern: Optional[str] = PERIOD2CRON[meter_type].format(
                minute=meter_offset.seconds % 3600 // 60,
                hour=meter_offset.seconds // 3600,
                day=meter_offset.days + 1
            )
            _LOGGER.debug('CRON pattern: %s', self._cron_pattern)
        else:
            self._cron_pattern = cron_pattern
        self._sensor_always_available: bool = sensor_always_available
        self._sensor_delta_values: bool = delta_values
        self._sensor_net_consumption: bool = net_consumption
        self._sensor_periodically_resetting: bool = periodically_resetting
        self._tariff: Optional[str] = tariff
        self._tariff_entity: Optional[str] = tariff_entity
        self._next_reset: Optional[datetime] = None
        self._current_tz: Optional[str] = None
        self._config_scheduler()

    def _config_scheduler(self) -> None:
        self.scheduler: Optional[CronSim] = CronSim(self._cron_pattern, dt_util.now(dt_util.get_default_time_zone())) if self._cron_pattern else None

    def start(self, attributes: Dict[str, Any]) -> None:
        """Initialize unit and state upon source initial update."""
        self._input_device_class = attributes.get(ATTR_DEVICE_CLASS)
        self._attr_native_unit_of_measurement = attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        self._attr_native_value: Union[Decimal, int, float, None] = 0
        self.async_write_ha_state()

    @staticmethod
    def _validate_state(state: Optional[State]) -> Optional[Decimal]:
        """Parse the state as a Decimal if available. Throws DecimalException if the state is not a number."""
        try:
            return None if state is None or state.state in [STATE_UNAVAILABLE, STATE_UNKNOWN] else Decimal(state.state)
        except DecimalException:
            return None

    def calculate_adjustment(self, old_state: Optional[State], new_state: State) -> Optional[Decimal]:
        """Calculate the adjustment based on the old and new state."""
        if (new_state_val := self._validate_state(new_state)) is None:
            _LOGGER.warning('Invalid state %s', new_state.state)
            return None
        if self._sensor_delta_values:
            return new_state_val
        if not self._sensor_periodically_resetting and self._last_valid_state is not None:
            return new_state_val - self._last_valid_state
        if (old_state_val := self._validate_state(old_state)) is not None:
            return new_state_val - old_state_val
        _LOGGER.debug('%s received an invalid state change coming from %s (%s > %s)', self.name, self._sensor_source_id, old_state.state if old_state else None, new_state_val)
        return None

    @callback
    def async_reading(self, event: Event[EventStateChangedData]) -> None:
