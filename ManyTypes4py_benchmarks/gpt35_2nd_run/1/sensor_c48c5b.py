from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException, InvalidOperation
import logging
from typing import Any
import voluptuous as vol
from homeassistant.components.sensor import ATTR_LAST_RESET, DEVICE_CLASS_UNITS, RestoreSensor, SensorDeviceClass, SensorExtraStoredData, SensorStateClass
from homeassistant.components.sensor.recorder import _suggest_report_issue
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_DEVICE_CLASS, ATTR_UNIT_OF_MEASUREMENT, CONF_NAME, CONF_UNIQUE_ID, EVENT_CORE_CONFIG_UPDATE, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers import entity_platform, entity_registry as er
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_point_in_time, async_track_state_change_event
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.template import is_number
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from homeassistant.util.enum import try_parse_enum
from .const import ATTR_NEXT_RESET, ATTR_VALUE, BIMONTHLY, CONF_CRON_PATTERN, CONF_METER, CONF_METER_DELTA_VALUES, CONF_METER_NET_CONSUMPTION, CONF_METER_OFFSET, CONF_METER_PERIODICALLY_RESETTING, CONF_METER_TYPE, CONF_SENSOR_ALWAYS_AVAILABLE, CONF_SOURCE_SENSOR, CONF_TARIFF, CONF_TARIFF_ENTITY, CONF_TARIFFS, DAILY, DATA_TARIFF_SENSORS, DATA_UTILITY, HOURLY, MONTHLY, QUARTER_HOURLY, QUARTERLY, SERVICE_CALIBRATE_METER, SIGNAL_RESET_METER, WEEKLY, YEARLY

PERIOD2CRON: dict[str, str] = {QUARTER_HOURLY: '{minute}/15 * * * *', HOURLY: '{minute} * * * *', DAILY: '{minute} {hour} * * *', WEEKLY: '{minute} {hour} * * {day}', MONTHLY: '{minute} {hour} {day} * *', BIMONTHLY: '{minute} {hour} {day} */2 *', QUARTERLY: '{minute} {hour} {day} */3 *', YEARLY: '{minute} {hour} {day} 1/12 *'}
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

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Initialize Utility Meter config entry."""
    entry_id: str = config_entry.entry_id
    registry: er.EntityRegistry = er.async_get(hass)
    source_entity_id: str = er.async_validate_entity_id(registry, config_entry.options[CONF_SOURCE_SENSOR])
    device_info: dict[str, Any] = async_device_info_to_link_from_entity(hass, source_entity_id)
    cron_pattern: str = None
    delta_values: Any = config_entry.options[CONF_METER_DELTA_VALUES]
    meter_offset: timedelta = timedelta(days=config_entry.options[CONF_METER_OFFSET])
    meter_type: str = config_entry.options[CONF_METER_TYPE]
    if meter_type == 'none':
        meter_type = None
    name: str = config_entry.title
    net_consumption: bool = config_entry.options[CONF_METER_NET_CONSUMPTION]
    periodically_resetting: bool = config_entry.options[CONF_METER_PERIODICALLY_RESETTING]
    tariff_entity: str = hass.data[DATA_UTILITY][entry_id][CONF_TARIFF_ENTITY]
    sensor_always_available: bool = config_entry.options.get(CONF_SENSOR_ALWAYS_AVAILABLE, False)
    meters: list[UtilityMeterSensor] = []
    tariffs: list[str] = config_entry.options[CONF_TARIFFS]
    if not tariffs:
        meter_sensor: UtilityMeterSensor = UtilityMeterSensor(cron_pattern=cron_pattern, delta_values=delta_values, meter_offset=meter_offset, meter_type=meter_type, name=name, net_consumption=net_consumption, parent_meter=entry_id, periodically_resetting=periodically_resetting, source_entity=source_entity_id, tariff_entity=tariff_entity, tariff=None, unique_id=entry_id, device_info=device_info, sensor_always_available=sensor_always_available)
        meters.append(meter_sensor)
        hass.data[DATA_UTILITY][entry_id][DATA_TARIFF_SENSORS].append(meter_sensor)
    else:
        for tariff in tariffs:
            meter_sensor: UtilityMeterSensor = UtilityMeterSensor(cron_pattern=cron_pattern, delta_values=delta_values, meter_offset=meter_offset, meter_type=meter_type, name=f'{name} {tariff}', net_consumption=net_consumption, parent_meter=entry_id, periodically_resetting=periodically_resetting, source_entity=source_entity_id, tariff_entity=tariff_entity, tariff=tariff, unique_id=f'{entry_id}_{tariff}', device_info=device_info, sensor_always_available=sensor_always_available)
            meters.append(meter_sensor)
            hass.data[DATA_UTILITY][entry_id][DATA_TARIFF_SENSORS].append(meter_sensor)
    async_add_entities(meters)
    platform: entity_platform.EntityPlatform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_CALIBRATE_METER, {vol.Required(ATTR_VALUE): validate_is_number}, 'async_calibrate')

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    """Set up the utility meter sensor."""
    if discovery_info is None:
        _LOGGER.error("This platform is not available to configure from 'sensor:' in configuration.yaml")
        return
    meters: list[UtilityMeterSensor] = []
    for conf in discovery_info.values():
        meter: str = conf[CONF_METER]
        conf_meter_source: str = hass.data[DATA_UTILITY][meter][CONF_SOURCE_SENSOR]
        conf_meter_unique_id: str = hass.data[DATA_UTILITY][meter].get(CONF_UNIQUE_ID)
        conf_sensor_tariff: str = conf.get(CONF_TARIFF, 'single_tariff')
        conf_sensor_unique_id: str = f'{conf_meter_unique_id}_{conf_sensor_tariff}' if conf_meter_unique_id else None
        conf_meter_name: str = hass.data[DATA_UTILITY][meter].get(CONF_NAME, meter)
        conf_sensor_tariff: str = conf.get(CONF_TARIFF)
        suggested_entity_id: str = None
        if conf_sensor_tariff:
            conf_sensor_name: str = f'{conf_meter_name} {conf_sensor_tariff}'
            slug: str = slugify(f'{meter} {conf_sensor_tariff}')
            suggested_entity_id = f'sensor.{slug}'
        else:
            conf_sensor_name: str = conf_meter_name
        conf_meter_type: str = hass.data[DATA_UTILITY][meter].get(CONF_METER_TYPE)
        conf_meter_offset: timedelta = hass.data[DATA_UTILITY][meter][CONF_METER_OFFSET]
        conf_meter_delta_values: Any = hass.data[DATA_UTILITY][meter][CONF_METER_DELTA_VALUES]
        conf_meter_net_consumption: bool = hass.data[DATA_UTILITY][meter][CONF_METER_NET_CONSUMPTION]
        conf_meter_periodically_resetting: bool = hass.data[DATA_UTILITY][meter][CONF_METER_PERIODICALLY_RESETTING]
        conf_meter_tariff_entity: str = hass.data[DATA_UTILITY][meter].get(CONF_TARIFF_ENTITY)
        conf_cron_pattern: str = hass.data[DATA_UTILITY][meter].get(CONF_CRON_PATTERN)
        conf_sensor_always_available: bool = hass.data[DATA_UTILITY][meter][CONF_SENSOR_ALWAYS_AVAILABLE]
        meter_sensor: UtilityMeterSensor = UtilityMeterSensor(cron_pattern=conf_cron_pattern, delta_values=conf_meter_delta_values, meter_offset=conf_meter_offset, meter_type=conf_meter_type, name=conf_sensor_name, net_consumption=conf_meter_net_consumption, parent_meter=meter, periodically_resetting=conf_meter_periodically_resetting, source_entity=conf_meter_source, tariff_entity=conf_meter_tariff_entity, tariff=conf_sensor_tariff, unique_id=conf_sensor_unique_id, suggested_entity_id=suggested_entity_id, sensor_always_available=conf_sensor_always_available)
        meters.append(meter_sensor)
        hass.data[DATA_UTILITY][meter][DATA_TARIFF_SENSORS].append(meter_sensor)
    async_add_entities(meters)
    platform: entity_platform.EntityPlatform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_CALIBRATE_METER, {vol.Required(ATTR_VALUE): validate_is_number}, 'async_calibrate')

@dataclass
class UtilitySensorExtraStoredData(SensorExtraStoredData):
    """Object to hold extra stored data."""

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation of the utility sensor data."""
        data: dict[str, Any] = super().as_dict()
        data['last_period'] = str(self.last_period)
        if isinstance(self.last_reset, datetime):
            data['last_reset'] = self.last_reset.isoformat()
        data['last_valid_state'] = str(self.last_valid_state) if self.last_valid_state else None
        data['status'] = self.status
        data['input_device_class'] = str(self.input_device_class)
        return data

    @classmethod
    def from_dict(cls, restored: dict[str, Any]) -> UtilitySensorExtraStoredData:
        """Initialize a stored sensor state from a dict."""
        extra: SensorExtraStoredData = SensorExtraStoredData.from_dict(restored)
        if extra is None:
            return None
        try:
            last_period: Decimal = Decimal(restored['last_period'])
            last_reset: datetime = dt_util.parse_datetime(restored['last_reset'])
            last_valid_state: Decimal = Decimal(restored['last_valid_state']) if restored.get('last_valid_state') else None
            status: str = restored['status']
            input_device_class: SensorDeviceClass = try_parse_enum(SensorDeviceClass, restored.get('input_device_class'))
        except KeyError:
            return None
        except InvalidOperation:
            return None
        return cls(extra.native_value, extra.native_unit_of_measurement, last_period, last_reset, last_valid_state, status, input_device_class)

class UtilityMeterSensor(RestoreSensor):
    """Representation of an utility meter sensor."""
    _attr_translation_key: str = 'utility_meter'
    _attr_should_poll: bool = False
    _unrecorded_attributes: frozenset[str] = {ATTR_NEXT_RESET}

    def __init__(self, *, cron_pattern: str, delta_values: Any, meter_offset: timedelta, meter_type: str, name: str, net_consumption: bool, parent_meter: str, periodically_resetting: bool, source_entity: str, tariff_entity: str, tariff: str, unique_id: str, sensor_always_available: bool, suggested_entity_id: str = None, device_info: dict[str, Any] = None) -> None:
        """Initialize the Utility Meter sensor."""
        self._attr_unique_id: str = unique_id
        self._attr_device_info: dict[str, Any] = device_info
        self.entity_id: str = suggested_entity_id
        self._parent_meter: str = parent_meter
        self._sensor_source_id: str = source_entity
        self._last_period: Decimal = Decimal(0)
        self._last_reset: datetime = dt_util.utcnow()
        self._last_valid_state: Decimal = None
        self._collecting: Any = None
        self._attr_name: str = name
        self._input_device_class: SensorDeviceClass = None
        self._attr_native_unit_of_measurement: str = None
        self._period: str = meter_type
        if meter_type is not None:
            self._cron_pattern: str = PERIOD2CRON[meter_type].format(minute=meter_offset.seconds % 3600 // 60, hour=meter_offset.seconds // 3600, day=meter_offset.days + 1)
            _LOGGER.debug('CRON pattern: %s', self._cron_pattern)
        else:
            self._cron_pattern: str = cron_pattern
        self._sensor_always_available: bool = sensor_always_available
        self._sensor_delta_values: Any = delta_values
        self._sensor_net_consumption: bool = net_consumption
        self._sensor_periodically_resetting: bool = periodically_resetting
        self._tariff: str = tariff
        self._tariff_entity: str = tariff_entity
        self._next_reset: Any = None
        self._current_tz: Any = None
        self._config_scheduler()

    def _config_scheduler(self) -> None:
        self.scheduler: CronSim = CronSim(self._cron_pattern, dt_util.now(dt_util.get_default_time_zone())) if self._cron_pattern else None

    def start(self, attributes: dict[str, Any]) -> None:
        """Initialize unit and state upon source initial update."""
        self._input_device_class: str = attributes.get(ATTR_DEVICE_CLASS)
        self._attr_native_unit_of_measurement: str = attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        self._attr_native_value: int = 0
        self.async_write_ha_state()

    @staticmethod
    def _validate_state(state: State) -> Decimal:
        """Parse the state as a Decimal if available. Throws DecimalException if the state is not a number."""
        try:
            return None if state is None or state.state in [STATE_UNAVAILABLE, STATE_UNKNOWN] else Decimal(state.state)
        except DecimalException:
            return None

    def calculate_adjustment(self, old_state: State, new_state: State) -> Decimal:
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
    def async_reading(self, event: Event) -> None:
        """Handle the sensor state changes."""
        if (source_state := self.hass.states.get(self._sensor_source_id)) is None or source_state.state == STATE_UNAVAILABLE:
            if not self._sensor_always_available:
                self._attr_available: bool = False
                self.async_write_ha_state()
            return
        self._attr_available: bool = True
        old_state: State = event.data['old_state']
        new_state: State = event.data['new_state']
        if new_state is None:
            return
        new_state_attributes: dict[str, Any] = new_state.attributes or {}
        if (new_state_val := self._validate_state(new_state)) is None:
            _LOGGER.warning('%s received an invalid new state from %s : %s', self.name, self._sensor_source_id, new_state.state)
            return
        if self.native_value is None:
            for sensor in self.hass.data[DATA_UTILITY][self._parent_meter][DATA_TARIFF_SENSORS]:
                sensor.start(new_state_attributes)
                if self.native_unit_of_measurement is None:
                    _LOGGER.warning('Source sensor %s has no unit of measurement. Please %s', self._sensor_source_id, _suggest_report_issue(self.hass, self._sensor_source_id))
        if (adjustment := self.calculate_adjustment(old_state, new_state)) is not None and (self._sensor_net_consumption or adjustment >= 0):
            self._attr_native_value += adjustment
        self._input_device_class: str = new_state_attributes.get(ATTR_DEVICE_CLASS)
        self._attr_native_unit_of_measurement: str = new_state_attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        self._last_valid_state: Decimal = new_state_val
        self.async_write_ha_state()

    @callback
    def async_tariff_change(self, event: Event) -> None:
        """Handle tariff changes."""
        if (new_state := event.data['new_state']) is None:
            return
        self._change_status(new_state.state)

    def _change_status(self, tariff: str) -> None:
        if self._tariff == tariff:
            self._collecting: Any = async_track_state_change_event(self.hass, [self._sensor_source_id], self.async_reading)
        else:
            if self._collecting:
                self._collecting()
            self._collecting = None
        self._last_valid_state = None
        _LOGGER.debug('%s - %s - source <%s>', self.name, COLLECTING if self._collecting is not None else PAUSED, self._sensor_source_id)
        self.async_write_ha_state()

    async def _program_reset(self) -> None:
        """Program the reset of the utility meter."""
        if self.scheduler:
            self._next_reset: Any = next(self.scheduler)
            _LOGGER.debug('Next reset of %s is %s', self.entity_id, self._next_reset)
            self.async_on_remove(async_track_point_in_time(self.hass, self._async_reset_meter, self._next_reset))
            self.async_write_ha_state()

    async def _async_reset_meter(self, event: Event) -> None:
        """Reset the utility meter status."""
        await self._program_reset()
        await self.async_reset_meter(self._tariff_entity)

    async def async_reset_meter(self, entity_id: str) -> None:
        """Reset meter."""
        if self._tariff_entity is not None and self._tariff_entity != entity_id:
            return
        if self._tariff_entity is None and entity_id is not None and (self.entity_id != entity_id):
            return
        _LOGGER.debug('Reset utility meter <%s>', self.entity_id)
        self._last_reset: datetime = dt_util.utcnow()
        self._last_period: Decimal = Decimal