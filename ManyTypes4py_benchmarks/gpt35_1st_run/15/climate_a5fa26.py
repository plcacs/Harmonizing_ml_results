from __future__ import annotations
import collections
from typing import Any, List, Dict, Optional, Union
import voluptuous as vol
from homeassistant.components.climate import ATTR_TARGET_TEMP_HIGH, ATTR_TARGET_TEMP_LOW, FAN_AUTO, FAN_ON, PRESET_AWAY, PRESET_HOME, PRESET_NONE, PRESET_SLEEP, ClimateEntity, ClimateEntityFeature, HVACAction, HVACMode
from homeassistant.const import ATTR_ENTITY_ID, ATTR_TEMPERATURE, PRECISION_HALVES, PRECISION_TENTHS, STATE_OFF, STATE_ON, UnitOfTemperature
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv, device_registry as dr, entity_platform
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.unit_conversion import TemperatureConverter
from . import EcobeeConfigEntry, EcobeeData
from .const import _LOGGER, ATTR_ACTIVE_SENSORS, ATTR_AVAILABLE_SENSORS, DOMAIN, ECOBEE_AUX_HEAT_ONLY, ECOBEE_MODEL_TO_NAME, MANUFACTURER

ATTR_COOL_TEMP: str = 'cool_temp'
ATTR_END_DATE: str = 'end_date'
ATTR_END_TIME: str = 'end_time'
ATTR_FAN_MIN_ON_TIME: str = 'fan_min_on_time'
ATTR_FAN_MODE: str = 'fan_mode'
ATTR_HEAT_TEMP: str = 'heat_temp'
ATTR_RESUME_ALL: str = 'resume_all'
ATTR_START_DATE: str = 'start_date'
ATTR_START_TIME: str = 'start_time'
ATTR_VACATION_NAME: str = 'vacation_name'
ATTR_DST_ENABLED: str = 'dst_enabled'
ATTR_MIC_ENABLED: str = 'mic_enabled'
ATTR_AUTO_AWAY: str = 'auto_away'
ATTR_FOLLOW_ME: str = 'follow_me'
ATTR_SENSOR_LIST: str = 'device_ids'
ATTR_PRESET_MODE: str = 'preset_mode'
DEFAULT_RESUME_ALL: bool = False
PRESET_AWAY_INDEFINITELY: str = 'away_indefinitely'
PRESET_TEMPERATURE: str = 'temp'
PRESET_VACATION: str = 'vacation'
PRESET_HOLD_NEXT_TRANSITION: str = 'next_transition'
PRESET_HOLD_INDEFINITE: str = 'indefinite'
HAS_HEAT_PUMP: str = 'hasHeatPump'
DEFAULT_MIN_HUMIDITY: int = 15
DEFAULT_MAX_HUMIDITY: int = 50
HUMIDIFIER_MANUAL_MODE: str = 'manual'
ECOBEE_HVAC_TO_HASS: Dict[str, HVACMode] = collections.OrderedDict([('heat', HVACMode.HEAT), ('cool', HVACMode.COOL), ('auto', HVACMode.HEAT_COOL), ('off', HVACMode.OFF), (ECOBEE_AUX_HEAT_ONLY, HVACMode.HEAT)])
HASS_TO_ECOBEE_HVAC: Dict[HVACMode, str] = {v: k for k, v in ECOBEE_HVAC_TO_HASS.items() if k != ECOBEE_AUX_HEAT_ONLY}
ECOBEE_HVAC_ACTION_TO_HASS: Dict[str, HVACAction] = {'heatPump': HVACAction.HEATING, 'heatPump2': HVACAction.HEATING, 'heatPump3': HVACAction.HEATING, 'compCool1': HVACAction.COOLING, 'compCool2': HVACAction.COOLING, 'auxHeat1': HVACAction.HEATING, 'auxHeat2': HVACAction.HEATING, 'auxHeat3': HVACAction.HEATING, 'fan': HVACAction.FAN, 'humidifier': None, 'dehumidifier': HVACAction.DRYING, 'ventilator': HVACAction.FAN, 'economizer': HVACAction.FAN, 'compHotWater': None, 'auxHotWater': None, 'compWaterHeater': None}
ECOBEE_TO_HASS_PRESET: Dict[str, str] = {'Away': PRESET_AWAY, 'Home': PRESET_HOME, 'Sleep': PRESET_SLEEP}
HASS_TO_ECOBEE_PRESET: Dict[str, str] = {v: k for k, v in ECOBEE_TO_HASS_PRESET.items()}
PRESET_TO_ECOBEE_HOLD: Dict[str, str] = {PRESET_HOLD_NEXT_TRANSITION: 'nextTransition', PRESET_HOLD_INDEFINITE: 'indefinite'}
SERVICE_CREATE_VACATION: str = 'create_vacation'
SERVICE_DELETE_VACATION: str = 'delete_vacation'
SERVICE_RESUME_PROGRAM: str = 'resume_program'
SERVICE_SET_FAN_MIN_ON_TIME: str = 'set_fan_min_on_time'
SERVICE_SET_DST_MODE: str = 'set_dst_mode'
SERVICE_SET_MIC_MODE: str = 'set_mic_mode'
SERVICE_SET_OCCUPANCY_MODES: str = 'set_occupancy_modes'
SERVICE_SET_SENSORS_USED_IN_CLIMATE: str = 'set_sensors_used_in_climate'
DTGROUP_START_INCLUSIVE_MSG: str = f'{ATTR_START_DATE} and {ATTR_START_TIME} must be specified together'
DTGROUP_END_INCLUSIVE_MSG: str = f'{ATTR_END_DATE} and {ATTR_END_TIME} must be specified together'
CREATE_VACATION_SCHEMA: vol.Schema = vol.Schema({vol.Required(ATTR_ENTITY_ID): cv.entity_id, vol.Required(ATTR_VACATION_NAME): vol.All(cv.string, vol.Length(max=12)), vol.Required(ATTR_COOL_TEMP): vol.Coerce(float), vol.Required(ATTR_HEAT_TEMP): vol.Coerce(float), vol.Inclusive(ATTR_START_DATE, 'dtgroup_start', msg=DTGROUP_START_INCLUSIVE_MSG): ecobee_date, vol.Inclusive(ATTR_START_TIME, 'dtgroup_start', msg=DTGROUP_START_INCLUSIVE_MSG): ecobee_time, vol.Inclusive(ATTR_END_DATE, 'dtgroup_end', msg=DTGROUP_END_INCLUSIVE_MSG): ecobee_date, vol.Inclusive(ATTR_END_TIME, 'dtgroup_end', msg=DTGROUP_END_INCLUSIVE_MSG): ecobee_time, vol.Optional(ATTR_FAN_MODE, default='auto'): vol.Any('auto', 'on'), vol.Optional(ATTR_FAN_MIN_ON_TIME, default=0): vol.All(int, vol.Range(min=0, max=60))})
DELETE_VACATION_SCHEMA: vol.Schema = vol.Schema({vol.Required(ATTR_ENTITY_ID): cv.entity_id, vol.Required(ATTR_VACATION_NAME): vol.All(cv.string, vol.Length(max=12)})
RESUME_PROGRAM_SCHEMA: vol.Schema = vol.Schema({vol.Optional(ATTR_ENTITY_ID): cv.entity_ids, vol.Optional(ATTR_RESUME_ALL, default=DEFAULT_RESUME_ALL): cv.boolean})
SET_FAN_MIN_ON_TIME_SCHEMA: vol.Schema = vol.Schema({vol.Optional(ATTR_ENTITY_ID): cv.entity_ids, vol.Required(ATTR_FAN_MIN_ON_TIME): vol.Coerce(int)})
SUPPORT_FLAGS: ClimateEntityFeature = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE | ClimateEntityFeature.TARGET_TEMPERATURE_RANGE | ClimateEntityFeature.FAN_MODE

async def async_setup_entry(hass: HomeAssistant, config_entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the ecobee thermostat."""
    data: EcobeeData = config_entry.runtime_data
    entities: List[Thermostat] = []
    for index in range(len(data.ecobee.thermostats)):
        thermostat = data.ecobee.get_thermostat(index)
        if thermostat['modelNumber'] not in ECOBEE_MODEL_TO_NAME:
            _LOGGER.error('Model number for ecobee thermostat %s not recognized. Please visit this link to open a new issue: https://github.com/home-assistant/core/issues and include the following information: Unrecognized model number: %s', thermostat['name'], thermostat['modelNumber'])
        entities.append(Thermostat(data, index, thermostat, hass))
    async_add_entities(entities, True)
    platform = entity_platform.async_get_current_platform()

    def create_vacation_service(service: ServiceCall) -> None:
        """Create a vacation on the target thermostat."""
        entity_id: str = service.data[ATTR_ENTITY_ID]
        for thermostat in entities:
            if thermostat.entity_id == entity_id:
                thermostat.create_vacation(service.data)
                thermostat.schedule_update_ha_state(True)
                break

    def delete_vacation_service(service: ServiceCall) -> None:
        """Delete a vacation on the target thermostat."""
        entity_id: str = service.data[ATTR_ENTITY_ID]
        vacation_name: str = service.data[ATTR_VACATION_NAME]
        for thermostat in entities:
            if thermostat.entity_id == entity_id:
                thermostat.delete_vacation(vacation_name)
                thermostat.schedule_update_ha_state(True)
                break

    def fan_min_on_time_set_service(service: ServiceCall) -> None:
        """Set the minimum fan on time on the target thermostats."""
        entity_id: Optional[Union[str, List[str]]] = service.data.get(ATTR_ENTITY_ID)
        fan_min_on_time: int = service.data[ATTR_FAN_MIN_ON_TIME]
        if entity_id:
            target_thermostats = [entity for entity in entities if entity.entity_id in entity_id]
        else:
            target_thermostats = entities
        for thermostat in target_thermostats:
            thermostat.set_fan_min_on_time(str(fan_min_on_time))
            thermostat.schedule_update_ha_state(True)

    def resume_program_set_service(service: ServiceCall) -> None:
        """Resume the program on the target thermostats."""
        entity_id: Optional[Union[str, List[str]]] = service.data.get(ATTR_ENTITY_ID)
        resume_all: bool = service.data.get(ATTR_RESUME_ALL)
        if entity_id:
            target_thermostats = [entity for entity in entities if entity.entity_id in entity_id]
        else:
            target_thermostats = entities
        for thermostat in target_thermostats:
            thermostat.resume_program(resume_all)
            thermostat.schedule_update_ha_state(True)
    hass.services.async_register(DOMAIN, SERVICE_CREATE_VACATION, create_vacation_service, schema=CREATE_VACATION_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_DELETE_VACATION, delete_vacation_service, schema=DELETE_VACATION_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_SET_FAN_MIN_ON_TIME, fan_min_on_time_set_service, schema=SET_FAN_MIN_ON_TIME_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_RESUME_PROGRAM, resume_program_set_service, schema=RESUME_PROGRAM_SCHEMA)
    platform.async_register_entity_service(SERVICE_SET_DST_MODE, {vol.Required(ATTR_DST_ENABLED): cv.boolean}, 'set_dst_mode')
    platform.async_register_entity_service(SERVICE_SET_MIC_MODE, {vol.Required(ATTR_MIC_ENABLED): cv.boolean}, 'set_mic_mode')
    platform.async_register_entity_service(SERVICE_SET_OCCUPANCY_MODES, {vol.Optional(ATTR_AUTO_AWAY): cv.boolean, vol.Optional(ATTR_FOLLOW_ME): cv.boolean}, 'set_occupancy_modes')
    platform.async_register_entity_service(SERVICE_SET_SENSORS_USED_IN_CLIMATE, {vol.Optional(ATTR_PRESET_MODE): cv.string, vol.Required(ATTR_SENSOR_LIST): cv.ensure_list}, 'set_sensors_used_in_climate')

class Thermostat(ClimateEntity):
    """A thermostat class for Ecobee."""
    _attr_precision: str = PRECISION_TENTHS
    _attr_temperature_unit: str = UnitOfTemperature.FAHRENHEIT
    _attr_min_humidity: int = DEFAULT_MIN_HUMIDITY
    _attr_max_humidity: int = DEFAULT_MAX_HUMIDITY
    _attr_fan_modes: List[str] = [FAN_AUTO, FAN_ON]
    _attr_name: Optional[str] = None
    _attr_has_entity_name: bool = True
    _attr_translation_key: str = 'ecobee'

    def __init__(self, data: EcobeeData, thermostat_index: int, thermostat: Dict[str, Any], hass: HomeAssistant) -> None:
        """Initialize the thermostat."""
        self.data = data
        self.thermostat_index = thermostat_index
        self.thermostat = thermostat
        self._attr_unique_id: str = self.thermostat['identifier']
        self.vacation: Optional[str] = None
        self._last_active_hvac_mode: HVACMode = HVACMode.HEAT_COOL
        self._last_hvac_mode_before_aux_heat: HVACMode = HVACMode.HEAT_COOL
        self._hass: HomeAssistant = hass
        self._attr_hvac_modes: List[HVACMode] = []
        if self.settings['heatStages'] or self.settings['hasHeatPump']:
            self._attr_hvac_modes.append(HVACMode.HEAT)
        if self.settings['coolStages']:
            self._attr_hvac_modes.append(HVACMode.COOL)
        if len(self._attr_hvac_modes) == 2:
            self._attr_hvac_modes.insert(0, HVACMode.HEAT_COOL)
        self._attr_hvac_modes.append(HVACMode.OFF)
        self._sensors: List[str] = self.remote_sensors
        self._preset_modes: Dict[str, str] = {comfort['climateRef']: comfort['name'] for comfort in self.thermostat['program']['climates']}
        self.update_without_throttle: bool = False

    async def async_update(self) -> None:
        """Get the latest state from the thermostat."""
        if self.update_without_throttle:
            await self.data.update(no_throttle=True)
            self.update_without_throttle = False
        else:
            await self.data.update()
        self.thermostat = self.data.ecobee.get_thermostat(self.thermostat_index)
        if self.hvac_mode != HVACMode.OFF:
            self._last_active_hvac_mode = self.hvac_mode

    @property
    def available(self) -> bool:
        """Return if device is available."""
        return self.thermostat['runtime']['connected']

    @property
    def supported_features(self) -> ClimateEntityFeature:
        """Return the list of supported features."""
        supported: ClimateEntityFeature = SUPPORT_FLAGS
        if self.has_humidifier_control:
            supported = supported | ClimateEntityFeature.TARGET_HUMIDITY
        if len(self.hvac_modes) > 1 and HVACMode.OFF in self.hvac_modes:
            supported = supported | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
        return supported

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information for this ecobee thermostat."""
        try:
            model = f'{ECOBEE_MODEL_TO_NAME[self.thermostat['modelNumber']]} Thermostat'
        except KeyError:
            model = None
        return DeviceInfo(identifiers={(DOMAIN, self.thermostat['identifier'])}, manufacturer=MANUFACTURER, model=model, name=self.thermostat['name'])

    @property
    def current_temperature(self) -> float:
        """Return the current temperature."""
        return self.thermostat['runtime']['actualTemperature'] / 10.0

    @property
    def target_temperature_low(self) -> Optional[float]:
        """Return the lower bound temperature we try to reach."""
        if self.hvac_mode == HVACMode.HEAT_COOL:
            return self.thermostat['runtime']['desiredHeat'] / 10.0
        return None

    @property
    def target_temperature_high(self) -> Optional[float]:
        """Return the upper bound temperature we try to reach."""
        if self.hvac_mode == HVACMode.HEAT_COOL:
            return self.thermostat['runtime']['desiredCool'] / 10.0
        return None

    @property
    def target_temperature_step(self) -> str:
        """Set target temperature step to halves."""
        return PRECISION_HALVES

    @property
    def settings(self) -> Dict[str, Any]:
        """Return the settings of the thermostat."""
        return self.thermostat['settings']

    @property
    def has_humidifier_control(self) -> bool:
        """Return true if humidifier connected to thermostat and set to manual/on mode."""
        return bool(self.settings.get('hasHumidifier')) and self.settings.get('humidifierMode') == HUMIDIFIER_MANUAL_MODE

    @property
    def target_humidity(self) -> Optional[int]:
        """Return the desired humidity set point."""
        if self.has_humidifier_control:
            return self.thermostat['runtime']['desiredHumidity']
        return None

    @property
    def target_temperature(self) -> Optional[float]:
        """Return the temperature we try to reach."""
        if self.hvac_mode == HVACMode.HEAT_COOL:
            return None
        if self.hvac_mode == HVACMode.HEAT:
            return self.thermostat['runtime']['desiredHeat'] / 10.0
        if self.hvac_mode == HVACMode.COOL:
            return self.thermostat['runtime']['desiredCool'] / 10.0
        return None

    @property
    def fan(self) -> str:
        """Return the current fan status."""
        if 'fan' in self.thermostat['equipmentStatus']:
            return STATE_ON
        return STATE_OFF

    @property
    def fan_mode(self) -> str:
        """Return the fan setting."""
        return self.thermostat['runtime']['desiredFanMode']

    @property
    def preset_mode(self) -> Optional[str]:
        """Return current preset mode."""
        events = self.thermostat['events']
        for event in events:
            if not event['running']:
                continue
            if event['type'] == 'hold':
                if event['holdClimateRef'] == 'away' and is_indefinite_hold(event['startDate'], event['endDate']):
                    return PRESET_AWAY_INDEFINITELY
                if (name := self.comfort_settings.get(event['holdClimateRef'])):
                    return ECOBEE_TO_HASS_PRESET.get(name, name)
                return PRESET_TEMPERATURE
            if event['type'].startswith('auto'):
                return event['type'][4:].lower()
            if event['type'] == 'vacation':
                self.vacation = event['name']
                return PRESET_VACATION
        if (name := self.comfort_settings.get(self.thermostat['program']['currentClimateRef'])):
            return ECOBEE_TO_HASS_PRESET.get(name, name)
        return None

    @property
    def hvac_mode(self) -> HVACMode:
        """Return current operation."""
        return ECOBEE_HVAC_TO_HASS[self.settings['hvacMode']]

    @property
    def current_humidity(self) -> Optional[int]:
        """Return the current humidity."""
        try:
            return int(self.thermostat['runtime']['actualHumidity'])
        except KeyError:
            return None

    @property
    def hvac_action(self) -> HVACAction:
        """Return current HVAC action."""
        if self.thermostat['equipmentStatus'] == '':
            return HVACAction.IDLE
        actions = [ECOB