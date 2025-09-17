"""Support for Ecobee Thermostats."""

from __future__ import annotations
import collections
from typing import Any, Optional, Dict, List
import voluptuous as vol
from homeassistant.components.climate import (
    ATTR_TARGET_TEMP_HIGH,
    ATTR_TARGET_TEMP_LOW,
    FAN_AUTO,
    FAN_ON,
    PRESET_AWAY,
    PRESET_HOME,
    PRESET_NONE,
    PRESET_SLEEP,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    STATE_OFF,
    STATE_ON,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv, device_registry as dr, entity_platform
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.unit_conversion import TemperatureConverter
from homeassistant.config_entries import ConfigEntry
from . import EcobeeConfigEntry, EcobeeData
from .const import _LOGGER, ATTR_ACTIVE_SENSORS, ATTR_AVAILABLE_SENSORS, DOMAIN, ECOBEE_AUX_HEAT_ONLY, ECOBEE_MODEL_TO_NAME, MANUFACTURER
from .util import ecobee_date, ecobee_time, is_indefinite_hold

ATTR_COOL_TEMP = "cool_temp"
ATTR_END_DATE = "end_date"
ATTR_END_TIME = "end_time"
ATTR_FAN_MIN_ON_TIME = "fan_min_on_time"
ATTR_FAN_MODE = "fan_mode"
ATTR_HEAT_TEMP = "heat_temp"
ATTR_RESUME_ALL = "resume_all"
ATTR_START_DATE = "start_date"
ATTR_START_TIME = "start_time"
ATTR_VACATION_NAME = "vacation_name"
ATTR_DST_ENABLED = "dst_enabled"
ATTR_MIC_ENABLED = "mic_enabled"
ATTR_AUTO_AWAY = "auto_away"
ATTR_FOLLOW_ME = "follow_me"
ATTR_SENSOR_LIST = "device_ids"
ATTR_PRESET_MODE = "preset_mode"
DEFAULT_RESUME_ALL = False
PRESET_AWAY_INDEFINITELY = "away_indefinitely"
PRESET_TEMPERATURE = "temp"
PRESET_VACATION = "vacation"
PRESET_HOLD_NEXT_TRANSITION = "next_transition"
PRESET_HOLD_INDEFINITE = "indefinite"
HAS_HEAT_PUMP = "hasHeatPump"
DEFAULT_MIN_HUMIDITY = 15
DEFAULT_MAX_HUMIDITY = 50
HUMIDIFIER_MANUAL_MODE = "manual"

ECOBEE_HVAC_TO_HASS: Dict[str, HVACMode] = collections.OrderedDict(
    [("heat", HVACMode.HEAT), ("cool", HVACMode.COOL), ("auto", HVACMode.HEAT_COOL), ("off", HVACMode.OFF), (ECOBEE_AUX_HEAT_ONLY, HVACMode.HEAT)]
)
HASS_TO_ECOBEE_HVAC: Dict[HVACMode, str] = {v: k for k, v in ECOBEE_HVAC_TO_HASS.items() if k != ECOBEE_AUX_HEAT_ONLY}
ECOBEE_HVAC_ACTION_TO_HASS: Dict[str, Optional[HVACAction]] = {
    "heatPump": HVACAction.HEATING,
    "heatPump2": HVACAction.HEATING,
    "heatPump3": HVACAction.HEATING,
    "compCool1": HVACAction.COOLING,
    "compCool2": HVACAction.COOLING,
    "auxHeat1": HVACAction.HEATING,
    "auxHeat2": HVACAction.HEATING,
    "auxHeat3": HVACAction.HEATING,
    "fan": HVACAction.FAN,
    "humidifier": None,
    "dehumidifier": HVACAction.DRYING,
    "ventilator": HVACAction.FAN,
    "economizer": HVACAction.FAN,
    "compHotWater": None,
    "auxHotWater": None,
    "compWaterHeater": None,
}
ECOBEE_TO_HASS_PRESET: Dict[str, str] = {"Away": PRESET_AWAY, "Home": PRESET_HOME, "Sleep": PRESET_SLEEP}
HASS_TO_ECOBEE_PRESET: Dict[str, str] = {v: k for k, v in ECOBEE_TO_HASS_PRESET.items()}
PRESET_TO_ECOBEE_HOLD: Dict[str, str] = {PRESET_HOLD_NEXT_TRANSITION: "nextTransition", PRESET_HOLD_INDEFINITE: "indefinite"}
SERVICE_CREATE_VACATION = "create_vacation"
SERVICE_DELETE_VACATION = "delete_vacation"
SERVICE_RESUME_PROGRAM = "resume_program"
SERVICE_SET_FAN_MIN_ON_TIME = "set_fan_min_on_time"
SERVICE_SET_DST_MODE = "set_dst_mode"
SERVICE_SET_MIC_MODE = "set_mic_mode"
SERVICE_SET_OCCUPANCY_MODES = "set_occupancy_modes"
SERVICE_SET_SENSORS_USED_IN_CLIMATE = "set_sensors_used_in_climate"
DTGROUP_START_INCLUSIVE_MSG = f"{ATTR_START_DATE} and {ATTR_START_TIME} must be specified together"
DTGROUP_END_INCLUSIVE_MSG = f"{ATTR_END_DATE} and {ATTR_END_TIME} must be specified together"

CREATE_VACATION_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required(ATTR_VACATION_NAME): vol.All(cv.string, vol.Length(max=12)),
        vol.Required(ATTR_COOL_TEMP): vol.Coerce(float),
        vol.Required(ATTR_HEAT_TEMP): vol.Coerce(float),
        vol.Inclusive(ATTR_START_DATE, "dtgroup_start", msg=DTGROUP_START_INCLUSIVE_MSG): ecobee_date,
        vol.Inclusive(ATTR_START_TIME, "dtgroup_start", msg=DTGROUP_START_INCLUSIVE_MSG): ecobee_time,
        vol.Inclusive(ATTR_END_DATE, "dtgroup_end", msg=DTGROUP_END_INCLUSIVE_MSG): ecobee_date,
        vol.Inclusive(ATTR_END_TIME, "dtgroup_end", msg=DTGROUP_END_INCLUSIVE_MSG): ecobee_time,
        vol.Optional(ATTR_FAN_MODE, default="auto"): vol.Any("auto", "on"),
        vol.Optional(ATTR_FAN_MIN_ON_TIME, default=0): vol.All(int, vol.Range(min=0, max=60)),
    }
)
DELETE_VACATION_SCHEMA = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): cv.entity_id, vol.Required(ATTR_VACATION_NAME): vol.All(cv.string, vol.Length(max=12))}
)
RESUME_PROGRAM_SCHEMA = vol.Schema(
    {vol.Optional(ATTR_ENTITY_ID): cv.entity_ids, vol.Optional(ATTR_RESUME_ALL, default=DEFAULT_RESUME_ALL): cv.boolean}
)
SET_FAN_MIN_ON_TIME_SCHEMA = vol.Schema({vol.Optional(ATTR_ENTITY_ID): cv.entity_ids, vol.Required(ATTR_FAN_MIN_ON_TIME): vol.Coerce(int)})
SUPPORT_FLAGS = (
    ClimateEntityFeature.TARGET_TEMPERATURE
    | ClimateEntityFeature.PRESET_MODE
    | ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
    | ClimateEntityFeature.FAN_MODE
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the ecobee thermostat."""
    data: EcobeeData = config_entry.runtime_data
    entities: List[Thermostat] = []
    for index in range(len(data.ecobee.thermostats)):
        thermostat: Dict[str, Any] = data.ecobee.get_thermostat(index)
        if thermostat["modelNumber"] not in ECOBEE_MODEL_TO_NAME:
            _LOGGER.error(
                "Model number for ecobee thermostat %s not recognized. Please visit this link to open a new issue: https://github.com/home-assistant/core/issues and include the following information: Unrecognized model number: %s",
                thermostat["name"],
                thermostat["modelNumber"],
            )
        entities.append(Thermostat(data, index, thermostat, hass))
    async_add_entities(entities, True)
    platform = entity_platform.async_get_current_platform()

    def create_vacation_service(service: ServiceCall) -> None:
        """Create a vacation on the target thermostat."""
        entity_id = service.data[ATTR_ENTITY_ID]
        for thermostat in entities:
            if thermostat.entity_id == entity_id:
                thermostat.create_vacation(service.data)
                thermostat.schedule_update_ha_state(True)
                break

    def delete_vacation_service(service: ServiceCall) -> None:
        """Delete a vacation on the target thermostat."""
        entity_id = service.data[ATTR_ENTITY_ID]
        vacation_name = service.data[ATTR_VACATION_NAME]
        for thermostat in entities:
            if thermostat.entity_id == entity_id:
                thermostat.delete_vacation(vacation_name)
                thermostat.schedule_update_ha_state(True)
                break

    def fan_min_on_time_set_service(service: ServiceCall) -> None:
        """Set the minimum fan on time on the target thermostats."""
        entity_id = service.data.get(ATTR_ENTITY_ID)
        fan_min_on_time = service.data[ATTR_FAN_MIN_ON_TIME]
        if entity_id:
            target_thermostats = [entity for entity in entities if entity.entity_id in entity_id]
        else:
            target_thermostats = entities
        for thermostat in target_thermostats:
            thermostat.set_fan_min_on_time(str(fan_min_on_time))
            thermostat.schedule_update_ha_state(True)

    def resume_program_set_service(service: ServiceCall) -> None:
        """Resume the program on the target thermostats."""
        entity_id = service.data.get(ATTR_ENTITY_ID)
        resume_all = service.data.get(ATTR_RESUME_ALL)
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
    platform.async_register_entity_service(SERVICE_SET_DST_MODE, {vol.Required(ATTR_DST_ENABLED): cv.boolean}, "set_dst_mode")
    platform.async_register_entity_service(SERVICE_SET_MIC_MODE, {vol.Required(ATTR_MIC_ENABLED): cv.boolean}, "set_mic_mode")
    platform.async_register_entity_service(SERVICE_SET_OCCUPANCY_MODES, {vol.Optional(ATTR_AUTO_AWAY): cv.boolean, vol.Optional(ATTR_FOLLOW_ME): cv.boolean}, "set_occupancy_modes")
    platform.async_register_entity_service(
        SERVICE_SET_SENSORS_USED_IN_CLIMATE,
        {vol.Optional(ATTR_PRESET_MODE): cv.string, vol.Required(ATTR_SENSOR_LIST): cv.ensure_list},
        "set_sensors_used_in_climate",
    )


class Thermostat(ClimateEntity):
    """A thermostat class for Ecobee."""
    _attr_precision = PRECISION_TENTHS
    _attr_temperature_unit = UnitOfTemperature.FAHRENHEIT
    _attr_min_humidity = DEFAULT_MIN_HUMIDITY
    _attr_max_humidity = DEFAULT_MAX_HUMIDITY
    _attr_fan_modes = [FAN_AUTO, FAN_ON]
    _attr_name: Optional[str] = None
    _attr_has_entity_name = True
    _attr_translation_key = "ecobee"

    def __init__(self, data: EcobeeData, thermostat_index: int, thermostat: Dict[str, Any], hass: HomeAssistant) -> None:
        """Initialize the thermostat."""
        self.data: EcobeeData = data
        self.thermostat_index: int = thermostat_index
        self.thermostat: Dict[str, Any] = thermostat
        self._attr_unique_id: str = self.thermostat["identifier"]
        self.vacation: Optional[str] = None
        self._last_active_hvac_mode: HVACMode = HVACMode.HEAT_COOL
        self._last_hvac_mode_before_aux_heat: HVACMode = HVACMode.HEAT_COOL
        self._hass: HomeAssistant = hass
        self._attr_hvac_modes = []  # type: List[HVACMode]
        if self.settings["heatStages"] or self.settings["hasHeatPump"]:
            self._attr_hvac_modes.append(HVACMode.HEAT)
        if self.settings["coolStages"]:
            self._attr_hvac_modes.append(HVACMode.COOL)
        if len(self._attr_hvac_modes) == 2:
            self._attr_hvac_modes.insert(0, HVACMode.HEAT_COOL)
        self._attr_hvac_modes.append(HVACMode.OFF)
        self._sensors: List[str] = self.remote_sensors
        self._preset_modes: Dict[str, str] = {comfort["climateRef"]: comfort["name"] for comfort in self.thermostat["program"]["climates"]}
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
        return self.thermostat["runtime"]["connected"]

    @property
    def supported_features(self) -> int:
        """Return the list of supported features."""
        supported: int = SUPPORT_FLAGS
        if self.has_humidifier_control:
            supported = supported | ClimateEntityFeature.TARGET_HUMIDITY
        if len(self.hvac_modes) > 1 and HVACMode.OFF in self.hvac_modes:
            supported = supported | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
        return supported

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information for this ecobee thermostat."""
        try:
            model: Optional[str] = f"{ECOBEE_MODEL_TO_NAME[self.thermostat['modelNumber']]} Thermostat"
        except KeyError:
            model = None
        return DeviceInfo(identifiers={(DOMAIN, self.thermostat["identifier"])}, manufacturer=MANUFACTURER, model=model, name=self.thermostat["name"])

    @property
    def current_temperature(self) -> float:
        """Return the current temperature."""
        return self.thermostat["runtime"]["actualTemperature"] / 10.0

    @property
    def target_temperature_low(self) -> Optional[float]:
        """Return the lower bound temperature we try to reach."""
        if self.hvac_mode == HVACMode.HEAT_COOL:
            return self.thermostat["runtime"]["desiredHeat"] / 10.0
        return None

    @property
    def target_temperature_high(self) -> Optional[float]:
        """Return the upper bound temperature we try to reach."""
        if self.hvac_mode == HVACMode.HEAT_COOL:
            return self.thermostat["runtime"]["desiredCool"] / 10.0
        return None

    @property
    def target_temperature_step(self) -> float:
        """Set target temperature step to halves."""
        return PRECISION_HALVES

    @property
    def settings(self) -> Dict[str, Any]:
        """Return the settings of the thermostat."""
        return self.thermostat["settings"]

    @property
    def has_humidifier_control(self) -> bool:
        """Return true if humidifier connected to thermostat and set to manual/on mode."""
        return bool(self.settings.get("hasHumidifier")) and self.settings.get("humidifierMode") == HUMIDIFIER_MANUAL_MODE

    @property
    def target_humidity(self) -> Optional[int]:
        """Return the desired humidity set point."""
        if self.has_humidifier_control:
            return self.thermostat["runtime"]["desiredHumidity"]
        return None

    @property
    def target_temperature(self) -> Optional[float]:
        """Return the temperature we try to reach."""
        if self.hvac_mode == HVACMode.HEAT_COOL:
            return None
        if self.hvac_mode == HVACMode.HEAT:
            return self.thermostat["runtime"]["desiredHeat"] / 10.0
        if self.hvac_mode == HVACMode.COOL:
            return self.thermostat["runtime"]["desiredCool"] / 10.0
        return None

    @property
    def fan(self) -> str:
        """Return the current fan status."""
        if "fan" in self.thermostat["equipmentStatus"]:
            return STATE_ON
        return STATE_OFF

    @property
    def fan_mode(self) -> str:
        """Return the fan setting."""
        return self.thermostat["runtime"]["desiredFanMode"]

    @property
    def preset_mode(self) -> Optional[str]:
        """Return current preset mode."""
        events: List[Dict[str, Any]] = self.thermostat["events"]
        for event in events:
            if not event["running"]:
                continue
            if event["type"] == "hold":
                if event["holdClimateRef"] == "away" and is_indefinite_hold(event["startDate"], event["endDate"]):
                    return PRESET_AWAY_INDEFINITELY
                if (name := self.comfort_settings.get(event["holdClimateRef"])):
                    return ECOBEE_TO_HASS_PRESET.get(name, name)
                return PRESET_TEMPERATURE
            if event["type"].startswith("auto"):
                return event["type"][4:].lower()
            if event["type"] == "vacation":
                self.vacation = event["name"]
                return PRESET_VACATION
        if (name := self.comfort_settings.get(self.thermostat["program"]["currentClimateRef"])):
            return ECOBEE_TO_HASS_PRESET.get(name, name)
        return None

    @property
    def hvac_mode(self) -> HVACMode:
        """Return current operation."""
        return ECOBEE_HVAC_TO_HASS[self.settings["hvacMode"]]

    @property
    def current_humidity(self) -> Optional[int]:
        """Return the current humidity."""
        try:
            return int(self.thermostat["runtime"]["actualHumidity"])
        except KeyError:
            return None

    @property
    def hvac_action(self) -> HVACAction:
        """Return current HVAC action.

        Ecobee returns a CSV string with different equipment that is active.
        We are prioritizing any heating/cooling equipment, otherwise look at
        drying/fanning. Idle if nothing going on.
        We are unable to map all actions to HA equivalents.
        """
        if self.thermostat["equipmentStatus"] == "":
            return HVACAction.IDLE
        actions: List[HVACAction] = [
            ECOBEE_HVAC_ACTION_TO_HASS[status]
            for status in self.thermostat["equipmentStatus"].split(",")
            if ECOBEE_HVAC_ACTION_TO_HASS[status] is not None
        ]
        for action in (HVACAction.HEATING, HVACAction.COOLING, HVACAction.DRYING, HVACAction.FAN):
            if action in actions:
                return action
        return HVACAction.IDLE

    _unrecorded_attributes = frozenset({ATTR_AVAILABLE_SENSORS, ATTR_ACTIVE_SENSORS})

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return device specific state attributes."""
        status: str = self.thermostat["equipmentStatus"]
        return {
            "fan": self.fan,
            "climate_mode": self.comfort_settings.get(self.thermostat["program"]["currentClimateRef"]),
            "equipment_running": status,
            "fan_min_on_time": self.settings["fanMinOnTime"],
            ATTR_AVAILABLE_SENSORS: self.remote_sensor_devices,
            ATTR_ACTIVE_SENSORS: self.active_sensor_devices_in_preset_mode,
        }

    @property
    def remote_sensors(self) -> List[str]:
        """Return the remote sensor names of the thermostat."""
        sensors_info: List[Dict[str, Any]] = self.thermostat.get("remoteSensors", [])
        return [sensor["name"] for sensor in sensors_info if sensor.get("name")]

    @property
    def remote_sensor_devices(self) -> List[str]:
        """Return the remote sensor device name_by_user or name for the thermostat."""
        return sorted([f'{item["name_by_user"]} ({item["id"]})' for item in self.remote_sensor_ids_names])

    @property
    def remote_sensor_ids_names(self) -> List[Dict[str, Any]]:
        """Return the remote sensor device id and name_by_user for the thermostat."""
        sensors_info: List[Dict[str, Any]] = self.thermostat.get("remoteSensors", [])
        device_registry = dr.async_get(self._hass)
        return [
            {"id": device.id, "name_by_user": device.name_by_user if device.name_by_user else device.name}
            for device in device_registry.devices.values()
            for sensor_info in sensors_info
            if device.name == sensor_info["name"]
        ]

    @property
    def active_sensors_in_preset_mode(self) -> List[str]:
        """Return the currently active/participating sensors."""
        mode: str = self._preset_modes.get(self.preset_mode, "Home")
        return self._sensors_in_preset_mode(mode)

    @property
    def active_sensor_devices_in_preset_mode(self) -> List[str]:
        """Return the currently active/participating sensor devices."""
        mode: str = self._preset_modes.get(self.preset_mode, "Home")
        return self._sensor_devices_in_preset_mode(mode)

    def set_preset_mode(self, preset_mode: str) -> None:
        """Activate a preset."""
        preset_mode = HASS_TO_ECOBEE_PRESET.get(preset_mode, preset_mode)
        if preset_mode == self.preset_mode:
            return
        self.update_without_throttle = True
        if self.preset_mode == PRESET_VACATION:
            self.data.ecobee.delete_vacation(self.thermostat_index, self.vacation)  # type: ignore
        if preset_mode == PRESET_AWAY_INDEFINITELY:
            self.data.ecobee.set_climate_hold(self.thermostat_index, "away", "indefinite", self.hold_hours())
        elif preset_mode == PRESET_TEMPERATURE:
            self.set_temp_hold(self.current_temperature)
        elif preset_mode in (PRESET_HOLD_NEXT_TRANSITION, PRESET_HOLD_INDEFINITE):
            self.data.ecobee.set_climate_hold(
                self.thermostat_index, PRESET_TO_ECOBEE_HOLD[preset_mode], self.hold_preference(), self.hold_hours()
            )
        elif preset_mode == PRESET_NONE:
            self.data.ecobee.resume_program(self.thermostat_index)
        else:
            for climate_ref, name in self.comfort_settings.items():
                if name == preset_mode:
                    preset_mode = climate_ref
                    break
            else:
                _LOGGER.warning("Received unknown preset mode: %s", preset_mode)
            self.data.ecobee.set_climate_hold(self.thermostat_index, preset_mode, self.hold_preference(), self.hold_hours())

    @property
    def preset_modes(self) -> List[str]:
        """Return available preset modes."""
        return [ECOBEE_TO_HASS_PRESET.get(name, name) for name in self.comfort_settings.values()] + [PRESET_AWAY_INDEFINITELY]

    @property
    def comfort_settings(self) -> Dict[str, str]:
        """Return ecobee API comfort settings."""
        return {comfort["climateRef"]: comfort["name"] for comfort in self.thermostat["program"]["climates"]}

    def set_auto_temp_hold(self, heat_temp: Optional[float], cool_temp: Optional[float]) -> None:
        """Set temperature hold in auto mode."""
        if cool_temp is not None:
            cool_temp_setpoint: float = cool_temp
        else:
            cool_temp_setpoint = self.thermostat["runtime"]["desiredCool"] / 10.0
        if heat_temp is not None:
            heat_temp_setpoint: float = heat_temp
        else:
            heat_temp_setpoint = self.thermostat["runtime"]["desiredCool"] / 10.0
        self.data.ecobee.set_hold_temp(
            self.thermostat_index, cool_temp_setpoint, heat_temp_setpoint, self.hold_preference(), self.hold_hours()
        )
        _LOGGER.debug(
            "Setting ecobee hold_temp to: heat=%s, is=%s, cool=%s, is=%s",
            heat_temp,
            isinstance(heat_temp, (int, float)),
            cool_temp,
            isinstance(cool_temp, (int, float)),
        )
        self.update_without_throttle = True

    def set_fan_mode(self, fan_mode: str) -> None:
        """Set the fan mode.  Valid values are 'on' or 'auto'."""
        if fan_mode.lower() not in (FAN_ON, FAN_AUTO):
            error = "Invalid fan_mode value:  Valid values are 'on' or 'auto'"
            _LOGGER.error(error)
            return
        self.data.ecobee.set_fan_mode(self.thermostat_index, fan_mode, self.hold_preference(), holdHours=self.hold_hours())
        _LOGGER.debug("Setting fan mode to: %s", fan_mode)

    def set_temp_hold(self, temp: float) -> None:
        """Set temperature hold in modes other than auto.
        Ecobee API: It is good practice to set the heat and cool hold
        temperatures to be the same, if the thermostat is in either heat, cool,
        auxHeatOnly, or off mode. If the thermostat is in auto mode, an
        additional rule is required. The cool hold temperature must be greater
        than the heat hold temperature by at least the amount in the
        heatCoolMinDelta property.
        https://www.ecobee.com/home/developer/api/examples/ex5.shtml
        """
        if self.hvac_mode in (HVACMode.HEAT, HVACMode.COOL):
            heat_temp: float = temp
            cool_temp: float = temp
        else:
            delta: float = self.settings["heatCoolMinDelta"] / 10.0
            heat_temp = temp - delta
            cool_temp = temp + delta
        self.set_auto_temp_hold(heat_temp, cool_temp)

    def set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        low_temp: Optional[float] = kwargs.get(ATTR_TARGET_TEMP_LOW)
        high_temp: Optional[float] = kwargs.get(ATTR_TARGET_TEMP_HIGH)
        temp: Optional[float] = kwargs.get(ATTR_TEMPERATURE)
        if self.hvac_mode == HVACMode.HEAT_COOL and (low_temp is not None or high_temp is not None):
            self.set_auto_temp_hold(low_temp, high_temp)
        elif temp is not None:
            self.set_temp_hold(temp)
        else:
            _LOGGER.error("Missing valid arguments for set_temperature in %s", kwargs)

    def set_humidity(self, humidity: float) -> None:
        """Set the humidity level."""
        if not 0 <= humidity <= 100:
            raise ValueError(f"Invalid set_humidity value (must be in range 0-100): {humidity}")
        self.data.ecobee.set_humidity(self.thermostat_index, int(humidity))
        self.update_without_throttle = True

    def set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set HVAC mode (auto, auxHeatOnly, cool, heat, off)."""
        ecobee_value: Optional[str] = HASS_TO_ECOBEE_HVAC.get(hvac_mode)
        if ecobee_value is None:
            _LOGGER.error("Invalid mode for set_hvac_mode: %s", hvac_mode)
            return
        self.data.ecobee.set_hvac_mode(self.thermostat_index, ecobee_value)
        self.update_without_throttle = True

    def set_fan_min_on_time(self, fan_min_on_time: str) -> None:
        """Set the minimum fan on time."""
        self.data.ecobee.set_fan_min_on_time(self.thermostat_index, fan_min_on_time)
        self.update_without_throttle = True

    def resume_program(self, resume_all: bool) -> None:
        """Resume the thermostat schedule program."""
        self.data.ecobee.resume_program(self.thermostat_index, "true" if resume_all else "false")
        self.update_without_throttle = True

    def set_sensors_used_in_climate(self, device_ids: List[str], preset_mode: Optional[str] = None) -> None:
        """Set the sensors used on a climate for a thermostat."""
        if preset_mode is None:
            preset_mode = self.preset_mode  # type: ignore
        elif preset_mode not in self._preset_modes.values():
            if self.preset_modes:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="invalid_preset",
                    translation_placeholders={"options": ", ".join(self._preset_modes.values())},
                )
        device_registry = dr.async_get(self.hass)
        sensor_names: List[str] = []
        sensor_ids: List[str] = []
        for device_id in device_ids:
            device = device_registry.async_get(device_id)
            if device and device.name:
                r_sensors: List[Dict[str, Any]] = self.thermostat.get("remoteSensors", [])
                ecobee_identifier = next((identifier for identifier in device.identifiers if identifier[0] == "ecobee"), None)
                if ecobee_identifier:
                    code = ecobee_identifier[1]
                    for r_sensor in r_sensors:
                        if len(code) == 4 and r_sensor.get("code") == code or (len(code) != 4 and r_sensor.get("type") == "thermostat"):
                            sensor_ids.append(r_sensor.get("id"))
                    sensor_names.append(device.name)
        if not set(sensor_names).issubset(set(self._sensors)) or not sensor_names:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_sensor",
                translation_placeholders={
                    "options": ", ".join(
                        [f'{item["name_by_user"]} ({item["id"]})' for item in self.remote_sensor_ids_names]
                    )
                },
            )
        if len(device_ids) != len(sensor_ids):
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key="sensor_lookup_failed")
        current_sensors_in_climate = self._sensors_in_preset_mode(preset_mode)
        if set(sensor_names) == set(current_sensors_in_climate):
            _LOGGER.debug("This action would not be an update, current sensors on climate (%s) are: %s", preset_mode, ", ".join(current_sensors_in_climate))
            return
        _LOGGER.debug("Setting sensors %s to be used on thermostat %s for program %s", sensor_names, self.device_info.get("name"), preset_mode)
        self.data.ecobee.update_climate_sensors(self.thermostat_index, preset_mode, sensor_ids=sensor_ids)
        self.update_without_throttle = True

    def _sensors_in_preset_mode(self, preset_mode: str) -> List[str]:
        """Return current sensors used in climate."""
        climates: List[Dict[str, Any]] = self.thermostat["program"]["climates"]
        for climate in climates:
            if climate.get("name") == preset_mode:
                return [sensor["name"] for sensor in climate["sensors"]]
        return []

    def _sensor_devices_in_preset_mode(self, preset_mode: str) -> List[str]:
        """Return current sensor device name_by_user or name used in climate."""
        device_registry = dr.async_get(self._hass)
        sensor_names: List[str] = self._sensors_in_preset_mode(preset_mode)
        return sorted(
            [
                device.name_by_user if device.name_by_user else device.name
                for device in device_registry.devices.values()
                for sensor_name in sensor_names
                if device.name == sensor_name
            ]
        )

    def hold_preference(self) -> str:
        """Return user preference setting for hold time."""
        device_preference: str = self.settings["holdAction"]
        hold_pref_map: Dict[str, str] = {"useEndTime2hour": "holdHours", "useEndTime4hour": "holdHours", "indefinite": "indefinite"}
        return hold_pref_map.get(device_preference, "nextTransition")

    def hold_hours(self) -> Optional[int]:
        """Return user preference setting for hold duration in hours."""
        device_preference: str = self.settings["holdAction"]
        hold_hours_map: Dict[str, int] = {"useEndTime2hour": 2, "useEndTime4hour": 4}
        return hold_hours_map.get(device_preference)

    def create_vacation(self, service_data: Dict[str, Any]) -> None:
        """Create a vacation with user-specified parameters."""
        vacation_name: str = service_data[ATTR_VACATION_NAME]
        cool_temp: float = TemperatureConverter.convert(service_data[ATTR_COOL_TEMP], self.hass.config.units.temperature_unit, UnitOfTemperature.FAHRENHEIT)
        heat_temp: float = TemperatureConverter.convert(service_data[ATTR_HEAT_TEMP], self.hass.config.units.temperature_unit, UnitOfTemperature.FAHRENHEIT)
        start_date: Optional[str] = service_data.get(ATTR_START_DATE)
        start_time: Optional[str] = service_data.get(ATTR_START_TIME)
        end_date: Optional[str] = service_data.get(ATTR_END_DATE)
        end_time: Optional[str] = service_data.get(ATTR_END_TIME)
        fan_mode: str = service_data[ATTR_FAN_MODE]
        fan_min_on_time: Any = service_data[ATTR_FAN_MIN_ON_TIME]
        kwargs: Dict[str, Any] = {key: value for key, value in {"start_date": start_date, "start_time": start_time, "end_date": end_date, "end_time": end_time, "fan_mode": fan_mode, "fan_min_on_time": fan_min_on_time}.items() if value is not None}
        _LOGGER.debug(
            "Creating a vacation on thermostat %s with name %s, cool temp %s, heat temp %s, and the following other parameters: %s",
            self.name,
            vacation_name,
            cool_temp,
            heat_temp,
            kwargs,
        )
        self.data.ecobee.create_vacation(self.thermostat_index, vacation_name, cool_temp, heat_temp, **kwargs)

    def delete_vacation(self, vacation_name: str) -> None:
        """Delete a vacation with the specified name."""
        _LOGGER.debug("Deleting a vacation on thermostat %s with name %s", self.name, vacation_name)
        self.data.ecobee.delete_vacation(self.thermostat_index, vacation_name)

    def turn_on(self) -> None:
        """Set the thermostat to the last active HVAC mode."""
        _LOGGER.debug("Turning on ecobee thermostat %s in %s mode", self.name, self._last_active_hvac_mode)
        self.set_hvac_mode(self._last_active_hvac_mode)

    def set_dst_mode(self, dst_enabled: bool) -> None:
        """Enable/disable automatic daylight savings time."""
        self.data.ecobee.set_dst_mode(self.thermostat_index, dst_enabled)

    def set_mic_mode(self, mic_enabled: bool) -> None:
        """Enable/disable Alexa mic (only for Ecobee 4)."""
        self.data.ecobee.set_mic_mode(self.thermostat_index, mic_enabled)

    def set_occupancy_modes(self, auto_away: Optional[bool] = None, follow_me: Optional[bool] = None) -> None:
        """Enable/disable Smart Home/Away and Follow Me modes."""
        self.data.ecobee.set_occupancy_modes(self.thermostat_index, auto_away, follow_me)
