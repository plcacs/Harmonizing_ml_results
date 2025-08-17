"""Support for Ecobee Thermostats."""

from __future__ import annotations

import collections
from typing import Any, Optional, Dict, List, Set, FrozenSet, OrderedDict, cast

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
from homeassistant.helpers import (
    config_validation as cv,
    device_registry as dr,
    entity_platform,
)
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.unit_conversion import TemperatureConverter

from . import EcobeeConfigEntry, EcobeeData
from .const import (
    _LOGGER,
    ATTR_ACTIVE_SENSORS,
    ATTR_AVAILABLE_SENSORS,
    DOMAIN,
    ECOBEE_AUX_HEAT_ONLY,
    ECOBEE_MODEL_TO_NAME,
    MANUFACTURER,
)
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

ECOBEE_HVAC_TO_HASS: OrderedDict[str, HVACMode] = collections.OrderedDict(
    [
        ("heat", HVACMode.HEAT),
        ("cool", HVACMode.COOL),
        ("auto", HVACMode.HEAT_COOL),
        ("off", HVACMode.OFF),
        (ECOBEE_AUX_HEAT_ONLY, HVACMode.HEAT),
    ]
)

HASS_TO_ECOBEE_HVAC: Dict[HVACMode, str] = {
    v: k for k, v in ECOBEE_HVAC_TO_HASS.items() if k != ECOBEE_AUX_HEAT_ONLY
}

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

ECOBEE_TO_HASS_PRESET: Dict[str, str] = {
    "Away": PRESET_AWAY,
    "Home": PRESET_HOME,
    "Sleep": PRESET_SLEEP,
}
HASS_TO_ECOBEE_PRESET: Dict[str, str] = {v: k for k, v in ECOBEE_TO_HASS_PRESET.items()}

PRESET_TO_ECOBEE_HOLD: Dict[str, str] = {
    PRESET_HOLD_NEXT_TRANSITION: "nextTransition",
    PRESET_HOLD_INDEFINITE: "indefinite",
}

SERVICE_CREATE_VACATION = "create_vacation"
SERVICE_DELETE_VACATION = "delete_vacation"
SERVICE_RESUME_PROGRAM = "resume_program"
SERVICE_SET_FAN_MIN_ON_TIME = "set_fan_min_on_time"
SERVICE_SET_DST_MODE = "set_dst_mode"
SERVICE_SET_MIC_MODE = "set_mic_mode"
SERVICE_SET_OCCUPANCY_MODES = "set_occupancy_modes"
SERVICE_SET_SENSORS_USED_IN_CLIMATE = "set_sensors_used_in_climate"

DTGROUP_START_INCLUSIVE_MSG = (
    f"{ATTR_START_DATE} and {ATTR_START_TIME} must be specified together"
)

DTGROUP_END_INCLUSIVE_MSG = (
    f"{ATTR_END_DATE} and {ATTR_END_TIME} must be specified together"
)

CREATE_VACATION_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required(ATTR_VACATION_NAME): vol.All(cv.string, vol.Length(max=12)),
        vol.Required(ATTR_COOL_TEMP): vol.Coerce(float),
        vol.Required(ATTR_HEAT_TEMP): vol.Coerce(float),
        vol.Inclusive(
            ATTR_START_DATE, "dtgroup_start", msg=DTGROUP_START_INCLUSIVE_MSG
        ): ecobee_date,
        vol.Inclusive(
            ATTR_START_TIME, "dtgroup_start", msg=DTGROUP_START_INCLUSIVE_MSG
        ): ecobee_time,
        vol.Inclusive(
            ATTR_END_DATE, "dtgroup_end", msg=DTGROUP_END_INCLUSIVE_MSG
        ): ecobee_date,
        vol.Inclusive(
            ATTR_END_TIME, "dtgroup_end", msg=DTGROUP_END_INCLUSIVE_MSG
        ): ecobee_time,
        vol.Optional(ATTR_FAN_MODE, default="auto"): vol.Any("auto", "on"),
        vol.Optional(ATTR_FAN_MIN_ON_TIME, default=0): vol.All(
            int, vol.Range(min=0, max=60)
        ),
    }
)

DELETE_VACATION_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Required(ATTR_VACATION_NAME): vol.All(cv.string, vol.Length(max=12)),
    }
)

RESUME_PROGRAM_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Optional(ATTR_RESUME_ALL, default=DEFAULT_RESUME_ALL): cv.boolean,
    }
)

SET_FAN_MIN_ON_TIME_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Required(ATTR_FAN_MIN_ON_TIME): vol.Coerce(int),
    }
)

SUPPORT_FLAGS = (
    ClimateEntityFeature.TARGET_TEMPERATURE
    | ClimateEntityFeature.PRESET_MODE
    | ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
    | ClimateEntityFeature.FAN_MODE
)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: EcobeeConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the ecobee thermostat."""
    data = config_entry.runtime_data
    entities: List[Thermostat] = []

    for index in range(len(data.ecobee.thermostats)):
        thermostat = data.ecobee.get_thermostat(index)
        if thermostat["modelNumber"] not in ECOBEE_MODEL_TO_NAME:
            _LOGGER.error(
                (
                    "Model number for ecobee thermostat %s not recognized. "
                    "Please visit this link to open a new issue: "
                    "https://github.com/home-assistant/core/issues "
                    "and include the following information: "
                    "Unrecognized model number: %s"
                ),
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
            target_thermostats = [
                entity for entity in entities if entity.entity_id in entity_id
            ]
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
            target_thermostats = [
                entity for entity in entities if entity.entity_id in entity_id
            ]
        else:
            target_thermostats = entities

        for thermostat in target_thermostats:
            thermostat.resume_program(resume_all)
            thermostat.schedule_update_ha_state(True)

    hass.services.async_register(
        DOMAIN,
        SERVICE_CREATE_VACATION,
        create_vacation_service,
        schema=CREATE_VACATION_SCHEMA,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_DELETE_VACATION,
        delete_vacation_service,
        schema=DELETE_VACATION_SCHEMA,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_SET_FAN_MIN_ON_TIME,
        fan_min_on_time_set_service,
        schema=SET_FAN_MIN_ON_TIME_SCHEMA,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESUME_PROGRAM,
        resume_program_set_service,
        schema=RESUME_PROGRAM_SCHEMA,
    )

    platform.async_register_entity_service(
        SERVICE_SET_DST_MODE,
        {vol.Required(ATTR_DST_ENABLED): cv.boolean},
        "set_dst_mode",
    )

    platform.async_register_entity_service(
        SERVICE_SET_MIC_MODE,
        {vol.Required(ATTR_MIC_ENABLED): cv.boolean},
        "set_mic_mode",
    )

    platform.async_register_entity_service(
        SERVICE_SET_OCCUPANCY_MODES,
        {
            vol.Optional(ATTR_AUTO_AWAY): cv.boolean,
            vol.Optional(ATTR_FOLLOW_ME): cv.boolean,
        },
        "set_occupancy_modes",
    )

    platform.async_register_entity_service(
        SERVICE_SET_SENSORS_USED_IN_CLIMATE,
        {
            vol.Optional(ATTR_PRESET_MODE): cv.string,
            vol.Required(ATTR_SENSOR_LIST): cv.ensure_list,
        },
        "set_sensors_used_in_climate",
    )

class Thermostat(ClimateEntity):
    """A thermostat class for Ecobee."""

    _attr_precision: float = PRECISION_TENTHS
    _attr_temperature_unit: str = UnitOfTemperature.FAHRENHEIT
    _attr_min_humidity: int = DEFAULT_MIN_HUMIDITY
    _attr_max_humidity: int = DEFAULT_MAX_HUMIDITY
    _attr_fan_modes: List[str] = [FAN_AUTO, FAN_ON]
    _attr_name: Optional[str] = None
    _attr_has_entity_name: bool = True
    _attr_translation_key: str = "ecobee"
    _unrecorded_attributes: FrozenSet[str] = frozenset({ATTR_AVAILABLE_SENSORS, ATTR_ACTIVE_SENSORS})

    def __init__(
        self,
        data: EcobeeData,
        thermostat_index: int,
        thermostat: Dict[str, Any],
        hass: HomeAssistant,
    ) -> None:
        """Initialize the thermostat."""
        self.data = data
        self.thermostat_index = thermostat_index
        self.thermostat = thermostat
        self._attr_unique_id = self.thermostat["identifier"]
        self.vacation: Optional[str] = None
        self._last_active_hvac_mode: HVACMode = HVACMode.HEAT_COOL
        self._last_hvac_mode_before_aux_heat: HVACMode = HVACMode.HEAT_COOL
        self._hass = hass

        self._attr_hvac_modes: List[HVACMode] = []
        if self.settings["heatStages"] or self.settings["hasHeatPump"]:
            self._attr_hvac_modes.append(HVACMode.HEAT)
        if self.settings["coolStages"]:
            self._attr_hvac_modes.append(HVACMode.COOL)
        if len(self._attr_hvac_modes) == 2:
            self._attr_hvac_modes.insert(0, HVACMode.HEAT_COOL)
        self._attr_hvac_modes.append(HVACMode.OFF)
        self._sensors: List[str] = self.remote_sensors
        self._preset_modes: Dict[str, str] = {
            comfort["climateRef"]: comfort["name"]
            for comfort in self.thermostat["program"]["climates"]
        }
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
    def supported_features(self) -> ClimateEntityFeature:
        """Return the list of supported features."""
        supported = SUPPORT_FLAGS
        if self.has_humidifier_control:
            supported = supported | ClimateEntityFeature.TARGET_HUMIDITY
        if len(self.hvac_modes) > 1 and HVACMode.OFF in self.hvac_modes:
            supported = (
                supported | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
            )
        return supported

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information for this ecobee thermostat."""
        model: Optional[str] = None
        try:
            model = f"{ECOBEE_MODEL_TO_NAME[self.thermostat['modelNumber']]} Thermostat"
        except KeyError:
            pass

        return DeviceInfo(
            identifiers={(DOMAIN, self.thermostat["identifier"])},
            manufacturer=MANUFACTURER,
            model=model,
            name=self.thermostat["name"],
        )

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
    def