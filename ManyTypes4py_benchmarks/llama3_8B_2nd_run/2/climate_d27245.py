from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from homeassistant.components.climate import FAN_AUTO, PRESET_AWAY, PRESET_HOME, SWING_BOTH, SWING_HORIZONTAL, SWING_OFF, SWING_ON, SWING_VERTICAL, ClimateEntity, ClimateEntityFeature, HVACAction, HVACMode
from homeassistant.const import ATTR_TEMPERATURE, PRECISION_TENTHS, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.typing import VolDictType
from . import TadoConfigEntry
from .const import *
from .coordinator import TadoDataUpdateCoordinator
from .entity import TadoZoneEntity
from .helper import decide_duration, decide_overlay_mode, generate_supported_fanmodes

_LOGGER = logging.getLogger(__name__)

SERVICE_CLIMATE_TIMER = 'set_climate_timer'
ATTR_TIME_PERIOD = 'time_period'
ATTR_REQUESTED_OVERLAY = 'requested_overlay'
CLIMATE_TIMER_SCHEMA = {vol.Required(ATTR_TEMPERATURE): vol.Coerce(float), vol.Exclusive(ATTR_TIME_PERIOD, CONST_EXCLUSIVE_OVERLAY_GROUP): vol.All(cv.time_period, cv.positive_timedelta, lambda td: td.total_seconds()), vol.Exclusive(ATTR_REQUESTED_OVERLAY, CONST_EXCLUSIVE_OVERLAY_GROUP): vol.In(CONST_OVERLAY_TADO_OPTIONS)}
SERVICE_TEMP_OFFSET = 'set_climate_temperature_offset'
ATTR_OFFSET = 'offset'
CLIMATE_TEMP_OFFSET_SCHEMA = {vol.Required(ATTR_OFFSET, default=0): vol.Coerce(float)}

async def async_setup_entry(hass: HomeAssistant, entry: TadoConfigEntry, async_add_entities: Callable[[List[ClimateEntity]], None]) -> None:
    """Set up the Tado climate platform."""
    tado = entry.runtime_data.coordinator
    entities = await _generate_entities(tado)
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_CLIMATE_TIMER, CLIMATE_TIMER_SCHEMA, 'set_timer')
    platform.async_register_entity_service(SERVICE_TEMP_OFFSET, CLIMATE_TEMP_OFFSET_SCHEMA, 'set_temp_offset')
    async_add_entities(entities, True)

async def _generate_entities(tado: TadoDataUpdateCoordinator) -> List[TadoClimate]:
    """Create all climate entities."""
    entities: List[TadoClimate] = []
    for zone in tado.zones:
        if zone['type'] in [TYPE_HEATING, TYPE_AIR_CONDITIONING]:
            entity = await create_climate_entity(tado, zone['name'], zone['id'], zone['devices'][0])
            if entity:
                entities.append(entity)
    return entities

async def create_climate_entity(tado: TadoDataUpdateCoordinator, name: str, zone_id: str, device_info: Any) -> Optional[TadoClimate]:
    """Create a Tado climate entity."""
    capabilities = await tado.get_capabilities(zone_id)
    _LOGGER.debug('Capabilities for zone %s: %s', zone_id, capabilities)
    zone_type = capabilities['type']
    support_flags: ClimateEntityFeature = ClimateEntityFeature.PRESET_MODE | ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
    supported_hvac_modes: List[HVACMode] = []
    supported_fan_modes: Optional[List[Any]] = None
    supported_swing_modes: Optional[List[Any]] = None
    heat_temperatures: Optional[Dict[str, float]] = None
    cool_temperatures: Optional[Dict[str, float]] = None
    if zone_type == TYPE_AIR_CONDITIONING:
        for mode in ORDERED_KNOWN_TADO_MODES:
            if mode not in capabilities:
                continue
            supported_hvac_modes.append(TADO_TO_HA_HVAC_MODE_MAP[mode])
            if TADO_SWING_SETTING in capabilities[mode] or TADO_VERTICAL_SWING_SETTING in capabilities[mode] or TADO_HORIZONTAL_SWING_SETTING in capabilities[mode]:
                support_flags |= ClimateEntityFeature.SWING_MODE
                supported_swing_modes = []
                if TADO_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_ON])
                if TADO_VERTICAL_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(SWING_VERTICAL)
                if TADO_HORIZONTAL_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(SWING_HORIZONTAL)
                if SWING_HORIZONTAL in supported_swing_modes and SWING_VERTICAL in supported_swing_modes:
                    supported_swing_modes.append(SWING_BOTH)
                supported_swing_modes.append(TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_OFF])
            if TADO_FANSPEED_SETTING not in capabilities[mode] and TADO_FANLEVEL_SETTING not in capabilities[mode]:
                continue
            support_flags |= ClimateEntityFeature.FAN_MODE
            if supported_fan_modes:
                continue
            if TADO_FANSPEED_SETTING in capabilities[mode]:
                supported_fan_modes = generate_supported_fanmodes(TADO_TO_HA_FAN_MODE_MAP_LEGACY, capabilities[mode][TADO_FANSPEED_SETTING])
            else:
                supported_fan_modes = generate_supported_fanmodes(TADO_TO_HA_FAN_MODE_MAP, capabilities[mode][TADO_FANLEVEL_SETTING])
        cool_temperatures = capabilities[CONST_MODE_COOL]['temperatures']
    else:
        supported_hvac_modes.append(HVACMode.HEAT)
    if CONST_MODE_HEAT in capabilities:
        heat_temperatures = capabilities[CONST_MODE_HEAT]['temperatures']
    if heat_temperatures is None and 'temperatures' in capabilities:
        heat_temperatures = capabilities['temperatures']
    if cool_temperatures is None and heat_temperatures is None:
        _LOGGER.debug('Not adding zone %s since it has no temperatures', name)
        return None
    heat_min_temp: Optional[float] = None
    heat_max_temp: Optional[float] = None
    heat_step: Optional[float] = None
    cool_min_temp: Optional[float] = None
    cool_max_temp: Optional[float] = None
    cool_step: Optional[float] = None
    if heat_temperatures is not None:
        heat_min_temp = float(heat_temperatures['celsius']['min'])
        heat_max_temp = float(heat_temperatures['celsius']['max'])
        heat_step = heat_temperatures['celsius'].get('step', PRECISION_TENTHS)
    if cool_temperatures is not None:
        cool_min_temp = float(cool_temperatures['celsius']['min'])
        cool_max_temp = float(cool_temperatures['celsius']['max'])
        cool_step = cool_temperatures['celsius'].get('step', PRECISION_TENTHS)
    auto_geofencing_supported: bool = await tado.get_auto_geofencing_supported()
    return TadoClimate(tado, name, zone_id, zone_type, supported_hvac_modes, support_flags, device_info, capabilities, auto_geofencing_supported, heat_min_temp, heat_max_temp, heat_step, cool_min_temp, cool_max_temp, cool_step, supported_fan_modes, supported_swing_modes)

class TadoClimate(TadoZoneEntity, ClimateEntity):
    """Representation of a Tado climate entity."""
    _attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS
    _attr_name: Optional[str] = None
    _attr_translation_key: str = DOMAIN
    _available: bool = False

    def __init__(self, coordinator: TadoDataUpdateCoordinator, zone_name: str, zone_id: str, zone_type: str, supported_hvac_modes: List[HVACMode], support_flags: ClimateEntityFeature, device_info: Any, capabilities: Dict[str, Any], auto_geofencing_supported: bool, heat_min_temp: Optional[float], heat_max_temp: Optional[float], heat_step: Optional[float], cool_min_temp: Optional[float], cool_max_temp: Optional[float], cool_step: Optional[float], supported_fan_modes: Optional[List[Any]], supported_swing_modes: Optional[List[Any]]) -> None:
        """Initialize of Tado climate entity."""
        self._tado = coordinator
        super().__init__(zone_name, coordinator.home_id, zone_id, coordinator)
        self.zone_id = zone_id
        self.zone_type = zone_type
        self._device_info = device_info
        self._device_id = self._device_info['shortSerialNo']
        self._ac_device = zone_type == TYPE_AIR_CONDITIONING
        self._attr_hvac_modes = supported_hvac_modes
        self._attr_fan_modes = supported_fan_modes
        self._attr_supported_features = support_flags
        self._cur_temp: Optional[float] = None
        self._cur_humidity: Optional[float] = None
        self._attr_swing_modes = supported_swing_modes
        self._heat_min_temp = heat_min_temp
        self._heat_max_temp = heat_max_temp
        self._heat_step = heat_step
        self._cool_min_temp = cool_min_temp
        self._cool_max_temp = cool_max_temp
        self._cool_step = cool_step
        self._target_temp: Optional[float] = None
        self._current_tado_fan_speed: str = CONST_FAN_OFF
        self._current_tado_fan_level: str = CONST_FAN_OFF
        self._current_tado_hvac_mode: str = CONST_MODE_OFF
        self._current_tado_hvac_action: HVACAction = HVACAction.OFF
        self._current_tado_swing_mode: str = TADO_SWING_OFF
        self._current_tado_vertical_swing: str = TADO_SWING_OFF
        self._current_tado_horizontal_swing: str = TADO_SWING_OFF
        self._current_tado_capabilities: Dict[str, Any] = capabilities
        self._auto_geofencing_supported: bool = auto_geofencing_supported
        self._tado_zone_data: Dict[str, Any] = {}
        self._tado_geofence_data: Optional[Dict[str, Any]] = None
        self._tado_zone_temp_offset: Dict[str, float] = {}
        self._async_update_zone