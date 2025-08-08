"""The lookin integration climate platform."""
from __future__ import annotations
import logging
from typing import Any, Final, cast
from aiolookin import Climate, MeteoSensor, Remote
from aiolookin.models import UDPCommandType, UDPEvent
from homeassistant.components.climate import ATTR_HVAC_MODE, FAN_AUTO, FAN_HIGH, FAN_LOW, FAN_MIDDLE, SWING_BOTH, SWING_OFF, ClimateEntity, ClimateEntityFeature, HVACMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, PRECISION_WHOLE, Platform, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN, TYPE_TO_PLATFORM
from .coordinator import LookinDataUpdateCoordinator
from .entity import LookinCoordinatorEntity
from .models import LookinData
LOOKIN_FAN_MODE_IDX_TO_HASS = [FAN_AUTO, FAN_LOW, FAN_MIDDLE, FAN_HIGH]
LOOKIN_SWING_MODE_IDX_TO_HASS = [SWING_OFF, SWING_BOTH]
LOOKIN_HVAC_MODE_IDX_TO_HASS = [HVACMode.OFF, HVACMode.AUTO, HVACMode.COOL,
    HVACMode.HEAT, HVACMode.DRY, HVACMode.FAN_ONLY]
HASS_TO_LOOKIN_HVAC_MODE = {mode: idx for idx, mode in enumerate(
    LOOKIN_HVAC_MODE_IDX_TO_HASS)}
HASS_TO_LOOKIN_FAN_MODE = {mode: idx for idx, mode in enumerate(
    LOOKIN_FAN_MODE_IDX_TO_HASS)}
HASS_TO_LOOKIN_SWING_MODE = {mode: idx for idx, mode in enumerate(
    LOOKIN_SWING_MODE_IDX_TO_HASS)}
MIN_TEMP = 16
MAX_TEMP = 30
LOGGER = logging.getLogger(__name__)


async def func_i3whlj7i(hass, config_entry, async_add_entities):
    """Set up the climate platform for lookin from a config entry."""
    lookin_data = hass.data[DOMAIN][config_entry.entry_id]
    entities = []
    for remote in lookin_data.devices:
        if TYPE_TO_PLATFORM.get(remote['Type']) != Platform.CLIMATE:
            continue
        uuid = remote['UUID']
        coordinator = lookin_data.device_coordinators[uuid]
        device = cast(Climate, coordinator.data)
        entities.append(ConditionerEntity(uuid=uuid, device=device,
            lookin_data=lookin_data, coordinator=coordinator))
    async_add_entities(entities)


class ConditionerEntity(LookinCoordinatorEntity, ClimateEntity):
    """An aircon or heat pump."""
    _attr_current_humidity = None
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_supported_features = (ClimateEntityFeature.TARGET_TEMPERATURE |
        ClimateEntityFeature.FAN_MODE | ClimateEntityFeature.SWING_MODE |
        ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON)
    _attr_fan_modes = LOOKIN_FAN_MODE_IDX_TO_HASS
    _attr_swing_modes = LOOKIN_SWING_MODE_IDX_TO_HASS
    _attr_hvac_modes = LOOKIN_HVAC_MODE_IDX_TO_HASS
    _attr_min_temp = MIN_TEMP
    _attr_max_temp = MAX_TEMP
    _attr_target_temperature_step = PRECISION_WHOLE

    def __init__(self, uuid, device, lookin_data, coordinator):
        """Init the ConditionerEntity."""
        super().__init__(coordinator, uuid, device, lookin_data)
        self._async_update_from_data()

    @property
    def func_tfq8zf2i(self):
        return cast(Climate, self.coordinator.data)

    async def func_xhwp0pvc(self, hvac_mode):
        """Set the hvac mode of the device."""
        if (mode := HASS_TO_LOOKIN_HVAC_MODE.get(hvac_mode)) is None:
            return
        self._climate.hvac_mode = mode
        await self._async_update_conditioner()

    async def func_ayqhkogx(self, **kwargs):
        """Set the temperature of the device."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._climate.temp_celsius = int(temperature)
        lookin_index = LOOKIN_HVAC_MODE_IDX_TO_HASS
        if (hvac_mode := kwargs.get(ATTR_HVAC_MODE)):
            self._climate.hvac_mode = HASS_TO_LOOKIN_HVAC_MODE[hvac_mode]
        elif self._climate.hvac_mode == lookin_index.index(HVACMode.OFF):
            if self._meteo_coordinator:
                meteo_data = self._meteo_coordinator.data
                if not (current_temp := meteo_data.temperature):
                    self._climate.hvac_mode = lookin_index.index(HVACMode.AUTO)
                elif current_temp >= self._climate.temp_celsius:
                    self._climate.hvac_mode = lookin_index.index(HVACMode.COOL)
                else:
                    self._climate.hvac_mode = lookin_index.index(HVACMode.HEAT)
            else:
                self._climate.hvac_mode = lookin_index.index(HVACMode.AUTO)
        await self._async_update_conditioner()

    async def func_8k0tz4ez(self, fan_mode):
        """Set the fan mode of the device."""
        if (mode := HASS_TO_LOOKIN_FAN_MODE.get(fan_mode)) is None:
            return
        self._climate.fan_mode = mode
        await self._async_update_conditioner()

    async def func_ujquf65q(self, swing_mode):
        """Set the swing mode of the device."""
        if (mode := HASS_TO_LOOKIN_SWING_MODE.get(swing_mode)) is None:
            return
        self._climate.swing_mode = mode
        await self._async_update_conditioner()

    async def func_u4gn7knz(self):
        """Update the conditioner state from the climate data."""
        self.coordinator.async_set_updated_data(self._climate)
        await self._lookin_protocol.update_conditioner(uuid=self.
            _attr_unique_id, status=self._climate.to_status)

    def func_jex6yp0a(self):
        """Update attrs from data."""
        if self._meteo_coordinator:
            temperature = self._meteo_coordinator.data.temperature
            humidity = int(self._meteo_coordinator.data.humidity)
        else:
            temperature = humidity = None
        self._attr_current_temperature = temperature
        self._attr_current_humidity = humidity
        self._attr_target_temperature = self._climate.temp_celsius
        self._attr_fan_mode = LOOKIN_FAN_MODE_IDX_TO_HASS[self._climate.
            fan_mode]
        self._attr_swing_mode = LOOKIN_SWING_MODE_IDX_TO_HASS[self._climate
            .swing_mode]
        self._attr_hvac_mode = LOOKIN_HVAC_MODE_IDX_TO_HASS[self._climate.
            hvac_mode]

    @callback
    def func_ha05q4qx(self, event):
        """Update temperature and humidity from UDP event."""
        self._attr_current_temperature = float(int(event.value[:4], 16)) / 10
        self._attr_current_humidity = float(int(event.value[-4:], 16)) / 10

    @callback
    def func_z5dpi8vo(self):
        """Handle updated data from the coordinator."""
        self._async_update_from_data()
        super()._handle_coordinator_update()

    @callback
    def func_kapd5fsh(self, event):
        """Process an update pushed via UDP."""
        LOGGER.debug('Processing push message for %s: %s', self.entity_id,
            event)
        self._climate.update_from_status(event.value)
        self.coordinator.async_set_updated_data(self._climate)

    async def func_jd8lhqle(self):
        """Call when the entity is added to hass."""
        self.async_on_remove(self._lookin_udp_subs.subscribe_event(self.
            _lookin_device.id, UDPCommandType.ir, self._uuid, self.
            _async_push_update))
        self.async_on_remove(self._lookin_udp_subs.subscribe_event(self.
            _lookin_device.id, UDPCommandType.meteo, None, self.
            _async_update_meteo_from_value))
        return await super().async_added_to_hass()
