"""Support for humidifiers."""
from __future__ import annotations
from enum import StrEnum
from typing import Any, cast
from aiocomelit import ComelitSerialBridgeObject
from aiocomelit.const import CLIMATE
from homeassistant.components.humidifier import MODE_AUTO, MODE_NORMAL, HumidifierAction, HumidifierDeviceClass, HumidifierEntity, HumidifierEntityFeature
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN
from .coordinator import ComelitConfigEntry, ComelitSerialBridge

class HumidifierComelitMode(StrEnum):
    """Serial Bridge humidifier modes."""
    AUTO = 'A'
    OFF = 'O'
    LOWER = 'L'
    UPPER = 'U'

class HumidifierComelitCommand(StrEnum):
    """Serial Bridge humidifier commands."""
    OFF = 'off'
    ON = 'on'
    MANUAL = 'man'
    SET = 'set'
    AUTO = 'auto'
    LOWER = 'lower'
    UPPER = 'upper'
MODE_TO_ACTION = {MODE_AUTO: HumidifierComelitCommand.AUTO, MODE_NORMAL: HumidifierComelitCommand.MANUAL}

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up Comelit humidifiers."""
    coordinator = cast(ComelitSerialBridge, config_entry.runtime_data)
    entities = []
    for device in coordinator.data[CLIMATE].values():
        entities.append(ComelitHumidifierEntity(coordinator, device, config_entry.entry_id, active_mode=HumidifierComelitMode.LOWER, active_action=HumidifierAction.DRYING, set_command=HumidifierComelitCommand.LOWER, device_class=HumidifierDeviceClass.DEHUMIDIFIER))
        entities.append(ComelitHumidifierEntity(coordinator, device, config_entry.entry_id, active_mode=HumidifierComelitMode.UPPER, active_action=HumidifierAction.HUMIDIFYING, set_command=HumidifierComelitCommand.UPPER, device_class=HumidifierDeviceClass.HUMIDIFIER))
    async_add_entities(entities)

class ComelitHumidifierEntity(CoordinatorEntity[ComelitSerialBridge], HumidifierEntity):
    """Humidifier device."""
    _attr_supported_features = HumidifierEntityFeature.MODES
    _attr_available_modes = [MODE_NORMAL, MODE_AUTO]
    _attr_min_humidity = 10
    _attr_max_humidity = 90
    _attr_has_entity_name = True

    def __init__(self, coordinator, device, config_entry_entry_id, active_mode, active_action, set_command, device_class):
        """Init light entity."""
        self._api = coordinator.api
        self._device = device
        super().__init__(coordinator)
        self._attr_unique_id = f'{config_entry_entry_id}-{device.index}-{device_class}'
        self._attr_device_info = coordinator.platform_device_info(device, device_class)
        self._attr_device_class = device_class
        self._attr_translation_key = device_class.value
        self._active_mode = active_mode
        self._active_action = active_action
        self._set_command = set_command

    @property
    def _humidifier(self):
        """Return humidifier device data."""
        return self.coordinator.data[CLIMATE][self._device.index].val[1]

    @property
    def _api_mode(self):
        """Return device mode."""
        return self._humidifier[2]

    @property
    def _api_active(self):
        """Return device active/idle."""
        return self._humidifier[1]

    @property
    def _api_automatic(self):
        """Return device in automatic/manual mode."""
        return self._humidifier[3] == HumidifierComelitMode.AUTO

    @property
    def target_humidity(self):
        """Set target humidity."""
        return self._humidifier[4] / 10

    @property
    def current_humidity(self):
        """Return current humidity."""
        return self._humidifier[0] / 10

    @property
    def is_on(self):
        """Return true is humidifier is on."""
        return self._api_mode == self._active_mode

    @property
    def mode(self):
        """Return current mode."""
        return MODE_AUTO if self._api_automatic else MODE_NORMAL

    @property
    def action(self):
        """Return current action."""
        if self._api_mode == HumidifierComelitMode.OFF:
            return HumidifierAction.OFF
        if self._api_active and self._api_mode == self._active_mode:
            return self._active_action
        return HumidifierAction.IDLE

    async def async_set_humidity(self, humidity):
        """Set new target humidity."""
        if self.mode == HumidifierComelitMode.OFF:
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key='humidity_while_off')
        await self.coordinator.api.set_humidity_status(self._device.index, HumidifierComelitCommand.MANUAL)
        await self.coordinator.api.set_humidity_status(self._device.index, HumidifierComelitCommand.SET, humidity)

    async def async_set_mode(self, mode):
        """Set humidifier mode."""
        await self.coordinator.api.set_humidity_status(self._device.index, MODE_TO_ACTION[mode])

    async def async_turn_on(self, **kwargs):
        """Turn on."""
        await self.coordinator.api.set_humidity_status(self._device.index, self._set_command)

    async def async_turn_off(self, **kwargs):
        """Turn off."""
        await self.coordinator.api.set_humidity_status(self._device.index, HumidifierComelitCommand.OFF)