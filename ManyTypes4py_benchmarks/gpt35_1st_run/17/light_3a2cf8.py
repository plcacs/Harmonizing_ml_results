from __future__ import annotations
import logging
from typing import Any

from homeassistant.components.light import ATTR_BRIGHTNESS, ColorMode, LightEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import HomeworksData
from .const import CONF_ADDR, CONF_CONTROLLER_ID, CONF_DIMMERS, CONF_RATE, DOMAIN
from .entity import HomeworksEntity

_LOGGER: logging.Logger

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    data: HomeworksData = hass.data[DOMAIN][entry.entry_id]
    controller = data.controller
    controller_id: str = entry.options[CONF_CONTROLLER_ID]
    entities = []
    for dimmer in entry.options.get(CONF_DIMMERS, []):
        entity = HomeworksLight(controller, controller_id, dimmer[CONF_ADDR], dimmer[CONF_NAME], dimmer[CONF_RATE])
        entities.append(entity)
    async_add_entities(entities, True)

class HomeworksLight(HomeworksEntity, LightEntity):
    _attr_color_mode: ColorMode = ColorMode.BRIGHTNESS
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.BRIGHTNESS}

    def __init__(self, controller, controller_id: str, addr: str, name: str, rate: int) -> None:
        super().__init__(controller, controller_id, addr, 0, None)
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, f'{controller_id}.{addr}')}, name=name)
        self._rate: int = rate
        self._level: int = 0
        self._prev_level: int = 0

    async def async_added_to_hass(self) -> None:
        signal: str = f'homeworks_entity_{self._controller_id}_{self._addr}'
        _LOGGER.debug('connecting %s', signal)
        self.async_on_remove(async_dispatcher_connect(self.hass, signal, self._update_callback))
        self._controller.request_dimmer_level(self._addr)

    def turn_on(self, **kwargs: Any) -> None:
        if ATTR_BRIGHTNESS in kwargs:
            new_level: int = kwargs[ATTR_BRIGHTNESS]
        elif self._prev_level == 0:
            new_level = 255
        else:
            new_level = self._prev_level
        self._set_brightness(new_level)

    def turn_off(self, **kwargs: Any) -> None:
        self._set_brightness(0)

    @property
    def brightness(self) -> int:
        return self._level

    def _set_brightness(self, level: int) -> None:
        self._controller.fade_dim(float(level * 100.0 / 255.0), self._rate, 0, self._addr)

    @property
    def is_on(self) -> bool:
        return self._level != 0

    @callback
    def _update_callback(self, msg_type: str, values: list[float]) -> None:
        if msg_type == HW_LIGHT_CHANGED:
            self._level = int(values[1] * 255.0 / 100.0)
            if self._level != 0:
                self._prev_level = self._level
            self.async_write_ha_state()
