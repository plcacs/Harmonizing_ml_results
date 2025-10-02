from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import Any
from pyfibaro.fibaro_device import DeviceModel
from homeassistant.const import ATTR_ARMED, ATTR_BATTERY_LEVEL
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.typing import StateType, ConfigType

_LOGGER: logging.Logger

class FibaroEntity(Entity):
    _attr_should_poll: bool = False

    def __init__(self, fibaro_device: DeviceModel) -> None:
        self.fibaro_device: DeviceModel
        self.controller: Any
        self.ha_id: Any
        self._attr_name: str
        self._attr_unique_id: str
        self._attr_device_info: Any
        self._attr_entity_registry_visible_default: bool

    async def async_added_to_hass(self) -> None:
        pass

    def _update_callback(self) -> None:
        pass

    @property
    def level(self) -> StateType:
        pass

    @property
    def level2(self) -> StateType:
        pass

    def dont_know_message(self, cmd: str) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def set_level2(self, level: int) -> None:
        pass

    def call_turn_on(self) -> None:
        pass

    def call_turn_off(self) -> None:
        pass

    def call_set_color(self, red: int, green: int, blue: int, white: int) -> None:
        pass

    def action(self, cmd: str, *args: Any) -> None:
        pass

    @property
    def current_binary_state(self) -> StateType:
        pass

    @property
    def extra_state_attributes(self) -> ConfigType:
        pass

    def update(self) -> None:
        pass
