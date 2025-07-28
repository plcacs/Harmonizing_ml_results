from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import Any, Optional
from pyfibaro.fibaro_device import DeviceModel
from homeassistant.const import ATTR_ARMED, ATTR_BATTERY_LEVEL
from homeassistant.helpers.entity import Entity

_LOGGER = logging.getLogger(__name__)


class FibaroEntity(Entity):
    _attr_should_poll: bool = False

    def __init__(self, fibaro_device: DeviceModel) -> None:
        self.fibaro_device: DeviceModel = fibaro_device
        self.controller = fibaro_device.fibaro_controller
        self.ha_id = fibaro_device.ha_id
        self._attr_name: str = fibaro_device.friendly_name
        self._attr_unique_id: str = fibaro_device.unique_id_str
        self._attr_device_info = self.controller.get_device_info(fibaro_device)
        if not fibaro_device.visible:
            self._attr_entity_registry_visible_default = False

    async def async_added_to_hass(self) -> None:
        self.controller.register(self.fibaro_device.fibaro_id, self._update_callback)

    def _update_callback(self) -> None:
        self.schedule_update_ha_state(True)

    @property
    def level(self) -> Optional[int]:
        if self.fibaro_device.value.has_value:
            return self.fibaro_device.value.int_value()
        return None

    @property
    def level2(self) -> Optional[int]:
        if self.fibaro_device.value_2.has_value:
            return self.fibaro_device.value_2.int_value()
        return None

    def dont_know_message(self, cmd: str) -> None:
        _LOGGER.warning('Not sure how to %s: %s (available actions: %s)', cmd, str(self.ha_id), str(self.fibaro_device.actions))

    def set_level(self, level: int) -> None:
        self.action('setValue', level)
        if self.fibaro_device.value.has_value:
            self.fibaro_device.properties['value'] = level
        if self.fibaro_device.has_brightness:
            self.fibaro_device.properties['brightness'] = level

    def set_level2(self, level: int) -> None:
        self.action('setValue2', level)
        if self.fibaro_device.value_2.has_value:
            self.fibaro_device.properties['value2'] = level

    def call_turn_on(self) -> None:
        self.action('turnOn')

    def call_turn_off(self) -> None:
        self.action('turnOff')

    def call_set_color(self, red: int, green: int, blue: int, white: int) -> None:
        red = int(max(0, min(255, red)))
        green = int(max(0, min(255, green)))
        blue = int(max(0, min(255, blue)))
        white = int(max(0, min(255, white)))
        color_str: str = f'{red},{green},{blue},{white}'
        self.fibaro_device.properties['color'] = color_str
        self.action('setColor', str(red), str(green), str(blue), str(white))

    def action(self, cmd: str, *args: Any) -> None:
        if cmd in self.fibaro_device.actions:
            self.fibaro_device.execute_action(cmd, args)
            _LOGGER.debug('-> %s.%s%s called', str(self.ha_id), str(cmd), str(args))
        else:
            self.dont_know_message(cmd)

    @property
    def current_binary_state(self) -> bool:
        return self.fibaro_device.value.bool_value(False)

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
        attr: dict[str, Any] = {'fibaro_id': self.fibaro_device.fibaro_id}
        if self.fibaro_device.has_battery_level:
            attr[ATTR_BATTERY_LEVEL] = self.fibaro_device.battery_level
        if self.fibaro_device.has_armed:
            attr[ATTR_ARMED] = self.fibaro_device.armed
        return attr

    def update(self) -> None:
        if self.fibaro_device.has_dead:
            self._attr_available = not self.fibaro_device.dead