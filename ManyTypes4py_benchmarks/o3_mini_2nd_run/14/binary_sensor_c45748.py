from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional
from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.restore_state import RestoreEntity
from .const import DOMAIN, GATEWAYS_KEY
from .entity import XiaomiDevice

_LOGGER = logging.getLogger(__name__)
NO_CLOSE = 'no_close'
ATTR_OPEN_SINCE = 'Open since'
MOTION = 'motion'
NO_MOTION = 'no_motion'
ATTR_LAST_ACTION = 'last_action'
ATTR_NO_MOTION_SINCE = 'No motion since'
DENSITY = 'density'
ATTR_DENSITY = 'Density'


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Perform the setup for Xiaomi devices."""
    entities: list[Any] = []
    gateway = hass.data[DOMAIN][GATEWAYS_KEY][config_entry.entry_id]
    for entity in gateway.devices['binary_sensor']:
        model = entity['model']
        if model in ('motion', 'sensor_motion', 'sensor_motion.aq2'):
            entities.append(XiaomiMotionSensor(entity, hass, gateway, config_entry))
        elif model in ('magnet', 'sensor_magnet', 'sensor_magnet.aq2'):
            entities.append(XiaomiDoorSensor(entity, gateway, config_entry))
        elif model == 'sensor_wleak.aq1':
            entities.append(XiaomiWaterLeakSensor(entity, gateway, config_entry))
        elif model in ('smoke', 'sensor_smoke'):
            entities.append(XiaomiSmokeSensor(entity, gateway, config_entry))
        elif model in ('natgas', 'sensor_natgas'):
            entities.append(XiaomiNatgasSensor(entity, gateway, config_entry))
        elif model in (
            'switch',
            'sensor_switch',
            'sensor_switch.aq2',
            'sensor_switch.aq3',
            'remote.b1acn01',
        ):
            if 'proto' not in entity or int(entity['proto'][0:1]) == 1:
                data_key = 'status'
            else:
                data_key = 'button_0'
            entities.append(
                XiaomiButton(entity, 'Switch', data_key, hass, gateway, config_entry)
            )
        elif model in (
            '86sw1',
            'sensor_86sw1',
            'sensor_86sw1.aq1',
            'remote.b186acn01',
            'remote.b186acn02',
        ):
            if 'proto' not in entity or int(entity['proto'][0:1]) == 1:
                data_key = 'channel_0'
            else:
                data_key = 'button_0'
            entities.append(
                XiaomiButton(entity, 'Wall Switch', data_key, hass, gateway, config_entry)
            )
        elif model in (
            '86sw2',
            'sensor_86sw2',
            'sensor_86sw2.aq1',
            'remote.b286acn01',
            'remote.b286acn02',
        ):
            if 'proto' not in entity or int(entity['proto'][0:1]) == 1:
                data_key_left = 'channel_0'
                data_key_right = 'channel_1'
            else:
                data_key_left = 'button_0'
                data_key_right = 'button_1'
            entities.append(
                XiaomiButton(entity, 'Wall Switch (Left)', data_key_left, hass, gateway, config_entry)
            )
            entities.append(
                XiaomiButton(entity, 'Wall Switch (Right)', data_key_right, hass, gateway, config_entry)
            )
            entities.append(
                XiaomiButton(entity, 'Wall Switch (Both)', 'dual_channel', hass, gateway, config_entry)
            )
        elif model in ('cube', 'sensor_cube', 'sensor_cube.aqgl01'):
            entities.append(XiaomiCube(entity, hass, gateway, config_entry))
        elif model in ('vibration', 'vibration.aq1'):
            entities.append(XiaomiVibration(entity, 'Vibration', 'status', gateway, config_entry))
        else:
            _LOGGER.warning('Unmapped Device Model %s', model)
    async_add_entities(entities)


class XiaomiBinarySensor(XiaomiDevice, BinarySensorEntity):
    """Representation of a base XiaomiBinarySensor."""

    def __init__(
        self,
        device: Dict[str, Any],
        name: str,
        xiaomi_hub: Any,
        data_key: str,
        device_class: Optional[str],
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the XiaomiBinarySensor."""
        self._data_key: str = data_key
        self._device_class: Optional[str] = device_class
        self._density: int = 0
        super().__init__(device, name, xiaomi_hub, config_entry)

    @property
    def is_on(self) -> bool:
        """Return true if sensor is on."""
        return self._state

    @property
    def device_class(self) -> Optional[str]:
        """Return the class of binary sensor."""
        return self._device_class

    def update(self) -> None:
        """Update the sensor state."""
        _LOGGER.debug('Updating xiaomi sensor (%s) by polling', self._sid)
        self._get_from_hub(self._sid)


class XiaomiNatgasSensor(XiaomiBinarySensor):
    """Representation of a XiaomiNatgasSensor."""

    def __init__(
        self, device: Dict[str, Any], xiaomi_hub: Any, config_entry: ConfigEntry
    ) -> None:
        """Initialize the XiaomiNatgasSensor."""
        self._density: Optional[int] = None
        super().__init__(device, 'Natgas Sensor', xiaomi_hub, 'alarm', 'gas', config_entry)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        attrs: Dict[str, Any] = {ATTR_DENSITY: self._density}
        attrs.update(super().extra_state_attributes)
        return attrs

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self._state = False

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        """Parse data sent by gateway."""
        if DENSITY in data:
            self._density = int(data.get(DENSITY))
        value = data.get(self._data_key)
        if value is None:
            return False
        if value in ('1', '2'):
            if self._state:
                return False
            self._state = True
            return True
        if value == '0':
            if self._state:
                self._state = False
                return True
            return False
        return False


class XiaomiMotionSensor(XiaomiBinarySensor):
    """Representation of a XiaomiMotionSensor."""

    def __init__(
        self,
        device: Dict[str, Any],
        hass: HomeAssistant,
        xiaomi_hub: Any,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the XiaomiMotionSensor."""
        self._hass: HomeAssistant = hass
        self._no_motion_since: int = 0
        self._unsub_set_no_motion: Optional[Callable[[], None]] = None
        if 'proto' not in device or int(device['proto'][0:1]) == 1:
            data_key = 'status'
        else:
            data_key = 'motion_status'
        super().__init__(device, 'Motion Sensor', xiaomi_hub, data_key, 'motion', config_entry)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        attrs: Dict[str, Any] = {ATTR_NO_MOTION_SINCE: self._no_motion_since}
        attrs.update(super().extra_state_attributes)
        return attrs

    @callback
    def _async_set_no_motion(self, now: Any) -> None:
        """Set state to False."""
        self._unsub_set_no_motion = None
        self._state = False
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self._state = False

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        """Parse data sent by gateway."""
        if raw_data['cmd'] == 'heartbeat':
            _LOGGER.debug(
                'Skipping heartbeat of the motion sensor. It can introduce an incorrect state because of a firmware bug (https://github.com/home-assistant/core/pull/11631#issuecomment-357507744)'
            )
            return False
        if NO_MOTION in data:
            self._no_motion_since = data[NO_MOTION]
            self._state = False
            return True
        value = data.get(self._data_key)
        if value is None:
            return False
        if value == MOTION:
            if self._data_key == 'motion_status':
                if self._unsub_set_no_motion:
                    self._unsub_set_no_motion()
                self._unsub_set_no_motion = async_call_later(self._hass, 120, self._async_set_no_motion)
            if self.entity_id is not None:
                self._hass.bus.async_fire('xiaomi_aqara.motion', {'entity_id': self.entity_id})
            self._no_motion_since = 0
            if self._state:
                return False
            self._state = True
            return True
        return False


class XiaomiDoorSensor(XiaomiBinarySensor, RestoreEntity):
    """Representation of a XiaomiDoorSensor."""

    def __init__(
        self, device: Dict[str, Any], xiaomi_hub: Any, config_entry: ConfigEntry
    ) -> None:
        """Initialize the XiaomiDoorSensor."""
        self._open_since: int = 0
        if 'proto' not in device or int(device['proto'][0:1]) == 1:
            data_key = 'status'
        else:
            data_key = 'window_status'
        super().__init__(device, 'Door Window Sensor', xiaomi_hub, data_key, BinarySensorDeviceClass.OPENING, config_entry)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        attrs: Dict[str, Any] = {ATTR_OPEN_SINCE: self._open_since}
        attrs.update(super().extra_state_attributes)
        return attrs

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        state = await self.async_get_last_state()
        if state is None:
            return
        self._state = state.state == 'on'

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        """Parse data sent by gateway."""
        self._attr_should_poll = False  # type: ignore
        if NO_CLOSE in data:
            self._open_since = data[NO_CLOSE]
            return True
        value = data.get(self._data_key)
        if value is None:
            return False
        if value == 'open':
            self._attr_should_poll = True  # type: ignore
            if self._state:
                return False
            self._state = True
            return True
        if value == 'close':
            self._open_since = 0
            if self._state:
                self._state = False
                return True
            return False
        return False


class XiaomiWaterLeakSensor(XiaomiBinarySensor):
    """Representation of a XiaomiWaterLeakSensor."""

    def __init__(
        self, device: Dict[str, Any], xiaomi_hub: Any, config_entry: ConfigEntry
    ) -> None:
        """Initialize the XiaomiWaterLeakSensor."""
        if 'proto' not in device or int(device['proto'][0:1]) == 1:
            data_key = 'status'
        else:
            data_key = 'wleak_status'
        super().__init__(device, 'Water Leak Sensor', xiaomi_hub, data_key, BinarySensorDeviceClass.MOISTURE, config_entry)

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self._state = False

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        """Parse data sent by gateway."""
        self._attr_should_poll = False  # type: ignore
        value = data.get(self._data_key)
        if value is None:
            return False
        if value == 'leak':
            self._attr_should_poll = True  # type: ignore
            if self._state:
                return False
            self._state = True
            return True
        if value == 'no_leak':
            if self._state:
                self._state = False
                return True
            return False
        return False


class XiaomiSmokeSensor(XiaomiBinarySensor):
    """Representation of a XiaomiSmokeSensor."""

    def __init__(
        self, device: Dict[str, Any], xiaomi_hub: Any, config_entry: ConfigEntry
    ) -> None:
        """Initialize the XiaomiSmokeSensor."""
        self._density: int = 0
        super().__init__(device, 'Smoke Sensor', xiaomi_hub, 'alarm', 'smoke', config_entry)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        attrs: Dict[str, Any] = {ATTR_DENSITY: self._density}
        attrs.update(super().extra_state_attributes)
        return attrs

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self._state = False

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        """Parse data sent by gateway."""
        if DENSITY in data:
            self._density = int(data.get(DENSITY))
        value = data.get(self._data_key)
        if value is None:
            return False
        if value in ('1', '2'):
            if self._state:
                return False
            self._state = True
            return True
        if value == '0':
            if self._state:
                self._state = False
                return True
            return False
        return False


class XiaomiVibration(XiaomiBinarySensor):
    """Representation of a Xiaomi Vibration Sensor."""

    def __init__(
        self,
        device: Dict[str, Any],
        name: str,
        data_key: str,
        xiaomi_hub: Any,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the XiaomiVibration."""
        self._last_action: Optional[str] = None
        super().__init__(device, name, xiaomi_hub, data_key, None, config_entry)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        attrs: Dict[str, Any] = {ATTR_LAST_ACTION: self._last_action}
        attrs.update(super().extra_state_attributes)
        return attrs

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self._state = False

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        """Parse data sent by gateway."""
        value = data.get(self._data_key)
        if value is None:
            return False
        if value not in ('vibrate', 'tilt', 'free_fall', 'actively'):
            _LOGGER.warning('Unsupported movement_type detected: %s', value)
            return False
        self.hass.bus.async_fire(
            'xiaomi_aqara.movement', {'entity_id': self.entity_id, 'movement_type': value}
        )
        self._last_action = value
        return True


class XiaomiButton(XiaomiBinarySensor):
    """Representation of a Xiaomi Button."""

    def __init__(
        self,
        device: Dict[str, Any],
        name: str,
        data_key: str,
        hass: HomeAssistant,
        xiaomi_hub: Any,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the XiaomiButton."""
        self._hass: HomeAssistant = hass
        self._last_action: Optional[str] = None
        super().__init__(device, name, xiaomi_hub, data_key, None, config_entry)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        attrs: Dict[str, Any] = {ATTR_LAST_ACTION: self._last_action}
        attrs.update(super().extra_state_attributes)
        return attrs

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self._state = False

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        """Parse data sent by gateway."""
        value = data.get(self._data_key)
        if value is None:
            return False

        if value == 'long_click_press':
            self._state = True
            click_type = 'long_click_press'
        elif value == 'long_click_release':
            self._state = False
            click_type = 'hold'
        elif value == 'click':
            click_type = 'single'
        elif value == 'double_click':
            click_type = 'double'
        elif value == 'both_click':
            click_type = 'both'
        elif value == 'double_both_click':
            click_type = 'double_both'
        elif value == 'shake':
            click_type = 'shake'
        elif value == 'long_click':
            click_type = 'long'
        elif value == 'long_both_click':
            click_type = 'long_both'
        else:
            _LOGGER.warning('Unsupported click_type detected: %s', value)
            return False
        self._hass.bus.async_fire(
            'xiaomi_aqara.click', {'entity_id': self.entity_id, 'click_type': click_type}
        )
        self._last_action = click_type
        return True


class XiaomiCube(XiaomiBinarySensor):
    """Representation of a Xiaomi Cube."""

    def __init__(
        self,
        device: Dict[str, Any],
        hass: HomeAssistant,
        xiaomi_hub: Any,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the Xiaomi Cube."""
        self._hass: HomeAssistant = hass
        self._last_action: Optional[str] = None
        if 'proto' not in device or int(device['proto'][0:1]) == 1:
            data_key = 'status'
        else:
            data_key = 'cube_status'
        super().__init__(device, 'Cube', xiaomi_hub, data_key, None, config_entry)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        attrs: Dict[str, Any] = {ATTR_LAST_ACTION: self._last_action}
        attrs.update(super().extra_state_attributes)
        return attrs

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self._state = False

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        """Parse data sent by gateway."""
        if self._data_key in data:
            self._hass.bus.async_fire(
                'xiaomi_aqara.cube_action',
                {'entity_id': self.entity_id, 'action_type': data[self._data_key]},
            )
            self._last_action = data[self._data_key]
        if 'rotate' in data:
            action_value = float(
                data['rotate']
                if isinstance(data['rotate'], int)
                else data['rotate'].replace(',', '.')
            )
            self._hass.bus.async_fire(
                'xiaomi_aqara.cube_action',
                {'entity_id': self.entity_id, 'action_type': 'rotate', 'action_value': action_value},
            )
            self._last_action = 'rotate'
        if 'rotate_degree' in data:
            action_value = float(
                data['rotate_degree']
                if isinstance(data['rotate_degree'], int)
                else data['rotate_degree'].replace(',', '.')
            )
            self._hass.bus.async_fire(
                'xiaomi_aqara.cube_action',
                {'entity_id': self.entity_id, 'action_type': 'rotate', 'action_value': action_value},
            )
            self._last_action = 'rotate'
        return True