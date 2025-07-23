"""Support for Xiaomi Aqara binary sensors."""
import logging
from typing import Any, Dict, List
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN, GATEWAYS_KEY
from .entity import XiaomiDevice

_LOGGER = logging.getLogger(__name__)

ATTR_LOAD_POWER = 'load_power'
ATTR_POWER_CONSUMED = 'power_consumed'
ATTR_IN_USE = 'in_use'
LOAD_POWER = 'load_power'
POWER_CONSUMED = 'power_consumed'
ENERGY_CONSUMED = 'energy_consumed'
IN_USE = 'inuse'

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Perform the setup for Xiaomi devices."""
    entities: List[XiaomiGenericSwitch] = []
    gateway = hass.data[DOMAIN][GATEWAYS_KEY][config_entry.entry_id]
    for device in gateway.devices.get('switch', []):
        model: str = device.get('model', '')
        if model == 'plug':
            if 'proto' not in device or int(device.get('proto', '0')[:1]) == 1:
                data_key: str = 'status'
            else:
                data_key = 'channel_0'
            entities.append(XiaomiGenericSwitch(
                device=device,
                name='Plug',
                data_key=data_key,
                supports_power_consumption=True,
                xiaomi_hub=gateway,
                config_entry=config_entry
            ))
        elif model in ('ctrl_neutral1', 'ctrl_neutral1.aq1', 'switch_b1lacn02', 'switch.b1lacn02'):
            entities.append(XiaomiGenericSwitch(
                device=device,
                name='Wall Switch',
                data_key='channel_0',
                supports_power_consumption=False,
                xiaomi_hub=gateway,
                config_entry=config_entry
            ))
        elif model in ('ctrl_ln1', 'ctrl_ln1.aq1', 'switch_b1nacn02', 'switch.b1nacn02'):
            entities.append(XiaomiGenericSwitch(
                device=device,
                name='Wall Switch LN',
                data_key='channel_0',
                supports_power_consumption=False,
                xiaomi_hub=gateway,
                config_entry=config_entry
            ))
        elif model in ('ctrl_neutral2', 'ctrl_neutral2.aq1', 'switch_b2lacn02', 'switch.b2lacn02'):
            entities.append(XiaomiGenericSwitch(
                device=device,
                name='Wall Switch Left',
                data_key='channel_0',
                supports_power_consumption=False,
                xiaomi_hub=gateway,
                config_entry=config_entry
            ))
            entities.append(XiaomiGenericSwitch(
                device=device,
                name='Wall Switch Right',
                data_key='channel_1',
                supports_power_consumption=False,
                xiaomi_hub=gateway,
                config_entry=config_entry
            ))
        elif model in ('ctrl_ln2', 'ctrl_ln2.aq1', 'switch_b2nacn02', 'switch.b2nacn02'):
            entities.append(XiaomiGenericSwitch(
                device=device,
                name='Wall Switch LN Left',
                data_key='channel_0',
                supports_power_consumption=False,
                xiaomi_hub=gateway,
                config_entry=config_entry
            ))
            entities.append(XiaomiGenericSwitch(
                device=device,
                name='Wall Switch LN Right',
                data_key='channel_1',
                supports_power_consumption=False,
                xiaomi_hub=gateway,
                config_entry=config_entry
            ))
        elif model in ('86plug', 'ctrl_86plug', 'ctrl_86plug.aq1'):
            if 'proto' not in device or int(device.get('proto', '0')[:1]) == 1:
                data_key = 'status'
            else:
                data_key = 'channel_0'
            entities.append(XiaomiGenericSwitch(
                device=device,
                name='Wall Plug',
                data_key=data_key,
                supports_power_consumption=True,
                xiaomi_hub=gateway,
                config_entry=config_entry
            ))
    async_add_entities(entities)

class XiaomiGenericSwitch(XiaomiDevice, SwitchEntity):
    """Representation of a XiaomiPlug."""

    def __init__(
        self,
        device: Dict[str, Any],
        name: str,
        data_key: str,
        supports_power_consumption: bool,
        xiaomi_hub: Any,
        config_entry: ConfigEntry
    ) -> None:
        """Initialize the XiaomiPlug."""
        self._data_key: str = data_key
        self._in_use: Any = None
        self._load_power: Any = None
        self._power_consumed: Any = None
        self._supports_power_consumption: bool = supports_power_consumption
        self._attr_should_poll: bool = supports_power_consumption
        super().__init__(device, name, xiaomi_hub, config_entry)

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend, if any."""
        if self._data_key == 'status':
            return 'mdi:power-plug'
        return 'mdi:power-socket'

    @property
    def is_on(self) -> bool:
        """Return true if it is on."""
        return self._state

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        if self._supports_power_consumption:
            attrs: Dict[str, Any] = {
                ATTR_IN_USE: self._in_use,
                ATTR_LOAD_POWER: self._load_power,
                ATTR_POWER_CONSUMED: self._power_consumed
            }
        else:
            attrs = {}
        attrs.update(super().extra_state_attributes)
        return attrs

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on."""
        if self._write_to_hub(self._sid, **{self._data_key: 'on'}):
            self._state = True
            self.schedule_update_ha_state()

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off."""
        if self._write_to_hub(self._sid, **{self._data_key: 'off'}):
            self._state = False
            self.schedule_update_ha_state()

    def parse_data(self, data: Dict[str, Any], raw_data: Any) -> bool:
        """Parse data sent by gateway."""
        if IN_USE in data:
            self._in_use = int(data[IN_USE])
            if not self._in_use:
                self._load_power = 0
        for key in (POWER_CONSUMED, ENERGY_CONSUMED):
            if key in data:
                self._power_consumed = round(float(data[key]), 2)
                break
        if LOAD_POWER in data:
            self._load_power = round(float(data[LOAD_POWER]), 2)
        value: Any = data.get(self._data_key)
        if value not in ['on', 'off']:
            return False
        state: bool = value == 'on'
        if self._state == state:
            return False
        self._state = state
        return True

    def update(self) -> None:
        """Get data from hub."""
        _LOGGER.debug('Update data from hub: %s', self._name)
        self._get_from_hub(self._sid)
