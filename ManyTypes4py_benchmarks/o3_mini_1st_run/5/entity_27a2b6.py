from typing import Any, Dict, Optional
import functools
import logging

from pyinsteon import devices
from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity import Entity

from .const import (
    DOMAIN,
    SIGNAL_ADD_DEFAULT_LINKS,
    SIGNAL_LOAD_ALDB,
    SIGNAL_PRINT_ALDB,
    SIGNAL_REMOVE_ENTITY,
    SIGNAL_SAVE_DEVICES,
    STATE_NAME_LABEL_MAP,
)
from .utils import print_aldb_to_log

_LOGGER: logging.Logger = logging.getLogger(__name__)


class InsteonEntity(Entity):
    _attr_should_poll: bool = False

    def __init__(self, device: devices.Device, group: int) -> None:
        self._insteon_device_group: Any = device.groups[group]
        self._insteon_device: devices.Device = device

    def __hash__(self) -> int:
        return hash(self._insteon_device)

    @property
    def address(self) -> str:
        return str(self._insteon_device.address)

    @property
    def group(self) -> int:
        return self._insteon_device_group.group

    @property
    def unique_id(self) -> str:
        if self._insteon_device_group.group == 1:
            uid: str = self._insteon_device.id
        else:
            uid = f'{self._insteon_device.id}_{self._insteon_device_group.group}'
        return uid

    @property
    def name(self) -> str:
        if (description := self._insteon_device.description) is None:
            description = 'Unknown Device'
        if (extension := self._get_label()):
            extension = f' {extension}'
        return f'{description} {self._insteon_device.address}{extension}'

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        return {'insteon_address': self.address, 'insteon_group': self.group}

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, str(self._insteon_device.address))},
            manufacturer='SmartLabs, Inc',
            model=f'{self._insteon_device.model} ({self._insteon_device.cat!r}, 0x{self._insteon_device.subcat:02x})',
            name=f'{self._insteon_device.description} {self._insteon_device.address}',
            sw_version=f'{self._insteon_device.firmware:02x} Engine Version: {self._insteon_device.engine_version}',
            via_device=(DOMAIN, str(devices.modem.address)),
            configuration_url=f'homeassistant://insteon/device/config/{self._insteon_device.id}',
        )

    @callback
    def async_entity_update(self, name: str, address: str, value: Any, group: int) -> None:
        _LOGGER.debug('Received update for device %s group %d value %s', address, group, value)
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        _LOGGER.debug(
            'Tracking updates for device %s group %d name %s',
            self.address,
            self.group,
            self._insteon_device_group.name,
        )
        self._insteon_device_group.subscribe(self.async_entity_update)
        load_signal: str = f'{self.entity_id}_{SIGNAL_LOAD_ALDB}'
        self.async_on_remove(
            async_dispatcher_connect(self.hass, load_signal, self._async_read_aldb)
        )
        print_signal: str = f'{self.entity_id}_{SIGNAL_PRINT_ALDB}'
        async_dispatcher_connect(self.hass, print_signal, self._print_aldb)
        default_links_signal: str = f'{self.entity_id}_{SIGNAL_ADD_DEFAULT_LINKS}'
        async_dispatcher_connect(self.hass, default_links_signal, self._async_add_default_links)
        remove_signal: str = f'{self._insteon_device.address.id}_{SIGNAL_REMOVE_ENTITY}'
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                remove_signal,
                functools.partial(self.async_remove, force_remove=True),
            )
        )

    async def async_will_remove_from_hass(self) -> None:
        _LOGGER.debug(
            'Remove tracking updates for device %s group %d name %s',
            self.address,
            self.group,
            self._insteon_device_group.name,
        )
        self._insteon_device_group.unsubscribe(self.async_entity_update)

    async def _async_read_aldb(self, reload: bool) -> None:
        await self._insteon_device.aldb.async_load(refresh=reload)
        self._print_aldb()
        async_dispatcher_send(self.hass, SIGNAL_SAVE_DEVICES)

    def _print_aldb(self) -> None:
        print_aldb_to_log(self._insteon_device.aldb)

    def get_device_property(self, name: str) -> Any:
        if (prop := self._insteon_device.properties.get(name)) is not None:
            return prop.value
        return None

    def _get_label(self) -> str:
        label: str = ''
        if len(self._insteon_device.groups) > 1:
            if self._insteon_device_group.name in STATE_NAME_LABEL_MAP:
                label = STATE_NAME_LABEL_MAP[self._insteon_device_group.name]
            else:
                label = f'Group {self.group:d}'
        return label

    async def _async_add_default_links(self) -> None:
        await self._insteon_device.async_add_default_links()