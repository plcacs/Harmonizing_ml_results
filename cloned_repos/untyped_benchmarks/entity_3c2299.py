"""Entity for Zigbee Home Automation."""
from __future__ import annotations
import asyncio
from collections.abc import Callable
from functools import partial
import logging
from typing import Any
from propcache.api import cached_property
from zha.mixins import LogMixin
from homeassistant.const import ATTR_MANUFACTURER, ATTR_MODEL, ATTR_NAME, EntityCategory
from homeassistant.core import State, callback
from homeassistant.helpers.device_registry import CONNECTION_ZIGBEE, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import UNDEFINED, UndefinedType
from .const import DOMAIN
from .helpers import SIGNAL_REMOVE_ENTITIES, EntityData, convert_zha_error_to_ha_error
_LOGGER = logging.getLogger(__name__)

class ZHAEntity(LogMixin, RestoreEntity, Entity):
    """ZHA eitity."""
    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, entity_data, *args, **kwargs):
        """Init ZHA entity."""
        super().__init__(*args, **kwargs)
        self.entity_data = entity_data
        self._unsubs = []
        if self.entity_data.entity.icon is not None:
            self._attr_icon = self.entity_data.entity.icon
        meta = self.entity_data.entity.info_object
        self._attr_unique_id = meta.unique_id
        if meta.entity_category is not None:
            self._attr_entity_category = EntityCategory(meta.entity_category)
        self._attr_entity_registry_enabled_default = meta.entity_registry_enabled_default
        if meta.translation_key is not None:
            self._attr_translation_key = meta.translation_key

    @cached_property
    def name(self):
        """Return the name of the entity."""
        meta = self.entity_data.entity.info_object
        original_name = super().name
        if original_name not in (UNDEFINED, None) or meta.fallback_name is None:
            return original_name
        self._attr_name = meta.fallback_name
        return super().name

    @property
    def available(self):
        """Return entity availability."""
        return self.entity_data.entity.available

    @property
    def device_info(self):
        """Return a device description for device registry."""
        zha_device_info = self.entity_data.device_proxy.device_info
        ieee = zha_device_info['ieee']
        zha_gateway = self.entity_data.device_proxy.gateway_proxy.gateway
        return DeviceInfo(connections={(CONNECTION_ZIGBEE, ieee)}, identifiers={(DOMAIN, ieee)}, manufacturer=zha_device_info[ATTR_MANUFACTURER], model=zha_device_info[ATTR_MODEL], name=zha_device_info[ATTR_NAME], via_device=(DOMAIN, str(zha_gateway.state.node_info.ieee)))

    @callback
    def _handle_entity_events(self, event):
        """Entity state changed."""
        self.debug('Handling event from entity: %s', event)
        self.async_write_ha_state()

    async def async_added_to_hass(self):
        """Run when about to be added to hass."""
        self.remove_future = self.hass.loop.create_future()
        self._unsubs.append(self.entity_data.entity.on_all_events(self._handle_entity_events))
        remove_signal = f'{SIGNAL_REMOVE_ENTITIES}_group_{self.entity_data.group_proxy.group.group_id}' if self.entity_data.is_group_entity and self.entity_data.group_proxy is not None else f'{SIGNAL_REMOVE_ENTITIES}_{self.entity_data.device_proxy.device.ieee}'
        self._unsubs.append(async_dispatcher_connect(self.hass, remove_signal, partial(self.async_remove, force_remove=True)))
        self.entity_data.device_proxy.gateway_proxy.register_entity_reference(self.entity_id, self.entity_data, self.device_info, self.remove_future)
        if (state := (await self.async_get_last_state())) is None:
            return
        self.restore_external_state_attributes(state)

    @callback
    def restore_external_state_attributes(self, state):
        """Restore ephemeral external state from Home Assistant back into ZHA."""

    async def async_will_remove_from_hass(self):
        """Disconnect entity object when removed."""
        for unsub in self._unsubs[:]:
            unsub()
            self._unsubs.remove(unsub)
        await super().async_will_remove_from_hass()
        self.remove_future.set_result(True)

    @convert_zha_error_to_ha_error
    async def async_update(self):
        """Update the entity."""
        await self.entity_data.entity.async_update()
        self.async_write_ha_state()

    def log(self, level, msg, *args, **kwargs):
        """Log a message."""
        msg = f'%s: {msg}'
        args = (self.entity_id, *args)
        _LOGGER.log(level, msg, *args, **kwargs)