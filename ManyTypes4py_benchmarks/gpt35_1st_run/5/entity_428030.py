from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Set, Union, List, Dict, Optional
from zwave_js_server.const import NodeStatus
from zwave_js_server.exceptions import BaseZwaveJSServerError
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.value import SetValueResult, Value as ZwaveValue, get_value_id_str
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import callback, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.typing import UNDEFINED
from .const import DOMAIN, EVENT_VALUE_UPDATED, LOGGER
from .discovery import ZwaveDiscoveryInfo

EVENT_VALUE_REMOVED: str = 'value removed'
EVENT_DEAD: str = 'dead'
EVENT_ALIVE: str = 'alive'

class ZWaveBaseEntity(Entity):
    _attr_should_poll: bool = False
    _attr_has_entity_name: bool = True

    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo) -> None:
        self.config_entry: ConfigEntry = config_entry
        self.driver: Driver = driver
        self.info: ZwaveDiscoveryInfo = info
        self.watched_value_ids: Set[str] = {self.info.primary_value.value_id}
        if self.info.additional_value_ids_to_watch:
            self.watched_value_ids = self.watched_value_ids.union(self.info.additional_value_ids_to_watch)
        self._attr_name: str = self.generate_name()
        self._attr_unique_id: str = get_unique_id(driver, self.info.primary_value.value_id)
        if self.info.entity_registry_enabled_default is False:
            self._attr_entity_registry_enabled_default: bool = False
        if self.info.entity_category is not None:
            self._attr_entity_category: str = self.info.entity_category
        if self.info.assumed_state:
            self._attr_assumed_state: bool = True
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={get_device_id(driver, self.info.node)})

    @callback
    def on_value_update(self) -> None:
        pass

    async def _async_poll_value(self, value_or_id: Union[ZwaveValue, str]) -> None:
        pass

    async def async_poll_value(self, refresh_all_values: bool) -> None:
        pass

    async def async_added_to_hass(self) -> None:
        pass

    def generate_name(self, include_value_name: bool = False, alternate_value_name: Optional[str] = None, additional_info: Optional[List[str]] = None, name_prefix: Optional[str] = None) -> str:
        pass

    @property
    def available(self) -> bool:
        pass

    @callback
    def _node_status_alive_or_dead(self, event_data: Dict[str, Any]) -> None:
        pass

    @callback
    def _value_changed(self, event_data: Dict[str, Any]) -> None:
        pass

    @callback
    def _value_removed(self, event_data: Dict[str, Any]) -> None:
        pass

    @callback
    def get_zwave_value(self, value_property: str, command_class: Optional[int] = None, endpoint: Optional[int] = None, value_property_key: Optional[int] = None, add_to_watched_value_ids: bool = True, check_all_endpoints: bool = False) -> Optional[ZwaveValue]:
        pass

    async def _async_set_value(self, value: ZwaveValue, new_value: Any, options: Optional[Dict[str, Any]] = None, wait_for_result: Optional[bool] = None) -> SetValueResult:
        pass
