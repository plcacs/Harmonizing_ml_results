from __future__ import annotations
import logging
from typing import Any, cast

def get_accessory(hass: HomeAssistant, driver: HomeDriver, state: State, aid: int, config: dict[str, Any]) -> Accessory:
    ...

class HomeAccessory(Accessory):
    def __init__(self, hass: HomeAssistant, driver: HomeDriver, name: str, entity_id: str, aid: int, config: dict[str, Any], *args: Any, category: int = CATEGORY_OTHER, device_id: str = None, **kwargs: Any) -> None:
        ...

    def _update_available_from_state(self, new_state: State) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    @ha_callback
    @pyhap_callback
    def run(self) -> None:
        ...

    @ha_callback
    def async_update_event_state_callback(self, event: Event) -> None:
        ...

    @ha_callback
    def async_update_state_callback(self, new_state: State) -> None:
        ...

    @ha_callback
    def async_update_linked_battery_callback(self, event: Event) -> None:
        ...

    @ha_callback
    def async_update_linked_battery_charging_callback(self, event: Event) -> None:
        ...

    @ha_callback
    def async_update_battery(self, battery_level: Any, battery_charging: Any) -> None:
        ...

    @ha_callback
    def async_update_state(self, new_state: State) -> None:
        ...

    @ha_callback
    def async_call_service(self, domain: str, service: str, service_data: dict[str, Any], value: Any = None) -> None:
        ...

    @ha_callback
    def async_reload(self) -> None:
        ...

    @ha_callback
    def async_stop(self) -> None:
        ...

    async def stop(self) -> None:
        ...

class HomeBridge(Bridge):
    def __init__(self, hass: HomeAssistant, driver: HomeDriver, name: str) -> None:
        ...

    def setup_message(self) -> None:
        ...

    async def async_get_snapshot(self, info: dict[str, Any]) -> bytes:
        ...

class HomeDriver(AccessoryDriver):
    def __init__(self, hass: HomeAssistant, entry_id: str, bridge_name: str, entry_title: str, iid_storage: AccessoryIIDStorage, **kwargs: Any) -> None:
        ...

    @pyhap_callback
    def pair(self, client_username_bytes: bytes, client_public: bytes, client_permissions: Any) -> bool:
        ...

    @pyhap_callback
    def unpair(self, client_uuid: UUID) -> None:
        ...

class HomeIIDManager(IIDManager):
    def __init__(self, iid_storage: AccessoryIIDStorage) -> None:
        ...

    def get_iid_for_obj(self, obj: Any) -> int:
        ...
