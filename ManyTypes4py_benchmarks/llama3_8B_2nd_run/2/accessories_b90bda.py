from __future__ import annotations
import logging
from typing import Any, cast
from uuid import UUID
from pyhap.accessory import Accessory
from pyhap.accessory_driver import AccessoryDriver
from pyhap.characteristic import Characteristic
from pyhap.const import CATEGORY_OTHER
from pyhap.iid_manager import IIDManager
from pyhap.service import Service
from pyhap.util import callback as pyhap_callback
from homeassistant.components.cover import CoverDeviceClass, CoverEntityFeature
from homeassistant.components.media_player import MediaPlayerDeviceClass
from homeassistant.components.remote import RemoteEntityFeature
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.switch import SwitchDeviceClass
from homeassistant.const import (
    ATTR_BATTERY_CHARGING,
    ATTR_BATTERY_LEVEL,
    ATTR_DEVICE_CLASS,
    ATTR_ENTITY_ID,
    ATTR_HW_VERSION,
    ATTR_MANUFACTURER,
    ATTR_MODEL,
    ATTR_SERVICE,
    ATTR_SUPPORTED_FEATURES,
    ATTR_SW_VERSION,
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_NAME,
    CONF_TYPE,
    LIGHT_LUX,
    PERCENTAGE,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
    __version__,
)

def get_accessory(hass: HomeAssistant, driver: AccessoryDriver, state: State, aid: UUID, config: dict) -> Accessory | None:
    ...

class HomeAccessory(Accessory):
    """Adapter class for Accessory."""

    def __init__(self, hass: HomeAssistant, driver: AccessoryDriver, name: str, entity_id: str, aid: UUID, config: dict, *args, category: int = CATEGORY_OTHER, device_id: str | None = None, **kwargs) -> None:
        ...

    @ha_callback
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
    def async_update_battery(self, battery_level: float | None, battery_charging: bool | None) -> None:
        ...

    @ha_callback
    def async_update_state(self, new_state: State) -> None:
        ...

    @ha_callback
    def async_call_service(self, domain: str, service: str, service_data: dict, value: Any | None) -> None:
        ...

    @ha_callback
    def async_reload(self) -> None:
        ...

    @ha_callback
    def async_stop(self) -> None:
        ...

class HomeBridge(Bridge):
    """Adapter class for Bridge."""

    def __init__(self, hass: HomeAssistant, driver: AccessoryDriver, name: str) -> None:
        ...

    def setup_message(self) -> None:
        ...

    async def async_get_snapshot(self, info: dict) -> bytes:
        ...

class HomeDriver(AccessoryDriver):
    """Adapter class for AccessoryDriver."""

    def __init__(self, hass: HomeAssistant, entry_id: str, bridge_name: str, entry_title: str, iid_storage: IIDManager, **kwargs) -> None:
        ...

    @pyhap_callback
    def pair(self, client_username_bytes: bytes, client_public: str, client_permissions: str) -> bool:
        ...

    @pyhap_callback
    def unpair(self, client_uuid: UUID) -> None:
        ...

class HomeIIDManager(IIDManager):
    """IID Manager that remembers IIDs between restarts."""

    def __init__(self, iid_storage: IIDManager) -> None:
        ...

    def get_iid_for_obj(self, obj: Any) -> UUID:
        ...
