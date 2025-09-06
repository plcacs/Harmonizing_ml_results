from __future__ import annotations
import logging
from typing import Any, List
from homeassistant.components.network import async_get_source_ip
from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import slugify
from .const import DOMAIN, SWITCH_TYPE_DEFLECTION, SWITCH_TYPE_PORTFORWARD, SWITCH_TYPE_PROFILE, SWITCH_TYPE_WIFINETWORK, WIFI_STANDARD, MeshRoles
from .coordinator import FRITZ_DATA_KEY, AvmWrapper, FritzConfigEntry, FritzData, FritzDevice, SwitchInfo, device_filter_out_from_trackers
from .entity import FritzBoxBaseEntity, FritzDeviceBase

_LOGGER = logging.getLogger(__name__)

async def _async_deflection_entities_list(avm_wrapper: AvmWrapper, device_friendly_name: str) -> List[FritzBoxDeflectionSwitch]:
    ...

async def _async_port_entities_list(avm_wrapper: AvmWrapper, device_friendly_name: str, local_ip: str) -> List[FritzBoxPortSwitch]:
    ...

async def _async_wifi_entities_list(avm_wrapper: AvmWrapper, device_friendly_name: str) -> List[FritzBoxWifiSwitch]:
    ...

async def _async_profile_entities_list(avm_wrapper: AvmWrapper, data_fritz: FritzData) -> List[FritzBoxProfileSwitch]:
    ...

async def async_all_entities_list(avm_wrapper: AvmWrapper, device_friendly_name: str, data_fritz: FritzData, local_ip: str) -> List[Entity]:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class FritzBoxBaseCoordinatorSwitch(CoordinatorEntity[AvmWrapper], SwitchEntity):
    ...

class FritzBoxBaseSwitch(FritzBoxBaseEntity, SwitchEntity):
    ...

class FritzBoxPortSwitch(FritzBoxBaseSwitch):
    ...

class FritzBoxDeflectionSwitch(FritzBoxBaseCoordinatorSwitch):
    ...

class FritzBoxProfileSwitch(FritzDeviceBase, SwitchEntity):
    ...

class FritzBoxWifiSwitch(FritzBoxBaseSwitch):
    ...
