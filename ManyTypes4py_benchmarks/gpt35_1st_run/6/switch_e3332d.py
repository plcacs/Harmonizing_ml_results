from __future__ import annotations
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar
from tplink_omada_client import OmadaSiteClient, SwitchPortOverrides
from tplink_omada_client.definitions import GatewayPortMode, PoEMode, PortType
from tplink_omada_client.devices import OmadaDevice, OmadaGateway, OmadaGatewayPortConfig, OmadaGatewayPortStatus, OmadaSwitch, OmadaSwitchPortDetails
from tplink_omada_client.omadasiteclient import GatewayPortSettings
from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import OmadaConfigEntry
from .controller import OmadaGatewayCoordinator, OmadaSwitchPortCoordinator
from .coordinator import OmadaCoordinator
from .entity import OmadaDeviceEntity

TPort = TypeVar('TPort')
TDevice = TypeVar('TDevice', bound='OmadaDevice')
TCoordinator = TypeVar('TCoordinator', bound='OmadaCoordinator[Any]')

async def async_setup_entry(hass: HomeAssistant, config_entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

def _get_switch_port_base_name(port: Any) -> str:
    ...

@dataclass(frozen=True, kw_only=True)
class OmadaDevicePortSwitchEntityDescription(SwitchEntityDescription, Generic[TCoordinator, TDevice, TPort]):
    ...

@dataclass(frozen=True, kw_only=True)
class OmadaSwitchPortSwitchEntityDescription(OmadaDevicePortSwitchEntityDescription[OmadaSwitchPortCoordinator, OmadaSwitch, OmadaSwitchPortDetails]):
    ...

@dataclass(frozen=True, kw_only=True)
class OmadaGatewayPortConfigSwitchEntityDescription(OmadaDevicePortSwitchEntityDescription[OmadaGatewayCoordinator, OmadaGateway, OmadaGatewayPortConfig]):
    ...

@dataclass(frozen=True, kw_only=True)
class OmadaGatewayPortStatusSwitchEntityDescription(OmadaDevicePortSwitchEntityDescription[OmadaGatewayCoordinator, OmadaGateway, OmadaGatewayPortStatus]):
    ...

async def _wan_connect_disconnect(client: Any, device: Any, port: Any, enable: bool, ipv6: bool) -> None:
    ...

SWITCH_PORT_DETAILS_SWITCHES: list[OmadaSwitchPortSwitchEntityDescription] = [...]
GATEWAY_PORT_STATUS_SWITCHES: list[OmadaGatewayPortStatusSwitchEntityDescription] = [...]
GATEWAY_PORT_CONFIG_SWITCHES: list[OmadaGatewayPortConfigSwitchEntityDescription] = [...]

class OmadaDevicePortSwitchEntity(OmadaDeviceEntity[TCoordinator], SwitchEntity, Generic[TCoordinator, TDevice, TPort]):
    ...

    def __init__(self, coordinator: TCoordinator, device: TDevice, port_details: TPort, port_id: str, entity_description: OmadaDevicePortSwitchEntityDescription) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def _async_turn_on_off(self, enable: bool) -> None:
        ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    def _do_update(self) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...
