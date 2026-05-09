from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast
from aioshelly.block_device import Block
from aioshelly.exceptions import DeviceConnectionError, InvalidAuthError, RpcCallError
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo
from homeassistant.helpers.entity import Entity, EntityDescription
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import RegistryEntry
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import CONF_SLEEP_PERIOD, LOGGER
from .coordinator import ShellyBlockCoordinator, ShellyConfigEntry, ShellyRpcCoordinator
from .utils import async_remove_shelly_entity, get_block_entity_name, get_rpc_entity_name, get_rpc_key_instances

@callback
def async_setup_entry_attribute_entities(hass: HomeAssistant, config_entry: er.RegistryEntry, async_add_entities: Callable[[list[Entity]], None], sensors: Mapping[str, EntityDescription], sensor_class: type[Entity]) -> None:
    """Set up entities for attributes."""
    ...

@callback
def async_setup_block_attribute_entities(hass: HomeAssistant, async_add_entities: Callable[[list[Entity]], None], coordinator: ShellyBlockCoordinator, sensors: Mapping[str, EntityDescription], sensor_class: type[Entity]) -> None:
    """Set up entities for block attributes."""
    ...

@callback
def async_restore_block_attribute_entities(hass: HomeAssistant, config_entry: er.RegistryEntry, async_add_entities: Callable[[list[Entity]], None], coordinator: ShellyBlockCoordinator, sensors: Mapping[str, EntityDescription], sensor_class: type[Entity]) -> None:
    """Restore block attributes entities."""
    ...

@callback
def async_setup_entry_rpc(hass: HomeAssistant, config_entry: er.RegistryEntry, async_add_entities: Callable[[list[Entity]], None], sensors: Mapping[str, EntityDescription], sensor_class: type[Entity]) -> None:
    """Set up entities for RPC sensors."""
    ...

@callback
def async_setup_rpc_attribute_entities(hass: HomeAssistant, config_entry: er.RegistryEntry, async_add_entities: Callable[[list[Entity]], None], sensors: Mapping[str, EntityDescription], sensor_class: type[Entity]) -> None:
    """Set up entities for RPC attributes."""
    ...

@callback
def async_restore_rpc_attribute_entities(hass: HomeAssistant, config_entry: er.RegistryEntry, async_add_entities: Callable[[list[Entity]], None], coordinator: ShellyRpcCoordinator, sensors: Mapping[str, EntityDescription], sensor_class: type[Entity]) -> None:
    """Restore RPC attributes entities."""
    ...

@callback
def async_setup_entry_rest(hass: HomeAssistant, config_entry: er.RegistryEntry, async_add_entities: Callable[[list[Entity]], None], sensors: Mapping[str, EntityDescription], sensor_class: type[Entity]) -> None:
    """Set up entities for REST sensors."""
    ...

@dataclass(frozen=True)
class BlockEntityDescription(EntityDescription):
    """Class to describe a BLOCK entity."""
    name: str
    unit_fn: Callable[[StateType], str] | None
    value: Callable[[StateType], StateType]
    available: Callable[[ShellyBlockCoordinator, Block], bool] | None
    removal_condition: Callable[[ShellyBlockCoordinator, Block], bool] | None
    extra_state_attributes: Callable[[Block], Mapping[str, Any]] | None

@dataclass(frozen=True, kw_only=True)
class RpcEntityDescription(EntityDescription):
    """Class to describe a RPC entity."""
    name: str
    value: Callable[[dict], StateType] | None
    available: Callable[[ShellyRpcCoordinator, dict], bool] | None
    removal_condition: Callable[[ShellyRpcCoordinator, dict], bool] | None
    extra_state_attributes: Callable[[dict], Mapping[str, Any]] | None
    use_polling_coordinator: bool
    supported: Callable[[dict], bool]
    unit: str | None
    options_fn: Callable[[dict], Mapping[str, Any]] | None
    entity_class: type[Entity] | None

@dataclass(frozen=True)
class RestEntityDescription(EntityDescription):
    """Class to describe a REST entity."""
    name: str
    value: Callable[[dict], StateType] | None
    extra_state_attributes: Callable[[dict], Mapping[str, Any]] | None

class ShellyBlockEntity(CoordinatorEntity[ShellyBlockCoordinator]):
    """Helper class to represent a block entity."""
    ...

class ShellyRpcEntity(CoordinatorEntity[ShellyRpcCoordinator]):
    """Helper class to represent a rpc entity."""
    ...

class ShellyBlockAttributeEntity(ShellyBlockEntity, Entity):
    """Helper class to represent a block attribute."""
    ...

class ShellyRestAttributeEntity(CoordinatorEntity[ShellyBlockCoordinator]):
    """Class to load info from REST."""
    ...

class ShellyRpcAttributeEntity(ShellyRpcEntity, Entity):
    """Helper class to represent a rpc attribute."""
    ...

class ShellySleepingBlockAttributeEntity(ShellyBlockAttributeEntity):
    """Represent a shelly sleeping block attribute entity."""
    ...

class ShellySleepingRpcAttributeEntity(ShellyRpcAttributeEntity):
    """Helper class to represent a sleeping rpc attribute."""
    ...

def get_entity_class(sensor_class: type[Entity], description: EntityDescription) -> type[Entity]:
    """Return entity class."""
    ...
