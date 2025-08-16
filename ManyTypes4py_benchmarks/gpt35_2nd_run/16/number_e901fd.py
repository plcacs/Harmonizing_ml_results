from __future__ import annotations
from typing import TYPE_CHECKING, Any, Final, cast
from homeassistant.components.number import DOMAIN as NUMBER_PLATFORM, NumberEntity, NumberEntityDescription, NumberExtraStoredData, NumberMode, RestoreNumber
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.entity_registry import RegistryEntry
from .const import BLU_TRV_TIMEOUT, CONF_SLEEP_PERIOD, LOGGER, VIRTUAL_NUMBER_MODE_MAP
from .coordinator import ShellyBlockCoordinator, ShellyConfigEntry, ShellyRpcCoordinator
from .entity import BlockEntityDescription, RpcEntityDescription, ShellyRpcAttributeEntity, ShellySleepingBlockAttributeEntity, async_setup_entry_attribute_entities, async_setup_entry_rpc, async_remove_orphaned_entities, get_device_entry_gen, get_virtual_component_ids

class BlockNumberDescription(BlockEntityDescription, NumberEntityDescription):
    rest_path: str = ''
    rest_arg: str = ''

class RpcNumberDescription(RpcEntityDescription, NumberEntityDescription):
    max_fn: Callable | None = None
    min_fn: Callable | None = None
    step_fn: Callable | None = None
    mode_fn: Callable | None = None

class RpcNumber(ShellyRpcAttributeEntity, NumberEntity):
    def __init__(self, coordinator: ShellyRpcCoordinator, key: str, attribute: str, description: RpcNumberDescription) -> None:
        ...

    @property
    def native_value(self) -> float | None:
        ...

    async def async_set_native_value(self, value: float) -> None:
        ...

class RpcBluTrvNumber(RpcNumber):
    def __init__(self, coordinator: ShellyRpcCoordinator, key: str, attribute: str, description: RpcNumberDescription) -> None:
        ...

    async def async_set_native_value(self, value: float) -> None:
        ...

class RpcBluTrvExtTempNumber(RpcBluTrvNumber):
    _reported_value: float | None = None

    @property
    def native_value(self) -> float | None:
        ...

    async def async_set_native_value(self, value: float) -> None:
        ...

async def async_setup_entry(hass: HomeAssistant, config_entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class BlockSleepingNumber(ShellySleepingBlockAttributeEntity, RestoreNumber):
    def __init__(self, coordinator: ShellyBlockCoordinator, block: Block, attribute: str, description: BlockNumberDescription, entry: RegistryEntry | None = None) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @property
    def native_value(self) -> float | None:
        ...

    async def async_set_native_value(self, value: float) -> None:
        ...

    async def _set_state_full_path(self, path: str, params: dict[str, Any]) -> Any:
        ...
