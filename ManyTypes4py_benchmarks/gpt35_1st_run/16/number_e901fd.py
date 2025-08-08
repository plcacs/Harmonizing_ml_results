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
    _attr_native_max_value: float | None
    _attr_native_min_value: float | None
    _attr_native_step: float | None
    _attr_mode: NumberMode | None

    def native_value(self) -> float:
        ...

    async def async_set_native_value(self, value: float):
        ...

class RpcBluTrvNumber(RpcNumber):
    _attr_device_info: DeviceInfo

    async def async_set_native_value(self, value: float):
        ...

class RpcBluTrvExtTempNumber(RpcBluTrvNumber):
    _reported_value: float | None

    def native_value(self) -> float:
        ...

    async def async_set_native_value(self, value: float):
        ...

async def async_setup_entry(hass: HomeAssistant, config_entry: Any, async_add_entities: AddConfigEntryEntitiesCallback):
    ...

class BlockSleepingNumber(ShellySleepingBlockAttributeEntity, RestoreNumber):
    restored_data: NumberExtraStoredData | None

    def native_value(self) -> float | None:
        ...

    async def async_set_native_value(self, value: float):
        ...

    async def _set_state_full_path(self, path: str, params: dict[str, Any]):
        ...
