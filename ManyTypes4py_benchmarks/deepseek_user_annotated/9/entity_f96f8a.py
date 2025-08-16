"""Shelly entity helper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast, Optional, Union, TypeVar, Generic, Type

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
from .utils import (
    async_remove_shelly_entity,
    get_block_entity_name,
    get_rpc_entity_name,
    get_rpc_key_instances,
)

T = TypeVar('T')
CoordinatorT = TypeVar('CoordinatorT', bound=CoordinatorEntity)

@callback
def async_setup_entry_attribute_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    sensors: Mapping[tuple[str, str], 'BlockEntityDescription'],
    sensor_class: Callable[..., Entity],
) -> None:
    """Set up entities for attributes."""
    coordinator = config_entry.runtime_data.block
    assert coordinator
    if coordinator.device.initialized:
        async_setup_block_attribute_entities(
            hass, async_add_entities, coordinator, sensors, sensor_class
        )
    else:
        async_restore_block_attribute_entities(
            hass,
            config_entry,
            async_add_entities,
            coordinator,
            sensors,
            sensor_class,
        )

@callback
def async_setup_block_attribute_entities(
    hass: HomeAssistant,
    async_add_entities: AddEntitiesCallback,
    coordinator: ShellyBlockCoordinator,
    sensors: Mapping[tuple[str, str], 'BlockEntityDescription'],
    sensor_class: Callable[..., Entity],
) -> None:
    """Set up entities for block attributes."""
    entities: list[Entity] = []

    assert coordinator.device.blocks

    for block in coordinator.device.blocks:
        for sensor_id in block.sensor_ids:
            description = sensors.get((cast(str, block.type), sensor_id))
            if description is None:
                continue

            if getattr(block, sensor_id, None) is None:
                continue

            if description.removal_condition and description.removal_condition(
                coordinator.device.settings, block
            ):
                domain = sensor_class.__module__.split(".")[-1]
                unique_id = f"{coordinator.mac}-{block.description}-{sensor_id}"
                async_remove_shelly_entity(hass, domain, unique_id)
            else:
                entities.append(
                    sensor_class(coordinator, block, sensor_id, description)
                )

    if not entities:
        return

    async_add_entities(entities)

@callback
def async_restore_block_attribute_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    coordinator: ShellyBlockCoordinator,
    sensors: Mapping[tuple[str, str], 'BlockEntityDescription'],
    sensor_class: Callable[..., Entity],
) -> None:
    """Restore block attributes entities."""
    entities: list[Entity] = []

    ent_reg = er.async_get(hass)
    entries = er.async_entries_for_config_entry(ent_reg, config_entry.entry_id)

    domain = sensor_class.__module__.split(".")[-1]

    for entry in entries:
        if entry.domain != domain:
            continue

        attribute = entry.unique_id.split("-")[-1]
        block_type = entry.unique_id.split("-")[-2].split("_")[0]

        if description := sensors.get((block_type, attribute)):
            entities.append(
                sensor_class(coordinator, None, attribute, description, entry)
            )

    if not entities:
        return

    async_add_entities(entities)

@callback
def async_setup_entry_rpc(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    sensors: Mapping[str, 'RpcEntityDescription'],
    sensor_class: Callable[..., Entity],
) -> None:
    """Set up entities for RPC sensors."""
    coordinator = config_entry.runtime_data.rpc
    assert coordinator

    if coordinator.device.initialized:
        async_setup_rpc_attribute_entities(
            hass, config_entry, async_add_entities, sensors, sensor_class
        )
    else:
        async_restore_rpc_attribute_entities(
            hass, config_entry, async_add_entities, coordinator, sensors, sensor_class
        )

@callback
def async_setup_rpc_attribute_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    sensors: Mapping[str, 'RpcEntityDescription'],
    sensor_class: Callable[..., Entity],
) -> None:
    """Set up entities for RPC attributes."""
    coordinator = config_entry.runtime_data.rpc
    assert coordinator

    polling_coordinator = None
    if not (sleep_period := config_entry.data[CONF_SLEEP_PERIOD]):
        polling_coordinator = config_entry.runtime_data.rpc_poll
        assert polling_coordinator

    entities: list[Entity] = []
    for sensor_id in sensors:
        description = sensors[sensor_id]
        key_instances = get_rpc_key_instances(
            coordinator.device.status, description.key
        )

        for key in key_instances:
            if description.sub_key not in coordinator.device.status[
                key
            ] and not description.supported(coordinator.device.status[key]):
                continue

            if description.removal_condition and description.removal_condition(
                coordinator.device.config, coordinator.device.status, key
            ):
                domain = sensor_class.__module__.split(".")[-1]
                unique_id = f"{coordinator.mac}-{key}-{sensor_id}"
                async_remove_shelly_entity(hass, domain, unique_id)
            elif description.use_polling_coordinator:
                if not sleep_period:
                    entities.append(
                        get_entity_class(sensor_class, description)(
                            polling_coordinator, key, sensor_id, description
                        )
                    )
            else:
                entities.append(
                    get_entity_class(sensor_class, description)(
                        coordinator, key, sensor_id, description
                    )
                )
    if not entities:
        return

    async_add_entities(entities)

@callback
def async_restore_rpc_attribute_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    coordinator: ShellyRpcCoordinator,
    sensors: Mapping[str, 'RpcEntityDescription'],
    sensor_class: Callable[..., Entity],
) -> None:
    """Restore block attributes entities."""
    entities: list[Entity] = []

    ent_reg = er.async_get(hass)
    entries = er.async_entries_for_config_entry(ent_reg, config_entry.entry_id)

    domain = sensor_class.__module__.split(".")[-1]

    for entry in entries:
        if entry.domain != domain:
            continue

        key = entry.unique_id.split("-")[-2]
        attribute = entry.unique_id.split("-")[-1]

        if description := sensors.get(attribute):
            entities.append(
                get_entity_class(sensor_class, description)(
                    coordinator, key, attribute, description, entry
                )
            )

    if not entities:
        return

    async_add_entities(entities)

@callback
def async_setup_entry_rest(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    sensors: Mapping[str, 'RestEntityDescription'],
    sensor_class: Callable[..., Entity],
) -> None:
    """Set up entities for REST sensors."""
    coordinator = config_entry.runtime_data.rest
    assert coordinator

    async_add_entities(
        sensor_class(coordinator, sensor_id, sensors[sensor_id])
        for sensor_id in sensors
    )

@dataclass(frozen=True)
class BlockEntityDescription(EntityDescription):
    """Class to describe a BLOCK entity."""

    name: str = ""
    unit_fn: Optional[Callable[[dict], str]] = None
    value: Callable[[Any], Any] = lambda val: val
    available: Optional[Callable[[Block], bool]] = None
    removal_condition: Optional[Callable[[dict, Block], bool]] = None
    extra_state_attributes: Optional[Callable[[Block], Optional[dict]]] = None

@dataclass(frozen=True, kw_only=True)
class RpcEntityDescription(EntityDescription):
    """Class to describe a RPC entity."""

    name: str = ""
    sub_key: str
    value: Optional[Callable[[Any, Any], Any]] = None
    available: Optional[Callable[[dict], bool]] = None
    removal_condition: Optional[Callable[[dict, dict, str], bool]] = None
    extra_state_attributes: Optional[Callable[[dict, dict], Optional[dict]]] = None
    use_polling_coordinator: bool = False
    supported: Callable = lambda _: False
    unit: Optional[Callable[[dict], Optional[str]]] = None
    options_fn: Optional[Callable[[dict], list[str]]] = None
    entity_class: Optional[Callable] = None

@dataclass(frozen=True)
class RestEntityDescription(EntityDescription):
    """Class to describe a REST entity."""

    name: str = ""
    value: Optional[Callable[[dict, Any], Any]] = None
    extra_state_attributes: Optional[Callable[[dict], Optional[dict]]] = None

class ShellyBlockEntity(CoordinatorEntity[ShellyBlockCoordinator]):
    """Helper class to represent a block entity."""

    def __init__(self, coordinator: ShellyBlockCoordinator, block: Block) -> None:
        """Initialize Shelly entity."""
        super().__init__(coordinator)
        self.block: Block = block
        self._attr_name: str = get_block_entity_name(coordinator.device, block)
        self._attr_device_info: DeviceInfo = DeviceInfo(
            connections={(CONNECTION_NETWORK_MAC, coordinator.mac)}
        )
        self._attr_unique_id: str = f"{coordinator.mac}-{block.description}"

    async def async_added_to_hass(self) -> None:
        """When entity is added to HASS."""
        self.async_on_remove(self.coordinator.async_add_listener(self._update_callback))

    @callback
    def _update_callback(self) -> None:
        """Handle device update."""
        self.async_write_ha_state()

    async def set_state(self, **kwargs: Any) -> Any:
        """Set block state (HTTP request)."""
        LOGGER.debug("Setting state for entity %s, state: %s", self.name, kwargs)
        try:
            return await self.block.set_state(**kwargs)
        except DeviceConnectionError as err:
            self.coordinator.last_update_success = False
            raise HomeAssistantError(
                f"Setting state for entity {self.name} failed, state: {kwargs}, error:"
                f" {err!r}"
            ) from err
        except InvalidAuthError:
            await self.coordinator.async_shutdown_device_and_start_reauth()

class ShellyRpcEntity(CoordinatorEntity[ShellyRpcCoordinator]):
    """Helper class to represent a rpc entity."""

    def __init__(self, coordinator: ShellyRpcCoordinator, key: str) -> None:
        """Initialize Shelly entity."""
        super().__init__(coordinator)
        self.key: str = key
        self._attr_device_info: dict = {
            "connections": {(CONNECTION_NETWORK_MAC, coordinator.mac)}
        }
        self._attr_unique_id: str = f"{coordinator.mac}-{key}"
        self._attr_name: str = get_rpc_entity_name(coordinator.device, key)

    @property
    def available(self) -> bool:
        """Check if device is available and initialized or sleepy."""
        coordinator = self.coordinator
        return super().available and (
            coordinator.device.initialized or bool(coordinator.sleep_period)
        )

    @property
    def status(self) -> dict:
        """Device status by entity key."""
        return cast(dict, self.coordinator.device.status[self.key])

    async def async_added_to_hass(self) -> None:
        """When entity is added to HASS."""
        self.async_on_remove(self.coordinator.async_add_listener(self._update_callback))

    @callback
    def _update_callback(self) -> None:
        """Handle device update."""
        self.async_write_ha_state()

    async def call_rpc(
        self, method: str, params: Any, timeout: Optional[float] = None
    ) -> Any:
        """Call RPC method."""
        LOGGER.debug(
            "Call RPC for entity %s, method: %s, params: %s, timeout: %s",
            self.name,
            method,
            params,
            timeout,
        )
        try:
            if timeout:
                return await self.coordinator.device.call_rpc(method, params, timeout)
            return await self.coordinator.device.call_rpc(method, params)
        except DeviceConnectionError as err:
            self.coordinator.last_update_success = False
            raise HomeAssistantError(
                f"Call RPC for {self.name} connection error, method: {method}, params:"
                f" {params}, error: {err!r}"
            ) from err
        except RpcCallError as err:
            raise HomeAssistantError(
                f"Call RPC for {self.name} request error, method: {method}, params:"
                f" {params}, error: {err!r}"
            ) from err
        except InvalidAuthError:
            await self.coordinator.async_shutdown_device_and_start_reauth()

class ShellyBlockAttributeEntity(ShellyBlockEntity, Entity):
    """Helper class to represent a block attribute."""

    entity_description: BlockEntityDescription

    def __init__(
        self,
        coordinator: ShellyBlockCoordinator,
        block: Block,
        attribute: str,
        description: BlockEntityDescription,
    ) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, block)
        self.attribute: str = attribute
        self.entity_description: BlockEntityDescription = description
        self._attr_unique_id: str = f"{super().unique_id}-{self.attribute}"
        self._attr_name: str = get_block_entity_name(
            coordinator.device, block, description.name
        )

    @property
    def attribute_value(self) -> StateType:
        """Value of sensor."""
        if (value := getattr(self.block, self.attribute)) is None:
            return None
        return cast(StateType, self.entity_description.value(value))

    @property
    def available(self) -> bool:
        """Available."""
        available = super().available
        if not available or not self.entity_description.available or self.block is None:
            return available
        return self.entity_description.available(self.block)

    @property
    def extra_state_attributes(self) -> Optional[dict[str, Any]]:
        """Return the state attributes."""
        if self.entity_description.extra_state_attributes is None:
            return None
        return self.entity_description.extra_state_attributes(self.block)

class ShellyRestAttributeEntity(CoordinatorEntity[ShellyBlockCoordinator]):
    """Class to load info from REST."""

    entity_description: RestEntityDescription

    def __init__(
        self,
        coordinator: ShellyBlockCoordinator,
        attribute: str,
        description: RestEntityDescription,
    ) -> None:
        """Initialize sensor."""
        super().__init__(coordinator)
        self.block_coordinator: ShellyBlockCoordinator = coordinator
        self.attribute: str = attribute
        self.entity_description: RestEntityDescription = description
        self._attr_name: str = get_block_entity_name(
            coordinator.device, None, description.name
        )
        self._attr_unique_id: str = f"{coordinator.mac}-{attribute}"
        self._attr_device_info: DeviceInfo = DeviceInfo(
            connections={(CONNECTION_NETWORK_MAC, coordinator.mac)}
        )
        self._last_value: Optional[Any] = None

    @property
    def available(self) -> bool:
        """Available."""
        return self.block_coordinator.last_update_success

    @property
    def attribute_value(self) -> StateType:
        """Value of sensor."""
        if self.entity_description.value is not None:
            self._last_value = self.entity_description.value(
                self.block_coordinator.device.status, self._last_value
            )
        return self._last_value

class ShellyRpcAttributeEntity(ShellyRpcEntity, Entity):
    """Helper class to represent a rpc attribute."""

    entity_description: RpcEntityDescription

    def __init__(
        self,
        coordinator: ShellyRpcCoordinator,
        key: str,
        attribute: str,
        description: RpcEntityDescription,
    ) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, key)
        self.attribute: str = attribute
        self.entity_description: RpcEntityDescription = description
        self._attr_unique_id: str = f"{super().unique_id}-{attribute}"
        self._attr_name: str = get_rpc_entity_name(coordinator.device, key, description.name)
        self._last_value: Optional[Any] = None
        id_key = key.split(":")[-1]
        self._id: Optional[int] = int(id_key) if id_key.isnumeric() else None

        if description.unit is not None:
            self._attr_native_unit_of_measurement