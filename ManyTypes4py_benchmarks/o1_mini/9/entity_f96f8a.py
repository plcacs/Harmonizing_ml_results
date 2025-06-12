"""Shelly entity helper."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast, Optional, Type, TypeVar, Union, Tuple
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

T = TypeVar('T', bound=Entity)


@callback
def async_setup_entry_attribute_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    sensors: Mapping[Tuple[str, str], Union[BlockEntityDescription, RpcEntityDescription]],
    sensor_class: Type[Entity],
) -> None:
    """Set up entities for attributes."""
    coordinator = config_entry.runtime_data.block
    assert coordinator
    if coordinator.device.initialized:
        async_setup_block_attribute_entities(hass, async_add_entities, coordinator, sensors, sensor_class)
    else:
        async_restore_block_attribute_entities(
            hass, config_entry, async_add_entities, coordinator, sensors, sensor_class
        )


@callback
def async_setup_block_attribute_entities(
    hass: HomeAssistant,
    async_add_entities: AddEntitiesCallback,
    coordinator: ShellyBlockCoordinator,
    sensors: Mapping[Tuple[str, str], Union[BlockEntityDescription, RpcEntityDescription]],
    sensor_class: Type[Entity],
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
            if description.removal_condition and description.removal_condition(coordinator.device.settings, block):
                domain = sensor_class.__module__.split('.')[-1]
                unique_id = f'{coordinator.mac}-{block.description}-{sensor_id}'
                async_remove_shelly_entity(hass, domain, unique_id)
            else:
                entities.append(sensor_class(coordinator, block, sensor_id, description))
    if not entities:
        return
    async_add_entities(entities)


@callback
def async_restore_block_attribute_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    coordinator: ShellyBlockCoordinator,
    sensors: Mapping[Tuple[str, str], Union[BlockEntityDescription, RpcEntityDescription]],
    sensor_class: Type[Entity],
) -> None:
    """Restore block attributes entities."""
    entities: list[Entity] = []
    ent_reg = er.async_get(hass)
    entries = er.async_entries_for_config_entry(ent_reg, config_entry.entry_id)
    domain = sensor_class.__module__.split('.')[-1]
    for entry in entries:
        if entry.domain != domain:
            continue
        unique_id_parts = entry.unique_id.split('-')
        if len(unique_id_parts) < 3:
            continue
        attribute = unique_id_parts[-1]
        block_type = unique_id_parts[-2].split('_')[0]
        description = sensors.get((block_type, attribute))
        if description:
            entities.append(sensor_class(coordinator, None, attribute, description, entry))
    if not entities:
        return
    async_add_entities(entities)


@callback
def async_setup_entry_rpc(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    sensors: Mapping[str, RpcEntityDescription],
    sensor_class: Type[Entity],
) -> None:
    """Set up entities for RPC sensors."""
    coordinator = config_entry.runtime_data.rpc
    assert coordinator
    if coordinator.device.initialized:
        async_setup_rpc_attribute_entities(hass, config_entry, async_add_entities, sensors, sensor_class)
    else:
        async_restore_rpc_attribute_entities(
            hass, config_entry, async_add_entities, coordinator, sensors, sensor_class
        )


@callback
def async_setup_rpc_attribute_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    sensors: Mapping[str, RpcEntityDescription],
    sensor_class: Type[Entity],
) -> None:
    """Set up entities for RPC attributes."""
    coordinator = config_entry.runtime_data.rpc
    assert coordinator
    polling_coordinator: Optional[ShellyRpcCoordinator] = None
    sleep_period = config_entry.data.get(CONF_SLEEP_PERIOD)
    if not sleep_period:
        polling_coordinator = config_entry.runtime_data.rpc_poll
        assert polling_coordinator
    entities: list[Entity] = []
    for sensor_id, description in sensors.items():
        key_instances = get_rpc_key_instances(coordinator.device.status, description.key)
        for key in key_instances:
            if (
                description.sub_key not in coordinator.device.status[key]
                and not description.supported(coordinator.device.status[key])
            ):
                continue
            if description.removal_condition and description.removal_condition(
                coordinator.device.config, coordinator.device.status, key
            ):
                domain = sensor_class.__module__.split('.')[-1]
                unique_id = f'{coordinator.mac}-{key}-{sensor_id}'
                async_remove_shelly_entity(hass, domain, unique_id)
            elif description.use_polling_coordinator:
                if not sleep_period:
                    entity_class = get_entity_class(sensor_class, description)
                    entities.append(entity_class(polling_coordinator, key, sensor_id, description))
            else:
                entity_class = get_entity_class(sensor_class, description)
                entities.append(entity_class(coordinator, key, sensor_id, description))
    if not entities:
        return
    async_add_entities(entities)


@callback
def async_restore_rpc_attribute_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    coordinator: ShellyRpcCoordinator,
    sensors: Mapping[str, RpcEntityDescription],
    sensor_class: Type[Entity],
) -> None:
    """Restore block attributes entities."""
    entities: list[Entity] = []
    ent_reg = er.async_get(hass)
    entries = er.async_entries_for_config_entry(ent_reg, config_entry.entry_id)
    domain = sensor_class.__module__.split('.')[-1]
    for entry in entries:
        if entry.domain != domain:
            continue
        unique_id_parts = entry.unique_id.split('-')
        if len(unique_id_parts) < 3:
            continue
        key = unique_id_parts[-2]
        attribute = unique_id_parts[-1]
        description = sensors.get(attribute)
        if description:
            entities.append(
                get_entity_class(sensor_class, description)(coordinator, key, attribute, description, entry)
            )
    if not entities:
        return
    async_add_entities(entities)


@callback
def async_setup_entry_rest(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback,
    sensors: Mapping[str, RestEntityDescription],
    sensor_class: Type[Entity],
) -> None:
    """Set up entities for REST sensors."""
    coordinator = config_entry.runtime_data.rest
    assert coordinator
    entities = (sensor_class(coordinator, sensor_id, sensors[sensor_id]) for sensor_id in sensors)
    async_add_entities(entities)


@dataclass(frozen=True)
class BlockEntityDescription(EntityDescription):
    """Class to describe a BLOCK entity."""
    name: str = ''
    unit_fn: Optional[Callable[[Any], str]] = None
    value: Callable[[Any], Any] = lambda val: val
    available: Optional[Callable[[Block], bool]] = None
    removal_condition: Optional[Callable[[Any, Block], bool]] = None
    extra_state_attributes: Optional[Callable[[Block], dict]] = None


@dataclass(frozen=True)
class RpcEntityDescription(EntityDescription):
    """Class to describe a RPC entity."""
    name: str = ''
    value: Optional[Callable[[Any, Any], Any]] = None
    available: Optional[Callable[[Any], bool]] = None
    removal_condition: Optional[Callable[[Any, Any, str], bool]] = None
    extra_state_attributes: Optional[Callable[[Any], dict]] = None
    use_polling_coordinator: bool = False
    supported: Callable[[Any], bool] = lambda _: False
    unit: Optional[Callable[[Any], str]] = None
    options_fn: Optional[Callable[[Any], Mapping[Any, Any]]] = None
    entity_class: Optional[Type[T]] = None


@dataclass(frozen=True)
class RestEntityDescription(EntityDescription):
    """Class to describe a REST entity."""
    name: str = ''
    value: Optional[Callable[[Any], Any]] = None
    extra_state_attributes: Optional[Callable[[Any], dict]] = None


class ShellyBlockEntity(CoordinatorEntity[ShellyBlockCoordinator]):
    """Helper class to represent a block entity."""

    block: Block

    def __init__(self, coordinator: ShellyBlockCoordinator, block: Block) -> None:
        """Initialize Shelly entity."""
        super().__init__(coordinator)
        self.block = block
        self._attr_name: str = get_block_entity_name(coordinator.device, block)
        self._attr_device_info: DeviceInfo = DeviceInfo(connections={(CONNECTION_NETWORK_MAC, coordinator.mac)})
        self._attr_unique_id: str = f'{coordinator.mac}-{block.description}'

    async def async_added_to_hass(self) -> None:
        """When entity is added to HASS."""
        self.async_on_remove(self.coordinator.async_add_listener(self._update_callback))

    @callback
    def _update_callback(self) -> None:
        """Handle device update."""
        self.async_write_ha_state()

    async def set_state(self, **kwargs: Any) -> Any:
        """Set block state (HTTP request)."""
        LOGGER.debug('Setting state for entity %s, state: %s', self.name, kwargs)
        try:
            return await self.block.set_state(**kwargs)
        except DeviceConnectionError as err:
            self.coordinator.last_update_success = False
            raise HomeAssistantError(
                f'Setting state for entity {self.name} failed, state: {kwargs}, error: {err!r}'
            ) from err
        except InvalidAuthError:
            await self.coordinator.async_shutdown_device_and_start_reauth()


class ShellyRpcEntity(CoordinatorEntity[ShellyRpcCoordinator]):
    """Helper class to represent a rpc entity."""

    key: str

    def __init__(self, coordinator: ShellyRpcCoordinator, key: str) -> None:
        """Initialize Shelly entity."""
        super().__init__(coordinator)
        self.key = key
        self._attr_device_info: DeviceInfo = DeviceInfo(connections={(CONNECTION_NETWORK_MAC, coordinator.mac)})
        self._attr_unique_id: str = f'{coordinator.mac}-{key}'
        self._attr_name: str = get_rpc_entity_name(coordinator.device, key)

    @property
    def available(self) -> bool:
        """Check if device is available and initialized or sleepy."""
        coordinator = self.coordinator
        return super().available and (coordinator.device.initialized or bool(coordinator.sleep_period))

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

    async def call_rpc(self, method: str, params: Any, timeout: Optional[float] = None) -> Any:
        """Call RPC method."""
        LOGGER.debug(
            'Call RPC for entity %s, method: %s, params: %s, timeout: %s',
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
                f'Call RPC for {self.name} connection error, method: {method}, params: {params}, error: {err!r}'
            ) from err
        except RpcCallError as err:
            raise HomeAssistantError(
                f'Call RPC for {self.name} request error, method: {method}, params: {params}, error: {err!r}'
            ) from err
        except InvalidAuthError:
            await self.coordinator.async_shutdown_device_and_start_reauth()


class ShellyBlockAttributeEntity(ShellyBlockEntity, Entity):
    """Helper class to represent a block attribute."""

    attribute: str
    entity_description: BlockEntityDescription

    def __init__(
        self,
        coordinator: ShellyBlockCoordinator,
        block: Optional[Block],
        attribute: str,
        description: BlockEntityDescription,
        entry: Optional[RegistryEntry] = None,
    ) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, block)
        self.attribute = attribute
        self.entity_description: BlockEntityDescription = description
        self._attr_unique_id = f'{super().unique_id}-{self.attribute}'
        self._attr_name = get_block_entity_name(coordinator.device, block, description.name)

    @property
    def attribute_value(self) -> Optional[StateType]:
        """Value of sensor."""
        value = getattr(self.block, self.attribute, None)
        if value is None:
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
    def extra_state_attributes(self) -> Optional[dict]:
        """Return the state attributes."""
        if self.entity_description.extra_state_attributes is None:
            return None
        return self.entity_description.extra_state_attributes(self.block)


class ShellyRestAttributeEntity(CoordinatorEntity[ShellyBlockCoordinator]):
    """Class to load info from REST."""

    attribute: str
    entity_description: RestEntityDescription
    _last_value: Any

    def __init__(
        self,
        coordinator: ShellyBlockCoordinator,
        attribute: str,
        description: RestEntityDescription,
    ) -> None:
        """Initialize sensor."""
        super().__init__(coordinator)
        self.block_coordinator = coordinator
        self.attribute = attribute
        self.entity_description: RestEntityDescription = description
        self._attr_name: str = get_block_entity_name(coordinator.device, None, description.name)
        self._attr_unique_id: str = f'{coordinator.mac}-{attribute}'
        self._attr_device_info: DeviceInfo = DeviceInfo(connections={(CONNECTION_NETWORK_MAC, coordinator.mac)})
        self._last_value: Any = None

    @property
    def available(self) -> bool:
        """Available."""
        return self.block_coordinator.last_update_success

    @property
    def attribute_value(self) -> Any:
        """Value of sensor."""
        if self.entity_description.value is not None:
            self._last_value = self.entity_description.value(self.block_coordinator.device.status, self._last_value)
        return self._last_value


class ShellyRpcAttributeEntity(ShellyRpcEntity, Entity):
    """Helper class to represent a rpc attribute."""

    attribute: str
    entity_description: RpcEntityDescription
    _last_value: Any
    _id: Optional[int]
    option_map: Mapping[Any, Any]
    reversed_option_map: Mapping[Any, Any]

    def __init__(
        self,
        coordinator: ShellyRpcCoordinator,
        key: str,
        attribute: str,
        description: RpcEntityDescription,
        entry: Optional[RegistryEntry] = None,
    ) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, key)
        self.attribute = attribute
        self.entity_description: RpcEntityDescription = description
        self._attr_unique_id = f'{super().unique_id}-{attribute}'
        self._attr_name = get_rpc_entity_name(coordinator.device, key, description.name)
        self._last_value: Any = None
        id_key = key.split(':')[-1]
        self._id: Optional[int] = int(id_key) if id_key.isnumeric() else None
        if description.unit is not None:
            self._attr_native_unit_of_measurement = description.unit(coordinator.device.config[key])
        self.option_map: Mapping[Any, Any] = {}
        self.reversed_option_map: Mapping[Any, Any] = {}
        if 'enum' in key:
            titles = self.coordinator.device.config[key]['meta']['ui']['titles']
            options = self.coordinator.device.config[key]['options']
            self.option_map = {
                opt: titles[opt] if titles.get(opt) is not None else opt for opt in options
            }
            self.reversed_option_map = {tit: opt for opt, tit in self.option_map.items()}

    @property
    def sub_status(self) -> Any:
        """Device status by entity key."""
        return self.status[self.entity_description.sub_key]

    @property
    def attribute_value(self) -> Any:
        """Value of sensor."""
        if self.entity_description.value is not None:
            self._last_value = self.entity_description.value(
                self.status.get(self.entity_description.sub_key), self._last_value
            )
        else:
            self._last_value = self.sub_status
        return self._last_value

    @property
    def available(self) -> bool:
        """Available."""
        available = super().available
        if not available or not self.entity_description.available:
            return available
        return self.entity_description.available(self.sub_status)


class ShellySleepingBlockAttributeEntity(ShellyBlockAttributeEntity):
    """Represent a shelly sleeping block attribute entity."""

    last_state: Optional[Any]

    def __init__(
        self,
        coordinator: ShellyBlockCoordinator,
        block: Optional[Block],
        attribute: str,
        description: BlockEntityDescription,
        entry: Optional[RegistryEntry] = None,
    ) -> None:
        """Initialize the sleeping sensor."""
        self.last_state = None
        self.coordinator = coordinator
        self.attribute = attribute
        self.block = block
        self.entity_description: BlockEntityDescription = description
        self._attr_device_info: DeviceInfo = DeviceInfo(connections={(CONNECTION_NETWORK_MAC, coordinator.mac)})
        if block is not None:
            self._attr_unique_id = f'{self.coordinator.mac}-{block.description}-{attribute}'
            self._attr_name = get_block_entity_name(self.coordinator.device, block, self.entity_description.name)
        elif entry is not None:
            self._attr_unique_id = entry.unique_id
            self._attr_name = cast(str, entry.original_name)

    @callback
    def _update_callback(self) -> None:
        """Handle device update."""
        if self.block is not None or not self.coordinator.device.initialized:
            super()._update_callback()
            return
        unique_id_parts = self._attr_unique_id.split('-')
        if len(unique_id_parts) < 3:
            return
        _, entity_block, entity_sensor = unique_id_parts
        assert self.coordinator.device.blocks
        for block in self.coordinator.device.blocks:
            if block.description != entity_block:
                continue
            for sensor_id in block.sensor_ids:
                if sensor_id != entity_sensor:
                    continue
                self.block = block
                LOGGER.debug('Entity %s attached to block', self.name)
                super()._update_callback()
                return

    async def async_update(self) -> None:
        """Update the entity."""
        LOGGER.info('Entity %s comes from a sleeping device, update is not possible', self.entity_id)


class ShellySleepingRpcAttributeEntity(ShellyRpcAttributeEntity):
    """Helper class to represent a sleeping rpc attribute."""

    last_state: Optional[Any]

    def __init__(
        self,
        coordinator: ShellyRpcCoordinator,
        key: str,
        attribute: str,
        description: RpcEntityDescription,
        entry: Optional[RegistryEntry] = None,
    ) -> None:
        """Initialize the sleeping sensor."""
        self.last_state = None
        self.coordinator = coordinator
        self.key = key
        self.attribute = attribute
        self.entity_description: RpcEntityDescription = description
        self._attr_device_info: DeviceInfo = DeviceInfo(connections={(CONNECTION_NETWORK_MAC, coordinator.mac)})
        self._attr_unique_id: str = f'{coordinator.mac}-{key}-{attribute}'
        self._last_value: Any = None
        if coordinator.device.initialized:
            self._attr_name = get_rpc_entity_name(coordinator.device, key, description.name)
        elif entry is not None:
            self._attr_name = cast(str, entry.original_name)

    async def async_update(self) -> None:
        """Update the entity."""
        LOGGER.info('Entity %s comes from a sleeping device, update is not possible', self.entity_id)


def get_entity_class(sensor_class: Type[T], description: Union[BlockEntityDescription, RpcEntityDescription, RestEntityDescription]) -> Type[T]:
    """Return entity class."""
    if hasattr(description, 'entity_class') and description.entity_class is not None:
        return description.entity_class
    return sensor_class
