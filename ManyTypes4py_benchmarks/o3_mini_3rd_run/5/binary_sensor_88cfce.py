from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final, Optional, cast, Dict, Tuple

from aioshelly.const import RPC_GENERATIONS
from homeassistant.components.binary_sensor import (
    DOMAIN as BINARY_SENSOR_PLATFORM,
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.const import STATE_ON, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from .const import CONF_SLEEP_PERIOD
from .coordinator import ShellyConfigEntry, ShellyRpcCoordinator
from .entity import (
    BlockEntityDescription,
    RestEntityDescription,
    RpcEntityDescription,
    ShellyBlockAttributeEntity,
    ShellyRestAttributeEntity,
    ShellyRpcAttributeEntity,
    ShellySleepingBlockAttributeEntity,
    ShellySleepingRpcAttributeEntity,
    async_setup_entry_attribute_entities,
    async_setup_entry_rest,
    async_setup_entry_rpc,
)
from .utils import async_remove_orphaned_entities, get_device_entry_gen, get_virtual_component_ids, is_block_momentary_input, is_rpc_momentary_input


@dataclass(frozen=True, kw_only=True)
class BlockBinarySensorDescription(BlockEntityDescription, BinarySensorEntityDescription):
    available: Optional[Callable[[Any], bool]] = None
    value: Optional[Callable[[Any], Any]] = None
    extra_state_attributes: Optional[Callable[[Any], dict]] = None
    removal_condition: Optional[Callable[[Any], bool]] = None


@dataclass(frozen=True, kw_only=True)
class RpcBinarySensorDescription(RpcEntityDescription, BinarySensorEntityDescription):
    has_entity_name: Optional[bool] = None
    value: Optional[Callable[[Any, Any], Any]] = None
    supported: Optional[Callable[[Any], bool]] = None
    removal_condition: Optional[Callable[[Any], bool]] = None
    entity_class: Optional[type[RpcBinarySensor]] = None


@dataclass(frozen=True, kw_only=True)
class RestBinarySensorDescription(RestEntityDescription, BinarySensorEntityDescription):
    value: Optional[Callable[[Any, Any], Any]] = None


class RpcBinarySensor(ShellyRpcAttributeEntity, BinarySensorEntity):
    @property
    def is_on(self) -> bool:
        return bool(self.attribute_value)


class RpcBluTrvBinarySensor(RpcBinarySensor):
    def __init__(
        self,
        coordinator: ShellyRpcCoordinator,
        key: str,
        attribute: Any,
        description: RpcBinarySensorDescription,
    ) -> None:
        super().__init__(coordinator, key, attribute, description)
        ble_addr: str = coordinator.device.config[key]['addr']
        self._attr_device_info = DeviceInfo(connections={(CONNECTION_BLUETOOTH, ble_addr)})


SENSORS: Final[Dict[Tuple[str, str], BlockBinarySensorDescription]] = {
    ('device', 'overtemp'): BlockBinarySensorDescription(
        key='device|overtemp',
        name='Overheating',
        device_class=BinarySensorDeviceClass.PROBLEM,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    ('device', 'overpower'): BlockBinarySensorDescription(
        key='device|overpower',
        name='Overpowering',
        device_class=BinarySensorDeviceClass.PROBLEM,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    ('light', 'overpower'): BlockBinarySensorDescription(
        key='light|overpower',
        name='Overpowering',
        device_class=BinarySensorDeviceClass.PROBLEM,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    ('relay', 'overpower'): BlockBinarySensorDescription(
        key='relay|overpower',
        name='Overpowering',
        device_class=BinarySensorDeviceClass.PROBLEM,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    ('sensor', 'dwIsOpened'): BlockBinarySensorDescription(
        key='sensor|dwIsOpened',
        name='Door',
        device_class=BinarySensorDeviceClass.OPENING,
        available=lambda block: cast(int, block.dwIsOpened) != -1,
    ),
    ('sensor', 'flood'): BlockBinarySensorDescription(
        key='sensor|flood',
        name='Flood',
        device_class=BinarySensorDeviceClass.MOISTURE,
    ),
    ('sensor', 'gas'): BlockBinarySensorDescription(
        key='sensor|gas',
        name='Gas',
        device_class=BinarySensorDeviceClass.GAS,
        translation_key='gas',
        value=lambda value: value in ['mild', 'heavy'],
        extra_state_attributes=lambda block: {'detected': block.gas},
    ),
    ('sensor', 'smoke'): BlockBinarySensorDescription(
        key='sensor|smoke',
        name='Smoke',
        device_class=BinarySensorDeviceClass.SMOKE,
    ),
    ('sensor', 'vibration'): BlockBinarySensorDescription(
        key='sensor|vibration',
        name='Vibration',
        device_class=BinarySensorDeviceClass.VIBRATION,
    ),
    ('input', 'input'): BlockBinarySensorDescription(
        key='input|input',
        name='Input',
        device_class=BinarySensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        removal_condition=is_block_momentary_input,
    ),
    ('relay', 'input'): BlockBinarySensorDescription(
        key='relay|input',
        name='Input',
        device_class=BinarySensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        removal_condition=is_block_momentary_input,
    ),
    ('device', 'input'): BlockBinarySensorDescription(
        key='device|input',
        name='Input',
        device_class=BinarySensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        removal_condition=is_block_momentary_input,
    ),
    ('sensor', 'extInput'): BlockBinarySensorDescription(
        key='sensor|extInput',
        name='External input',
        device_class=BinarySensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
    ),
    ('sensor', 'motion'): BlockBinarySensorDescription(
        key='sensor|motion',
        name='Motion',
        device_class=BinarySensorDeviceClass.MOTION,
    ),
}

REST_SENSORS: Final[Dict[str, RestBinarySensorDescription]] = {
    'cloud': RestBinarySensorDescription(
        key='cloud',
        name='Cloud',
        value=lambda status, _: status['cloud']['connected'],
        device_class=BinarySensorDeviceClass.CONNECTIVITY,
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    )
}

RPC_SENSORS: Final[Dict[str, RpcBinarySensorDescription]] = {
    'input': RpcBinarySensorDescription(
        key='input',
        sub_key='state',
        name='Input',
        device_class=BinarySensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        removal_condition=is_rpc_momentary_input,
    ),
    'cloud': RpcBinarySensorDescription(
        key='cloud',
        sub_key='connected',
        name='Cloud',
        device_class=BinarySensorDeviceClass.CONNECTIVITY,
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    'external_power': RpcBinarySensorDescription(
        key='devicepower',
        sub_key='external',
        name='External power',
        value=lambda status, _: status['present'],
        device_class=BinarySensorDeviceClass.POWER,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    'overtemp': RpcBinarySensorDescription(
        key='switch',
        sub_key='errors',
        name='Overheating',
        device_class=BinarySensorDeviceClass.PROBLEM,
        value=lambda status, _: False if status is None else 'overtemp' in status,
        entity_category=EntityCategory.DIAGNOSTIC,
        supported=lambda status: status.get('apower') is not None,
    ),
    'overpower': RpcBinarySensorDescription(
        key='switch',
        sub_key='errors',
        name='Overpowering',
        device_class=BinarySensorDeviceClass.PROBLEM,
        value=lambda status, _: False if status is None else 'overpower' in status,
        entity_category=EntityCategory.DIAGNOSTIC,
        supported=lambda status: status.get('apower') is not None,
    ),
    'overvoltage': RpcBinarySensorDescription(
        key='switch',
        sub_key='errors',
        name='Overvoltage',
        device_class=BinarySensorDeviceClass.PROBLEM,
        value=lambda status, _: False if status is None else 'overvoltage' in status,
        entity_category=EntityCategory.DIAGNOSTIC,
        supported=lambda status: status.get('apower') is not None,
    ),
    'overcurrent': RpcBinarySensorDescription(
        key='switch',
        sub_key='errors',
        name='Overcurrent',
        device_class=BinarySensorDeviceClass.PROBLEM,
        value=lambda status, _: False if status is None else 'overcurrent' in status,
        entity_category=EntityCategory.DIAGNOSTIC,
        supported=lambda status: status.get('apower') is not None,
    ),
    'smoke': RpcBinarySensorDescription(
        key='smoke',
        sub_key='alarm',
        name='Smoke',
        device_class=BinarySensorDeviceClass.SMOKE,
    ),
    'restart': RpcBinarySensorDescription(
        key='sys',
        sub_key='restart_required',
        name='Restart required',
        device_class=BinarySensorDeviceClass.PROBLEM,
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    'boolean': RpcBinarySensorDescription(
        key='boolean',
        sub_key='value',
        has_entity_name=True,
    ),
    'calibration': RpcBinarySensorDescription(
        key='blutrv',
        sub_key='errors',
        name='Calibration',
        device_class=BinarySensorDeviceClass.PROBLEM,
        value=lambda status, _: False if status is None else 'not_calibrated' in status,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_class=RpcBluTrvBinarySensor,
    ),
    'flood': RpcBinarySensorDescription(
        key='flood',
        sub_key='alarm',
        name='Flood',
        device_class=BinarySensorDeviceClass.MOISTURE,
    ),
    'mute': RpcBinarySensorDescription(
        key='flood',
        sub_key='mute',
        name='Mute',
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    if get_device_entry_gen(config_entry) in RPC_GENERATIONS:
        if config_entry.data[CONF_SLEEP_PERIOD]:
            async_setup_entry_rpc(
                hass, config_entry, async_add_entities, RPC_SENSORS, RpcSleepingBinarySensor
            )
        else:
            coordinator: ShellyRpcCoordinator = config_entry.runtime_data.rpc  # type: ignore
            assert coordinator is not None
            async_setup_entry_rpc(
                hass, config_entry, async_add_entities, RPC_SENSORS, RpcBinarySensor
            )
            virtual_binary_sensor_ids = get_virtual_component_ids(
                coordinator.device.config, BINARY_SENSOR_PLATFORM
            )
            async_remove_orphaned_entities(
                hass, config_entry.entry_id, coordinator.mac, BINARY_SENSOR_PLATFORM, virtual_binary_sensor_ids, 'boolean'
            )
        return

    if config_entry.data[CONF_SLEEP_PERIOD]:
        async_setup_entry_attribute_entities(
            hass, config_entry, async_add_entities, SENSORS, BlockSleepingBinarySensor
        )
    else:
        async_setup_entry_attribute_entities(
            hass, config_entry, async_add_entities, SENSORS, BlockBinarySensor
        )
        async_setup_entry_rest(
            hass, config_entry, async_add_entities, REST_SENSORS, RestBinarySensor
        )


class BlockBinarySensor(ShellyBlockAttributeEntity, BinarySensorEntity):
    @property
    def is_on(self) -> bool:
        return bool(self.attribute_value)


class RestBinarySensor(ShellyRestAttributeEntity, BinarySensorEntity):
    @property
    def is_on(self) -> bool:
        return bool(self.attribute_value)


class BlockSleepingBinarySensor(ShellySleepingBlockAttributeEntity, BinarySensorEntity, RestoreEntity):
    last_state: Optional[Any] = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.last_state = await self.async_get_last_state()

    @property
    def is_on(self) -> Optional[bool]:
        if self.block is not None:
            return bool(self.attribute_value)
        if self.last_state is None:
            return None
        return self.last_state.state == STATE_ON


class RpcSleepingBinarySensor(ShellySleepingRpcAttributeEntity, BinarySensorEntity, RestoreEntity):
    last_state: Optional[Any] = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.last_state = await self.async_get_last_state()

    @property
    def is_on(self) -> Optional[bool]:
        if self.coordinator.device.initialized:
            return bool(self.attribute_value)
        if self.last_state is None:
            return None
        return self.last_state.state == STATE_ON