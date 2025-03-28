"""Switch for Shelly."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional
from aioshelly.block_device import Block
from aioshelly.const import MODEL_2, MODEL_25, MODEL_WALL_DISPLAY, RPC_GENERATIONS
from homeassistant.components.switch import DOMAIN as SWITCH_PLATFORM, SwitchEntity, SwitchEntityDescription
from homeassistant.const import STATE_ON, EntityCategory
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.entity_registry import RegistryEntry
from homeassistant.helpers.restore_state import RestoreEntity
from .const import CONF_SLEEP_PERIOD, MOTION_MODELS
from .coordinator import ShellyBlockCoordinator, ShellyConfigEntry, ShellyRpcCoordinator
from .entity import BlockEntityDescription, RpcEntityDescription, ShellyBlockEntity, ShellyRpcAttributeEntity, ShellyRpcEntity, ShellySleepingBlockAttributeEntity, async_setup_entry_attribute_entities, async_setup_rpc_attribute_entities
from .utils import async_remove_orphaned_entities, async_remove_shelly_entity, get_device_entry_gen, get_rpc_key_ids, get_virtual_component_ids, is_block_channel_type_light, is_rpc_channel_type_light, is_rpc_thermostat_internal_actuator, is_rpc_thermostat_mode

@dataclass(frozen=True, kw_only=True)
class BlockSwitchDescription(BlockEntityDescription, SwitchEntityDescription):
    """Class to describe a BLOCK switch."""
MOTION_SWITCH: BlockSwitchDescription = BlockSwitchDescription(key='sensor|motionActive', name='Motion detection', entity_category=EntityCategory.CONFIG)

@dataclass(frozen=True, kw_only=True)
class RpcSwitchDescription(RpcEntityDescription, SwitchEntityDescription):
    """Class to describe a RPC virtual switch."""
RPC_VIRTUAL_SWITCH: RpcSwitchDescription = RpcSwitchDescription(key='boolean', sub_key='value')
RPC_SCRIPT_SWITCH: RpcSwitchDescription = RpcSwitchDescription(key='script', sub_key='running', entity_registry_enabled_default=False, entity_category=EntityCategory.CONFIG)

async def async_setup_entry(hass: HomeAssistant, config_entry: ShellyConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up switches for device."""
    device_gen: int = get_device_entry_gen(config_entry)
    if device_gen in RPC_GENERATIONS:
        await async_setup_rpc_entry(hass, config_entry, async_add_entities)
        return
    await async_setup_block_entry(hass, config_entry, async_add_entities)

@callback
def async_setup_block_entry(hass, config_entry, async_add_entities):
    """Set up entities for block device."""
    coordinator: ShellyBlockCoordinator = config_entry.runtime_data.block
    assert coordinator
    if coordinator.model in MOTION_MODELS:
        async_setup_entry_attribute_entities(hass, config_entry, async_add_entities, {('sensor', 'motionActive'): MOTION_SWITCH}, BlockSleepingMotionSwitch)
        return
    if config_entry.data.get(CONF_SLEEP_PERIOD):
        return
    if coordinator.model in {MODEL_2, MODEL_25} and coordinator.device.settings.get('mode') != 'relay':
        return
    relay_blocks: List[Block] = []
    assert coordinator.device.blocks is not None
    for block in coordinator.device.blocks:
        if block.type != 'relay':
            continue
        if block.channel is not None:
            channel: int = int(block.channel)
            if is_block_channel_type_light(coordinator.device.settings, channel):
                continue
        relay_blocks.append(block)
        unique_id: str = f'{coordinator.mac}-{block.type}_{block.channel}'
        async_remove_shelly_entity(hass, 'light', unique_id)
    if not relay_blocks:
        return
    async_add_entities((BlockRelaySwitch(coordinator, block) for block in relay_blocks))

@callback
def async_setup_rpc_entry(hass, config_entry, async_add_entities):
    """Set up entities for RPC device."""
    coordinator: ShellyRpcCoordinator = config_entry.runtime_data.rpc
    assert coordinator
    switch_key_ids: List[int] = get_rpc_key_ids(coordinator.device.status, 'switch')
    switch_ids: List[int] = []
    for id_ in switch_key_ids:
        if is_rpc_channel_type_light(coordinator.device.config, id_):
            continue
        if coordinator.model == MODEL_WALL_DISPLAY:
            if not is_rpc_thermostat_mode(id_, coordinator.device.status):
                unique_id: str = f'{coordinator.mac}-thermostat:{id_}'
                async_remove_shelly_entity(hass, 'climate', unique_id)
            elif is_rpc_thermostat_internal_actuator(coordinator.device.status):
                continue
        switch_ids.append(id_)
        unique_id: str = f'{coordinator.mac}-switch:{id_}'
        async_remove_shelly_entity(hass, 'light', unique_id)
    async_setup_rpc_attribute_entities(hass, config_entry, async_add_entities, {'boolean': RPC_VIRTUAL_SWITCH}, RpcVirtualSwitch)
    async_setup_rpc_attribute_entities(hass, config_entry, async_add_entities, {'script': RPC_SCRIPT_SWITCH}, RpcScriptSwitch)
    virtual_switch_ids: List[int] = get_virtual_component_ids(coordinator.device.config, SWITCH_PLATFORM)
    async_remove_orphaned_entities(hass, config_entry.entry_id, coordinator.mac, SWITCH_PLATFORM, virtual_switch_ids, 'boolean')
    async_remove_orphaned_entities(hass, config_entry.entry_id, coordinator.mac, SWITCH_PLATFORM, coordinator.device.status, 'script')
    if not switch_ids:
        return
    async_add_entities((RpcRelaySwitch(coordinator, id_) for id_ in switch_ids))

class BlockSleepingMotionSwitch(ShellySleepingBlockAttributeEntity, RestoreEntity, SwitchEntity):
    """Entity that controls Motion Sensor on Block based Shelly devices."""
    entity_description: BlockSwitchDescription
    _attr_translation_key: str = 'motion_switch'
    last_state: Optional[State]

    def __init__(self, coordinator, block, attribute, description, entry=None):
        """Initialize the sleeping sensor."""
        super().__init__(coordinator, block, attribute, description, entry)
        self.last_state = None

    @property
    def is_on(self):
        """If motion is active."""
        if self.block is not None:
            return bool(self.block.motionActive)
        if self.last_state is None:
            return None
        return self.last_state.state == STATE_ON

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Activate switch."""
        await self.coordinator.device.set_shelly_motion_detection(True)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Deactivate switch."""
        await self.coordinator.device.set_shelly_motion_detection(False)
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        last_state: Optional[State] = await self.async_get_last_state()
        if last_state is not None:
            self.last_state = last_state

class BlockRelaySwitch(ShellyBlockEntity, SwitchEntity):
    """Entity that controls a relay on Block based Shelly devices."""
    control_result: Optional[Dict[str, Any]]

    def __init__(self, coordinator, block):
        """Initialize relay switch."""
        super().__init__(coordinator, block)
        self.control_result = None

    @property
    def is_on(self):
        """If switch is on."""
        if self.control_result:
            return cast(bool, self.control_result.get('ison', False))
        return bool(self.block.output)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on relay."""
        self.control_result = await self.set_state(turn='on')
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off relay."""
        self.control_result = await self.set_state(turn='off')
        self.async_write_ha_state()

    @callback
    def _update_callback(self):
        """When device updates, clear control result that overrides state."""
        self.control_result = None
        super()._update_callback()

class RpcRelaySwitch(ShellyRpcEntity, SwitchEntity):
    """Entity that controls a relay on RPC based Shelly devices."""
    _id: int

    def __init__(self, coordinator, id_):
        """Initialize relay switch."""
        super().__init__(coordinator, f'switch:{id_}')
        self._id = id_

    @property
    def is_on(self):
        """If switch is on."""
        return bool(self.status.get('output', False))

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on relay."""
        await self.call_rpc('Switch.Set', {'id': self._id, 'on': True})

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off relay."""
        await self.call_rpc('Switch.Set', {'id': self._id, 'on': False})

class RpcVirtualSwitch(ShellyRpcAttributeEntity, SwitchEntity):
    """Entity that controls a virtual boolean component on RPC based Shelly devices."""
    entity_description: RpcSwitchDescription
    _attr_has_entity_name: bool = True

    @property
    def is_on(self):
        """If switch is on."""
        return bool(self.attribute_value)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on relay."""
        await self.call_rpc('Boolean.Set', {'id': self._id, 'value': True})

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off relay."""
        await self.call_rpc('Boolean.Set', {'id': self._id, 'value': False})

class RpcScriptSwitch(ShellyRpcAttributeEntity, SwitchEntity):
    """Entity that controls a script component on RPC based Shelly devices."""
    entity_description: RpcSwitchDescription
    _attr_has_entity_name: bool = True

    @property
    def is_on(self):
        """If switch is on."""
        return bool(self.status.get('running', False))

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on relay."""
        await self.call_rpc('Script.Start', {'id': self._id})

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off relay."""
        await self.call_rpc('Script.Stop', {'id': self._id})