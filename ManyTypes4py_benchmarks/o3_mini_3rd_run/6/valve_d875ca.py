"""Valve for Shelly."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, cast

from aioshelly.block_device import Block
from aioshelly.const import BLOCK_GENERATIONS, MODEL_GAS
from homeassistant.components.valve import (
    ValveDeviceClass,
    ValveEntity,
    ValveEntityDescription,
    ValveEntityFeature,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import ShellyBlockCoordinator, ShellyConfigEntry
from .entity import BlockEntityDescription, ShellyBlockAttributeEntity, async_setup_block_attribute_entities
from .utils import async_remove_shelly_entity, get_device_entry_gen

@dataclass(kw_only=True, frozen=True)
class BlockValveDescription(BlockEntityDescription, ValveEntityDescription):
    """Class to describe a BLOCK valve."""
    # The available callable receives a Block and returns a bool.
    available: Callable[[Block], bool] = lambda block: block.valve not in ('failure', 'checking')
    removal_condition: Callable[[Any, Block], bool] = lambda _, block: block.valve in ('not_connected', 'unknown')

GAS_VALVE: BlockValveDescription = BlockValveDescription(
    key='valve|valve',
    name='Valve',
)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up valves for device."""
    if get_device_entry_gen(config_entry) in BLOCK_GENERATIONS:
        async_setup_block_entry(hass, config_entry, async_add_entities)

@callback
def async_setup_block_entry(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up valve for device."""
    coordinator: Optional[ShellyBlockCoordinator] = config_entry.runtime_data.block
    assert coordinator is not None and coordinator.device.blocks
    if coordinator.model == MODEL_GAS:
        async_setup_block_attribute_entities(
            hass, async_add_entities, coordinator, {('valve', 'valve'): GAS_VALVE}, BlockShellyValve
        )
        unique_id: str = f'{coordinator.mac}-valve_0-valve'
        async_remove_shelly_entity(hass, 'switch', unique_id)

class BlockShellyValve(ShellyBlockAttributeEntity, ValveEntity):
    """Entity that controls a valve on block based Shelly devices."""
    _attr_device_class = ValveDeviceClass.GAS
    _attr_supported_features = ValveEntityFeature.OPEN | ValveEntityFeature.CLOSE

    def __init__(
        self,
        coordinator: ShellyBlockCoordinator,
        block: Block,
        attribute: Tuple[str, str],
        description: BlockValveDescription
    ) -> None:
        """Initialize block valve."""
        super().__init__(coordinator, block, attribute, description)
        self.control_result: Optional[Dict[str, Any]] = None
        self._attr_is_closed: bool = bool(self.attribute_value == 'closed')

    @property
    def is_closing(self) -> bool:
        """Return if the valve is closing."""
        if self.control_result:
            return cast(bool, self.control_result['state'] == 'closing')
        return self.attribute_value == 'closing'

    @property
    def is_opening(self) -> bool:
        """Return if the valve is opening."""
        if self.control_result:
            return cast(bool, self.control_result['state'] == 'opening')
        return self.attribute_value == 'opening'

    async def async_open_valve(self, **kwargs: Any) -> None:
        """Open valve."""
        self.control_result = await self.set_state(go='open')
        self.async_write_ha_state()

    async def async_close_valve(self, **kwargs: Any) -> None:
        """Close valve."""
        self.control_result = await self.set_state(go='close')
        self.async_write_ha_state()

    @callback
    def _update_callback(self) -> None:
        """When device updates, clear control result that overrides state."""
        self.control_result = None
        self._attr_is_closed = bool(self.attribute_value == 'closed')
        super()._update_callback()