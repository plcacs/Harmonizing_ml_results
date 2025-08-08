from __future__ import annotations
from typing import Any

from homematicip.aio.device import (
    AsyncBrandSwitch2, AsyncBrandSwitchMeasuring, AsyncDinRailSwitch, AsyncDinRailSwitch4,
    AsyncFullFlushInputSwitch, AsyncFullFlushSwitchMeasuring, AsyncHeatingSwitch2, AsyncMultiIOBox,
    AsyncOpenCollector8Module, AsyncPlugableSwitch, AsyncPlugableSwitchMeasuring,
    AsyncPrintedCircuitBoardSwitch2, AsyncPrintedCircuitBoardSwitchBattery, AsyncWiredSwitch8
)
from homematicip.aio.group import AsyncExtendedLinkedSwitchingGroup, AsyncSwitchingGroup
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN
from .entity import ATTR_GROUP_MEMBER_UNREACHABLE, HomematicipGenericEntity
from .hap import HomematicipHAP

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    hap: HomematicipHAP = hass.data[DOMAIN][config_entry.unique_id]
    entities: list[HomematicipGroupSwitch] = [HomematicipGroupSwitch(hap, group) for group in hap.home.groups if isinstance(group, (AsyncExtendedLinkedSwitchingGroup, AsyncSwitchingGroup))]
    for device in hap.home.devices:
        if isinstance(device, AsyncBrandSwitchMeasuring):
            pass
        elif isinstance(device, (AsyncPlugableSwitchMeasuring, AsyncFullFlushSwitchMeasuring)):
            entities.append(HomematicipSwitchMeasuring(hap, device))
        elif isinstance(device, AsyncWiredSwitch8):
            entities.extend((HomematicipMultiSwitch(hap, device, channel=channel) for channel in range(1, 9)))
        elif isinstance(device, AsyncDinRailSwitch):
            entities.append(HomematicipMultiSwitch(hap, device, channel=1))
        elif isinstance(device, AsyncDinRailSwitch4):
            entities.extend((HomematicipMultiSwitch(hap, device, channel=channel) for channel in range(1, 5)))
        elif isinstance(device, (AsyncPlugableSwitch, AsyncPrintedCircuitBoardSwitchBattery, AsyncFullFlushInputSwitch)):
            entities.append(HomematicipSwitch(hap, device))
        elif isinstance(device, AsyncOpenCollector8Module):
            entities.extend((HomematicipMultiSwitch(hap, device, channel=channel) for channel in range(1, 9)))
        elif isinstance(device, (AsyncBrandSwitch2, AsyncPrintedCircuitBoardSwitch2, AsyncHeatingSwitch2, AsyncMultiIOBox)):
            entities.extend((HomematicipMultiSwitch(hap, device, channel=channel) for channel in range(1, 3))
    async_add_entities(entities)

class HomematicipMultiSwitch(HomematicipGenericEntity, SwitchEntity):
    """Representation of the HomematicIP multi switch."""

    def __init__(self, hap: HomematicipHAP, device: Any, channel: int = 1, is_multi_channel: bool = True) -> None:
        super().__init__(hap, device, channel=channel, is_multi_channel=is_multi_channel)

    @property
    def is_on(self) -> bool:
        return self._device.functionalChannels[self._channel].on

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self._device.turn_on(self._channel)

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._device.turn_off(self._channel)

class HomematicipSwitch(HomematicipMultiSwitch, SwitchEntity):
    """Representation of the HomematicIP switch."""

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        super().__init__(hap, device, is_multi_channel=False)

class HomematicipGroupSwitch(HomematicipGenericEntity, SwitchEntity):
    """Representation of the HomematicIP switching group."""

    def __init__(self, hap: HomematicipHAP, device: Any, post: str = 'Group') -> None:
        device.modelType = f'HmIP-{post}'
        super().__init__(hap, device, post)

    @property
    def is_on(self) -> bool:
        return self._device.on

    @property
    def available(self) -> bool:
        return True

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        state_attr: dict[str, Any] = super().extra_state_attributes
        if self._device.unreach:
            state_attr[ATTR_GROUP_MEMBER_UNREACHABLE] = True
        return state_attr

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self._device.turn_on()

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._device.turn_off()

class HomematicipSwitchMeasuring(HomematicipSwitch):
    """Representation of the HomematicIP measuring switch."""
