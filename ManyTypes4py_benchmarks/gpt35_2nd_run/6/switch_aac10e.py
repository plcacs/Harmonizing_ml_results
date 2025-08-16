from __future__ import annotations
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class HomematicipMultiSwitch(HomematicipGenericEntity, SwitchEntity):

    def __init__(self, hap: HomematicipHAP, device: Any, channel: int = 1, is_multi_channel: bool = True) -> None:

    @property
    def is_on(self) -> bool:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:

class HomematicipSwitch(HomematicipMultiSwitch, SwitchEntity):

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:

class HomematicipGroupSwitch(HomematicipGenericEntity, SwitchEntity):

    def __init__(self, hap: HomematicipHAP, device: Any, post: str = 'Group') -> None:

    @property
    def is_on(self) -> bool:

    @property
    def available(self) -> bool:

    @property
    def extra_state_attributes(self) -> dict:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:

class HomematicipSwitchMeasuring(HomematicipSwitch):

