from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable

def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: Callable[[List[SwitchEntity]], None]) -> None:
    ...

class HomeKitSwitch(HomeKitEntity, SwitchEntity):
    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def is_on(self) -> bool:
        ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        ...

class HomeKitFaucet(HomeKitEntity, SwitchEntity):
    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def is_on(self) -> bool:
        ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...

class HomeKitValve(HomeKitEntity, SwitchEntity):
    def get_characteristic_types(self) -> List[str]:
        ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

class DeclarativeCharacteristicSwitch(CharacteristicEntity, SwitchEntity):
    def __init__(self, conn: HKDevice, info: Dict[str, int], char: Characteristic, description: DeclarativeSwitchEntityDescription) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def is_on(self) -> bool:
        ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...
