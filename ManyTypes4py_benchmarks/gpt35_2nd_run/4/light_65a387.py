from __future__ import annotations
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class HomematicipLight(HomematicipGenericEntity, LightEntity):

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:

    @property
    def is_on(self) -> bool:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:

class HomematicipLightMeasuring(HomematicipLight):

class HomematicipMultiDimmer(HomematicipGenericEntity, LightEntity):

    def __init__(self, hap: HomematicipHAP, device: Any, channel: int = 1, is_multi_channel: bool = True) -> None:

    @property
    def is_on(self) -> bool:

    @property
    def brightness(self) -> int:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:

class HomematicipDimmer(HomematicipMultiDimmer, LightEntity):

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:

class HomematicipNotificationLight(HomematicipGenericEntity, LightEntity):

    def __init__(self, hap: HomematicipHAP, device: Any, channel: int, post: str) -> None:

    @property
    def is_on(self) -> bool:

    @property
    def brightness(self) -> int:

    @property
    def hs_color(self) -> Tuple[float, float]:

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:

    @property
    def unique_id(self) -> str:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:

class HomematicipNotificationLightV2(HomematicipNotificationLight, LightEntity):

    def __init__(self, hap: HomematicipHAP, device: Any, channel: int, post: str) -> None:

    @property
    def effect_list(self) -> List[OpticalSignalBehaviour]:

    @property
    def effect(self) -> OpticalSignalBehaviour:

    @property
    def is_on(self) -> bool:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:

def _convert_color(color: Tuple[float, float]) -> RGBColorState:
