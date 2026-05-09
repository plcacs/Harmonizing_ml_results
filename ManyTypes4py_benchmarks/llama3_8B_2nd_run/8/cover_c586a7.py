from typing import Any

async def async_setup_entry(
    hass: HomeAssistant, 
    config_entry: ConfigEntry, 
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    ...

class XiaomiGenericCover(XiaomiDevice, CoverEntity):
    """Representation of a XiaomiGenericCover."""

    def __init__(self, 
        device: Any, 
        name: str, 
        data_key: str, 
        xiaomi_hub: Any, 
        config_entry: ConfigEntry
    ) -> None:
        ...

    @property
    def current_cover_position(self) -> int:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    def close_cover(self, **kwargs: Any) -> None:
        ...

    def open_cover(self, **kwargs: Any) -> None:
        ...

    def stop_cover(self, **kwargs: Any) -> None:
        ...

    def set_cover_position(self, **kwargs: Any) -> None:
        ...

    def parse_data(self, 
        data: Any, 
        raw_data: Any
    ) -> bool:
        ...
