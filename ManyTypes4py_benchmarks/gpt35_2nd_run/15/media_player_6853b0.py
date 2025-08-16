from typing import Any, Dict, List, Tuple

def _get_sources_from_dict(data: Dict[str, Any]) -> List[Any]:
    ...

def _get_sources(config_entry: ConfigEntry) -> List[Any]:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class MonopriceZone(MediaPlayerEntity):
    _attr_device_class: str = MediaPlayerDeviceClass.RECEIVER
    _attr_supported_features: int = MediaPlayerEntityFeature.VOLUME_MUTE | MediaPlayerEntityFeature.VOLUME_SET | MediaPlayerEntityFeature.VOLUME_STEP | MediaPlayerEntityFeature.TURN_ON | MediaPlayerEntityFeature.TURN_OFF | MediaPlayerEntityFeature.SELECT_SOURCE
    _attr_has_entity_name: bool = True
    _attr_name: str = None

    def __init__(self, monoprice: Any, sources: List[Any], namespace: str, zone_id: int) -> None:
        ...

    def update(self) -> None:
        ...

    @property
    def entity_registry_enabled_default(self) -> bool:
        ...

    @property
    def media_title(self) -> str:
        ...

    def snapshot(self) -> None:
        ...

    def restore(self) -> None:
        ...

    def select_source(self, source: str) -> None:
        ...

    def turn_on(self) -> None:
        ...

    def turn_off(self) -> None:
        ...

    def mute_volume(self, mute: bool) -> None:
        ...

    def set_volume_level(self, volume: float) -> None:
        ...

    def volume_up(self) -> None:
        ...

    def volume_down(self) -> None:
        ...
