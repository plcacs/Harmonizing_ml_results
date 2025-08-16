from typing import Any, Dict, Optional, Tuple

class InsteonEntity(Entity):
    _attr_should_poll: bool = False
    _insteon_device_group: Any
    _insteon_device: Any

    def __init__(self, device: Any, group: int) -> None:
        ...

    def __hash__(self) -> int:
        ...

    @property
    def address(self) -> str:
        ...

    @property
    def group(self) -> int:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    @property
    def device_info(self) -> Dict[str, Any]:
        ...

    @callback
    def async_entity_update(self, name: str, address: str, value: Any, group: int) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...

    async def _async_read_aldb(self, reload: bool) -> None:
        ...

    def _print_aldb(self) -> None:
        ...

    def get_device_property(self, name: str) -> Optional[Any]:
        ...

    def _get_label(self) -> str:
        ...

    async def _async_add_default_links(self) -> None:
        ...
