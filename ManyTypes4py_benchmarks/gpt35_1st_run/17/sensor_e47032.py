from typing import Any, Dict, List, Optional

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

    def __init__(self, syncthing: Any, server_id: str, folder_id: str, folder_label: str, version: str) -> None:

    @property
    def native_value(self) -> Optional[str]:

    @property
    def available(self) -> bool:

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:

    async def async_update_status(self) -> None:

    def subscribe(self) -> None:

    def unsubscribe(self) -> None:

    async def async_added_to_hass(self) -> None:

    def _filter_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
