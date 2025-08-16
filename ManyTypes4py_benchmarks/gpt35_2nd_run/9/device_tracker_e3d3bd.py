from typing import Any, Dict, List, Optional

async def async_setup_entry(hass: HomeAssistant, config_entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class OmadaClientScannerEntity(CoordinatorEntity[OmadaClientsCoordinator], ScannerEntity):
    _client_details: Optional[Any] = None

    def __init__(self, site_id: str, client_id: str, display_name: str, coordinator: OmadaClientsCoordinator) -> None:
        ...

    def _do_update(self) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    @property
    def ip_address(self) -> Optional[str]:
        ...

    @property
    def mac_address(self) -> str:
        ...

    @property
    def hostname(self) -> Optional[str]:
        ...

    @property
    def is_connected(self) -> bool:
        ...

    @property
    def unique_id(self) -> str:
        ...
