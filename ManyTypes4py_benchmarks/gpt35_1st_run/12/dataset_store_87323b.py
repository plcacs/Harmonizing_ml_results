from asyncio import Event
from datetime import datetime
from typing import Any, Dict, Optional
from propcache.api import cached_property
from python_otbr_api import tlv_parser
from python_otbr_api.tlv_parser import MeshcopTLVType
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.singleton import singleton
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util, ulid as ulid_util

BORDER_AGENT_DISCOVERY_TIMEOUT: int = 30
DATA_STORE: str = 'thread.datasets'
STORAGE_KEY: str = 'thread.datasets'
STORAGE_VERSION_MAJOR: int = 1
STORAGE_VERSION_MINOR: int = 4
SAVE_DELAY: int = 10

class DatasetPreferredError(HomeAssistantError):
    """Raised when attempting to delete the preferred dataset."""

class DatasetEntry:
    created: datetime
    id: str

    def __init__(self) -> None:
        self.created = dt_util.utcnow()
        self.id = ulid_util.ulid_now()

    @property
    def channel(self) -> Optional[int]: ...

    @cached_property
    def dataset(self) -> Dict[MeshcopTLVType, Any]: ...

    @property
    def extended_pan_id(self) -> str: ...

    @property
    def network_name(self) -> Optional[str]: ...

    @property
    def pan_id(self) -> str: ...

    def to_json(self) -> Dict[str, Any]: ...

class DatasetStoreStore(Store):
    async def _async_migrate_func(self, old_major_version: int, old_minor_version: int, old_data: Dict[str, Any]) -> Dict[str, Any]: ...

class DatasetStore:
    def __init__(self, hass: HomeAssistant) -> None: ...

    @callback
    def async_add(self, source: str, tlv: str, preferred_border_agent_id: Optional[str], preferred_extended_address: Optional[str]) -> None: ...

    @callback
    def async_delete(self, dataset_id: str) -> None: ...

    @callback
    def async_get(self, dataset_id: str) -> Optional[DatasetEntry]: ...

    @callback
    def async_set_preferred_border_agent(self, dataset_id: str, border_agent_id: Optional[str], extended_address: Optional[str]) -> None: ...

    @property
    @callback
    def preferred_dataset(self) -> Optional[str]: ...

    @preferred_dataset.setter
    @callback
    def preferred_dataset(self, dataset_id: str) -> None: ...

    async def _set_preferred_dataset_if_only_network(self, dataset_id: str, extended_address: str) -> None: ...

    async def async_load(self) -> None: ...

    @callback
    def async_schedule_save(self) -> None: ...

    @callback
    def _data_to_save(self) -> Dict[str, Any]: ...

@singleton(DATA_STORE)
async def async_get_store(hass: HomeAssistant) -> DatasetStore: ...

async def async_add_dataset(hass: HomeAssistant, source: str, tlv: str, *, preferred_border_agent_id: Optional[str] = None, preferred_extended_address: Optional[str] = None) -> None: ...

async def async_get_dataset(hass: HomeAssistant, dataset_id: str) -> Optional[str]: ...

async def async_get_preferred_dataset(hass: HomeAssistant) -> Optional[str]: ...
