from __future__ import annotations
from asyncio import Event, Task, wait
import dataclasses
from datetime import datetime
import logging
from typing import Any, Callable, Dict, Optional, Set
from propcache.api import cached_property
from python_otbr_api import tlv_parser
from python_otbr_api.tlv_parser import MeshcopTLVType
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.singleton import singleton
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util, ulid as ulid_util
from . import discovery

BORDER_AGENT_DISCOVERY_TIMEOUT: int = 30
DATA_STORE: str = 'thread.datasets'
STORAGE_KEY: str = 'thread.datasets'
STORAGE_VERSION_MAJOR: int = 1
STORAGE_VERSION_MINOR: int = 4
SAVE_DELAY: int = 10
_LOGGER = logging.getLogger(__name__)


class DatasetPreferredError(HomeAssistantError):
    """Raised when attempting to delete the preferred dataset."""


@dataclasses.dataclass(frozen=True)
class DatasetEntry:
    created: datetime = dataclasses.field(default_factory=dt_util.utcnow)
    id: str = dataclasses.field(default_factory=ulid_util.ulid_now)
    preferred_border_agent_id: Optional[str] = None
    preferred_extended_address: Optional[str] = None
    source: Any = None
    tlv: bytes = b""

    @property
    def channel(self) -> Optional[int]:
        """Return channel as an integer."""
        dataset: Dict[Any, Any] = self.dataset
        if (channel := dataset.get(MeshcopTLVType.CHANNEL)) is None:
            return None
        return cast(tlv_parser.Channel, channel).channel

    @cached_property
    def dataset(self) -> Dict[Any, Any]:
        """Return the dataset in dict format."""
        return tlv_parser.parse_tlv(self.tlv)

    @property
    def extended_pan_id(self) -> str:
        """Return extended PAN ID as a hex string."""
        return str(self.dataset[MeshcopTLVType.EXTPANID])

    @property
    def network_name(self) -> Optional[str]:
        """Return network name as a string."""
        if (name := self.dataset.get(MeshcopTLVType.NETWORKNAME)) is None:
            return None
        return cast(tlv_parser.NetworkName, name).name

    @property
    def pan_id(self) -> str:
        """Return PAN ID as a hex string."""
        return str(self.dataset.get(MeshcopTLVType.PANID))

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON serializable representation for storage."""
        return {
            'created': self.created.isoformat(),
            'id': self.id,
            'preferred_border_agent_id': self.preferred_border_agent_id,
            'preferred_extended_address': self.preferred_extended_address,
            'source': self.source,
            'tlv': self.tlv,
        }


class DatasetStoreStore(Store):
    """Store Thread datasets."""

    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Migrate to the new version."""
        if old_major_version == 1:
            data: Dict[str, Any] = old_data
            if old_minor_version < 2:
                datasets: Dict[str, DatasetEntry] = {}
                preferred_dataset: Optional[str] = old_data['preferred_dataset']
                for dataset in old_data['datasets']:
                    created: datetime = cast(datetime, dt_util.parse_datetime(dataset['created']))
                    entry = DatasetEntry(
                        created=created,
                        id=dataset['id'],
                        preferred_border_agent_id=None,
                        preferred_extended_address=None,
                        source=dataset['source'],
                        tlv=dataset['tlv'],
                    )
                    if MeshcopTLVType.EXTPANID not in entry.dataset or MeshcopTLVType.ACTIVETIMESTAMP not in entry.dataset:
                        _LOGGER.warning("Dropped invalid Thread dataset '%s'", entry.tlv)
                        if entry.id == preferred_dataset:
                            preferred_dataset = None
                        continue
                    if entry.extended_pan_id in datasets:
                        if datasets[entry.extended_pan_id].id == preferred_dataset:
                            _LOGGER.warning(
                                "Dropped duplicated Thread dataset '%s' (duplicate of preferred dataset '%s')",
                                entry.tlv, datasets[entry.extended_pan_id].tlv
                            )
                            continue
                        new_timestamp = cast(tlv_parser.Timestamp, entry.dataset[MeshcopTLVType.ACTIVETIMESTAMP])
                        old_timestamp = cast(
                            tlv_parser.Timestamp,
                            datasets[entry.extended_pan_id].dataset[MeshcopTLVType.ACTIVETIMESTAMP]
                        )
                        if old_timestamp.seconds >= new_timestamp.seconds or (old_timestamp.seconds == new_timestamp.seconds and old_timestamp.ticks >= new_timestamp.ticks):
                            _LOGGER.warning(
                                "Dropped duplicated Thread dataset '%s' (duplicate of '%s')",
                                entry.tlv, datasets[entry.extended_pan_id].tlv
                            )
                            continue
                        _LOGGER.warning(
                            "Dropped duplicated Thread dataset '%s' (duplicate of '%s')",
                            datasets[entry.extended_pan_id].tlv, entry.tlv
                        )
                    datasets[entry.extended_pan_id] = entry
                data = {
                    'preferred_dataset': preferred_dataset,
                    'datasets': [dataset.to_json() for dataset in datasets.values()],
                }
            if old_minor_version < 4:
                for dataset in data['datasets']:
                    dataset['preferred_border_agent_id'] = None
                    dataset['preferred_extended_address'] = None
        return data


class DatasetStore:
    """Class to hold a collection of thread datasets."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the dataset store."""
        self.hass: HomeAssistant = hass
        self.datasets: Dict[str, DatasetEntry] = {}
        self._preferred_dataset: Optional[str] = None
        self._set_preferred_dataset_task: Optional[Task] = None
        self._store: DatasetStoreStore = DatasetStoreStore(
            hass, STORAGE_VERSION_MAJOR, STORAGE_KEY, atomic_writes=True, minor_version=STORAGE_VERSION_MINOR
        )

    @callback
    def async_add(
        self,
        source: Any,
        tlv: bytes,
        preferred_border_agent_id: Optional[str],
        preferred_extended_address: Optional[str],
    ) -> None:
        """Add dataset, does nothing if it already exists."""
        dataset: Dict[Any, Any] = tlv_parser.parse_tlv(tlv)
        if MeshcopTLVType.EXTPANID not in dataset or MeshcopTLVType.ACTIVETIMESTAMP not in dataset:
            raise HomeAssistantError('Invalid dataset')
        if preferred_border_agent_id is not None and preferred_extended_address is None:
            raise HomeAssistantError('Must set preferred extended address with preferred border agent ID')
        for entry in self.datasets.values():
            if entry.dataset == dataset:
                if preferred_extended_address and entry.preferred_extended_address is None:
                    self.async_set_preferred_border_agent(entry.id, preferred_border_agent_id, preferred_extended_address)
                return
        if (entry := next((entry for entry in self.datasets.values() if entry.dataset[MeshcopTLVType.EXTPANID] == dataset[MeshcopTLVType.EXTPANID]), None)):
            new_timestamp = cast(tlv_parser.Timestamp, dataset[MeshcopTLVType.ACTIVETIMESTAMP])
            old_timestamp = cast(tlv_parser.Timestamp, entry.dataset[MeshcopTLVType.ACTIVETIMESTAMP])
            if old_timestamp.seconds >= new_timestamp.seconds or (old_timestamp.seconds == new_timestamp.seconds and old_timestamp.ticks >= new_timestamp.ticks):
                _LOGGER.warning(
                    "Got dataset with same extended PAN ID and same or older active timestamp, old dataset: '%s', new dataset: '%s'",
                    entry.tlv, tlv
                )
                return
            _LOGGER.debug(
                "Updating dataset with same extended PAN ID and newer active timestamp, old dataset: '%s', new dataset: '%s'",
                entry.tlv, tlv
            )
            self.datasets[entry.id] = dataclasses.replace(self.datasets[entry.id], tlv=tlv)
            self.async_schedule_save()
            if preferred_extended_address and entry.preferred_extended_address is None:
                self.async_set_preferred_border_agent(entry.id, preferred_border_agent_id, preferred_extended_address)
            return
        entry = DatasetEntry(
            preferred_border_agent_id=preferred_border_agent_id,
            preferred_extended_address=preferred_extended_address,
            source=source,
            tlv=tlv,
        )
        self.datasets[entry.id] = entry
        self.async_schedule_save()
        if self._preferred_dataset is None and preferred_extended_address and (not self._set_preferred_dataset_task):
            self._set_preferred_dataset_task = self.hass.async_create_task(
                self._set_preferred_dataset_if_only_network(entry.id, preferred_extended_address)
            )

    @callback
    def async_delete(self, dataset_id: str) -> None:
        """Delete dataset."""
        if self._preferred_dataset == dataset_id:
            raise DatasetPreferredError('attempt to remove preferred dataset')
        del self.datasets[dataset_id]
        self.async_schedule_save()

    @callback
    def async_get(self, dataset_id: str) -> Optional[DatasetEntry]:
        """Get dataset by id."""
        return self.datasets.get(dataset_id)

    @callback
    def async_set_preferred_border_agent(
        self, dataset_id: str, border_agent_id: Optional[str], extended_address: Optional[str]
    ) -> None:
        """Set preferred border agent id and extended address of a dataset."""
        if border_agent_id is not None and extended_address is None:
            raise HomeAssistantError('Must set preferred extended address with preferred border agent ID')
        self.datasets[dataset_id] = dataclasses.replace(
            self.datasets[dataset_id],
            preferred_border_agent_id=border_agent_id,
            preferred_extended_address=extended_address,
        )
        self.async_schedule_save()

    @property
    @callback
    def preferred_dataset(self) -> Optional[str]:
        """Get the id of the preferred dataset."""
        return self._preferred_dataset

    @preferred_dataset.setter
    @callback
    def preferred_dataset(self, dataset_id: str) -> None:
        """Set the preferred dataset."""
        if dataset_id not in self.datasets:
            raise KeyError('unknown dataset')
        self._preferred_dataset = dataset_id
        self.async_schedule_save()

    async def _set_preferred_dataset_if_only_network(self, dataset_id: str, extended_address: str) -> None:
        """Set the preferred dataset, unless there are other routers present."""
        _LOGGER.debug('_set_preferred_dataset_if_only_network called for router %s', extended_address)
        own_router_evt: Event = Event()
        other_router_evt: Event = Event()

        @callback
        def router_discovered(key: Any, data: Any) -> None:
            """Handle router discovered."""
            _LOGGER.debug('discovered router with ext addr %s', data.extended_address)
            if data.extended_address == extended_address:
                own_router_evt.set()
                return
            other_router_evt.set()

        thread_discovery = discovery.ThreadRouterDiscovery(self.hass, router_discovered, lambda key: None)
        await thread_discovery.async_start()
        found_own_router: Task = self.hass.async_create_task(own_router_evt.wait())
        found_other_router: Task = self.hass.async_create_task(other_router_evt.wait())
        pending: Set[Task] = {found_own_router, found_other_router}
        done, pending = await wait(pending, timeout=BORDER_AGENT_DISCOVERY_TIMEOUT)
        if found_other_router in done:
            _LOGGER.debug('Other router found, do not set dataset as default')
        elif found_own_router in pending:
            _LOGGER.debug('Own router not found, do not set dataset as default')
        else:
            _LOGGER.debug('No other router found, set dataset as default')
            self.preferred_dataset = dataset_id
        for task in pending:
            task.cancel()
        await thread_discovery.async_stop()

    async def async_load(self) -> None:
        """Load the datasets."""
        data: Optional[Dict[str, Any]] = await self._store.async_load()
        datasets: Dict[str, DatasetEntry] = {}
        preferred_dataset: Optional[str] = None
        if data is not None:
            for dataset in data['datasets']:
                created: datetime = cast(datetime, dt_util.parse_datetime(dataset['created']))
                datasets[dataset['id']] = DatasetEntry(
                    created=created,
                    id=dataset['id'],
                    preferred_border_agent_id=dataset['preferred_border_agent_id'],
                    preferred_extended_address=dataset['preferred_extended_address'],
                    source=dataset['source'],
                    tlv=dataset['tlv'],
                )
            preferred_dataset = data['preferred_dataset']
        self.datasets = datasets
        self._preferred_dataset = preferred_dataset

    @callback
    def async_schedule_save(self) -> None:
        """Schedule saving the dataset store."""
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY)

    @callback
    def _data_to_save(self) -> Dict[str, Any]:
        """Return data of datasets to store in a file."""
        data: Dict[str, Any] = {}
        data['datasets'] = [dataset.to_json() for dataset in self.datasets.values()]
        data['preferred_dataset'] = self._preferred_dataset
        return data


@singleton(DATA_STORE)
async def async_get_store(hass: HomeAssistant) -> DatasetStore:
    """Get the dataset store."""
    store: DatasetStore = DatasetStore(hass)
    await store.async_load()
    return store


async def async_add_dataset(
    hass: HomeAssistant,
    source: Any,
    tlv: bytes,
    *,
    preferred_border_agent_id: Optional[str] = None,
    preferred_extended_address: Optional[str] = None,
) -> None:
    """Add a dataset."""
    store: DatasetStore = await async_get_store(hass)
    store.async_add(source, tlv, preferred_border_agent_id, preferred_extended_address)


async def async_get_dataset(hass: HomeAssistant, dataset_id: str) -> Optional[bytes]:
    """Get a dataset."""
    store: DatasetStore = await async_get_store(hass)
    if (entry := store.async_get(dataset_id)) is None:
        return None
    return entry.tlv


async def async_get_preferred_dataset(hass: HomeAssistant) -> Optional[bytes]:
    """Get the preferred dataset."""
    store: DatasetStore = await async_get_store(hass)
    if (preferred_dataset := store.preferred_dataset) is None or (entry := store.async_get(preferred_dataset)) is None:
        return None
    return entry.tlv
