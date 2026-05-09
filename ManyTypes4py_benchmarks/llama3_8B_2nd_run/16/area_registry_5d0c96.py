from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from homeassistant.const import ATTR_DEVICE_CLASS
from homeassistant.core import HomeAssistant, callback
from homeassistant.util.dt import utc_from_timestamp, utcnow
from homeassistant.util.event_type import EventType
from homeassistant.util.hass_dict import HassKey
from . import device_registry as dr, entity_registry as er
from .json import json_bytes, json_fragment
from .normalized_name_base_registry import NormalizedNameBaseRegistryEntry, NormalizedNameBaseRegistryItems
from .registry import BaseRegistry, RegistryIndexType
from .singleton import singleton
from .storage import Store
from .typing import UNDEFINED, UndefinedType
if TYPE_CHECKING:
    from propcache.api import cached_property as under_cached_property
else:
    from propcache.api import under_cached_property

DATA_REGISTRY: HassKey = 'area_registry'
EVENT_AREA_REGISTRY_UPDATED: EventType = EventType('area_registry_updated')
STORAGE_KEY: str = 'core.area_registry'
STORAGE_VERSION_MAJOR: int = 1
STORAGE_VERSION_MINOR: int = 8

class _AreaStoreData(TypedDict):
    """Data type for individual area. Used in AreasRegistryStoreData."""
    pass

class AreasRegistryStoreData(TypedDict):
    """Store data type for AreaRegistry."""
    areas: list[AreaEntry]

class EventAreaRegistryUpdatedData(TypedDict):
    """EventAreaRegistryUpdated data."""
    action: Literal['create', 'remove', 'update']
    area_id: str

@dataclass(frozen=True, kw_only=True, slots=True)
class AreaEntry(NormalizedNameBaseRegistryEntry):
    """Area Registry Entry."""
    labels: set[str]
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, init=False)

    @under_cached_property
    def json_fragment(self) -> str:
        """Return a JSON representation of this AreaEntry."""
        return json_fragment(json_bytes({'aliases': list(self.aliases), 'area_id': self.id, 'floor_id': self.floor_id, 'humidity_entity_id': self.humidity_entity_id, 'icon': self.icon, 'labels': list(self.labels), 'name': self.name, 'picture': self.picture, 'temperature_entity_id': self.temperature_entity_id, 'created_at': self.created_at.timestamp(), 'modified_at': self.modified_at.timestamp()}))

class AreaRegistryStore(Store[AreasRegistryStoreData]):
    """Store area registry data."""
    async def _async_migrate_func(self, old_major_version: int, old_minor_version: int, old_data: AreasRegistryStoreData) -> AreasRegistryStoreData:
        """Migrate to the new version."""
        # ...

class AreaRegistryItems(NormalizedNameBaseRegistryItems[AreaEntry]):
    """Class to hold area registry items."""
    def __init__(self) -> None:
        """Initialize the area registry items."""
        super().__init__()
        self._labels_index: defaultdict[set[str], dict[str, bool]] = defaultdict(dict)
        self._floors_index: defaultdict[set[str], dict[str, bool]] = defaultdict(dict)

    def _index_entry(self, key: str, entry: AreaEntry) -> None:
        """Index an entry."""
        super()._index_entry(key, entry)
        if entry.floor_id is not None:
            self._floors_index[entry.floor_id][key] = True
        for label in entry.labels:
            self._labels_index[label][key] = True

    def _unindex_entry(self, key: str, replacement_entry: AreaEntry | None = None) -> None:
        super()._unindex_entry(key, replacement_entry)
        entry = self.data[key]
        if labels := entry.labels:
            for label in labels:
                self._unindex_entry_value(key, label, self._labels_index)
        if floor_id := entry.floor_id:
            self._unindex_entry_value(key, floor_id, self._floors_index)

    def get_areas_for_label(self, label: str) -> list[AreaEntry]:
        """Get areas for label."""
        data = self.data
        return [data[key] for key in self._labels_index.get(label, [])]

    def get_areas_for_floor(self, floor: str) -> list[AreaEntry]:
        """Get areas for floor."""
        data = self.data
        return [data[key] for key in self._floors_index.get(floor, [])]

class AreaRegistry(BaseRegistry[AreasRegistryStoreData]):
    """Class to hold a registry of areas."""
    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the area registry."""
        self.hass: HomeAssistant = hass
        self._store: AreaRegistryStore = AreaRegistryStore(hass, STORAGE_VERSION_MAJOR, STORAGE_KEY, atomic_writes=True, minor_version=STORAGE_VERSION_MINOR)

    @callback
    def async_get_area(self, area_id: str) -> AreaEntry | None:
        """Get area by id."""
        return self._area_data.get(area_id)

    @callback
    def async_get_area_by_name(self, name: str) -> AreaEntry | None:
        """Get area by name."""
        return self.areas.get_by_name(name)

    @callback
    def async_list_areas(self) -> list[AreaEntry]:
        """Get all areas."""
        return self.areas.values()

    @callback
    def async_get_or_create(self, name: str) -> AreaEntry:
        """Get or create an area."""
        if (area := self.async_get_area_by_name(name)):
            return area
        return self.async_create(name)

    def _generate_id(self, name: str) -> str:
        """Generate area ID."""
        return self.areas.generate_id_from_name(name)

    @callback
    def async_create(self, name: str, *, aliases: set[str] | None = None, floor_id: str | None = None, humidity_entity_id: str | None = None, icon: str | None = None, labels: set[str] | None = None, picture: str | None = None, temperature_entity_id: str | None = None) -> AreaEntry:
        """Create a new area."""
        # ...

    @callback
    def async_delete(self, area_id: str) -> None:
        """Delete area."""
        # ...

    @callback
    def async_update(self, area_id: str, *, aliases: set[str] | None = None, floor_id: str | None = None, humidity_entity_id: str | None = None, icon: str | None = None, labels: set[str] | None = None, name: str | None = None, picture: str | None = None, temperature_entity_id: str | None = None) -> AreaEntry:
        """Update name of area."""
        # ...

    async def async_load(self) -> None:
        """Load the area registry."""
        # ...

    @callback
    def _data_to_save(self) -> AreasRegistryStoreData:
        """Return data of area registry to store in a file."""
        # ...

    @callback
    def _async_setup_cleanup(self) -> None:
        """Set up the area registry cleanup."""
        # ...

@callback
@singleton(DATA_REGISTRY)
def async_get(hass: HomeAssistant) -> AreaRegistry:
    """Get area registry."""
    return AreaRegistry(hass)

async def async_load(hass: HomeAssistant) -> None:
    """Load area registry."""
    assert DATA_REGISTRY not in hass.data
    await async_get(hass).async_load()

@callback
def async_entries_for_floor(registry: AreaRegistry, floor_id: str) -> list[AreaEntry]:
    """Return entries that match a floor."""
    return registry.areas.get_areas_for_floor(floor_id)

@callback
def async_entries_for_label(registry: AreaRegistry, label_id: str) -> list[AreaEntry]:
    """Return entries that match a label."""
    return registry.areas.get_areas_for_label(label_id)

def _validate_temperature_entity(hass: HomeAssistant, entity_id: str) -> None:
    """Validate temperature entity."""
    # ...

def _validate_humidity_entity(hass: HomeAssistant, entity_id: str) -> None:
    """Validate humidity entity."""
    # ...
