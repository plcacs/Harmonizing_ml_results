"""Helpers for HomeKit data stored in HA storage."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional, cast
from aiohomekit.characteristic_cache import Pairing, StorageLayout
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store
from .const import DOMAIN, ENTITY_MAP
from typing import TypedDict

ENTITY_MAP_STORAGE_KEY = f'{DOMAIN}-entity-map'
ENTITY_MAP_STORAGE_VERSION = 1
ENTITY_MAP_SAVE_DELAY = 10
_LOGGER = logging.getLogger(__name__)

class EntityMapStorage:
    """Holds a cache of entity structure data from a paired HomeKit device.

    HomeKit has a cacheable entity map that describes how an IP or BLE
    endpoint is structured. This object holds the latest copy of that data.

    An endpoint is made of accessories, services and characteristics. It is
    safe to cache this data until the c# discovery data changes.

    Caching this data means we can add HomeKit devices to HA immediately at
    start even if discovery hasn't seen them yet or they are out of range. It
    is also important for BLE devices - accessing the entity structure is
    very slow for these devices.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Create a new entity map store."""
        self.hass: HomeAssistant = hass
        self.store: Store[StorageLayout] = Store[StorageLayout](
            hass, ENTITY_MAP_STORAGE_VERSION, ENTITY_MAP_STORAGE_KEY
        )
        self.storage_data: Dict[str, Pairing] = {}

    async def async_initialize(self) -> None:
        """Get the pairing cache data."""
        if not (raw_storage := (await self.store.async_load())):
            return
        self.storage_data = raw_storage.get('pairings', {})

    def get_map(self, homekit_id: str) -> Optional[Pairing]:
        """Get a pairing cache item."""
        return self.storage_data.get(homekit_id)

    @callback
    def async_create_or_update_map(
        self,
        homekit_id: str,
        config_num: int,
        accessories: list[dict[str, Any]],
        broadcast_key: Optional[bytes] = None,
        state_num: Optional[int] = None,
    ) -> Pairing:
        """Create a new pairing cache."""
        _LOGGER.debug('Creating or updating entity map for %s', homekit_id)
        data = Pairing(
            config_num=config_num,
            accessories=accessories,
            broadcast_key=broadcast_key,
            state_num=state_num,
        )
        self.storage_data[homekit_id] = data
        self._async_schedule_save()
        return data

    @callback
    def async_delete_map(self, homekit_id: str) -> None:
        """Delete pairing cache."""
        removed_one = False
        for hkid in (homekit_id, homekit_id.lower()):
            if hkid not in self.storage_data:
                continue
            _LOGGER.debug('Deleting entity map for %s', hkid)
            self.storage_data.pop(hkid)
            removed_one = True
        if removed_one:
            self._async_schedule_save()

    @callback
    def _async_schedule_save(self) -> None:
        """Schedule saving the entity map cache."""
        self.store.async_delay_save(self._data_to_save, ENTITY_MAP_SAVE_DELAY)

    @callback
    def _data_to_save(self) -> StorageLayout:
        """Return data of entity map to store in a file."""
        return StorageLayout(pairings=self.storage_data)

async def async_get_entity_storage(hass: HomeAssistant) -> EntityMapStorage:
    """Get entity storage."""
    if ENTITY_MAP in hass.data:
        map_storage: EntityMapStorage = cast(EntityMapStorage, hass.data[ENTITY_MAP])
        return map_storage
    map_storage = hass.data[ENTITY_MAP] = EntityMapStorage(hass)
    await map_storage.async_initialize()
    return map_storage
