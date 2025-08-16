from __future__ import annotations
import logging
from typing import Any, Dict
from aiohomekit.characteristic_cache import Pairing, StorageLayout
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store
from .const import DOMAIN, ENTITY_MAP

ENTITY_MAP_STORAGE_KEY: str = f'{DOMAIN}-entity-map'
ENTITY_MAP_STORAGE_VERSION: int = 1
ENTITY_MAP_SAVE_DELAY: int = 10
_LOGGER: logging.Logger = logging.getLogger(__name__)

class EntityMapStorage:
    def __init__(self, hass: HomeAssistant) -> None:
        self.hass: HomeAssistant = hass
        self.store: Store[StorageLayout] = Store[StorageLayout](hass, ENTITY_MAP_STORAGE_VERSION, ENTITY_MAP_STORAGE_KEY)
        self.storage_data: Dict[str, Pairing] = {}

    async def async_initialize(self) -> None:
        if not (raw_storage := (await self.store.async_load())):
            return
        self.storage_data = raw_storage.get('pairings', {})

    def get_map(self, homekit_id: str) -> Pairing:
        return self.storage_data.get(homekit_id)

    @callback
    def async_create_or_update_map(self, homekit_id: str, config_num: int, accessories: Any, broadcast_key: Any = None, state_num: Any = None) -> Pairing:
        _LOGGER.debug('Creating or updating entity map for %s', homekit_id)
        data: Pairing = Pairing(config_num=config_num, accessories=accessories, broadcast_key=broadcast_key, state_num=state_num)
        self.storage_data[homekit_id] = data
        self._async_schedule_save()
        return data

    @callback
    def async_delete_map(self, homekit_id: str) -> None:
        removed_one: bool = False
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
        self.store.async_delay_save(self._data_to_save, ENTITY_MAP_SAVE_DELAY)

    @callback
    def _data_to_save(self) -> StorageLayout:
        return StorageLayout(pairings=self.storage_data)

async def async_get_entity_storage(hass: HomeAssistant) -> EntityMapStorage:
    if ENTITY_MAP in hass.data:
        map_storage = hass.data[ENTITY_MAP]
        return map_storage
    map_storage = hass.data[ENTITY_MAP] = EntityMapStorage(hass)
    await map_storage.async_initialize()
    return map_storage
