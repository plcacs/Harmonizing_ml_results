from typing import Any, Dict, List, Optional

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class FolderSensor(SensorEntity):
    _attr_should_poll: bool = False
    _attr_translation_key: str = 'syncthing'
    STATE_ATTRIBUTES: Dict[str, str] = {'errors': 'errors', 'globalBytes': 'global_bytes', 'globalDeleted': 'global_deleted', 'globalDirectories': 'global_directories', 'globalFiles': 'global_files', 'globalSymlinks': 'global_symlinks', 'globalTotalItems': 'global_total_items', 'ignorePatterns': 'ignore_patterns', 'inSyncBytes': 'in_sync_bytes', 'inSyncFiles': 'in_sync_files', 'invalid': 'invalid', 'localBytes': 'local_bytes', 'localDeleted': 'local_deleted', 'localDirectories': 'local_directories', 'localFiles': 'local_files', 'localSymlinks': 'local_symlinks', 'localTotalItems': 'local_total_items', 'needBytes': 'need_bytes', 'needDeletes': 'need_deletes', 'needDirectories': 'need_directories', 'needFiles': 'need_files', 'needSymlinks': 'need_symlinks', 'needTotalItems': 'need_total_items', 'pullErrors': 'pull_errors', 'state': 'state'}

    def __init__(self, syncthing: aiosyncthing.Syncthing, server_id: str, folder_id: str, folder_label: str, version: str) -> None:
        ...

    @property
    def native_value(self) -> Any:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_update_status(self) -> None:
        ...

    def subscribe(self) -> None:
        ...

    def unsubscribe(self) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...
    
    def _filter_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ...
