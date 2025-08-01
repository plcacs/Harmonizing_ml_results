"""Support for monitoring the Syncthing instance."""
import aiosyncthing
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from .const import (
    DOMAIN,
    FOLDER_PAUSED_RECEIVED,
    FOLDER_SUMMARY_RECEIVED,
    SCAN_INTERVAL,
    SERVER_AVAILABLE,
    SERVER_UNAVAILABLE,
    STATE_CHANGED_RECEIVED,
)
from typing import Any, Dict, List, Optional

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Syncthing sensors."""
    syncthing: aiosyncthing.Syncthing = hass.data[DOMAIN][config_entry.entry_id]
    try:
        config: Dict[str, Any] = await syncthing.system.config()
        version: Dict[str, Any] = await syncthing.system.version()
    except aiosyncthing.exceptions.SyncthingError as exception:
        raise PlatformNotReady from exception
    server_id: str = syncthing.server_id
    entities: List[FolderSensor] = [
        FolderSensor(syncthing, server_id, folder["id"], folder["label"], version["version"])
        for folder in config["folders"]
    ]
    async_add_entities(entities)

class FolderSensor(SensorEntity):
    """A Syncthing folder sensor."""

    _attr_should_poll: bool = False
    _attr_translation_key: str = "syncthing"
    STATE_ATTRIBUTES: Dict[str, str] = {
        "errors": "errors",
        "globalBytes": "global_bytes",
        "globalDeleted": "global_deleted",
        "globalDirectories": "global_directories",
        "globalFiles": "global_files",
        "globalSymlinks": "global_symlinks",
        "globalTotalItems": "global_total_items",
        "ignorePatterns": "ignore_patterns",
        "inSyncBytes": "in_sync_bytes",
        "inSyncFiles": "in_sync_files",
        "invalid": "invalid",
        "localBytes": "local_bytes",
        "localDeleted": "local_deleted",
        "localDirectories": "local_directories",
        "localFiles": "local_files",
        "localSymlinks": "local_symlinks",
        "localTotalItems": "local_total_items",
        "needBytes": "need_bytes",
        "needDeletes": "need_deletes",
        "needDirectories": "need_directories",
        "needFiles": "need_files",
        "needSymlinks": "need_symlinks",
        "needTotalItems": "need_total_items",
        "pullErrors": "pull_errors",
        "state": "state",
    }

    def __init__(
        self,
        syncthing: aiosyncthing.Syncthing,
        server_id: str,
        folder_id: str,
        folder_label: str,
        version: str,
    ) -> None:
        """Initialize the sensor."""
        self._syncthing: aiosyncthing.Syncthing = syncthing
        self._server_id: str = server_id
        self._folder_id: str = folder_id
        self._folder_label: str = folder_label
        self._state: Optional[Dict[str, Any]] = None
        self._unsub_timer: Optional[Any] = None
        self._short_server_id: str = server_id.split("-")[0]
        self._attr_name: str = f"{self._short_server_id} {folder_id} {folder_label}"
        self._attr_unique_id: str = f"{self._short_server_id}-{folder_id}"
        self._attr_device_info: DeviceInfo = DeviceInfo(
            entry_type=DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, self._server_id)},
            manufacturer="Syncthing Team",
            name=f"Syncthing ({syncthing.url})",
            sw_version=version,
        )

    @property
    def native_value(self) -> str:
        """Return the state of the sensor."""
        return self._state["state"]

    @property
    def available(self) -> bool:
        """Could the device be accessed during the last update call."""
        return self._state is not None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return self._state

    async def async_update_status(self) -> None:
        """Request folder status and update state."""
        try:
            state: Dict[str, Any] = await self._syncthing.database.status(self._folder_id)
        except aiosyncthing.exceptions.SyncthingError:
            self._state = None
        else:
            self._state = self._filter_state(state)
        self.async_write_ha_state()

    def subscribe(self) -> None:
        """Start polling syncthing folder status."""
        if self._unsub_timer is None:

            async def refresh(event_time: Any) -> None:
                """Get the latest data from Syncthing."""
                await self.async_update_status()

            self._unsub_timer = async_track_time_interval(self.hass, refresh, SCAN_INTERVAL)

    @callback
    def unsubscribe(self) -> None:
        """Stop polling syncthing folder status."""
        if self._unsub_timer is not None:
            self._unsub_timer()
            self._unsub_timer = None

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""

        @callback
        def handle_folder_summary(event: Dict[str, Any]) -> None:
            if self._state is not None:
                self._state = self._filter_state(event["data"]["summary"])
                self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{FOLDER_SUMMARY_RECEIVED}-{self._server_id}-{self._folder_id}",
                handle_folder_summary,
            )
        )

        @callback
        def handle_state_changed(event: Dict[str, Any]) -> None:
            if self._state is not None:
                self._state["state"] = event["data"]["to"]
                self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{STATE_CHANGED_RECEIVED}-{self._server_id}-{self._folder_id}",
                handle_state_changed,
            )
        )

        @callback
        def handle_folder_paused(event: Dict[str, Any]) -> None:
            if self._state is not None:
                self._state["state"] = "paused"
                self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{FOLDER_PAUSED_RECEIVED}-{self._server_id}-{self._folder_id}",
                handle_folder_paused,
            )
        )

        @callback
        def handle_server_unavailable() -> None:
            self._state = None
            self.unsubscribe()
            self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SERVER_UNAVAILABLE}-{self._server_id}",
                handle_server_unavailable,
            )
        )

        async def handle_server_available() -> None:
            self.subscribe()
            await self.async_update_status()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SERVER_AVAILABLE}-{self._server_id}",
                handle_server_available,
            )
        )
        self.subscribe()
        self.async_on_remove(self.unsubscribe)
        await self.async_update_status()

    def _filter_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state_filtered: Dict[str, Any] = {
            self.STATE_ATTRIBUTES[key]: value
            for key, value in state.items()
            if key in self.STATE_ATTRIBUTES
        }
        if state_filtered["state"] == "":
            state_filtered["state"] = "paused"
        state_filtered["id"] = self._folder_id
        state_filtered["label"] = self._folder_label
        return state_filtered
