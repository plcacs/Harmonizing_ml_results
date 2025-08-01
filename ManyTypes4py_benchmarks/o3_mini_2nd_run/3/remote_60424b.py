"""Support for Harmony Hub devices."""
from __future__ import annotations
from collections.abc import Iterable
import json
import logging
from typing import Any, Optional, Tuple, Dict, List
import voluptuous as vol
from homeassistant.components.remote import (
    ATTR_ACTIVITY,
    ATTR_DELAY_SECS,
    ATTR_DEVICE,
    ATTR_HOLD_SECS,
    ATTR_NUM_REPEATS,
    DEFAULT_DELAY_SECS,
    RemoteEntity,
    RemoteEntityFeature,
)
from homeassistant.core import HassJob, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import VolDictType
from homeassistant.config_entries import ConfigEntry
from .const import (
    ACTIVITY_POWER_OFF,
    ATTR_ACTIVITY_STARTING,
    ATTR_DEVICES_LIST,
    ATTR_LAST_ACTIVITY,
    DOMAIN,
    HARMONY_OPTIONS_UPDATE,
    PREVIOUS_ACTIVE_ACTIVITY,
    SERVICE_CHANGE_CHANNEL,
    SERVICE_SYNC,
)
from .data import HarmonyConfigEntry, HarmonyData
from .entity import HarmonyEntity
from .subscriber import HarmonyCallback

_LOGGER = logging.getLogger(__name__)
PARALLEL_UPDATES = 0

ATTR_CHANNEL = "channel"
HARMONY_CHANGE_CHANNEL_SCHEMA = {vol.Required(ATTR_CHANNEL): cv.positive_int}


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the Harmony config entry."""
    data: HarmonyData = entry.runtime_data
    _LOGGER.debug("HarmonyData : %s", data)
    default_activity: Optional[str] = entry.options.get(ATTR_ACTIVITY)
    delay_secs: int = entry.options.get(ATTR_DELAY_SECS, DEFAULT_DELAY_SECS)
    harmony_conf_file: str = hass.config.path(f"harmony_{entry.unique_id}.conf")
    device = HarmonyRemote(data, default_activity, delay_secs, harmony_conf_file)
    async_add_entities([device])
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_SYNC, None, "sync")
    platform.async_register_entity_service(SERVICE_CHANGE_CHANNEL, HARMONY_CHANGE_CHANNEL_SCHEMA, "change_channel")


class HarmonyRemote(HarmonyEntity, RemoteEntity, RestoreEntity):
    """Remote representation used to control a Harmony device."""

    _attr_supported_features = RemoteEntityFeature.ACTIVITY
    _attr_name: Optional[str] = None

    def __init__(self, data: HarmonyData, activity: Optional[str], delay_secs: int, out_path: str) -> None:
        """Initialize HarmonyRemote class."""
        super().__init__(data=data)
        self._state: Optional[bool] = None
        self._current_activity: str = ACTIVITY_POWER_OFF
        self.default_activity: Optional[str] = activity
        self._activity_starting: Optional[str] = None
        self._is_initial_update: bool = True
        self.delay_secs: int = delay_secs
        self._last_activity: Optional[str] = None
        self._config_path: str = out_path
        self._attr_unique_id: str = data.unique_id
        self._attr_device_info = self._data.device_info(DOMAIN)

    async def _async_update_options(self, data: Dict[str, Any]) -> None:
        """Change options when the options flow does."""
        if ATTR_DELAY_SECS in data:
            self.delay_secs = data[ATTR_DELAY_SECS]
        if ATTR_ACTIVITY in data:
            self.default_activity = data[ATTR_ACTIVITY]

    def _setup_callbacks(self) -> None:
        self.async_on_remove(
            self._data.async_subscribe(
                HarmonyCallback(
                    connected=HassJob(self.async_got_connected),
                    disconnected=HassJob(self.async_got_disconnected),
                    config_updated=HassJob(self.async_new_config),
                    activity_starting=HassJob(self.async_new_activity),
                    activity_started=HassJob(self.async_new_activity_finished),
                )
            )
        )

    @callback
    def async_new_activity_finished(self, activity_info: Tuple[int, str]) -> None:
        """Call for finished updated current activity."""
        self._activity_starting = None
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Complete the initialization."""
        await super().async_added_to_hass()
        _LOGGER.debug("%s: Harmony Hub added", self._data.name)
        self.async_on_remove(self._async_clear_disconnection_delay)
        self._setup_callbacks()
        self.async_on_remove(
            async_dispatcher_connect(self.hass, f"{HARMONY_OPTIONS_UPDATE}-{self.unique_id}", self._async_update_options)
        )
        await self.async_new_config()
        last_state = await self.async_get_last_state()
        if not last_state:
            return
        if ATTR_LAST_ACTIVITY not in last_state.attributes:
            return
        if self.is_on:
            return
        self._last_activity = last_state.attributes[ATTR_LAST_ACTIVITY]

    @property
    def current_activity(self) -> str:
        """Return the current activity."""
        return self._current_activity

    @property
    def activity_list(self) -> List[str]:
        """Return the available activities."""
        return self._data.activity_names

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Add platform specific attributes."""
        return {
            ATTR_ACTIVITY_STARTING: self._activity_starting,
            ATTR_DEVICES_LIST: self._data.device_names,
            ATTR_LAST_ACTIVITY: self._last_activity,
        }

    @property
    def is_on(self) -> bool:
        """Return False if PowerOff is the current activity, otherwise True."""
        return self._current_activity not in [None, "PowerOff"]

    @callback
    def async_new_activity(self, activity_info: Tuple[int, str]) -> None:
        """Call for updating the current activity."""
        activity_id, activity_name = activity_info
        _LOGGER.debug("%s: activity reported as: %s", self._data.name, activity_name)
        self._current_activity = activity_name
        if self._is_initial_update:
            self._is_initial_update = False
        else:
            self._activity_starting = activity_name
        if activity_id != -1:
            self._last_activity = activity_name
        self._state = bool(activity_id != -1)
        self.async_write_ha_state()

    async def async_new_config(self, _: Any = None) -> None:
        """Call for updating the current activity."""
        _LOGGER.debug("%s: configuration has been updated", self._data.name)
        self.async_new_activity(self._data.current_activity)
        await self.hass.async_add_executor_job(self.write_config_file)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Start an activity from the Harmony device."""
        _LOGGER.debug("%s: Turn On", self._data.name)
        activity: Optional[str] = kwargs.get(ATTR_ACTIVITY, self.default_activity)
        if not activity or activity == PREVIOUS_ACTIVE_ACTIVITY:
            if self._last_activity:
                activity = self._last_activity
            elif (all_activities := self._data.activity_names):
                activity = all_activities[0]
        if activity:
            await self._data.async_start_activity(activity)
        else:
            _LOGGER.error("%s: No activity specified with turn_on service", self._data.name)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Start the PowerOff activity."""
        await self._data.async_power_off()

    async def async_send_command(self, command: Any, **kwargs: Any) -> None:
        """Send a list of commands to one device."""
        _LOGGER.debug("%s: Send Command", self._data.name)
        if (device := kwargs.get(ATTR_DEVICE)) is None:
            _LOGGER.error("%s: Missing required argument: device", self._data.name)
            return
        num_repeats: Any = kwargs[ATTR_NUM_REPEATS]
        delay_secs: int = kwargs.get(ATTR_DELAY_SECS, self.delay_secs)
        hold_secs: Any = kwargs[ATTR_HOLD_SECS]
        await self._data.async_send_command(command, device, num_repeats, delay_secs, hold_secs)

    async def change_channel(self, channel: int) -> None:
        """Change the channel using Harmony remote."""
        await self._data.change_channel(channel)

    async def sync(self) -> None:
        """Sync the Harmony device with the web service."""
        if await self._data.sync():
            await self.hass.async_add_executor_job(self.write_config_file)

    def write_config_file(self) -> None:
        """Write Harmony configuration file.

        This is a handy way for users to figure out the available commands for automations.
        """
        _LOGGER.debug("%s: Writing hub configuration to file: %s", self._data.name, self._config_path)
        json_config = self._data.json_config
        if json_config is None:
            _LOGGER.warning("%s: No configuration received from hub", self._data.name)
            return
        try:
            with open(self._config_path, "w+", encoding="utf-8") as file_out:
                json.dump(json_config, file_out, sort_keys=True, indent=4)
        except OSError as exc:
            _LOGGER.error("%s: Unable to write HUB configuration to %s: %s", self._data.name, self._config_path, exc)