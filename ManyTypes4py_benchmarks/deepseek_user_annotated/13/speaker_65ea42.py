"""Base class for common speaker tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Coroutine
import contextlib
import datetime
from functools import partial
import logging
import time
from typing import TYPE_CHECKING, Any, cast, Dict, List, Optional, Set, Tuple, Union

import defusedxml.ElementTree as ET
from soco.core import SoCo
from soco.events_base import Event as SonosEvent, SubscriptionBase
from soco.exceptions import SoCoException, SoCoUPnPException
from soco.plugins.plex import PlexPlugin
from soco.plugins.sharelink import ShareLinkPlugin
from soco.snapshot import Snapshot
from sonos_websocket import SonosWebsocket

from homeassistant.components.media_player import DOMAIN as MP_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
    dispatcher_send,
)
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from .alarms import SonosAlarms
from .const import (
    AVAILABILITY_TIMEOUT,
    BATTERY_SCAN_INTERVAL,
    DATA_SONOS,
    DOMAIN,
    SCAN_INTERVAL,
    SONOS_CHECK_ACTIVITY,
    SONOS_CREATE_ALARM,
    SONOS_CREATE_AUDIO_FORMAT_SENSOR,
    SONOS_CREATE_BATTERY,
    SONOS_CREATE_LEVELS,
    SONOS_CREATE_MEDIA_PLAYER,
    SONOS_CREATE_MIC_SENSOR,
    SONOS_CREATE_SWITCHES,
    SONOS_FALLBACK_POLL,
    SONOS_REBOOTED,
    SONOS_SPEAKER_ACTIVITY,
    SONOS_SPEAKER_ADDED,
    SONOS_STATE_PLAYING,
    SONOS_STATE_TRANSITIONING,
    SONOS_STATE_UPDATED,
    SONOS_VANISHED,
    SUBSCRIPTION_TIMEOUT,
)
from .exception import S1BatteryMissing, SonosSubscriptionsFailed, SonosUpdateError
from .favorites import SonosFavorites
from .helpers import soco_error
from .media import SonosMedia
from .statistics import ActivityStatistics, EventStatistics

if TYPE_CHECKING:
    from . import SonosData

NEVER_TIME = -1200.0
RESUB_COOLDOWN_SECONDS = 10.0
EVENT_CHARGING = {
    "CHARGING": True,
    "NOT_CHARGING": False,
}
SUBSCRIPTION_SERVICES = {
    "alarmClock",
    "avTransport",
    "contentDirectory",
    "deviceProperties",
    "renderingControl",
    "zoneGroupTopology",
}
SUPPORTED_VANISH_REASONS = ("powered off", "sleeping", "switch to bluetooth", "upgrade")
UNUSED_DEVICE_KEYS = ["SPID", "TargetRoomName"]

_LOGGER = logging.getLogger(__name__)


class SonosSpeaker:
    """Representation of a Sonos speaker."""

    def __init__(
        self,
        hass: HomeAssistant,
        soco: SoCo,
        speaker_info: Dict[str, Any],
        zone_group_state_sub: Optional[SubscriptionBase],
    ) -> None:
        """Initialize a SonosSpeaker."""
        self.hass: HomeAssistant = hass
        self.data: SonosData = hass.data[DATA_SONOS]
        self.soco: SoCo = soco
        self.websocket: Optional[SonosWebsocket] = None
        self.household_id: str = soco.household_id
        self.media: SonosMedia = SonosMedia(hass, soco)
        self._plex_plugin: Optional[PlexPlugin] = None
        self._share_link_plugin: Optional[ShareLinkPlugin] = None
        self.available: bool = True

        # Device information
        self.hardware_version: str = speaker_info["hardware_version"]
        self.software_version: str = speaker_info["software_version"]
        self.mac_address: str = speaker_info["mac_address"]
        self.model_name: str = speaker_info["model_name"]
        self.model_number: str = speaker_info["model_number"]
        self.uid: str = speaker_info["uid"]
        self.version: str = speaker_info["display_version"]
        self.zone_name: str = speaker_info["zone_name"]

        # Subscriptions and events
        self.subscriptions_failed: bool = False
        self._subscriptions: List[SubscriptionBase] = []
        if zone_group_state_sub:
            zone_group_state_sub.callback = self.async_dispatch_event
            self._subscriptions.append(zone_group_state_sub)
        self._subscription_lock: Optional[asyncio.Lock] = None
        self._last_activity: float = NEVER_TIME
        self._last_event_cache: Dict[str, Any] = {}
        self.activity_stats: ActivityStatistics = ActivityStatistics(self.zone_name)
        self.event_stats: EventStatistics = EventStatistics(self.zone_name)
        self._resub_cooldown_expires_at: Optional[float] = None

        # Scheduled callback handles
        self._poll_timer: Optional[Callable] = None

        # Dispatcher handles
        self.dispatchers: List[Callable] = []

        # Battery
        self.battery_info: Dict[str, Any] = {}
        self._last_battery_event: Optional[datetime.datetime] = None
        self._battery_poll_timer: Optional[Callable] = None

        # Volume / Sound
        self.volume: Optional[int] = None
        self.muted: Optional[bool] = None
        self.cross_fade: Optional[bool] = None
        self.balance: Optional[Tuple[int, int]] = None
        self.bass: Optional[int] = None
        self.treble: Optional[int] = None
        self.loudness: Optional[bool] = None

        # Home theater
        self.audio_delay: Optional[int] = None
        self.dialog_level: Optional[bool] = None
        self.night_mode: Optional[bool] = None
        self.sub_enabled: Optional[bool] = None
        self.sub_crossover: Optional[int] = None
        self.sub_gain: Optional[int] = None
        self.surround_enabled: Optional[bool] = None
        self.surround_mode: Optional[bool] = None
        self.surround_level: Optional[int] = None
        self.music_surround_level: Optional[int] = None

        # Misc features
        self.buttons_enabled: Optional[bool] = None
        self.mic_enabled: Optional[bool] = None
        self.status_light: Optional[bool] = None

        # Grouping
        self.coordinator: Optional[SonosSpeaker] = None
        self.sonos_group: List[SonosSpeaker] = [self]
        self.sonos_group_entities: List[str] = []
        self.soco_snapshot: Optional[Snapshot] = None
        self.snapshot_group: List[SonosSpeaker] = []
        self._group_members_missing: Set[str] = set()

    async def async_setup(
        self, entry: ConfigEntry, has_battery: bool, dispatches: List[Tuple[Any, ...]]
    ) -> None:
        # ... rest of the methods remain the same with their existing type hints ...
        pass

    # ... rest of the class implementation remains the same ...
