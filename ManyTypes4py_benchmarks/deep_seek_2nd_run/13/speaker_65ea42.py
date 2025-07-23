"""Base class for common speaker tasks."""
from __future__ import annotations
import asyncio
from collections.abc import Callable, Collection, Coroutine
import contextlib
import datetime
from functools import partial
import logging
import time
from typing import TYPE_CHECKING, Any, cast, Optional, Dict, List, Set, Tuple, Union
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
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send, dispatcher_send
from homeassistant.helpers.event import async_track_time_interval, CALLBACK_TYPE
from homeassistant.util import dt as dt_util
from .alarms import SonosAlarms
from .const import AVAILABILITY_TIMEOUT, BATTERY_SCAN_INTERVAL, DATA_SONOS, DOMAIN, SCAN_INTERVAL, SONOS_CHECK_ACTIVITY, SONOS_CREATE_ALARM, SONOS_CREATE_AUDIO_FORMAT_SENSOR, SONOS_CREATE_BATTERY, SONOS_CREATE_LEVELS, SONOS_CREATE_MEDIA_PLAYER, SONOS_CREATE_MIC_SENSOR, SONOS_CREATE_SWITCHES, SONOS_FALLBACK_POLL, SONOS_REBOOTED, SONOS_SPEAKER_ACTIVITY, SONOS_SPEAKER_ADDED, SONOS_STATE_PLAYING, SONOS_STATE_TRANSITIONING, SONOS_STATE_UPDATED, SONOS_VANISHED, SUBSCRIPTION_TIMEOUT
from .exception import S1BatteryMissing, SonosSubscriptionsFailed, SonosUpdateError
from .favorites import SonosFavorites
from .helpers import soco_error
from .media import SonosMedia
from .statistics import ActivityStatistics, EventStatistics

if TYPE_CHECKING:
    from . import SonosData

NEVER_TIME = -1200.0
RESUB_COOLDOWN_SECONDS = 10.0
EVENT_CHARGING = {'CHARGING': True, 'NOT_CHARGING': False}
SUBSCRIPTION_SERVICES = {'alarmClock', 'avTransport', 'contentDirectory', 'deviceProperties', 'renderingControl', 'zoneGroupTopology'}
SUPPORTED_VANISH_REASONS = ('powered off', 'sleeping', 'switch to bluetooth', 'upgrade')
UNUSED_DEVICE_KEYS = ['SPID', 'TargetRoomName']
_LOGGER = logging.getLogger(__name__)

class SonosSpeaker:
    """Representation of a Sonos speaker."""

    def __init__(
        self,
        hass: HomeAssistant,
        soco: SoCo,
        speaker_info: Dict[str, Any],
        zone_group_state_sub: Optional[SubscriptionBase]
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
        self.hardware_version: str = speaker_info['hardware_version']
        self.software_version: str = speaker_info['software_version']
        self.mac_address: str = speaker_info['mac_address']
        self.model_name: str = speaker_info['model_name']
        self.model_number: str = speaker_info['model_number']
        self.uid: str = speaker_info['uid']
        self.version: str = speaker_info['display_version']
        self.zone_name: str = speaker_info['zone_name']
        self.subscriptions_failed: bool = False
        self._subscriptions: List[SubscriptionBase] = []
        if zone_group_state_sub:
            zone_group_state_sub.callback = self.async_dispatch_event
            self._subscriptions.append(zone_group_state_sub)
        self._subscription_lock: Optional[asyncio.Lock] = None
        self._last_activity: float = NEVER_TIME
        self._last_event_cache: Dict[str, Dict[str, str]] = {}
        self.activity_stats: ActivityStatistics = ActivityStatistics(self.zone_name)
        self.event_stats: EventStatistics = EventStatistics(self.zone_name)
        self._resub_cooldown_expires_at: Optional[float] = None
        self._poll_timer: Optional[CALLBACK_TYPE] = None
        self.dispatchers: List[Any] = []
        self.battery_info: Dict[str, Any] = {}
        self._last_battery_event: Optional[datetime.datetime] = None
        self._battery_poll_timer: Optional[CALLBACK_TYPE] = None
        self.volume: Optional[int] = None
        self.muted: Optional[bool] = None
        self.cross_fade: Optional[bool] = None
        self.balance: Optional[Tuple[int, int]] = None
        self.bass: Optional[int] = None
        self.treble: Optional[int] = None
        self.loudness: Optional[bool] = None
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
        self.buttons_enabled: Optional[bool] = None
        self.mic_enabled: Optional[bool] = None
        self.status_light: Optional[bool] = None
        self.coordinator: Optional[SonosSpeaker] = None
        self.sonos_group: List[SonosSpeaker] = [self]
        self.sonos_group_entities: List[str] = []
        self.soco_snapshot: Optional[Snapshot] = None
        self.snapshot_group: List[SonosSpeaker] = []
        self._group_members_missing: Set[str] = set()

    async def async_setup(self, entry: ConfigEntry, has_battery: bool, dispatches: List[Tuple[str, ...]]) -> None:
        """Complete setup in async context."""
        if has_battery:
            self._battery_poll_timer = async_track_time_interval(self.hass, self.async_poll_battery, BATTERY_SCAN_INTERVAL)
        self.websocket = SonosWebsocket(self.soco.ip_address, player_id=self.soco.uid, session=async_get_clientsession(self.hass))
        dispatch_pairs = (
            (SONOS_CHECK_ACTIVITY, self.async_check_activity),
            (SONOS_SPEAKER_ADDED, self.async_update_group_for_uid),
            (f'{SONOS_REBOOTED}-{self.soco.uid}', self.async_rebooted),
            (f'{SONOS_SPEAKER_ACTIVITY}-{self.soco.uid}', self.speaker_activity),
            (f'{SONOS_VANISHED}-{self.soco.uid}', self.async_vanished)
        )
        for signal, target in dispatch_pairs:
            entry.async_on_unload(async_dispatcher_connect(self.hass, signal, target))
        for dispatch in dispatches:
            async_dispatcher_send(self.hass, *dispatch)
        await self.async_subscribe()

    def setup(self, entry: ConfigEntry) -> None:
        """Run initial setup of the speaker."""
        self.media.play_mode = self.soco.play_mode
        self.update_volume()
        self.update_groups()
        if self.is_coordinator:
            self.media.poll_media()
        dispatches = [(SONOS_CREATE_LEVELS, self)]
        if (audio_format := self.soco.soundbar_audio_input_format):
            dispatches.append((SONOS_CREATE_AUDIO_FORMAT_SENSOR, self, audio_format))
        has_battery = False
        try:
            self.battery_info = self.fetch_battery_info()
        except SonosUpdateError:
            _LOGGER.debug('No battery available for %s', self.zone_name)
        else:
            has_battery = True
            dispatcher_send(self.hass, SONOS_CREATE_BATTERY, self)
        if (mic_enabled := self.soco.mic_enabled) is not None:
            self.mic_enabled = mic_enabled
            dispatches.append((SONOS_CREATE_MIC_SENSOR, self))
        if (new_alarms := [alarm.alarm_id for alarm in self.alarms if alarm.zone.uid == self.soco.uid]):
            dispatches.append((SONOS_CREATE_ALARM, self, new_alarms))
        dispatches.append((SONOS_CREATE_SWITCHES, self))
        dispatches.append((SONOS_CREATE_MEDIA_PLAYER, self))
        dispatches.append((SONOS_SPEAKER_ADDED, self.soco.uid))
        self.hass.create_task(self.async_setup(entry, has_battery, dispatches))

    def write_entity_states(self) -> None:
        """Write states for associated SonosEntity instances."""
        dispatcher_send(self.hass, f'{SONOS_STATE_UPDATED}-{self.soco.uid}')

    @callback
    def async_write_entity_states(self) -> None:
        """Write states for associated SonosEntity instances."""
        async_dispatcher_send(self.hass, f'{SONOS_STATE_UPDATED}-{self.soco.uid}')

    @property
    def alarms(self) -> SonosAlarms:
        """Return the SonosAlarms instance for this household."""
        return self.data.alarms[self.household_id]

    @property
    def favorites(self) -> SonosFavorites:
        """Return the SonosFavorites instance for this household."""
        return self.data.favorites[self.household_id]

    @property
    def is_coordinator(self) -> bool:
        """Return true if player is a coordinator."""
        return self.coordinator is None

    @property
    def plex_plugin(self) -> PlexPlugin:
        """Cache the PlexPlugin instance for this speaker."""
        if not self._plex_plugin:
            self._plex_plugin = PlexPlugin(self.soco)
        return self._plex_plugin

    @property
    def share_link(self) -> ShareLinkPlugin:
        """Cache the ShareLinkPlugin instance for this speaker."""
        if not self._share_link_plugin:
            self._share_link_plugin = ShareLinkPlugin(self.soco)
        return self._share_link_plugin

    @property
    def subscription_address(self) -> str:
        """Return the current subscription callback address."""
        assert len(self._subscriptions) > 0
        addr, port = self._subscriptions[0].event_listener.address
        return ':'.join([addr, str(port)])

    @property
    def missing_subscriptions(self) -> Set[str]:
        """Return a list of missing service subscriptions."""
        subscribed_services = {sub.service.service_type for sub in self._subscriptions}
        return SUBSCRIPTION_SERVICES - subscribed_services

    def log_subscription_result(self, result: Union[Any, Exception], event: str, level: int = logging.DEBUG) -> None:
        """Log a message if a subscription action (create/renew/stop) results in an exception."""
        if not isinstance(result, Exception):
            return
        if isinstance(result, asyncio.exceptions.TimeoutError):
            message = 'Request timed out'
            exc_info = None
        else:
            message = str(result)
            exc_info = result if not str(result) else None
        _LOGGER.log(level, '%s failed for %s: %s', event, self.zone_name, message, exc_info=exc_info)

    async def async_subscribe(self) -> None:
        """Initiate event subscriptions under an async lock."""
        if not self._subscription_lock:
            self._subscription_lock = asyncio.Lock()
        async with self._subscription_lock:
            try:
                await self._async_subscribe()
            except SonosSubscriptionsFailed:
                _LOGGER.warning('Creating subscriptions failed for %s', self.zone_name)
                await self._async_offline()

    async def _async_subscribe(self) -> None:
        """Create event subscriptions."""
        subscriptions = [self._subscribe(getattr(self.soco, service), self.async_dispatch_event) for service in self.missing_subscriptions]
        if not subscriptions:
            return
        _LOGGER.debug('Creating subscriptions for %s', self.zone_name)
        results = await asyncio.gather(*subscriptions, return_exceptions=True)
        for result in results:
            self.log_subscription_result(result, 'Creating subscription', logging.WARNING)
        if any((isinstance(result, Exception) for result in results)):
            raise SonosSubscriptionsFailed
        if not self._poll_timer:
            self._poll_timer = async_track_time_interval(self.hass, partial(async_dispatcher_send, self.hass, f'{SONOS_FALLBACK_POLL}-{self.soco.uid}'), SCAN_INTERVAL)

    async def _subscribe(self, target: Any, sub_callback: Callable[[SonosEvent], None]) -> SubscriptionBase:
        """Create a Sonos subscription."""
        subscription = await target.subscribe(auto_renew=True, requested_timeout=SUBSCRIPTION_TIMEOUT)
        subscription.callback = sub_callback
        subscription.auto_renew_fail = self.async_renew_failed
        self._subscriptions.append(subscription)
        return subscription

    async def async_unsubscribe(self) -> None:
        """Cancel all subscriptions."""
        if not self._subscriptions:
            return
        _LOGGER.debug('Unsubscribing from events for %s', self.zone_name)
        results = await asyncio.gather(*(subscription.unsubscribe() for subscription in self._subscriptions), return_exceptions=True)
        for result in results:
            self.log_subscription_result(result, 'Unsubscribe')
        self._subscriptions = []

    @callback
    def async_renew_failed(self, exception: Exception) -> None:
        """Handle a failed subscription renewal."""
        self.hass.async_create_background_task(self._async_renew_failed(exception), 'sonos renew failed', eager_start=True)

    async def _async_renew_failed(self, exception: Exception) -> None:
        """Mark the speaker as offline after a subscription renewal failure.

        This is to reset the state to allow a future clean subscription attempt.
        """
        if not self.available:
            return
        self.log_subscription_result(exception, 'Subscription renewal', logging.WARNING)
        await self.async_offline()

    @callback
    def async_dispatch_event(self, event: SonosEvent) -> None:
        """Handle callback event and route as needed."""
        if self._poll_timer:
            _LOGGER.debug('Received event, cancelling poll timer for %s', self.zone_name)
            self._poll_timer()
            self._poll_timer = None
        self.speaker_activity(f'{event.service.service_type} subscription')
        self.event_stats.receive(event)
        if (last_event := self._last_event_cache.get(event.service.service_type)):
            if event.variables.items() <= last_event.items():
                self.event_stats.duplicate(event)
                return
        self._last_event_cache[event.service.service_type] = event.variables
        dispatcher = self._event_dispatchers[event.service.service_type]
        dispatcher(self, event)

    @callback
    def async_dispatch_alarms(self, event: SonosEvent) -> None:
        """Add the soco instance associated with the event to the callback."""
        if 'alarm_list_version' not in event.variables:
            return
        self.hass.async_create_background_task(self.alarms.async_process_event(event, self), 'sonos process event', eager_start=True)

    @callback
    def async_dispatch_device_properties(self, event: SonosEvent) -> None:
        """Update device properties from an event."""
        self.event_stats.process(event)
        self.hass.async_create_background_task(self.async_update_device_properties(event), 'sonos device properties', eager_start=True)

    async def async_update_device_properties(self, event: SonosEvent) -> None:
        """Update device properties from an event."""
        if 'mic_enabled' in event.variables:
            mic_exists = self.mic_enabled is not None
            self.mic_enabled = bool(int(event.variables['mic_enabled']))
            if not mic_exists:
                async_dispatcher_send(self.hass, SONOS_CREATE_MIC_SENSOR, self)
        if (more_info := event.variables.get('more_info')):
            await self.async_update_battery_info(more_info)
        self.async_write_entity_states()

    @callback
    def async_dispatch_favorites(self, event: SonosEvent) -> None:
        """Add the soco instance associated with the event to the callback."""
        if 'favorites_update_id' not in event.variables:
            return
        if 'container_update_i_ds'