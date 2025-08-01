from __future__ import annotations
import asyncio
import contextlib
import datetime
import logging
import time
from collections.abc import Callable, Collection
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
from homeassistant.helpers.event import async_track_time_interval
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

NEVER_TIME: float = -1200.0
RESUB_COOLDOWN_SECONDS: float = 10.0
EVENT_CHARGING: Dict[str, bool] = {'CHARGING': True, 'NOT_CHARGING': False}
SUBSCRIPTION_SERVICES: Set[str] = {
    'alarmClock', 'avTransport', 'contentDirectory', 'deviceProperties', 'renderingControl', 'zoneGroupTopology'
}
SUPPORTED_VANISH_REASONS: Tuple[str, ...] = ('powered off', 'sleeping', 'switch to bluetooth', 'upgrade')
UNUSED_DEVICE_KEYS: List[str] = ['SPID', 'TargetRoomName']
_LOGGER = logging.getLogger(__name__)


class SonosSpeaker:
    """Representation of a Sonos speaker."""

    _event_dispatchers: Dict[str, Callable[[SonosSpeaker, SonosEvent], None]] = {
        'AlarmClock': SonosSpeaker.async_dispatch_alarms,
        'AVTransport': SonosSpeaker.async_dispatch_media_update,
        'ContentDirectory': SonosSpeaker.async_dispatch_favorites,
        'DeviceProperties': SonosSpeaker.async_dispatch_device_properties,
        'RenderingControl': SonosSpeaker.async_update_volume,
        'ZoneGroupTopology': SonosSpeaker.async_update_groups,
    }

    def __init__(self, hass: HomeAssistant, soco: SoCo, speaker_info: Dict[str, Any],
                 zone_group_state_sub: Optional[SubscriptionBase]) -> None:
        """Initialize a SonosSpeaker."""
        self.hass: HomeAssistant = hass
        self.data: Any = hass.data[DATA_SONOS]
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
        self._last_event_cache: Dict[str, Any] = {}
        self.activity_stats: ActivityStatistics = ActivityStatistics(self.zone_name)
        self.event_stats: EventStatistics = EventStatistics(self.zone_name)
        self._resub_cooldown_expires_at: Optional[float] = None
        self._poll_timer: Optional[Callable[[], Any]] = None
        self.dispatchers: List[Any] = []
        self.battery_info: Dict[str, Any] = {}
        self._last_battery_event: Optional[datetime.datetime] = None
        self._battery_poll_timer: Optional[Callable[[], Any]] = None
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
        self.buttons_enabled: Optional[Any] = None
        self.mic_enabled: Optional[bool] = None
        self.status_light: Optional[Any] = None
        self.coordinator: Optional[SonosSpeaker] = None
        self.sonos_group: List[SonosSpeaker] = [self]
        self.sonos_group_entities: List[str] = []
        self.soco_snapshot: Optional[Snapshot] = None
        self.snapshot_group: List[SonosSpeaker] = []
        self._group_members_missing: Set[str] = set()

    async def async_setup(self, entry: ConfigEntry, has_battery: bool,
                          dispatches: List[Tuple[Any, ...]]) -> None:
        """Complete setup in async context."""
        if has_battery:
            self._battery_poll_timer = async_track_time_interval(self.hass, self.async_poll_battery, BATTERY_SCAN_INTERVAL)
        self.websocket = SonosWebsocket(self.soco.ip_address, player_id=self.soco.uid,
                                        session=async_get_clientsession(self.hass))
        dispatch_pairs: List[Tuple[str, Callable[[Any], Any]]] = [
            (SONOS_CHECK_ACTIVITY, self.async_check_activity),
            (SONOS_SPEAKER_ADDED, self.async_update_group_for_uid),
            (f'{SONOS_REBOOTED}-{self.soco.uid}', self.async_rebooted),
            (f'{SONOS_SPEAKER_ACTIVITY}-{self.soco.uid}', self.speaker_activity),
            (f'{SONOS_VANISHED}-{self.soco.uid}', self.async_vanished),
        ]
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
        dispatches: List[Tuple[Any, ...]] = [(SONOS_CREATE_LEVELS, self)]
        if (audio_format := self.soco.soundbar_audio_input_format):
            dispatches.append((SONOS_CREATE_AUDIO_FORMAT_SENSOR, self, audio_format))
        has_battery: bool = False
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
        subscribed_services: Set[str] = {sub.service.service_type for sub in self._subscriptions}
        return SUBSCRIPTION_SERVICES - subscribed_services

    def log_subscription_result(self, result: Any, event: str, level: int = logging.DEBUG) -> None:
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
        subscriptions = [
            self._subscribe(getattr(self.soco, service), self.async_dispatch_event)
            for service in self.missing_subscriptions
        ]
        if not subscriptions:
            return
        _LOGGER.debug('Creating subscriptions for %s', self.zone_name)
        results = await asyncio.gather(*subscriptions, return_exceptions=True)
        for result in results:
            self.log_subscription_result(result, 'Creating subscription', logging.WARNING)
        if any(isinstance(result, Exception) for result in results):
            raise SonosSubscriptionsFailed
        if not self._poll_timer:
            self._poll_timer = async_track_time_interval(
                self.hass,
                partial(async_dispatcher_send, self.hass, f'{SONOS_FALLBACK_POLL}-{self.soco.uid}'),
                SCAN_INTERVAL
            )

    async def _subscribe(self, target: Any, sub_callback: Callable[[SonosEvent], None]) -> None:
        """Create a Sonos subscription."""
        subscription: SubscriptionBase = await target.subscribe(auto_renew=True, requested_timeout=SUBSCRIPTION_TIMEOUT)
        subscription.callback = sub_callback
        subscription.auto_renew_fail = self.async_renew_failed
        self._subscriptions.append(subscription)

    async def async_unsubscribe(self) -> None:
        """Cancel all subscriptions."""
        if not self._subscriptions:
            return
        _LOGGER.debug('Unsubscribing from events for %s', self.zone_name)
        results = await asyncio.gather(*(subscription.unsubscribe() for subscription in self._subscriptions),
                                        return_exceptions=True)
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
        self.hass.async_create_background_task(self.alarms.async_process_event(event, self),
                                                 'sonos process event', eager_start=True)

    @callback
    def async_dispatch_device_properties(self, event: SonosEvent) -> None:
        """Update device properties from an event."""
        self.event_stats.process(event)
        self.hass.async_create_background_task(self.async_update_device_properties(event),
                                                 'sonos device properties', eager_start=True)

    async def async_update_device_properties(self, event: SonosEvent) -> None:
        """Update device properties from an event."""
        if 'mic_enabled' in event.variables:
            mic_exists: bool = self.mic_enabled is not None
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
        if 'container_update_i_ds' not in event.variables:
            return
        self.hass.async_create_background_task(self.favorites.async_process_event(event, self),
                                                 'sonos dispatch favorites', eager_start=True)

    @callback
    def async_dispatch_media_update(self, event: SonosEvent) -> None:
        """Update information about currently playing media from an event."""
        av_transport_uri: str = event.variables.get('av_transport_uri', '')
        current_track_uri: str = event.variables.get('current_track_uri', '')
        if av_transport_uri == current_track_uri and av_transport_uri.startswith('x-rincon:'):
            new_coordinator_uid: str = av_transport_uri.split(':')[-1]
            if (new_coordinator_speaker := self.data.discovered.get(new_coordinator_uid)):
                _LOGGER.debug('Media update coordinator (%s) received for %s',
                              new_coordinator_speaker.zone_name, self.zone_name)
                self.coordinator = new_coordinator_speaker
            else:
                _LOGGER.debug('Media update coordinator (%s) for %s not yet available',
                              new_coordinator_uid, self.zone_name)
            return
        if (crossfade := event.variables.get('current_crossfade_mode')):
            crossfade_bool: bool = bool(int(crossfade))
            if self.cross_fade != crossfade_bool:
                self.cross_fade = crossfade_bool
                self.async_write_entity_states()
        if (new_status := event.variables.get('transport_state')) is None:
            return
        if new_status == SONOS_STATE_TRANSITIONING:
            return
        self.event_stats.process(event)
        self.hass.async_add_executor_job(self.media.update_media_from_event, event.variables)

    @callback
    def async_update_volume(self, event: SonosEvent) -> None:
        """Update information about currently volume settings."""
        self.event_stats.process(event)
        variables: Dict[str, Any] = event.variables
        if 'volume' in variables:
            volume: Dict[str, Any] = variables['volume']
            self.volume = int(volume['Master'])
            if 'LF' in volume and 'RF' in volume:
                self.balance = (int(volume['LF']), int(volume['RF']))
        if 'mute' in variables:
            self.muted = variables['mute']['Master'] == '1'
        if (loudness := variables.get('loudness')):
            self.loudness = loudness['Master'] == '1'
        for bool_var in ('dialog_level', 'night_mode', 'sub_enabled', 'surround_enabled', 'surround_mode'):
            if bool_var in variables:
                setattr(self, bool_var, variables[bool_var] == '1')
        for int_var in ('audio_delay', 'bass', 'treble', 'sub_crossover', 'sub_gain', 'surround_level', 'music_surround_level'):
            if int_var in variables:
                setattr(self, int_var, variables[int_var])
        self.async_write_entity_states()

    @soco_error()
    def ping(self) -> None:
        """Test device availability. Failure will raise SonosUpdateError."""
        self.soco.renderingControl.GetVolume([('InstanceID', 0), ('Channel', 'Master')], timeout=1)

    @callback
    def speaker_activity(self, source: str) -> None:
        """Track the last activity on this speaker, set availability and resubscribe."""
        if self._resub_cooldown_expires_at:
            if time.monotonic() < self._resub_cooldown_expires_at:
                _LOGGER.debug('Activity on %s from %s while in cooldown, ignoring', self.zone_name, source)
                return
            self._resub_cooldown_expires_at = None
        _LOGGER.debug('Activity on %s from %s', self.zone_name, source)
        self._last_activity = time.monotonic()
        self.activity_stats.activity(source, self._last_activity)
        was_available: bool = self.available
        self.available = True
        if not was_available:
            self.async_write_entity_states()
            self.hass.async_create_task(self.async_subscribe(), eager_start=True)

    @callback
    def async_check_activity(self, now: Any) -> None:
        """Validate availability of the speaker based on recent activity."""
        if not self.available:
            return
        if time.monotonic() - self._last_activity < AVAILABILITY_TIMEOUT:
            return
        self.hass.async_create_background_task(self._async_check_activity(),
                                                 f'sonos {self.uid} {self.zone_name} ping', eager_start=True)

    async def _async_check_activity(self) -> None:
        """Validate availability of the speaker based on recent activity."""
        try:
            await self.hass.async_add_executor_job(self.ping)
        except SonosUpdateError:
            _LOGGER.warning('No recent activity and cannot reach %s, marking unavailable', self.zone_name)
            await self.async_offline()

    async def async_offline(self) -> None:
        """Handle removal of speaker when unavailable."""
        assert self._subscription_lock is not None
        async with self._subscription_lock:
            await self._async_offline()

    async def _async_offline(self) -> None:
        """Handle removal of speaker when unavailable."""
        if not self.available:
            return
        if self._resub_cooldown_expires_at is None and (not self.hass.is_stopping):
            self._resub_cooldown_expires_at = time.monotonic() + RESUB_COOLDOWN_SECONDS
            _LOGGER.debug('Starting resubscription cooldown for %s', self.zone_name)
        self.available = False
        self.async_write_entity_states()
        self._share_link_plugin = None
        if self._poll_timer:
            self._poll_timer()
            self._poll_timer = None
        await self.async_unsubscribe()
        self.data.discovery_known.discard(self.soco.uid)

    async def async_vanished(self, reason: str) -> None:
        """Handle removal of speaker when marked as vanished."""
        if not self.available:
            return
        _LOGGER.debug('%s has vanished (%s), marking unavailable', self.zone_name, reason)
        await self.async_offline()

    async def async_rebooted(self) -> None:
        """Handle a detected speaker reboot."""
        _LOGGER.debug('%s rebooted, reconnecting', self.zone_name)
        await self.async_offline()
        self.speaker_activity('reboot')

    @soco_error()
    def fetch_battery_info(self) -> Dict[str, Any]:
        """Fetch battery_info for the speaker."""
        battery_info: Dict[str, Any] = self.soco.get_battery_info()
        if not battery_info:
            raise S1BatteryMissing
        return battery_info

    async def async_update_battery_info(self, more_info: str) -> None:
        """Update battery info using a SonosEvent payload value."""
        battery_dict: Dict[str, str] = dict((x.split(':') for x in more_info.split(',')))
        for unused in UNUSED_DEVICE_KEYS:
            battery_dict.pop(unused, None)
        if not battery_dict:
            return
        if 'BattChg' not in battery_dict:
            _LOGGER.debug("Unknown device properties update for %s (%s), please report an issue: '%s'",
                          self.zone_name, self.model_name, more_info)
            return
        self._last_battery_event = dt_util.utcnow()
        is_charging: bool = EVENT_CHARGING[battery_dict['BattChg']]
        if not self._battery_poll_timer:
            new_battery: bool = not bool(self.battery_info)
            self.battery_info.update({'Level': int(battery_dict['BattPct']),
                                        'PowerSource': 'EXTERNAL' if is_charging else 'BATTERY'})
            if new_battery:
                _LOGGER.warning('S1 firmware detected on %s, battery info may update infrequently', self.zone_name)
                async_dispatcher_send(self.hass, SONOS_CREATE_BATTERY, self)
            return
        if is_charging == self.charging:
            self.battery_info.update({'Level': int(battery_dict['BattPct'])})
        elif not is_charging:
            self.battery_info['PowerSource'] = 'BATTERY'
        else:
            try:
                self.battery_info = await self.hass.async_add_executor_job(self.fetch_battery_info)
            except SonosUpdateError as err:
                _LOGGER.debug('Could not request current power source: %s', err)

    @property
    def power_source(self) -> Optional[str]:
        """Return the name of the current power source.

        Observed to be either BATTERY or SONOS_CHARGING_RING or USB_POWER.

        May be an empty dict if used with an S1 Move.
        """
        return self.battery_info.get('PowerSource')

    @property
    def charging(self) -> Optional[bool]:
        """Return the charging status of the speaker."""
        if self.power_source:
            return self.power_source != 'BATTERY'
        return None

    async def async_poll_battery(self, now: Optional[Any] = None) -> None:
        """Poll the device for the current battery state."""
        if not self.available:
            return
        if self._last_battery_event and dt_util.utcnow() - self._last_battery_event < BATTERY_SCAN_INTERVAL:
            return
        try:
            self.battery_info = await self.hass.async_add_executor_job(self.fetch_battery_info)
        except SonosUpdateError as err:
            _LOGGER.debug('Could not poll battery info: %s', err)
        else:
            self.async_write_entity_states()

    def update_groups(self) -> None:
        """Update group topology when polling."""
        self.hass.add_job(self.create_update_groups_coro())

    @callback
    def async_update_group_for_uid(self, uid: str) -> None:
        """Update group topology if uid is missing."""
        if uid not in self._group_members_missing:
            return
        missing_zone: str = self.data.discovered[uid].zone_name
        _LOGGER.debug('%s was missing, adding to %s group', missing_zone, self.zone_name)
        self.hass.async_create_task(self.create_update_groups_coro(), eager_start=True)

    @callback
    def async_update_groups(self, event: SonosEvent) -> None:
        """Handle callback for topology change event."""
        if (xml := event.variables.get('zone_group_state')):
            zgs = ET.fromstring(xml)
            for vanished_device in zgs.find('VanishedDevices') or []:
                if (reason := vanished_device.get('Reason')) not in SUPPORTED_VANISH_REASONS:
                    _LOGGER.debug('Ignoring %s marked %s as vanished with reason: %s',
                                  self.zone_name, vanished_device.get('ZoneName'), reason)
                    continue
                uid: str = vanished_device.get('UUID')  # type: ignore
                async_dispatcher_send(self.hass, f'{SONOS_VANISHED}-{uid}', reason)
        self.event_stats.process(event)
        self.hass.async_create_background_task(self.create_update_groups_coro(event),
                                                 name=f'sonos group update {self.zone_name}', eager_start=True)

    def create_update_groups_coro(self, event: Optional[SonosEvent] = None) -> asyncio.Future[Any]:
        """Handle callback for topology change event."""

        def _get_soco_group() -> List[str]:
            """Ask SoCo cache for existing topology."""
            coordinator_uid: str = self.soco.uid
            joined_uids: List[str] = []
            with contextlib.suppress(OSError, SoCoException):
                if self.soco.group and self.soco.group.coordinator:
                    coordinator_uid = self.soco.group.coordinator.uid
                    joined_uids = [p.uid for p in self.soco.group.members if p.uid != coordinator_uid and p.is_visible]
            return [coordinator_uid, *joined_uids]

        async def _async_extract_group(evt: Optional[SonosEvent]) -> List[str]:
            """Extract group layout from a topology event."""
            if evt and getattr(evt, 'zone_player_uui_ds_in_group', None):
                assert isinstance(evt.zone_player_uui_ds_in_group, str)
                return evt.zone_player_uui_ds_in_group.split(',')
            return await self.hass.async_add_executor_job(_get_soco_group)

        @callback
        def _async_regroup(group: List[str]) -> None:
            """Rebuild internal group layout."""
            _LOGGER.debug('async_regroup %s %s', self.zone_name, group)
            if group == [self.soco.uid] and self.sonos_group == [self] and self.sonos_group_entities:
                if self.coordinator is not None:
                    _LOGGER.debug('Zone %s Cleared coordinator [%s]', self.zone_name, self.coordinator.zone_name)
                    self.coordinator = None
                    self.async_write_entity_states()
                return
            entity_registry = er.async_get(self.hass)
            sonos_group: List[SonosSpeaker] = []
            sonos_group_entities: List[str] = []
            for uid in group:
                speaker = self.data.discovered.get(uid)
                if speaker:
                    self._group_members_missing.discard(uid)
                    sonos_group.append(speaker)
                    entity_id: Optional[str] = entity_registry.async_get_entity_id(MP_DOMAIN, DOMAIN, uid)
                    if entity_id is not None:
                        sonos_group_entities.append(entity_id)
                else:
                    self._group_members_missing.add(uid)
                    _LOGGER.debug('%s group member unavailable (%s), will try again', self.zone_name, uid)
                    return
            if self.sonos_group_entities == sonos_group_entities:
                return
            self.coordinator = None
            self.sonos_group = sonos_group
            self.sonos_group_entities = sonos_group_entities
            self.async_write_entity_states()
            for joined_uid in group[1:]:
                if (joined_speaker := self.data.discovered.get(joined_uid)):
                    joined_speaker.coordinator = self
                    joined_speaker.sonos_group = sonos_group
                    joined_speaker.sonos_group_entities = sonos_group_entities
                    _LOGGER.debug('Zone %s Set coordinator [%s]', joined_speaker.zone_name, self.zone_name)
                    joined_speaker.async_write_entity_states()
            _LOGGER.debug('Regrouped %s: %s', self.zone_name, self.sonos_group_entities)

        async def _async_handle_group_event(evt: Optional[SonosEvent]) -> None:
            """Get async lock and handle event."""
            async with self.data.topology_condition:
                group = await _async_extract_group(evt)
                if self.soco.uid == group[0]:
                    _async_regroup(group)
                    self.data.topology_condition.notify_all()
        return asyncio.ensure_future(_async_handle_group_event(event))

    @soco_error()
    def join(self, speakers: Collection[SonosSpeaker]) -> List[SonosSpeaker]:
        """Form a group with other players."""
        if self.coordinator:
            self.unjoin()
            group: List[SonosSpeaker] = [self]
        else:
            group = self.sonos_group.copy()
        for speaker in speakers:
            if speaker.soco.uid != self.soco.uid:
                if speaker not in group:
                    speaker.soco.join(self.soco)
                    speaker.coordinator = self
                    group.append(speaker)
        return group

    @staticmethod
    async def join_multi(hass: HomeAssistant, master: SonosSpeaker,
                         speakers: Collection[SonosSpeaker]) -> None:
        """Form a group with other players."""
        async with hass.data[DATA_SONOS].topology_condition:
            group: List[SonosSpeaker] = await hass.async_add_executor_job(master.join, speakers)
            await SonosSpeaker.wait_for_groups(hass, [group])

    @soco_error()
    def unjoin(self) -> None:
        """Unjoin the player from a group."""
        if self.sonos_group == [self]:
            return
        self.soco.unjoin()
        self.coordinator = None

    @staticmethod
    async def unjoin_multi(hass: HomeAssistant, speakers: Collection[SonosSpeaker]) -> None:
        """Unjoin several players from their group."""
        def _unjoin_all(speakers_iter: Collection[SonosSpeaker]) -> None:
            coordinators = [s for s in speakers_iter if s.is_coordinator]
            joined_speakers = [s for s in speakers_iter if not s.is_coordinator]
            for speaker in joined_speakers + coordinators:
                speaker.unjoin()
        async with hass.data[DATA_SONOS].topology_condition:
            await hass.async_add_executor_job(_unjoin_all, speakers)
            await SonosSpeaker.wait_for_groups(hass, [[s] for s in speakers])

    @soco_error()
    def snapshot(self, with_group: bool) -> None:
        """Snapshot the state of a player."""
        self.soco_snapshot = Snapshot(self.soco)
        self.soco_snapshot.snapshot()
        if with_group:
            self.snapshot_group = self.sonos_group.copy()
        else:
            self.snapshot_group = []

    @staticmethod
    async def snapshot_multi(hass: HomeAssistant, speakers: Collection[SonosSpeaker],
                             with_group: bool) -> None:
        """Snapshot all the speakers and optionally their groups."""
        def _snapshot_all(speakers_iter: Collection[SonosSpeaker]) -> None:
            for speaker in speakers_iter:
                speaker.snapshot(with_group)
        speakers_set: Set[SonosSpeaker] = set(speakers)
        if with_group:
            for speaker in list(speakers_set):
                speakers_set.update(speaker.sonos_group)
        async with hass.data[DATA_SONOS].topology_condition:
            await hass.async_add_executor_job(_snapshot_all, speakers_set)

    @soco_error()
    def restore(self) -> None:
        """Restore a snapshotted state to a player."""
        try:
            assert self.soco_snapshot is not None
            self.soco_snapshot.restore()
        except (TypeError, AssertionError, AttributeError, SoCoException) as ex:
            _LOGGER.warning('Error on restore %s: %s', self.zone_name, ex)
        self.soco_snapshot = None
        self.snapshot_group = []

    @staticmethod
    async def restore_multi(hass: HomeAssistant, speakers: Collection[SonosSpeaker],
                            with_group: bool) -> None:
        """Restore snapshots for all the speakers."""
        def _restore_groups(speakers_iter: Collection[SonosSpeaker], wg: bool) -> List[List[SonosSpeaker]]:
            for speaker in (s for s in speakers_iter if s.is_coordinator):
                if speaker.media.playback_status == SONOS_STATE_PLAYING and 'Pause' in speaker.soco.available_actions:
                    try:
                        speaker.soco.pause()
                    except SoCoUPnPException as exc:
                        _LOGGER.debug('Pause failed during restore of %s: %s', speaker.zone_name,
                                      speaker.soco.available_actions, exc_info=exc)
            groups: List[List[SonosSpeaker]] = []
            if not wg:
                return groups
            speakers_to_unjoin: Set[SonosSpeaker] = set()
            for speaker in speakers_iter:
                if speaker.sonos_group == speaker.snapshot_group:
                    continue
                speakers_to_unjoin.update({s for s in speaker.sonos_group[1:] if s not in speaker.snapshot_group})
            for speaker in speakers_to_unjoin:
                speaker.unjoin()
            for speaker in (s for s in speakers_iter if s.snapshot_group):
                assert len(speaker.snapshot_group) > 0
                if speaker.snapshot_group[0] == speaker:
                    if speaker.snapshot_group not in (speaker.sonos_group, [speaker]):
                        speaker.join(speaker.snapshot_group)
                    groups.append(speaker.snapshot_group.copy())
            return groups

        def _restore_players(speakers_iter: Collection[SonosSpeaker]) -> None:
            for speaker in (s for s in speakers_iter if not s.is_coordinator):
                speaker.restore()
            for speaker in (s for s in speakers_iter if s.is_coordinator):
                speaker.restore()

        speakers_set: Set[SonosSpeaker] = {s for s in speakers if s.soco_snapshot}
        missing_snapshots: Set[SonosSpeaker] = set(speakers) - speakers_set
        if missing_snapshots:
            raise HomeAssistantError(f'Restore failed, speakers are missing snapshots: {[s.zone_name for s in missing_snapshots]}')
        if with_group:
            for speaker in [s for s in speakers_set if s.snapshot_group]:
                assert len(speaker.snapshot_group) > 0
                speakers_set.update(speaker.snapshot_group)
        async with hass.data[DATA_SONOS].topology_condition:
            groups: List[List[SonosSpeaker]] = await hass.async_add_executor_job(_restore_groups, speakers_set, with_group)
            await SonosSpeaker.wait_for_groups(hass, groups)
            await hass.async_add_executor_job(_restore_players, speakers_set)

    @staticmethod
    async def wait_for_groups(hass: HomeAssistant, groups: Collection[List[SonosSpeaker]]) -> None:
        """Wait until all groups are present, or timeout."""
        def _test_groups(grps: Collection[List[SonosSpeaker]]) -> bool:
            for group in grps:
                coordinator = group[0]
                current_group = coordinator.sonos_group
                if coordinator != current_group[0]:
                    return False
                if set(group[1:]) != set(current_group[1:]):
                    return False
            return True
        try:
            async with asyncio.timeout(5):
                while not _test_groups(groups):
                    await hass.data[DATA_SONOS].topology_condition.wait()
        except TimeoutError:
            _LOGGER.warning('Timeout waiting for target groups %s', groups)
        any_speaker = next(iter(hass.data[DATA_SONOS].discovered.values()))
        any_speaker.soco.zone_group_state.clear_cache()

    @soco_error()
    def update_volume(self) -> None:
        """Update information about current volume settings."""
        self.volume = self.soco.volume
        self.muted = self.soco.mute
