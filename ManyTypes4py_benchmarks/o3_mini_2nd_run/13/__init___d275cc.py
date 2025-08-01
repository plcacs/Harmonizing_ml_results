from __future__ import annotations
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import datetime
from functools import partial
import logging
import socket
from typing import Any, Callable, cast, Dict, Iterable, Optional, Set
from urllib.parse import urlparse

from aiohttp import ClientError
from requests.exceptions import Timeout
from soco import events_asyncio, zonegroupstate
import soco.config as soco_config
from soco.core import SoCo
from soco.events_base import Event as SonosEvent, SubscriptionBase
from soco.exceptions import SoCoException
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components import ssdp
from homeassistant.components.media_player import DOMAIN as MP_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOSTS, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, device_registry as dr, issue_registry as ir
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later, async_track_time_interval
from homeassistant.helpers.service_info.ssdp import ATTR_UPNP_MODEL_NAME, ATTR_UPNP_UDN, SsdpServiceInfo
from homeassistant.helpers.typing import ConfigType
from homeassistant.util.async_ import create_eager_task

from .alarms import SonosAlarms
from .const import (
    AVAILABILITY_CHECK_INTERVAL,
    DATA_SONOS,
    DATA_SONOS_DISCOVERY_MANAGER,
    DISCOVERY_INTERVAL,
    DOMAIN,
    PLATFORMS,
    SONOS_CHECK_ACTIVITY,
    SONOS_REBOOTED,
    SONOS_SPEAKER_ACTIVITY,
    SONOS_VANISHED,
    SUB_FAIL_ISSUE_ID,
    SUB_FAIL_URL,
    SUBSCRIPTION_TIMEOUT,
    UPNP_ST,
)
from .exception import SonosUpdateError
from .favorites import SonosFavorites
from .helpers import sync_get_visible_zones
from .speaker import SonosSpeaker

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_ADVERTISE_ADDR: str = 'advertise_addr'
CONF_INTERFACE_ADDR: str = 'interface_addr'
DISCOVERY_IGNORED_MODELS: list[str] = ['Sonos Boost']
ZGS_SUBSCRIPTION_TIMEOUT: int = 2

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                MP_DOMAIN: vol.All(
                    cv.deprecated(CONF_INTERFACE_ADDR),
                    vol.Schema(
                        {
                            vol.Optional(CONF_ADVERTISE_ADDR): cv.string,
                            vol.Optional(CONF_INTERFACE_ADDR): cv.string,
                            vol.Optional(CONF_HOSTS): vol.All(cv.ensure_list_csv, [cv.string]),
                        }
                    ),
                )
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)


@dataclass
class UnjoinData:
    """Class to track data necessary for unjoin coalescing."""
    event: asyncio.Event = field(default_factory=asyncio.Event)


class SonosData:
    """Storage class for platform global data."""

    def __init__(self) -> None:
        """Initialize the data."""
        self.discovered: OrderedDict[str, SonosSpeaker] = OrderedDict()
        self.favorites: Dict[str, Any] = {}
        self.alarms: Dict[str, Any] = {}
        self.topology_condition: asyncio.Condition = asyncio.Condition()
        self.hosts_heartbeat: Optional[CALLBACK_TYPE] = None
        self.discovery_known: Set[str] = set()
        self.boot_counts: Dict[str, int] = {}
        self.mdns_names: Dict[str, str] = {}
        self.entity_id_mappings: Dict[str, Any] = {}
        self.unjoin_data: Dict[str, UnjoinData] = {}


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Sonos component."""
    conf: Optional[dict[str, Any]] = config.get(DOMAIN)
    hass.data[DOMAIN] = conf or {}
    if conf is not None:
        hass.async_create_task(
            hass.config_entries.flow.async_init(DOMAIN, context={'source': config_entries.SOURCE_IMPORT})
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Sonos from a config entry."""
    soco_config.EVENTS_MODULE = events_asyncio
    soco_config.REQUEST_TIMEOUT = 9.5
    soco_config.ZGT_EVENT_FALLBACK = False
    zonegroupstate.EVENT_CACHE_TIMEOUT = SUBSCRIPTION_TIMEOUT
    if DATA_SONOS not in hass.data:
        hass.data[DATA_SONOS] = SonosData()
    data: SonosData = hass.data[DATA_SONOS]
    config: dict[str, Any] = hass.data[DOMAIN].get('media_player', {})
    hosts: list[str] = config.get(CONF_HOSTS, [])
    _LOGGER.debug('Reached async_setup_entry, config=%s', config)
    if (advertise_addr := config.get(CONF_ADVERTISE_ADDR)):
        soco_config.EVENT_ADVERTISE_IP = advertise_addr
    if (deprecated_address := config.get(CONF_INTERFACE_ADDR)):
        _LOGGER.warning(
            "'%s' is deprecated, enable %s in the Network integration (https://www.home-assistant.io/integrations/network/)",
            CONF_INTERFACE_ADDR,
            deprecated_address,
        )
    manager = hass.data[DATA_SONOS_DISCOVERY_MANAGER] = SonosDiscoveryManager(hass, entry, data, hosts)
    await manager.setup_platforms_and_discovery()
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a Sonos config entry."""
    unload_ok: bool = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    await hass.data[DATA_SONOS_DISCOVERY_MANAGER].async_shutdown()
    hass.data.pop(DATA_SONOS)
    hass.data.pop(DATA_SONOS_DISCOVERY_MANAGER)
    return unload_ok


class SonosDiscoveryManager:
    """Manage sonos discovery."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, data: SonosData, hosts: Iterable[str]) -> None:
        """Init discovery manager."""
        self.hass: HomeAssistant = hass
        self.entry: ConfigEntry = entry
        self.data: SonosData = data
        self.hosts: set[str] = set(hosts)
        self.hosts_in_error: Dict[str, bool] = {}
        self.discovery_lock: asyncio.Lock = asyncio.Lock()
        self.creation_lock: asyncio.Lock = asyncio.Lock()
        self._known_invisible: Set[SoCo] = set()
        self._manual_config_required: bool = bool(hosts)

    async def async_shutdown(self) -> None:
        """Stop all running tasks."""
        await self._async_stop_event_listener()
        self._stop_manual_heartbeat()

    def is_device_invisible(self, ip_address: str) -> bool:
        """Check if device at provided IP is known to be invisible."""
        return any((x for x in self._known_invisible if x.ip_address == ip_address))

    async def async_subscribe_to_zone_updates(self, ip_address: str) -> None:
        """Test subscriptions and create SonosSpeakers based on results."""
        soco: SoCo = SoCo(ip_address)
        await self.hass.async_add_executor_job(getattr, soco, 'household_id')
        sub: SubscriptionBase = await soco.zoneGroupTopology.subscribe()

        @callback
        def _async_add_visible_zones(subscription_succeeded: bool = False) -> None:
            """Determine visible zones and create SonosSpeaker instances."""
            zones_to_add: set[Any] = set()
            subscription: Optional[SubscriptionBase] = None
            visible_zones = soco.visible_zones
            self._known_invisible = soco.all_zones - visible_zones
            for zone in visible_zones:
                if zone.uid not in self.data.discovered:
                    zones_to_add.add(zone)
            if not zones_to_add:
                return
            self.hass.async_create_task(
                self.async_add_speakers(zones_to_add, subscription, soco.uid), 
                name="sonos-add-speakers",
            )

        async def async_subscription_failed(now: Any) -> None:
            """Fallback logic if the subscription callback never arrives."""
            addr, port = sub.event_listener.address
            listener_address: str = f'{addr}:{port}'
            if (advertise_ip := soco_config.EVENT_ADVERTISE_IP):
                listener_address += f' (advertising as {advertise_ip})'
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                SUB_FAIL_ISSUE_ID,
                is_fixable=False,
                severity=ir.IssueSeverity.ERROR,
                translation_key='subscriptions_failed',
                translation_placeholders={'device_ip': ip_address, 'listener_address': listener_address, 'sub_fail_url': SUB_FAIL_URL},
            )
            _LOGGER.warning('Subscription to %s failed, attempting to poll directly', ip_address)
            try:
                await sub.unsubscribe()
            except (ClientError, OSError, Timeout) as ex:
                _LOGGER.debug('Unsubscription from %s failed: %s', ip_address, ex)
            try:
                await self.hass.async_add_executor_job(soco.zone_group_state.poll, soco)
            except (OSError, SoCoException, Timeout) as ex:
                _LOGGER.warning('Fallback pollling to %s failed, setup cannot continue: %s', ip_address, ex)
                return
            _LOGGER.debug('Fallback ZoneGroupState poll to %s succeeded', ip_address)
            _async_add_visible_zones()

        cancel_failure_callback: Callable[[Any], Any] = async_call_later(
            self.hass, ZGS_SUBSCRIPTION_TIMEOUT, async_subscription_failed
        )

        @callback
        def _async_subscription_succeeded(event: SonosEvent) -> None:
            """Create SonosSpeakers when subscription callbacks successfully arrive."""
            _LOGGER.debug('Subscription to %s succeeded', ip_address)
            cancel_failure_callback()
            ir.async_delete_issue(self.hass, DOMAIN, SUB_FAIL_ISSUE_ID)
            _async_add_visible_zones(subscription_succeeded=True)

        sub.callback = _async_subscription_succeeded  # type: ignore[assignment]
        await asyncio.sleep(ZGS_SUBSCRIPTION_TIMEOUT * 2)
        try:
            await sub.unsubscribe()
        except ClientError as ex:
            _LOGGER.debug('Cleanup unsubscription from %s was rejected: %s', ip_address, ex)
        except (OSError, Timeout) as ex:
            _LOGGER.error('Cleanup unsubscription from %s failed: %s', ip_address, ex)

    async def _async_stop_event_listener(self, event: Optional[Event] = None) -> None:
        for speaker in self.data.discovered.values():
            speaker.activity_stats.log_report()
            speaker.event_stats.log_report()
        if (zgs := next((speaker.soco.zone_group_state for speaker in self.data.discovered.values()), None)):
            _LOGGER.debug('ZoneGroupState stats: (%s/%s) processed', zgs.processed_count, zgs.total_requests)
        await asyncio.gather(*(create_eager_task(speaker.async_offline()) for speaker in self.data.discovered.values()))
        if events_asyncio.event_listener:
            await events_asyncio.event_listener.async_stop()

    @callback
    def _stop_manual_heartbeat(self, event: Optional[Event] = None) -> None:
        if self.data.hosts_heartbeat:
            self.data.hosts_heartbeat()
            self.data.hosts_heartbeat = None

    async def async_add_speakers(
        self, socos: Iterable[Any], zgs_subscription: Optional[SubscriptionBase], zgs_subscription_uid: str
    ) -> None:
        """Create and set up new SonosSpeaker instances."""

        def _add_speakers() -> None:
            """Add all speakers in a single executor job."""
            for soco in socos:
                if soco.uid in self.data.discovered:
                    continue
                sub: Optional[SubscriptionBase] = None
                if soco.uid == zgs_subscription_uid and zgs_subscription:
                    sub = zgs_subscription
                self._add_speaker(soco, sub)

        async with self.creation_lock:
            await self.hass.async_add_executor_job(_add_speakers)

    def _add_speaker(self, soco: SoCo, zone_group_state_sub: Optional[SubscriptionBase]) -> None:
        """Create and set up a new SonosSpeaker instance."""
        try:
            speaker_info: dict[str, Any] = soco.get_speaker_info(True, timeout=7)
            if soco.uid not in self.data.boot_counts:
                self.data.boot_counts[soco.uid] = soco.boot_seqnum
            _LOGGER.debug('Adding new speaker: %s', speaker_info)
            speaker: SonosSpeaker = SonosSpeaker(self.hass, soco, speaker_info, zone_group_state_sub)
            self.data.discovered[soco.uid] = speaker
            for coordinator, coord_dict in ((SonosAlarms, self.data.alarms), (SonosFavorites, self.data.favorites)):
                if soco.household_id not in coord_dict:
                    new_coordinator = coordinator(self.hass, soco.household_id)
                    new_coordinator.setup(soco)
                    coord_dict[soco.household_id] = new_coordinator
            speaker.setup(self.entry)
        except (OSError, SoCoException, Timeout) as ex:
            _LOGGER.warning('Failed to add SonosSpeaker using %s: %s', soco, ex)

    async def async_poll_manual_hosts(self, now: Any = None) -> None:
        """Add and maintain Sonos devices from a manual configuration."""
        for host in self.hosts.copy():
            ip_addr: str = await self.hass.async_add_executor_job(socket.gethostbyname, host)
            soco: SoCo = SoCo(ip_addr)
            try:
                visible_zones = await self.hass.async_add_executor_job(sync_get_visible_zones, soco)
            except (OSError, SoCoException, Timeout, TimeoutError) as ex:
                if not self.hosts_in_error.get(ip_addr):
                    _LOGGER.warning('Could not get visible Sonos devices from %s: %s', ip_addr, ex)
                    self.hosts_in_error[ip_addr] = True
                else:
                    _LOGGER.debug('Could not get visible Sonos devices from %s: %s', ip_addr, ex)
                continue
            if self.hosts_in_error.pop(ip_addr, None):
                _LOGGER.warning('Connection reestablished to Sonos device %s', ip_addr)
            new_hosts: Set[str] = {x.ip_address for x in visible_zones if x.ip_address not in self.hosts}  # type: ignore[union-attr]
            if new_hosts:
                _LOGGER.debug('Adding to manual hosts: %s', new_hosts)
                self.hosts.update(new_hosts)
            if self.is_device_invisible(ip_addr):
                _LOGGER.debug('Discarding %s from manual hosts', ip_addr)
                self.hosts.discard(ip_addr)
        for host in self.hosts.copy():
            ip_addr = await self.hass.async_add_executor_job(socket.gethostbyname, host)
            soco = SoCo(ip_addr)
            if self.hosts_in_error.get(ip_addr):
                continue
            known_speaker: Optional[SonosSpeaker] = next(
                (speaker for speaker in self.data.discovered.values() if speaker.soco.ip_address == ip_addr), None
            )
            if not known_speaker:
                try:
                    await self._async_handle_discovery_message(soco.uid, ip_addr, 'manual zone scan')
                except (OSError, SoCoException, Timeout, TimeoutError) as ex:
                    _LOGGER.warning('Discovery message failed to %s : %s', ip_addr, ex)
            elif not known_speaker.available:
                try:
                    await self.hass.async_add_executor_job(known_speaker.ping)
                    async_dispatcher_send(self.hass, f'{SONOS_SPEAKER_ACTIVITY}-{soco.uid}', 'manual zone scan')
                except SonosUpdateError:
                    _LOGGER.debug('Manual poll to %s failed, keeping unavailable', ip_addr)
        self.data.hosts_heartbeat = async_call_later(
            self.hass, DISCOVERY_INTERVAL.total_seconds(), self.async_poll_manual_hosts
        )

    async def _async_handle_discovery_message(
        self, uid: str, discovered_ip: str, source: str, boot_seqnum: Optional[int] = None
    ) -> None:
        """Handle discovered player creation and activity."""
        async with self.discovery_lock:
            if not self.data.discovered:
                await self.async_subscribe_to_zone_updates(discovered_ip)
            elif uid not in self.data.discovered:
                if self.is_device_invisible(discovered_ip):
                    return
                await self.async_subscribe_to_zone_updates(discovered_ip)
            elif boot_seqnum and boot_seqnum > self.data.boot_counts[uid]:
                self.data.boot_counts[uid] = boot_seqnum
                async_dispatcher_send(self.hass, f'{SONOS_REBOOTED}-{uid}')
            else:
                async_dispatcher_send(self.hass, f'{SONOS_SPEAKER_ACTIVITY}-{uid}', source)

    @callback
    def _async_ssdp_discovered_player(self, info: SsdpServiceInfo, change: ssdp.SsdpChange) -> None:
        uid: str = info.upnp[ATTR_UPNP_UDN]
        if not uid.startswith('uuid:RINCON_'):
            return
        uid = uid[5:]
        if change == ssdp.SsdpChange.BYEBYE:
            _LOGGER.debug('ssdp:byebye received from %s', info.upnp.get('friendlyName', uid))
            reason: str = info.ssdp_headers.get('X-RINCON-REASON', 'ssdp:byebye')
            async_dispatcher_send(self.hass, f'{SONOS_VANISHED}-{uid}', reason)
            return
        self.async_discovered_player(
            'SSDP',
            info,
            cast(str, urlparse(info.ssdp_location).hostname),
            uid,
            info.ssdp_headers.get('X-RINCON-BOOTSEQ'),
            cast(Optional[str], info.upnp.get(ATTR_UPNP_MODEL_NAME)),
            None,
        )

    @callback
    def async_discovered_player(
        self,
        source: str,
        info: Any,
        discovered_ip: str,
        uid: str,
        boot_seqnum: Optional[str],
        model: Optional[str],
        mdns_name: Optional[str],
    ) -> None:
        """Handle discovery via ssdp or zeroconf."""
        if self._manual_config_required:
            _LOGGER.warning('Automatic discovery is working, Sonos hosts in configuration.yaml are not needed')
            self._manual_config_required = False
        if model in DISCOVERY_IGNORED_MODELS:
            _LOGGER.debug('Ignoring device: %s', info)
            return
        if self.is_device_invisible(discovered_ip):
            return
        if boot_seqnum:
            boot_seqnum_int: int = int(boot_seqnum)
            self.data.boot_counts.setdefault(uid, boot_seqnum_int)
        if mdns_name:
            self.data.mdns_names[uid] = mdns_name
        if uid not in self.data.discovery_known:
            _LOGGER.debug('New %s discovery uid=%s: %s', source, uid, info)
            self.data.discovery_known.add(uid)
        self.entry.async_create_background_task(
            self.hass, self._async_handle_discovery_message(uid, discovered_ip, 'discovery', boot_seqnum=cast(Optional[int], boot_seqnum)), 'sonos-handle_discovery_message'
        )

    async def setup_platforms_and_discovery(self) -> None:
        """Set up platforms and discovery."""
        await self.hass.config_entries.async_forward_entry_setups(self.entry, PLATFORMS)
        self.entry.async_on_unload(self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, self._async_stop_event_listener))
        _LOGGER.debug('Adding discovery job')
        if self.hosts:
            self.entry.async_on_unload(self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, self._stop_manual_heartbeat))
            await self.async_poll_manual_hosts()
        self.entry.async_on_unload(
            await ssdp.async_register_callback(self.hass, self._async_ssdp_discovered_player, {'st': UPNP_ST})
        )
        self.entry.async_on_unload(
            async_track_time_interval(
                self.hass, partial(async_dispatcher_send, self.hass, SONOS_CHECK_ACTIVITY), AVAILABILITY_CHECK_INTERVAL
            )
        )


async def async_remove_config_entry_device(hass: HomeAssistant, config_entry: ConfigEntry, device_entry: dr.DeviceEntry) -> bool:
    """Remove Sonos config entry from a device."""
    known_devices = hass.data[DATA_SONOS].discovered.keys()
    for identifier in device_entry.identifiers:
        if identifier[0] != DOMAIN:
            continue
        uid: str = identifier[1]
        if uid not in known_devices:
            return True
    return False
