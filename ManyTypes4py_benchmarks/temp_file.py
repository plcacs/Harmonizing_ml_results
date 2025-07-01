"""Support for exposing Home Assistant via Zeroconf."""
from __future__ import annotations
import contextlib
from contextlib import suppress
from fnmatch import translate
from functools import lru_cache, partial
from ipaddress import IPv4Address, IPv6Address
import logging
import re
import sys
from typing import TYPE_CHECKING, Any, Final, cast, Dict, List, Optional, Set, Tuple, Union
import voluptuous as vol
from zeroconf import BadTypeInNameException, InterfaceChoice, IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo
from homeassistant import config_entries
from homeassistant.components import network
from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE, EVENT_HOMEASSISTANT_STOP, __version__
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, discovery_flow, instance_id
from homeassistant.helpers.deprecation import DeprecatedConstant, all_with_deprecated_constants, check_if_deprecated_constant, dir_with_deprecated_constants
from homeassistant.helpers.discovery_flow import DiscoveryKey
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.network import NoURLAvailableError, get_url
from homeassistant.helpers.service_info.zeroconf import ATTR_PROPERTIES_ID as _ATTR_PROPERTIES_ID, ZeroconfServiceInfo as _ZeroconfServiceInfo
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import HomeKitDiscoveredIntegration, ZeroconfMatcher, async_get_homekit, async_get_zeroconf, bind_hass
from homeassistant.setup import async_when_setup_or_start
from .models import HaAsyncZeroconf, HaZeroconf
from .usage import install_multiple_zeroconf_catcher
_LOGGER: Final = logging.getLogger(__name__)
DOMAIN: Final = 'zeroconf'
ZEROCONF_TYPE: Final = '_home-assistant._tcp.local.'
HOMEKIT_TYPES: Final[List[str]] = ['_hap._tcp.local.', '_hap._udp.local.']
_HOMEKIT_MODEL_SPLITS: Final[Tuple[Optional[str], ...]] = (None, ' ', '-')
CONF_DEFAULT_INTERFACE: Final = 'default_interface'
CONF_IPV6: Final = 'ipv6'
DEFAULT_DEFAULT_INTERFACE: Final = True
DEFAULT_IPV6: Final = True
HOMEKIT_PAIRED_STATUS_FLAG: Final = 'sf'
HOMEKIT_MODEL_LOWER: Final = 'md'
HOMEKIT_MODEL_UPPER: Final = 'MD'
MAX_PROPERTY_VALUE_LEN: Final = 230
MAX_NAME_LEN: Final = 63
ATTR_DOMAIN: Final = 'domain'
ATTR_NAME: Final = 'name'
ATTR_PROPERTIES: Final = 'properties'
_DEPRECATED_ATTR_PROPERTIES_ID: Final = DeprecatedConstant(_ATTR_PROPERTIES_ID,
    'homeassistant.helpers.service_info.zeroconf.ATTR_PROPERTIES_ID', '2026.2')
CONFIG_SCHEMA: Final = vol.Schema({DOMAIN: vol.All(cv.deprecated(
    CONF_DEFAULT_INTERFACE), cv.deprecated(CONF_IPV6), vol.Schema({vol.
    Optional(CONF_DEFAULT_INTERFACE): cv.boolean, vol.Optional(CONF_IPV6,
    default=DEFAULT_IPV6): cv.boolean}))}, extra=vol.ALLOW_EXTRA)
_DEPRECATED_ZeroconfServiceInfo: Final = DeprecatedConstant(
    _ZeroconfServiceInfo,
    'homeassistant.helpers.service_info.zeroconf.ZeroconfServiceInfo', '2026.2'
    )


@bind_hass
async def async_get_instance(hass: HomeAssistant) ->HaZeroconf:
    """Get or create the shared HaZeroconf instance."""
    return cast(HaZeroconf, _async_get_instance(hass).zeroconf)


@bind_hass
async def async_get_async_instance(hass: HomeAssistant) ->HaAsyncZeroconf:
    """Get or create the shared HaAsyncZeroconf instance."""
    return _async_get_instance(hass)


@callback
def async_get_async_zeroconf(hass: HomeAssistant) ->HaAsyncZeroconf:
    """Get or create the shared HaAsyncZeroconf instance.

    This method must be run in the event loop, and is an alternative
    to the async_get_async_instance method when a coroutine cannot be used.
    """
    return _async_get_instance(hass)


def _async_get_instance(hass: HomeAssistant) ->HaAsyncZeroconf:
    if DOMAIN in hass.data:
        return cast(HaAsyncZeroconf, hass.data[DOMAIN])
    logging.getLogger('zeroconf').setLevel(logging.NOTSET)
    zeroconf = HaZeroconf(**_async_get_zc_args(hass))
    aio_zc = HaAsyncZeroconf(zc=zeroconf)
    install_multiple_zeroconf_catcher(zeroconf)

    async def _async_stop_zeroconf(_event: Event) ->None:
        """Stop Zeroconf."""
        await aio_zc.ha_async_close()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_stop_zeroconf)
    hass.data[DOMAIN] = aio_zc
    return aio_zc


@callback
def _async_zc_has_functional_dual_stack() ->bool:
    """Return true for platforms not supporting IP_ADD_MEMBERSHIP on an AF_INET6 socket.

    Zeroconf only supports a single listen socket at this time.
    """
    return not sys.platform.startswith('freebsd'
        ) and not sys.platform.startswith('darwin')


def _async_get_zc_args(hass: HomeAssistant) ->Dict[str, Any]:
    """Get zeroconf arguments from config."""
    zc_args: Dict[str, Any] = {'ip_version': IPVersion.V4Only}
    adapters = network.async_get_loaded_adapters(hass)
    ipv6 = False
    if _async_zc_has_functional_dual_stack():
        if any(adapter['enabled'] and adapter['ipv6'] for adapter in adapters):
            ipv6 = True
            zc_args['ip_version'] = IPVersion.All
    elif not any(adapter['enabled'] and adapter['ipv4'] for adapter in adapters
        ):
        zc_args['ip_version'] = IPVersion.V6Only
        ipv6 = True
    if not ipv6 and network.async_only_default_interface_enabled(adapters):
        zc_args['interfaces'] = InterfaceChoice.Default
    else:
        zc_args['interfaces'] = [str(source_ip) for source_ip in network.
            async_get_enabled_source_ips_from_adapters(adapters) if not
            source_ip.is_loopback and not (isinstance(source_ip,
            IPv6Address) and source_ip.is_global) and not (isinstance(
            source_ip, IPv6Address) and zc_args['ip_version'] == IPVersion.
            V4Only) and not (isinstance(source_ip, IPv4Address) and zc_args
            ['ip_version'] == IPVersion.V6Only)]
    return zc_args


async def async_setup(hass: HomeAssistant, config: ConfigType) ->bool:
    """Set up Zeroconf and make Home Assistant discoverable."""
    aio_zc = _async_get_instance(hass)
    zeroconf = cast(HaZeroconf, aio_zc.zeroconf)
    zeroconf_types = await async_get_zeroconf(hass)
    homekit_models = await async_get_homekit(hass)
    homekit_model_lookup, homekit_model_matchers = (
        _build_homekit_model_lookups(homekit_models))
    discovery = ZeroconfDiscovery(hass, zeroconf, zeroconf_types,
        homekit_model_lookup, homekit_model_matchers)
    await discovery.async_setup()

    async def _async_zeroconf_hass_start(hass: HomeAssistant, comp: str
        ) ->None:
        """Expose Home Assistant on zeroconf when it starts.

        Wait till started or otherwise HTTP is not up and running.
        """
        uuid = await instance_id.async_get(hass)
        await _async_register_hass_zc_service(hass, aio_zc, uuid)

    async def _async_zeroconf_hass_stop(_event: Event) ->None:
        await discovery.async_stop()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP,
        _async_zeroconf_hass_stop)
    async_when_setup_or_start(hass, 'frontend', _async_zeroconf_hass_start)
    return True


def _build_homekit_model_lookups(homekit_models: Dict[str,
    HomeKitDiscoveredIntegration]) ->Tuple[Dict[str,
    HomeKitDiscoveredIntegration], Dict[re.Pattern,
    HomeKitDiscoveredIntegration]]:
    """Build lookups for homekit models."""
    homekit_model_lookup: Dict[str, HomeKitDiscoveredIntegration] = {}
    homekit_model_matchers: Dict[re.Pattern, HomeKitDiscoveredIntegration] = {}
    for model, discovery in homekit_models.items():
        if '*' in model or '?' in model or '[' in model:
            homekit_model_matchers[_compile_fnmatch(model)] = discovery
        else:
            homekit_model_lookup[model] = discovery
    return homekit_model_lookup, homekit_model_matchers


def _filter_disallowed_characters(name: str) ->str:
    """Filter disallowed characters from a string.

    . is a reversed character for zeroconf.
    """
    return name.replace('.', ' ')


async def _async_register_hass_zc_service(hass: HomeAssistant, aio_zc:
    HaAsyncZeroconf, uuid: str) ->None:
    valid_location_name = _truncate_location_name_to_valid(
        _filter_disallowed_characters(hass.config.location_name or 'Home'))
    params: Dict[str, Any] = {'location_name': valid_location_name, 'uuid':
        uuid, 'version': __version__, 'external_url': '', 'internal_url':
        '', 'base_url': '', 'requires_api_password': True}
    with suppress(NoURLAvailableError):
        params['external_url'] = get_url(hass, allow_internal=False)
    with suppress(NoURLAvailableError):
        params['internal_url'] = get_url(hass, allow_external=False)
    params['base_url'] = params['external_url'] or params['internal_url']
    _suppress_invalid_properties(params)
    info = AsyncServiceInfo(ZEROCONF_TYPE, name=
        f'{valid_location_name}.{ZEROCONF_TYPE}', server=f'{uuid}.local.',
        parsed_addresses=await network.async_get_announce_addresses(hass),
        port=hass.http.server_port, properties=params)
    _LOGGER.info('Starting Zeroconf broadcast')
    await aio_zc.async_register_service(info, allow_name_change=True)


def _match_against_props(matcher: Dict[str, str], props: Dict[str, str]
    ) ->bool:
    """Check a matcher to ensure all values in props."""
    for key, value in matcher.items():
        prop_val = props.get(key)
        if prop_val is None or not _memorized_fnmatch(prop_val.lower(), value):
            return False
    return True


def is_homekit_paired(props: Dict[str, str]) ->bool:
    """Check properties to see if a device is homekit paired."""
    if HOMEKIT_PAIRED_STATUS_FLAG not in props:
        return False
    with contextlib.suppress(ValueError):
        return int(props[HOMEKIT_PAIRED_STATUS_FLAG]) == 0
    return False


class ZeroconfDiscovery:
    """Discovery via zeroconf."""

    def __init__(self, hass: HomeAssistant, zeroconf: HaZeroconf,
        zeroconf_types: Dict[str, List[ZeroconfMatcher]],
        homekit_model_lookups: Dict[str, HomeKitDiscoveredIntegration],
        homekit_model_matchers: Dict[re.Pattern, HomeKitDiscoveredIntegration]
        ) ->None:
        """Init discovery."""
        self.hass = hass
        self.zeroconf = zeroconf
        self.zeroconf_types = zeroconf_types
        self.homekit_model_lookups = homekit_model_lookups
        self.homekit_model_matchers = homekit_model_matchers
        self.async_service_browser: Optional[AsyncServiceBrowser] = None

    async def async_setup(self) ->None:
        """Start discovery."""
        types = list(self.zeroconf_types)
        types.extend(hk_type for hk_type in (ZEROCONF_TYPE, *HOMEKIT_TYPES) if
            hk_type not in self.zeroconf_types)
        _LOGGER.debug('Starting Zeroconf browser for: %s', types)
        self.async_service_browser = AsyncServiceBrowser(self.zeroconf,
            types, handlers=[self.async_service_update])
        async_dispatcher_connect(self.hass, config_entries.
            signal_discovered_config_entry_removed(DOMAIN), self.
            _handle_config_entry_removed)

    async def async_stop(self) ->None:
        """Cancel the service browser and stop processing the queue."""
        if self.async_service_browser:
            await self.async_service_browser.async_cancel()

    @callback
    def _handle_config_entry_removed(self, entry: config_entries.ConfigEntry
        ) ->None:
        """Handle config entry changes."""
        for discovery_key in entry.discovery_keys[DOMAIN]:
            if discovery_key.version != 1:
                continue
            _type = discovery_key.key[0]
            name = discovery_key.key[1]
            _LOGGER.debug('Rediscover service %s.%s', _type, name)
            self._async_service_update(self.zeroconf, _type, name)

    def _async_dismiss_discoveries(self, name: str) ->None:
        """Dismiss all discoveries for the given name."""
        for flow in self.hass.config_entries.flow.async_progress_by_init_data_type(
            _ZeroconfServiceInfo, lambda service_info: bool(service_info.
            name == name)):
            self.hass.config_entries.flow.async_abort(flow['flow_id'])

    @callback
    def async_service_update(self, zeroconf: HaZeroconf, service_type: str,
        name: str, state_change: ServiceStateChange) ->None:
        """Service state changed."""
        _LOGGER.debug('service_update: type=%s name=%s state_change=%s',
            service_type, name, state_change)
        if state_change is ServiceStateChange.Removed:
            self._async_dismiss_discoveries(name)
            return
        self._async_service_update(zeroconf, service_type, name)

    @callback
    def _async_service_update(self, zeroconf: HaZeroconf, service_type: str,
        name: str) ->None:
        """Service state added or changed."""
        try:
            async_service_info = AsyncServiceInfo(service_type, name)
        except BadTypeInNameException as ex:
            _LOGGER.debug('Bad name in zeroconf record: %s: %s', name, ex)
            return
        if async_service_info.load_from_cache(zeroconf):
            self._async_process_service_update(async_service_info,
                service_type, name)
        else:
            self.hass.async_create_background_task(self.
                _async_lookup_and_process_service_update(zeroconf,
                async_service_info, service_type, name), name=
                f'zeroconf lookup {name}.{service_type}')

    async def _async_lookup_and_process_service_update(self, zeroconf:
        HaZeroconf, async_service_info: AsyncServiceInfo, service_type: str,
        name: str) ->None:
        """Update and process a zeroconf update."""
        await async_service_info.async_request(zeroconf, 3000)
        self._async_process_service_update(async_service_info, service_type,
            name)

    @callback
    def _async_process_service_update(self, async_service_info:
        AsyncServiceInfo, service_type: str, name) ->None:
        """Process a zeroconf update."""
        info = info_from_service(async_service_info)
        if not info:
            _LOGGER.debug('Failed to get addresses for device %s', name)
            return
        _LOGGER.debug('Discovered new device %s %s', name, info)
        props = info.properties
        discovery_key = DiscoveryKey(domain=DOMAIN, key=(info.type, info.
            name), version=1)
        domain: Optional[str] = None
        if service_type in HOMEKIT_TYPES and (homekit_discovery :=
            async_get_homekit_discovery(self.homekit_model_lookups, self.
            homekit_model_matchers, props)):
            domain = homekit_discovery.domain
            discovery_flow.async_create_flow(self.hass, homekit_discovery.
                domain, {'source': config_entries.SOURCE_HOMEKIT}, info,
                discovery_key=discovery_key)
            if not is_homekit_paired(props
                ) and not homekit_discovery.always_discover:
                return
        if not (matchers := self.zeroconf_types.get(service_type)):
            return
        for matcher in matchers:
            if len(matcher) > 1:
                if ATTR_NAME in matcher and not _memorized_fnmatch(info.
                    name.lower(), matcher[ATTR_NAME]):
                    continue
                if ATTR_PROPERTIES in matcher and not _match_against_props(
                    matcher[ATTR_PROPERTIES], props):
                    continue
            matcher_domain = matcher[ATTR_DOMAIN]
            context = {'source': config_entries.SOURCE_ZEROCONF}
            if domain:
                context['alternative_domain'] = domain
            discovery_flow.async_create_
