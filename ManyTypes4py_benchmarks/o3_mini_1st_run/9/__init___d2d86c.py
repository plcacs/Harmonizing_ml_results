from __future__ import annotations
import contextlib
from contextlib import suppress
from fnmatch import translate
from functools import lru_cache, partial
from ipaddress import IPv4Address, IPv6Address
import logging
import re
import sys
from typing import Any, TYPE_CHECKING, cast, Callable, Pattern, Optional, Union, Dict, List, Tuple
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
from homeassistant.helpers.service_info.zeroconf import (
    ATTR_PROPERTIES_ID as _ATTR_PROPERTIES_ID,
    ZeroconfServiceInfo as _ZeroconfServiceInfo,
)
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import HomeKitDiscoveredIntegration, ZeroconfMatcher, async_get_homekit, async_get_zeroconf, bind_hass
from homeassistant.setup import async_when_setup_or_start

from .models import HaAsyncZeroconf, HaZeroconf
from .usage import install_multiple_zeroconf_catcher

_LOGGER: logging.Logger = logging.getLogger(__name__)

DOMAIN: str = 'zeroconf'
ZEROCONF_TYPE: str = '_home-assistant._tcp.local.'
HOMEKIT_TYPES: List[str] = ['_hap._tcp.local.', '_hap._udp.local.']
_HOMEKIT_MODEL_SPLITS: Tuple[Optional[str], str, str] = (None, ' ', '-')
CONF_DEFAULT_INTERFACE: str = 'default_interface'
CONF_IPV6: str = 'ipv6'
DEFAULT_DEFAULT_INTERFACE: bool = True
DEFAULT_IPV6: bool = True
HOMEKIT_PAIRED_STATUS_FLAG: str = 'sf'
HOMEKIT_MODEL_LOWER: str = 'md'
HOMEKIT_MODEL_UPPER: str = 'MD'
MAX_PROPERTY_VALUE_LEN: int = 230
MAX_NAME_LEN: int = 63
ATTR_DOMAIN: str = 'domain'
ATTR_NAME: str = 'name'
ATTR_PROPERTIES: str = 'properties'

_DEPRECATED_ATTR_PROPERTIES_ID: DeprecatedConstant = DeprecatedConstant(
    _ATTR_PROPERTIES_ID,
    'homeassistant.helpers.service_info.zeroconf.ATTR_PROPERTIES_ID',
    '2026.2',
)

CONFIG_SCHEMA: vol.Schema = vol.Schema(
    {
        DOMAIN: vol.All(
            cv.deprecated(CONF_DEFAULT_INTERFACE),
            cv.deprecated(CONF_IPV6),
            vol.Schema(
                {
                    vol.Optional(CONF_DEFAULT_INTERFACE): cv.boolean,
                    vol.Optional(CONF_IPV6, default=DEFAULT_IPV6): cv.boolean,
                }
            ),
        )
    },
    extra=vol.ALLOW_EXTRA,
)

_DEPRECATED_ZeroconfServiceInfo: DeprecatedConstant = DeprecatedConstant(
    _ZeroconfServiceInfo,
    'homeassistant.helpers.service_info.zeroconf.ZeroconfServiceInfo',
    '2026.2',
)


@bind_hass
async def async_get_instance(hass: HomeAssistant) -> HaZeroconf:
    """Get or create the shared HaZeroconf instance."""
    return cast(HaZeroconf, _async_get_instance(hass).zeroconf)


@bind_hass
async def async_get_async_instance(hass: HomeAssistant) -> HaAsyncZeroconf:
    """Get or create the shared HaAsyncZeroconf instance."""
    return _async_get_instance(hass)


@callback
def async_get_async_zeroconf(hass: HomeAssistant) -> HaAsyncZeroconf:
    """Get or create the shared HaAsyncZeroconf instance.

    This method must be run in the event loop, and is an alternative
    to the async_get_async_instance method when a coroutine cannot be used.
    """
    return _async_get_instance(hass)


def _async_get_instance(hass: HomeAssistant) -> HaAsyncZeroconf:
    if DOMAIN in hass.data:
        return cast(HaAsyncZeroconf, hass.data[DOMAIN])
    logging.getLogger('zeroconf').setLevel(logging.NOTSET)
    zeroconf: HaZeroconf = HaZeroconf(**_async_get_zc_args(hass))
    aio_zc: HaAsyncZeroconf = HaAsyncZeroconf(zc=zeroconf)
    install_multiple_zeroconf_catcher(zeroconf)

    async def _async_stop_zeroconf(_event: Event) -> None:
        """Stop Zeroconf."""
        await aio_zc.ha_async_close()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_stop_zeroconf)
    hass.data[DOMAIN] = aio_zc
    return aio_zc


@callback
def _async_zc_has_functional_dual_stack() -> bool:
    """Return true for platforms not supporting IP_ADD_MEMBERSHIP on an AF_INET6 socket.

    Zeroconf only supports a single listen socket at this time.
    """
    return not sys.platform.startswith('freebsd') and (not sys.platform.startswith('darwin'))


def _async_get_zc_args(hass: HomeAssistant) -> Dict[str, Any]:
    """Get zeroconf arguments from config."""
    zc_args: Dict[str, Any] = {'ip_version': IPVersion.V4Only}
    adapters: List[Dict[str, Any]] = network.async_get_loaded_adapters(hass)
    ipv6: bool = False
    if _async_zc_has_functional_dual_stack():
        if any((adapter['enabled'] and adapter['ipv6'] for adapter in adapters)):
            ipv6 = True
            zc_args['ip_version'] = IPVersion.All
    elif not any((adapter['enabled'] and adapter['ipv4'] for adapter in adapters)):
        zc_args['ip_version'] = IPVersion.V6Only
        ipv6 = True
    if not ipv6 and network.async_only_default_interface_enabled(adapters):
        zc_args['interfaces'] = InterfaceChoice.Default
    else:
        zc_args['interfaces'] = [
            str(source_ip)
            for source_ip in network.async_get_enabled_source_ips_from_adapters(adapters)
            if not source_ip.is_loopback
            and (not (isinstance(source_ip, IPv6Address) and source_ip.is_global))
            and (not (isinstance(source_ip, IPv6Address) and zc_args['ip_version'] == IPVersion.V4Only))
            and (not (isinstance(source_ip, IPv4Address) and zc_args['ip_version'] == IPVersion.V6Only))
        ]
    return zc_args


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Zeroconf and make Home Assistant discoverable."""
    aio_zc: HaAsyncZeroconf = _async_get_instance(hass)
    zeroconf: HaZeroconf = cast(HaZeroconf, aio_zc.zeroconf)
    zeroconf_types: Dict[str, List[Dict[str, Any]]] = await async_get_zeroconf(hass)
    homekit_models: Dict[str, HomeKitDiscoveredIntegration] = await async_get_homekit(hass)
    homekit_model_lookup, homekit_model_matchers = _build_homekit_model_lookups(homekit_models)
    discovery = ZeroconfDiscovery(hass, zeroconf, zeroconf_types, homekit_model_lookup, homekit_model_matchers)
    await discovery.async_setup()

    async def _async_zeroconf_hass_start(hass: HomeAssistant, comp: Any) -> None:
        """Expose Home Assistant on zeroconf when it starts.

        Wait till started or otherwise HTTP is not up and running.
        """
        uuid: str = await instance_id.async_get(hass)
        await _async_register_hass_zc_service(hass, aio_zc, uuid)

    async def _async_zeroconf_hass_stop(_event: Event) -> None:
        await discovery.async_stop()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _async_zeroconf_hass_stop)
    async_when_setup_or_start(hass, 'frontend', _async_zeroconf_hass_start)
    return True


def _build_homekit_model_lookups(
    homekit_models: Dict[str, HomeKitDiscoveredIntegration]
) -> Tuple[
    Dict[str, HomeKitDiscoveredIntegration],
    Dict[Pattern[str], HomeKitDiscoveredIntegration],
]:
    """Build lookups for homekit models."""
    homekit_model_lookup: Dict[str, HomeKitDiscoveredIntegration] = {}
    homekit_model_matchers: Dict[Pattern[str], HomeKitDiscoveredIntegration] = {}
    for model, discovery in homekit_models.items():
        if '*' in model or '?' in model or '[' in model:
            homekit_model_matchers[_compile_fnmatch(model)] = discovery
        else:
            homekit_model_lookup[model] = discovery
    return (homekit_model_lookup, homekit_model_matchers)


def _filter_disallowed_characters(name: str) -> str:
    """Filter disallowed characters from a string.

    . is a reversed character for zeroconf.
    """
    return name.replace('.', ' ')


async def _async_register_hass_zc_service(hass: HomeAssistant, aio_zc: HaAsyncZeroconf, uuid: str) -> None:
    valid_location_name: str = _truncate_location_name_to_valid(
        _filter_disallowed_characters(hass.config.location_name or 'Home')
    )
    params: Dict[str, Any] = {
        'location_name': valid_location_name,
        'uuid': uuid,
        'version': __version__,
        'external_url': '',
        'internal_url': '',
        'base_url': '',
        'requires_api_password': True,
    }
    with suppress(NoURLAvailableError):
        params['external_url'] = get_url(hass, allow_internal=False)
    with suppress(NoURLAvailableError):
        params['internal_url'] = get_url(hass, allow_external=False)
    params['base_url'] = params['external_url'] or params['internal_url']
    _suppress_invalid_properties(params)
    info: AsyncServiceInfo = AsyncServiceInfo(
        ZEROCONF_TYPE,
        name=f'{valid_location_name}.{ZEROCONF_TYPE}',
        server=f'{uuid}.local.',
        parsed_addresses=await network.async_get_announce_addresses(hass),
        port=hass.http.server_port,
        properties=params,
    )
    _LOGGER.info('Starting Zeroconf broadcast')
    await aio_zc.async_register_service(info, allow_name_change=True)


def _match_against_props(matcher: Dict[str, Any], props: Dict[str, Any]) -> bool:
    """Check a matcher to ensure all values in props."""
    for key, value in matcher.items():
        prop_val: Optional[Any] = props.get(key)
        if prop_val is None or not _memorized_fnmatch(prop_val.lower(), value):
            return False
    return True


def is_homekit_paired(props: Dict[str, Any]) -> bool:
    """Check properties to see if a device is homekit paired."""
    if HOMEKIT_PAIRED_STATUS_FLAG not in props:
        return False
    with contextlib.suppress(ValueError):
        return int(props[HOMEKIT_PAIRED_STATUS_FLAG]) == 0
    return False


class ZeroconfDiscovery:
    """Discovery via zeroconf."""

    def __init__(
        self,
        hass: HomeAssistant,
        zeroconf: HaZeroconf,
        zeroconf_types: Dict[str, List[Dict[str, Any]]],
        homekit_model_lookups: Dict[str, HomeKitDiscoveredIntegration],
        homekit_model_matchers: Dict[Pattern[str], HomeKitDiscoveredIntegration],
    ) -> None:
        """Init discovery."""
        self.hass: HomeAssistant = hass
        self.zeroconf: HaZeroconf = zeroconf
        self.zeroconf_types: Dict[str, List[Dict[str, Any]]] = zeroconf_types
        self.homekit_model_lookups: Dict[str, HomeKitDiscoveredIntegration] = homekit_model_lookups
        self.homekit_model_matchers: Dict[Pattern[str], HomeKitDiscoveredIntegration] = homekit_model_matchers
        self.async_service_browser: Optional[AsyncServiceBrowser] = None

    async def async_setup(self) -> None:
        """Start discovery."""
        types: List[str] = list(self.zeroconf_types)
        types.extend(
            (
                hk_type
                for hk_type in (ZEROCONF_TYPE, *HOMEKIT_TYPES)
                if hk_type not in self.zeroconf_types
            )
        )
        _LOGGER.debug('Starting Zeroconf browser for: %s', types)
        self.async_service_browser = AsyncServiceBrowser(
            self.zeroconf, types, handlers=[self.async_service_update]
        )
        async_dispatcher_connect(
            self.hass,
            config_entries.signal_discovered_config_entry_removed(DOMAIN),
            self._handle_config_entry_removed,
        )

    async def async_stop(self) -> None:
        """Cancel the service browser and stop processing the queue."""
        if self.async_service_browser:
            await self.async_service_browser.async_cancel()

    @callback
    def _handle_config_entry_removed(self, entry: Any) -> None:
        """Handle config entry changes."""
        for flow in self.hass.config_entries.flow.async_progress_by_init_data_type(
            _ZeroconfServiceInfo, lambda service_info: bool(service_info.name == entry.name)  # type: ignore[attr-defined]
        ):
            self.hass.config_entries.flow.async_abort(flow['flow_id'])
        # Rediscover service if needed.
        for discovery_key in entry.discovery_keys[DOMAIN]:
            if discovery_key.version != 1:
                continue
            _type, name = discovery_key.key  # type: ignore[assignment]
            _LOGGER.debug('Rediscover service %s.%s', _type, name)
            self._async_service_update(self.zeroconf, _type, name)

    @callback
    def _async_dismiss_discoveries(self, name: str) -> None:
        """Dismiss all discoveries for the given name."""
        for flow in self.hass.config_entries.flow.async_progress_by_init_data_type(
            _ZeroconfServiceInfo, lambda service_info: bool(service_info.name == name)
        ):
            self.hass.config_entries.flow.async_abort(flow['flow_id'])

    @callback
    def async_service_update(
        self, zeroconf: HaZeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        """Service state changed."""
        _LOGGER.debug('service_update: type=%s name=%s state_change=%s', service_type, name, state_change)
        if state_change is ServiceStateChange.Removed:
            self._async_dismiss_discoveries(name)
            return
        self._async_service_update(zeroconf, service_type, name)

    @callback
    def _async_service_update(self, zeroconf: HaZeroconf, service_type: str, name: str) -> None:
        """Service state added or changed."""
        try:
            async_service_info: AsyncServiceInfo = AsyncServiceInfo(service_type, name)
        except BadTypeInNameException as ex:
            _LOGGER.debug('Bad name in zeroconf record: %s: %s', name, ex)
            return
        if async_service_info.load_from_cache(zeroconf):
            self._async_process_service_update(async_service_info, service_type, name)
        else:
            self.hass.async_create_background_task(
                self._async_lookup_and_process_service_update(zeroconf, async_service_info, service_type, name),
                name=f'zeroconf lookup {name}.{service_type}',
            )

    async def _async_lookup_and_process_service_update(
        self, zeroconf: HaZeroconf, async_service_info: AsyncServiceInfo, service_type: str, name: str
    ) -> None:
        """Update and process a zeroconf update."""
        await async_service_info.async_request(zeroconf, 3000)
        self._async_process_service_update(async_service_info, service_type, name)

    @callback
    def _async_process_service_update(self, async_service_info: AsyncServiceInfo, service_type: str, name: str) -> None:
        """Process a zeroconf update."""
        info: Optional[_ZeroconfServiceInfo] = info_from_service(async_service_info)
        if not info:
            _LOGGER.debug('Failed to get addresses for device %s', name)
            return
        _LOGGER.debug('Discovered new device %s %s', name, info)
        props: Dict[str, Any] = info.properties
        discovery_key: DiscoveryKey = DiscoveryKey(domain=DOMAIN, key=(info.type, info.name), version=1)
        domain: Optional[str] = None
        if service_type in HOMEKIT_TYPES and (homekit_discovery := async_get_homekit_discovery(self.homekit_model_lookups, self.homekit_model_matchers, props)):
            domain = homekit_discovery.domain
            discovery_flow.async_create_flow(
                self.hass,
                homekit_discovery.domain,
                {'source': config_entries.SOURCE_HOMEKIT},
                info,
                discovery_key=discovery_key,
            )
            if not is_homekit_paired(props) and (not homekit_discovery.always_discover):
                return
        if not (matchers := self.zeroconf_types.get(service_type)):
            return
        for matcher in matchers:
            if len(matcher) > 1:
                if ATTR_NAME in matcher and (not _memorized_fnmatch(info.name.lower(), matcher[ATTR_NAME])):
                    continue
                if ATTR_PROPERTIES in matcher and (not _match_against_props(matcher[ATTR_PROPERTIES], props)):
                    continue
            matcher_domain: str = matcher[ATTR_DOMAIN]
            context: Dict[str, Any] = {'source': config_entries.SOURCE_ZEROCONF}
            if domain:
                context['alternative_domain'] = domain
            discovery_flow.async_create_flow(self.hass, matcher_domain, context, info, discovery_key=discovery_key)


def async_get_homekit_discovery(
    homekit_model_lookups: Dict[str, HomeKitDiscoveredIntegration],
    homekit_model_matchers: Dict[Pattern[str], HomeKitDiscoveredIntegration],
    props: Dict[str, Any],
) -> Optional[HomeKitDiscoveredIntegration]:
    """Handle a HomeKit discovery.

    Return the domain to forward the discovery data to.
    """
    model_val: Optional[Any] = props.get(HOMEKIT_MODEL_LOWER) or props.get(HOMEKIT_MODEL_UPPER)
    if not model_val or not isinstance(model_val, str):
        return None
    model: str = model_val
    for split_str in _HOMEKIT_MODEL_SPLITS:
        key: str = model.split(split_str)[0] if split_str else model
        if (discovery := homekit_model_lookups.get(key)):
            return discovery
    for pattern, discovery in homekit_model_matchers.items():
        if pattern.match(model):
            return discovery
    return None


def info_from_service(service: AsyncServiceInfo) -> Optional[_ZeroconfServiceInfo]:
    """Return prepared info from mDNS entries."""
    maybe_ip_addresses: Optional[List[Union[IPv4Address, IPv6Address]]] = service.ip_addresses_by_version(IPVersion.All)
    if not maybe_ip_addresses:
        return None
    ip_addresses: List[Union[IPv4Address, IPv6Address]] = maybe_ip_addresses  # type: ignore
    ip_address: Optional[Union[IPv4Address, IPv6Address]] = None
    for ip_addr in ip_addresses:
        if not ip_addr.is_link_local and (not ip_addr.is_unspecified):
            ip_address = ip_addr
            break
    if not ip_address:
        return None
    return _ZeroconfServiceInfo(
        ip_address=ip_address,
        ip_addresses=ip_addresses,
        port=service.port,
        hostname=service.server,  # type: ignore
        type=service.type,
        name=service.name,
        properties=service.decoded_properties,
    )


def _suppress_invalid_properties(properties: Dict[str, Any]) -> None:
    """Suppress any properties that will cause zeroconf to fail to startup."""
    for prop, prop_value in properties.items():
        if not isinstance(prop_value, str):
            continue
        if len(prop_value.encode('utf-8')) > MAX_PROPERTY_VALUE_LEN:
            _LOGGER.error(
                "The property '%s' was suppressed because it is longer than the maximum length of %d bytes: %s",
                prop,
                MAX_PROPERTY_VALUE_LEN,
                prop_value,
            )
            properties[prop] = ''


def _truncate_location_name_to_valid(location_name: str) -> str:
    """Truncate or return the location name usable for zeroconf."""
    if len(location_name.encode('utf-8')) < MAX_NAME_LEN:
        return location_name
    _LOGGER.warning(
        'The location name was truncated because it is longer than the maximum length of %d bytes: %s',
        MAX_NAME_LEN,
        location_name,
    )
    return location_name.encode('utf-8')[:MAX_NAME_LEN].decode('utf-8', 'ignore')


@lru_cache(maxsize=4096, typed=True)
def _compile_fnmatch(pattern: str) -> Pattern[str]:
    """Compile a fnmatch pattern."""
    return re.compile(translate(pattern))


@lru_cache(maxsize=1024, typed=True)
def _memorized_fnmatch(name: str, pattern: str) -> bool:
    """Memorized version of fnmatch that has a larger lru_cache.

    The default version of fnmatch only has a lru_cache of 256 entries.
    With many devices we quickly reach that limit and end up compiling
    the same pattern over and over again.

    Zeroconf has its own memorized fnmatch with its own lru_cache
    since the data is going to be relatively the same
    since the devices will not change frequently.
    """
    return bool(_compile_fnmatch(pattern).match(name))


__getattr__ = partial(check_if_deprecated_constant, module_globals=globals())
__dir__ = partial(dir_with_deprecated_constants, module_globals_keys=[*globals().keys()])
__all__ = all_with_deprecated_constants(globals())