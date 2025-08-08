from __future__ import annotations
import logging
import sys
from typing import TYPE_CHECKING, Any, Final, cast
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

_LOGGER: logging.Logger = logging.getLogger(__name__)
DOMAIN: Final[str] = 'zeroconf'
ZEROCONF_TYPE: Final[str] = '_home-assistant._tcp.local.'
HOMEKIT_TYPES: Final[list[str]] = ['_hap._tcp.local.', '_hap._udp.local.']
_HOMEKIT_MODEL_SPLITS: Final[tuple[str, str, str]] = (None, ' ', '-')
CONF_DEFAULT_INTERFACE: Final[str] = 'default_interface'
CONF_IPV6: Final[str] = 'ipv6'
DEFAULT_DEFAULT_INTERFACE: Final[bool] = True
DEFAULT_IPV6: Final[bool] = True
HOMEKIT_PAIRED_STATUS_FLAG: Final[str] = 'sf'
HOMEKIT_MODEL_LOWER: Final[str] = 'md'
HOMEKIT_MODEL_UPPER: Final[str] = 'MD'
MAX_PROPERTY_VALUE_LEN: Final[int] = 230
MAX_NAME_LEN: Final[int] = 63
ATTR_DOMAIN: Final[str] = 'domain'
ATTR_NAME: Final[str] = 'name'
ATTR_PROPERTIES: Final[str] = 'properties'
_DEPRECATED_ATTR_PROPERTIES_ID: Final[DeprecatedConstant] = DeprecatedConstant(_ATTR_PROPERTIES_ID, 'homeassistant.helpers.service_info.zeroconf.ATTR_PROPERTIES_ID', '2026.2')
_CONFIG_SCHEMA: Final[vol.Schema] = vol.Schema({
    DOMAIN: vol.All(
        cv.deprecated(CONF_DEFAULT_INTERFACE),
        cv.deprecated(CONF_IPV6),
        vol.Schema({
            vol.Optional(CONF_DEFAULT_INTERFACE): cv.boolean,
            vol.Optional(CONF_IPV6, default=DEFAULT_IPV6): cv.boolean
        })
    },
    extra=vol.ALLOW_EXTRA
)
_DEPRECATED_ZeroconfServiceInfo: Final[DeprecatedConstant] = DeprecatedConstant(_ZeroconfServiceInfo, 'homeassistant.helpers.service_info.zeroconf.ZeroconfServiceInfo', '2026.2')
