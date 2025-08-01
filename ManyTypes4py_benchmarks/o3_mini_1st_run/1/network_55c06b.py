from __future__ import annotations
from collections.abc import Callable
from contextlib import suppress
from ipaddress import ip_address
from typing import Optional
from aiohttp import hdrs
from hass_nabucasa import remote
import yarl
from homeassistant.components import http
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.loader import bind_hass
from homeassistant.util.network import is_ip_address, is_loopback, normalize_url
from .hassio import is_hassio

TYPE_URL_INTERNAL: str = 'internal_url'
TYPE_URL_EXTERNAL: str = 'external_url'
SUPERVISOR_NETWORK_HOST: str = 'homeassistant'


class NoURLAvailableError(HomeAssistantError):
    """An URL to the Home Assistant instance is not available."""


@bind_hass
def is_internal_request(hass: HomeAssistant) -> bool:
    """Test if the current request is internal."""
    try:
        get_url(hass, allow_external=False, allow_cloud=False, require_current_request=True)
    except NoURLAvailableError:
        return False
    return True


@bind_hass
def get_supervisor_network_url(hass: HomeAssistant, *, allow_ssl: bool = False) -> Optional[str]:
    """Get URL for home assistant within supervisor network."""
    if hass.config.api is None or not is_hassio(hass):
        return None
    scheme: str = 'http'
    if hass.config.api.use_ssl:
        if not allow_ssl:
            return None
        scheme = 'https'
    return str(yarl.URL.build(scheme=scheme, host=SUPERVISOR_NETWORK_HOST, port=hass.config.api.port))


def is_hass_url(hass: HomeAssistant, url: str) -> bool:
    """Return if the URL points at this Home Assistant instance."""
    parsed: yarl.URL = yarl.URL(url)
    if not parsed.is_absolute():
        return False
    if parsed.is_default_port():
        parsed = parsed.with_port(None)

    def host_ip() -> Optional[str]:
        if hass.config.api is None or is_loopback(ip_address(hass.config.api.local_ip)):
            return None
        return str(yarl.URL.build(scheme='http', host=hass.config.api.local_ip, port=hass.config.api.port))

    def cloud_url() -> Optional[str]:
        try:
            return _get_cloud_url(hass)
        except NoURLAvailableError:
            return None

    potential_base_factories: tuple[Callable[[], Optional[str]], ...] = (
        lambda: hass.config.internal_url,
        lambda: hass.config.external_url,
        cloud_url,
        host_ip,
        lambda: get_supervisor_network_url(hass, allow_ssl=True)
    )
    for potential_base_factory in potential_base_factories:
        potential_base: Optional[str] = potential_base_factory()
        if potential_base is None:
            continue
        potential_parsed: yarl.URL = yarl.URL(normalize_url(potential_base))
        if parsed.scheme == potential_parsed.scheme and parsed.authority == potential_parsed.authority:
            return True
    return False


@bind_hass
def get_url(
    hass: HomeAssistant,
    *,
    require_current_request: bool = False,
    require_ssl: bool = False,
    require_standard_port: bool = False,
    require_cloud: bool = False,
    allow_internal: bool = True,
    allow_external: bool = True,
    allow_cloud: bool = True,
    allow_ip: Optional[bool] = None,
    prefer_external: Optional[bool] = None,
    prefer_cloud: bool = False
) -> str:
    """Get a URL to this instance."""
    if require_current_request and http.current_request.get() is None:
        raise NoURLAvailableError
    if prefer_external is None:
        prefer_external = hass.config.api is not None and hass.config.api.use_ssl
    if allow_ip is None:
        allow_ip = hass.config.api is None or not hass.config.api.use_ssl
    order: list[str] = [TYPE_URL_INTERNAL, TYPE_URL_EXTERNAL]
    if prefer_external:
        order.reverse()
    for url_type in order:
        if allow_internal and url_type == TYPE_URL_INTERNAL and (not require_cloud):
            with suppress(NoURLAvailableError):
                return _get_internal_url(
                    hass,
                    allow_ip=allow_ip,
                    require_current_request=require_current_request,
                    require_ssl=require_ssl,
                    require_standard_port=require_standard_port
                )
        if require_cloud or (allow_external and url_type == TYPE_URL_EXTERNAL):
            with suppress(NoURLAvailableError):
                return _get_external_url(
                    hass,
                    allow_cloud=allow_cloud,
                    allow_ip=allow_ip,
                    prefer_cloud=prefer_cloud,
                    require_current_request=require_current_request,
                    require_ssl=require_ssl,
                    require_standard_port=require_standard_port,
                    require_cloud=require_cloud,
                )
            if require_cloud:
                raise NoURLAvailableError
    request_host: Optional[str] = _get_request_host()
    if require_current_request and request_host is not None and (hass.config.api is not None):
        scheme: str = 'https' if hass.config.api.use_ssl else 'http'
        current_url: yarl.URL = yarl.URL.build(scheme=scheme, host=request_host, port=hass.config.api.port)
        known_hostnames: list[str] = ['localhost']
        if is_hassio(hass):
            from homeassistant.components.hassio import get_host_info
            if (host_info := get_host_info(hass)):
                known_hostnames.extend([host_info['hostname'], f"{host_info['hostname']}.local"])
        if (
            (allow_ip and is_ip_address(request_host) and is_loopback(ip_address(request_host)))
            or (request_host in known_hostnames)
        ) and (not require_ssl or current_url.scheme == 'https') and (not require_standard_port or current_url.is_default_port()):
            return normalize_url(str(current_url))
    raise NoURLAvailableError


def _get_request_host() -> Optional[str]:
    """Get the host address of the current request."""
    request = http.current_request.get()
    if request is None:
        raise NoURLAvailableError
    host: Optional[str] = request.headers.get(hdrs.HOST)
    if host is None:
        return None
    if '[' in host:
        return host.partition('[')[2].partition(']')[0]
    if ':' in host:
        host = host.partition(':')[0]
    return host


@bind_hass
def _get_internal_url(
    hass: HomeAssistant,
    *,
    allow_ip: bool = True,
    require_current_request: bool = False,
    require_ssl: bool = False,
    require_standard_port: bool = False,
) -> str:
    """Get internal URL of this instance."""
    if hass.config.internal_url:
        internal_url: yarl.URL = yarl.URL(hass.config.internal_url)
        if (
            (not require_current_request or internal_url.host == _get_request_host())
            and (not require_ssl or internal_url.scheme == 'https')
            and (not require_standard_port or internal_url.is_default_port())
            and (allow_ip or not is_ip_address(str(internal_url.host)))
        ):
            return normalize_url(str(internal_url))
    if allow_ip and (not (require_ssl or hass.config.api is None or hass.config.api.use_ssl)):
        ip_url: yarl.URL = yarl.URL.build(scheme='http', host=hass.config.api.local_ip, port=hass.config.api.port)
        if (
            ip_url.host
            and (not is_loopback(ip_address(ip_url.host)))
            and (not require_current_request or ip_url.host == _get_request_host())
            and (not require_standard_port or ip_url.is_default_port())
        ):
            return normalize_url(str(ip_url))
    raise NoURLAvailableError


@bind_hass
def _get_external_url(
    hass: HomeAssistant,
    *,
    allow_cloud: bool = True,
    allow_ip: bool = True,
    prefer_cloud: bool = False,
    require_current_request: bool = False,
    require_ssl: bool = False,
    require_standard_port: bool = False,
    require_cloud: bool = False,
) -> str:
    """Get external URL of this instance."""
    if require_cloud:
        return _get_cloud_url(hass, require_current_request=require_current_request)
    if prefer_cloud and allow_cloud:
        with suppress(NoURLAvailableError):
            return _get_cloud_url(hass)
    if hass.config.external_url:
        external_url: yarl.URL = yarl.URL(hass.config.external_url)
        if (
            (allow_ip or not is_ip_address(str(external_url.host)))
            and (not require_current_request or external_url.host == _get_request_host())
            and (not require_standard_port or external_url.is_default_port())
            and (not require_ssl or (external_url.scheme == 'https' and (not is_ip_address(str(external_url.host)))))
        ):
            return normalize_url(str(external_url))
    if allow_cloud:
        with suppress(NoURLAvailableError):
            return _get_cloud_url(hass, require_current_request=require_current_request)
    raise NoURLAvailableError


@bind_hass
def _get_cloud_url(hass: HomeAssistant, require_current_request: bool = False) -> str:
    """Get external Home Assistant Cloud URL of this instance."""
    if 'cloud' in hass.config.components:
        from homeassistant.components.cloud import CloudNotAvailable, async_remote_ui_url
        try:
            cloud_url: yarl.URL = yarl.URL(async_remote_ui_url(hass))
        except CloudNotAvailable as err:
            raise NoURLAvailableError from err
        if not require_current_request or cloud_url.host == _get_request_host():
            return normalize_url(str(cloud_url))
    raise NoURLAvailableError


def is_cloud_connection(hass: HomeAssistant) -> bool:
    """Return True if the current connection is a nabucasa cloud connection."""
    if 'cloud' not in hass.config.components:
        return False
    return remote.is_cloud_request.get()