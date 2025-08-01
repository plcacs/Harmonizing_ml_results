"""Helper for aiohttp webclient stuff."""
from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import socket
from ssl import SSLContext
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self, cast, TypeVar, Literal, Optional, Union
import aiohttp
from aiohttp import web
from aiohttp.hdrs import CONTENT_TYPE, USER_AGENT
from aiohttp.web_exceptions import HTTPBadGateway, HTTPGatewayTimeout
from aiohttp_asyncmdnsresolver.api import AsyncDualMDNSResolver
from homeassistant import config_entries
from homeassistant.components import zeroconf
from homeassistant.const import APPLICATION_NAME, EVENT_HOMEASSISTANT_CLOSE, __version__
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.loader import bind_hass
from homeassistant.util import ssl as ssl_util
from homeassistant.util.hass_dict import HassKey
from homeassistant.util.json import json_loads
from .frame import warn_use
from .json import json_dumps
if TYPE_CHECKING:
    from aiohttp.typedefs import JSONDecoder
DATA_CONNECTOR: HassKey = HassKey('aiohttp_connector')
DATA_CLIENTSESSION: HassKey = HassKey('aiohttp_clientsession')
SERVER_SOFTWARE: str = f'{APPLICATION_NAME}/{__version__} aiohttp/{aiohttp.__version__} Python/{sys.version_info[0]}.{sys.version_info[1]}'
ENABLE_CLEANUP_CLOSED: bool = (3, 13, 0) <= sys.version_info < (3, 13, 1) or sys.version_info < (3, 12, 7)
WARN_CLOSE_MSG: str = 'closes the Home Assistant aiohttp session'
MAXIMUM_CONNECTIONS: int = 4096
MAXIMUM_CONNECTIONS_PER_HOST: int = 100

class HassClientResponse(aiohttp.ClientResponse):
    """aiohttp.ClientResponse with a json method that uses json_loads by default."""

    async def json(self, *args: Any, loads: Callable[[str], Any] = json_loads, **kwargs: Any) -> Any:
        """Send a json request and parse the json response."""
        return await super().json(*args, loads=loads, **kwargs)

class ChunkAsyncStreamIterator:
    """Async iterator for chunked streams.

    Based on aiohttp.streams.ChunkTupleAsyncStreamIterator, but yields
    bytes instead of tuple[bytes, bool].
    """
    __slots__ = ('_stream',)

    def __init__(self, stream: Any) -> None:
        """Initialize."""
        self._stream = stream

    def __aiter__(self) -> Self:
        """Iterate."""
        return self

    async def __anext__(self) -> bytes:
        """Yield next chunk."""
        rv = await self._stream.readchunk()
        if rv == (b'', False):
            raise StopAsyncIteration
        return rv[0]

@callback
@bind_hass
def async_get_clientsession(
    hass: HomeAssistant, 
    verify_ssl: bool = True, 
    family: int = socket.AF_UNSPEC, 
    ssl_cipher: ssl_util.SSLCipherList = ssl_util.SSLCipherList.PYTHON_DEFAULT
) -> aiohttp.ClientSession:
    """Return default aiohttp ClientSession.

    This method must be run in the event loop.
    """
    session_key = _make_key(verify_ssl, family, ssl_cipher)
    sessions = hass.data.setdefault(DATA_CLIENTSESSION, {})
    if session_key not in sessions:
        session = _async_create_clientsession(
            hass, 
            verify_ssl, 
            auto_cleanup_method=_async_register_default_clientsession_shutdown, 
            family=family, 
            ssl_cipher=ssl_cipher
        )
        sessions[session_key] = session
    else:
        session = sessions[session_key]
    return session

@callback
@bind_hass
def async_create_clientsession(
    hass: HomeAssistant, 
    verify_ssl: bool = True, 
    auto_cleanup: bool = True, 
    family: int = socket.AF_UNSPEC, 
    ssl_cipher: ssl_util.SSLCipherList = ssl_util.SSLCipherList.PYTHON_DEFAULT, 
    **kwargs: Any
) -> aiohttp.ClientSession:
    """Create a new ClientSession with kwargs, i.e. for cookies.

    If auto_cleanup is False, you need to call detach() after the session
    returned is no longer used. Default is True, the session will be
    automatically detached on homeassistant_stop or when being created
    in config entry setup, the config entry is unloaded.

    This method must be run in the event loop.
    """
    auto_cleanup_method: Optional[Callable[[HomeAssistant, aiohttp.ClientSession], None]] = None
    if auto_cleanup:
        auto_cleanup_method = _async_register_clientsession_shutdown
    return _async_create_clientsession(
        hass, 
        verify_ssl, 
        auto_cleanup_method=auto_cleanup_method, 
        family=family, 
        ssl_cipher=ssl_cipher, 
        **kwargs
    )

@callback
def _async_create_clientsession(
    hass: HomeAssistant, 
    verify_ssl: bool = True, 
    auto_cleanup_method: Optional[Callable[[HomeAssistant, aiohttp.ClientSession], None]] = None, 
    family: int = socket.AF_UNSPEC, 
    ssl_cipher: ssl_util.SSLCipherList = ssl_util.SSLCipherList.PYTHON_DEFAULT, 
    **kwargs: Any
) -> aiohttp.ClientSession:
    """Create a new ClientSession with kwargs, i.e. for cookies."""
    clientsession = aiohttp.ClientSession(
        connector=_async_get_connector(hass, verify_ssl, family, ssl_cipher), 
        json_serialize=json_dumps, 
        response_class=HassClientResponse, 
        **kwargs
    )
    clientsession._default_headers = MappingProxyType({USER_AGENT: SERVER_SOFTWARE})
    clientsession.close = warn_use(clientsession.close, WARN_CLOSE_MSG)
    if auto_cleanup_method:
        auto_cleanup_method(hass, clientsession)
    return clientsession

@bind_hass
async def async_aiohttp_proxy_web(
    hass: HomeAssistant, 
    request: web.Request, 
    web_coro: Awaitable[aiohttp.ClientResponse], 
    buffer_size: int = 102400, 
    timeout: int = 10
) -> Optional[web.StreamResponse]:
    """Stream websession request to aiohttp web response."""
    try:
        async with asyncio.timeout(timeout):
            req = await web_coro
    except asyncio.CancelledError:
        return None
    except TimeoutError as err:
        raise HTTPGatewayTimeout from err
    except aiohttp.ClientError as err:
        raise HTTPBadGateway from err
    try:
        return await async_aiohttp_proxy_stream(
            hass, 
            request, 
            req.content, 
            req.headers.get(CONTENT_TYPE)
        )
    finally:
        req.close()

@bind_hass
async def async_aiohttp_proxy_stream(
    hass: HomeAssistant, 
    request: web.Request, 
    stream: aiohttp.StreamReader, 
    content_type: Optional[str], 
    buffer_size: int = 102400, 
    timeout: int = 10
) -> web.StreamResponse:
    """Stream a stream to aiohttp web response."""
    response = web.StreamResponse()
    if content_type is not None:
        response.content_type = content_type
    await response.prepare(request)
    with suppress(TimeoutError, aiohttp.ClientError):
        while hass.is_running:
            async with asyncio.timeout(timeout):
                data = await stream.read(buffer_size)
            if not data:
                break
            await response.write(data)
    return response

@callback
def _async_register_clientsession_shutdown(hass: HomeAssistant, clientsession: aiohttp.ClientSession) -> None:
    """Register ClientSession close on Home Assistant shutdown or config entry unload.

    This method must be run in the event loop.
    """

    @callback
    def _async_close_websession(*_: Any) -> None:
        """Close websession."""
        clientsession.detach()
    unsub = hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_close_websession)
    if not (config_entry := config_entries.current_entry.get()):
        return
    config_entry.async_on_unload(unsub)
    config_entry.async_on_unload(_async_close_websession)

@callback
def _async_register_default_clientsession_shutdown(hass: HomeAssistant, clientsession: aiohttp.ClientSession) -> None:
    """Register default ClientSession close on Home Assistant shutdown.

    This method must be run in the event loop.
    """

    @callback
    def _async_close_websession(event: Event) -> None:
        """Close websession."""
        clientsession.detach()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_close_websession)

@callback
def _make_key(
    verify_ssl: bool = True, 
    family: int = socket.AF_UNSPEC, 
    ssl_cipher: ssl_util.SSLCipherList = ssl_util.SSLCipherList.PYTHON_DEFAULT
) -> tuple[bool, int, ssl_util.SSLCipherList]:
    """Make a key for connector or session pool."""
    return (verify_ssl, family, ssl_cipher)

class HomeAssistantTCPConnector(aiohttp.TCPConnector):
    """Home Assistant TCP Connector.

    Same as aiohttp.TCPConnector but with a longer cleanup_closed timeout.

    By default the cleanup_closed timeout is 2 seconds. This is too short
    for Home Assistant since we churn through a lot of connections. We set
    it to 60 seconds to reduce the overhead of aborting TLS connections
    that are likely already closed.
    """
    _cleanup_closed_period: float = 60.0

@callback
def _async_get_connector(
    hass: HomeAssistant, 
    verify_ssl: bool = True, 
    family: int = socket.AF_UNSPEC, 
    ssl_cipher: ssl_util.SSLCipherList = ssl_util.SSLCipherList.PYTHON_DEFAULT
) -> aiohttp.TCPConnector:
    """Return the connector pool for aiohttp.

    This method must be run in the event loop.
    """
    connector_key = _make_key(verify_ssl, family, ssl_cipher)
    connectors = hass.data.setdefault(DATA_CONNECTOR, {})
    if connector_key in connectors:
        return connectors[connector_key]
    if verify_ssl:
        ssl_context = ssl_util.client_context(ssl_cipher)
    else:
        ssl_context = ssl_util.client_context_no_verify(ssl_cipher)
    connector = HomeAssistantTCPConnector(
        family=family, 
        enable_cleanup_closed=ENABLE_CLEANUP_CLOSED, 
        ssl=ssl_context, 
        limit=MAXIMUM_CONNECTIONS, 
        limit_per_host=MAXIMUM_CONNECTIONS_PER_HOST, 
        resolver=_async_make_resolver(hass)
    )
    connectors[connector_key] = connector

    async def _async_close_connector(event: Event) -> None:
        """Close connector pool."""
        await connector.close()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_close_connector)
    return connector

@callback
def _async_make_resolver(hass: HomeAssistant) -> AsyncDualMDNSResolver:
    return AsyncDualMDNSResolver(async_zeroconf=zeroconf.async_get_async_zeroconf(hass))
