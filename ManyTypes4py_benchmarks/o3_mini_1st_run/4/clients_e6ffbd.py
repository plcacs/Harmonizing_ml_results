#!/usr/bin/env python3
import abc
import asyncio
import os
import ssl
from datetime import timedelta
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    cast,
)
from urllib.parse import urlparse
from urllib.request import proxy_bypass
from uuid import UUID

import orjson
from cachetools import TTLCache
from prometheus_client import Counter
from python_socks.async_.asyncio import Proxy
from typing_extensions import Self
from websockets import Subprotocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from websockets.legacy.client import Connect, WebSocketClientProtocol
from prefect.events import Event
from prefect.logging import get_logger
from prefect.settings import (
    PREFECT_API_KEY,
    PREFECT_API_TLS_INSECURE_SKIP_VERIFY,
    PREFECT_API_URL,
    PREFECT_CLOUD_API_URL,
    PREFECT_DEBUG_MODE,
    PREFECT_SERVER_ALLOW_EPHEMERAL_MODE,
)
from prefect.types._datetime import add_years, now

if TYPE_CHECKING:
    from prefect.events.filters import EventFilter

EVENTS_EMITTED: Counter = Counter(
    'prefect_events_emitted',
    'The number of events emitted by Prefect event clients',
    labelnames=['client'],
)
EVENTS_OBSERVED: Counter = Counter(
    'prefect_events_observed',
    'The number of events observed by Prefect event subscribers',
    labelnames=['client'],
)
EVENT_WEBSOCKET_CONNECTIONS: Counter = Counter(
    'prefect_event_websocket_connections',
    'The number of times Prefect event clients have connected to an event stream, broken down by direction (in/out) and connection (initial/reconnect)',
    labelnames=['client', 'direction', 'connection'],
)
EVENT_WEBSOCKET_CHECKPOINTS: Counter = Counter(
    'prefect_event_websocket_checkpoints',
    'The number of checkpoints performed by Prefect event clients',
    labelnames=['client'],
)

if TYPE_CHECKING:
    import logging

logger = get_logger(__name__)


def http_to_ws(url: str) -> str:
    return url.replace('https://', 'wss://').replace('http://', 'ws://').rstrip('/')


def events_in_socket_from_api_url(url: str) -> str:
    return http_to_ws(url) + '/events/in'


def events_out_socket_from_api_url(url: str) -> str:
    return http_to_ws(url) + '/events/out'


class WebsocketProxyConnect(Connect):
    def __init__(self, uri: str, **kwargs: Any) -> None:
        self.uri: str = uri
        self._kwargs: Dict[str, Any] = kwargs
        u = urlparse(uri)
        host: Optional[str] = u.hostname
        if not host:
            raise ValueError(f'Invalid URI {uri}, no hostname found')
        if u.scheme == 'ws':
            port: int = u.port or 80
            proxy_url: Optional[str] = os.environ.get('HTTP_PROXY')
        elif u.scheme == 'wss':
            port = u.port or 443
            proxy_url = os.environ.get('HTTPS_PROXY')
            kwargs['server_hostname'] = host
        else:
            raise ValueError("Unsupported scheme %s. Expected 'ws' or 'wss'. " % u.scheme)
        self._proxy: Optional[Proxy] = Proxy.from_url(proxy_url) if proxy_url and (not proxy_bypass(host)) else None
        self._host: str = host
        self._port: int = port
        if PREFECT_API_TLS_INSECURE_SKIP_VERIFY:
            ctx: ssl.SSLContext = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            self._kwargs.setdefault('ssl', ctx)

    async def _proxy_connect(self) -> WebSocketClientProtocol:
        if self._proxy:
            sock = await self._proxy.connect(dest_host=self._host, dest_port=self._port)
            self._kwargs['sock'] = sock
        super().__init__(self.uri, **self._kwargs)
        proto = await self.__await_impl__()
        return proto

    def __await__(self) -> Generator[Any, None, WebSocketClientProtocol]:
        return self._proxy_connect().__await__()


def websocket_connect(uri: str, **kwargs: Any) -> WebsocketProxyConnect:
    return WebsocketProxyConnect(uri, **kwargs)


def get_events_client(reconnection_attempts: int = 10, checkpoint_every: int = 700) -> "EventsClient":
    api_url: Optional[Any] = PREFECT_API_URL.value()
    if isinstance(api_url, str) and api_url.startswith(PREFECT_CLOUD_API_URL.value()):
        from __main__ import PrefectCloudEventsClient  # type: ignore
        return PrefectCloudEventsClient(reconnection_attempts=reconnection_attempts, checkpoint_every=checkpoint_every)
    elif api_url:
        from __main__ import PrefectEventsClient  # type: ignore
        return PrefectEventsClient(reconnection_attempts=reconnection_attempts, checkpoint_every=checkpoint_every)
    elif PREFECT_SERVER_ALLOW_EPHEMERAL_MODE:
        from prefect.server.api.server import SubprocessASGIServer
        server = SubprocessASGIServer()
        server.start()
        from __main__ import PrefectEventsClient  # type: ignore
        return PrefectEventsClient(api_url=server.api_url, reconnection_attempts=reconnection_attempts, checkpoint_every=checkpoint_every)
    else:
        raise ValueError('No Prefect API URL provided. Please set PREFECT_API_URL to the address of a running Prefect server.')


def get_events_subscriber(filter: Optional[Any] = None, reconnection_attempts: int = 10) -> "PrefectEventSubscriber":
    api_url: Optional[Any] = PREFECT_API_URL.value()
    if isinstance(api_url, str) and api_url.startswith(PREFECT_CLOUD_API_URL.value()):
        from __main__ import PrefectCloudEventSubscriber  # type: ignore
        return PrefectCloudEventSubscriber(filter=filter, reconnection_attempts=reconnection_attempts)
    elif api_url:
        return PrefectEventSubscriber(filter=filter, reconnection_attempts=reconnection_attempts)
    elif PREFECT_SERVER_ALLOW_EPHEMERAL_MODE:
        from prefect.server.api.server import SubprocessASGIServer
        server = SubprocessASGIServer()
        server.start()
        return PrefectEventSubscriber(api_url=server.api_url, filter=filter, reconnection_attempts=reconnection_attempts)
    else:
        raise ValueError('No Prefect API URL provided. Please set PREFECT_API_URL to the address of a running Prefect server.')


class EventsClient(abc.ABC):
    """The abstract interface for all Prefect Events clients"""

    @property
    def client_name(self) -> str:
        return self.__class__.__name__

    async def emit(self, event: Event) -> Any:
        """Emit a single event"""
        if not hasattr(self, '_in_context'):
            raise TypeError('Events may only be emitted while this client is being used as a context manager')
        try:
            return await self._emit(event)
        finally:
            EVENTS_EMITTED.labels(self.client_name).inc()

    @abc.abstractmethod
    async def _emit(self, event: Event) -> None:
        ...

    async def __aenter__(self) -> Self:
        self._in_context = True
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        del self._in_context
        return None


class NullEventsClient(EventsClient):
    """A Prefect Events client implementation that does nothing"""

    async def _emit(self, event: Event) -> None:
        pass


class AssertingEventsClient(EventsClient):
    """A Prefect Events client that records all events sent to it for inspection during tests."""
    last: ClassVar[Optional["AssertingEventsClient"]] = None
    all: ClassVar[List["AssertingEventsClient"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        AssertingEventsClient.last = self
        AssertingEventsClient.all.append(self)
        self.args: Tuple[Any, ...] = args
        self.kwargs: Dict[str, Any] = kwargs

    @classmethod
    def reset(cls) -> None:
        """Reset all captured instances and their events. For use between tests"""
        cls.last = None
        cls.all = []

    def pop_events(self) -> List[Event]:
        events: List[Event] = self.events  # type: ignore[attr-defined]
        self.events = []
        return events

    async def _emit(self, event: Event) -> None:
        self.events.append(event)  # type: ignore[attr-defined]

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        self.events: List[Event] = []
        return self


def _get_api_url_and_key(api_url: Optional[str], api_key: Optional[str]) -> Tuple[str, str]:
    api_url = api_url or PREFECT_API_URL.value()
    api_key = api_key or PREFECT_API_KEY.value()
    if not api_url or not api_key:
        raise ValueError('api_url and api_key must be provided or set in the Prefect configuration')
    return (api_url, api_key)


class PrefectEventsClient(EventsClient):
    """A Prefect Events client that streams events to a Prefect server"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        reconnection_attempts: int = 10,
        checkpoint_every: int = 700,
    ) -> None:
        """
        Args:
            api_url: The base URL for a Prefect server
            reconnection_attempts: When the client is disconnected, how many times
                the client should attempt to reconnect
            checkpoint_every: How often the client should sync with the server to
                confirm receipt of all previously sent events
        """
        api_url = api_url or PREFECT_API_URL.value()
        if not api_url:
            raise ValueError('api_url must be provided or set in the Prefect configuration')
        self._events_socket_url: str = events_in_socket_from_api_url(api_url)
        self._connect: WebsocketProxyConnect = websocket_connect(self._events_socket_url)
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._reconnection_attempts: int = reconnection_attempts
        self._unconfirmed_events: List[Event] = []
        self._checkpoint_every: int = checkpoint_every

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        await self._reconnect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self._websocket = None
        await self._connect.__aexit__(exc_type, exc_val, exc_tb)
        return await super().__aexit__(exc_type, exc_val, exc_tb)

    def _log_debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        message = f'EventsClient(id={id(self)}): ' + message
        logger.debug(message, *args, **kwargs)

    async def _reconnect(self) -> None:
        logger.debug('Reconnecting websocket connection.')
        if self._websocket:
            self._websocket = None
            await self._connect.__aexit__(None, None, None)
            logger.debug('Cleared existing websocket connection.')
        try:
            logger.debug('Opening websocket connection.')
            self._websocket = await self._connect.__aenter__()
            logger.debug('Pinging to ensure websocket connected.')
            assert self._websocket is not None
            pong = await self._websocket.ping()
            await pong
            logger.debug('Pong received. Websocket connected.')
        except Exception as e:
            logger.warning(
                'Unable to connect to %r. Please check your network settings to ensure websocket connections to the API are allowed. Otherwise event data (including task run data) may be lost. Reason: %s. Set PREFECT_DEBUG_MODE=1 to see the full error.',
                self._events_socket_url,
                str(e),
                exc_info=PREFECT_DEBUG_MODE,
            )
            raise
        events_to_resend: List[Event] = self._unconfirmed_events
        logger.debug('Resending %s unconfirmed events.', len(events_to_resend))
        self._unconfirmed_events = []
        for event in events_to_resend:
            await self.emit(event)
        logger.debug('Finished resending unconfirmed events.')

    async def _checkpoint(self, event: Event) -> None:
        assert self._websocket is not None
        self._unconfirmed_events.append(event)
        unconfirmed_count: int = len(self._unconfirmed_events)
        logger.debug('Added event id=%s to unconfirmed events list. There are now %s unconfirmed events.', event.id, unconfirmed_count)
        if unconfirmed_count < self._checkpoint_every:
            return
        logger.debug('Pinging to checkpoint unconfirmed events.')
        pong = await self._websocket.ping()
        await pong
        self._log_debug('Pong received. Events checkpointed.')
        self._unconfirmed_events = self._unconfirmed_events[unconfirmed_count:]
        EVENT_WEBSOCKET_CHECKPOINTS.labels(self.client_name).inc()

    async def _emit(self, event: Event) -> None:
        self._log_debug('Emitting event id=%s.', event.id)
        for i in range(self._reconnection_attempts + 1):
            self._log_debug('Emit reconnection attempt %s.', i)
            try:
                if not self._websocket or i > 0:
                    self._log_debug('Attempting websocket reconnection.')
                    await self._reconnect()
                    assert self._websocket is not None
                self._log_debug('Sending event id=%s.', event.id)
                await self._websocket.send(event.model_dump_json())
                self._log_debug('Checkpointing event id=%s.', event.id)
                await self._checkpoint(event)
                return
            except ConnectionClosed:
                self._log_debug('Got ConnectionClosed error.')
                if i == self._reconnection_attempts:
                    raise
                if i > 2:
                    logger.debug('Sleeping for 1 second before next reconnection attempt.')
                    await asyncio.sleep(1)


class AssertingPassthroughEventsClient(PrefectEventsClient):
    """A Prefect Events client that BOTH records all events sent to it for inspection
    during tests AND sends them to a Prefect server."""
    last: ClassVar[Optional["AssertingPassthroughEventsClient"]] = None
    all: ClassVar[List["AssertingPassthroughEventsClient"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        AssertingPassthroughEventsClient.last = self
        AssertingPassthroughEventsClient.all.append(self)
        self.args: Tuple[Any, ...] = args
        self.kwargs: Dict[str, Any] = kwargs

    @classmethod
    def reset(cls) -> None:
        cls.last = None
        cls.all = []

    def pop_events(self) -> List[Event]:
        events: List[Event] = self.events  # type: ignore[attr-defined]
        self.events = []
        return events

    async def _emit(self, event: Event) -> None:
        await super()._emit(event)
        self.events.append(event)  # type: ignore[attr-defined]

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        self.events: List[Event] = []
        return self


class PrefectCloudEventsClient(PrefectEventsClient):
    """A Prefect Events client that streams events to a Prefect Cloud Workspace"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reconnection_attempts: int = 10,
        checkpoint_every: int = 700,
    ) -> None:
        """
        Args:
            api_url: The base URL for a Prefect Cloud workspace
            api_key: The API of an actor with the manage_events scope
            reconnection_attempts: When the client is disconnected, how many times
                the client should attempt to reconnect
            checkpoint_every: How often the client should sync with the server to
                confirm receipt of all previously sent events
        """
        api_url, api_key = _get_api_url_and_key(api_url, api_key)
        super().__init__(api_url=api_url, reconnection_attempts=reconnection_attempts, checkpoint_every=checkpoint_every)
        self._connect = websocket_connect(
            self._events_socket_url, extra_headers={'Authorization': f'bearer {api_key}'}
        )


SEEN_EVENTS_SIZE: int = 500000
SEEN_EVENTS_TTL: int = 120


class PrefectEventSubscriber(AsyncIterator[Event]):
    """
    Subscribes to a Prefect event stream, yielding events as they occur.

    Example:

        from prefect.events.clients import PrefectEventSubscriber
        from prefect.events.filters import EventFilter, EventNameFilter

        filter = EventFilter(event=EventNameFilter(prefix=["prefect.flow-run."]))

        async with PrefectEventSubscriber(filter=filter) as subscriber:
            async for event in subscriber:
                print(event.occurred, event.resource.id, event.event)

    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        filter: Optional["EventFilter"] = None,
        reconnection_attempts: int = 10,
    ) -> None:
        """
        Args:
            api_url: The base URL for a Prefect Cloud workspace
            reconnection_attempts: When the client is disconnected, how many times
                the client should attempt to reconnect
        """
        self._api_key: Optional[str] = None
        if not api_url:
            api_url = cast(str, PREFECT_API_URL.value())
        from prefect.events.filters import EventFilter  # type: ignore
        self._filter: "EventFilter" = filter or EventFilter()
        self._seen_events: TTLCache[str, bool] = TTLCache(maxsize=SEEN_EVENTS_SIZE, ttl=SEEN_EVENTS_TTL)
        socket_url: str = events_out_socket_from_api_url(api_url)
        logger.debug('Connecting to %s', socket_url)
        self._connect: WebsocketProxyConnect = websocket_connect(socket_url, subprotocols=[Subprotocol('prefect')])
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._reconnection_attempts: int = reconnection_attempts
        if self._reconnection_attempts < 0:
            raise ValueError('reconnection_attempts must be a non-negative integer')

    @property
    def client_name(self) -> str:
        return self.__class__.__name__

    async def __aenter__(self) -> Self:
        try:
            await self._reconnect()
        finally:
            EVENT_WEBSOCKET_CONNECTIONS.labels(self.client_name, 'out', 'initial')
        return self

    async def _reconnect(self) -> None:
        logger.debug('Reconnecting...')
        if self._websocket:
            self._websocket = None
            await self._connect.__aexit__(None, None, None)
        self._websocket = await self._connect.__aenter__()
        logger.debug('  pinging...')
        assert self._websocket is not None
        pong = await self._websocket.ping()
        await pong
        logger.debug('  authenticating...')
        await self._websocket.send(orjson.dumps({'type': 'auth', 'token': self._api_key}).decode())
        try:
            message = orjson.loads(await self._websocket.recv())
            logger.debug('  auth result %s', message)
            assert message['type'] == 'auth_success', message.get('reason', '')
        except AssertionError as e:
            raise Exception(
                f'Unable to authenticate to the event stream. Please ensure the provided api_key you are using is valid for this environment. Reason: {e.args[0]}'
            )
        except ConnectionClosedError as e:
            reason = getattr(e.rcvd, 'reason', None)
            msg = 'Unable to authenticate to the event stream. Please ensure the '
            msg += 'provided api_key you are using is valid for this environment. '
            msg += f'Reason: {reason}' if reason else ''
            raise Exception(msg) from e
        from prefect.events.filters import EventOccurredFilter  # type: ignore
        self._filter.occurred = EventOccurredFilter(
            since=now('UTC') - timedelta(minutes=1), until=add_years(now('UTC'), 1)
        )
        logger.debug('  filtering events since %s...', self._filter.occurred.since)
        filter_message: Dict[str, Any] = {'type': 'filter', 'filter': self._filter.model_dump(mode='json')}
        await self._websocket.send(orjson.dumps(filter_message).decode())

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._websocket = None
        await self._connect.__aexit__(exc_type, exc_val, exc_tb)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        assert self._reconnection_attempts >= 0
        for i in range(self._reconnection_attempts + 1):
            try:
                if not self._websocket or i > 0:
                    try:
                        await self._reconnect()
                    finally:
                        EVENT_WEBSOCKET_CONNECTIONS.labels(self.client_name, 'out', 'reconnect')
                    assert self._websocket is not None
                while True:
                    message = orjson.loads(await self._websocket.recv())
                    event = Event.model_validate(message['event'])
                    if event.id in self._seen_events:
                        continue
                    self._seen_events[event.id] = True
                    try:
                        return event
                    finally:
                        EVENTS_OBSERVED.labels(self.client_name).inc()
            except ConnectionClosedOK:
                logger.debug('Connection closed with "OK" status')
                raise StopAsyncIteration
            except ConnectionClosed:
                logger.debug('Connection closed with %s/%s attempts', i + 1, self._reconnection_attempts)
                if i == self._reconnection_attempts:
                    raise
                if i > 2:
                    await asyncio.sleep(1)
        raise StopAsyncIteration


class PrefectCloudEventSubscriber(PrefectEventSubscriber):
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        filter: Optional["EventFilter"] = None,
        reconnection_attempts: int = 10,
    ) -> None:
        """
        Args:
            api_url: The base URL for a Prefect Cloud workspace
            api_key: The API of an actor with the manage_events scope
            reconnection_attempts: When the client is disconnected, how many times
                the client should attempt to reconnect
        """
        api_url, api_key = _get_api_url_and_key(api_url, api_key)
        super().__init__(api_url=api_url, filter=filter, reconnection_attempts=reconnection_attempts)
        self._api_key = api_key


class PrefectCloudAccountEventSubscriber(PrefectCloudEventSubscriber):
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        filter: Optional["EventFilter"] = None,
        reconnection_attempts: int = 10,
    ) -> None:
        """
        Args:
            api_url: The base URL for a Prefect Cloud workspace
            api_key: The API of an actor with the manage_events scope
            reconnection_attempts: When the client is disconnected, how many times
                the client should attempt to reconnect
        """
        api_url, api_key = _get_api_url_and_key(api_url, api_key)
        account_api_url, _, _ = api_url.partition('/workspaces/')
        super().__init__(api_url=account_api_url, filter=filter, reconnection_attempts=reconnection_attempts)
        self._api_key = api_key
