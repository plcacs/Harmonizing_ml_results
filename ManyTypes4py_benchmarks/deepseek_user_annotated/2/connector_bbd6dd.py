import asyncio
import functools
import logging
import random
import socket
import sys
import traceback
import warnings
from collections import OrderedDict, defaultdict, deque
from contextlib import suppress
from http import HTTPStatus
from itertools import chain, cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Deque,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    FrozenSet,
)

import aiohappyeyeballs

from . import hdrs, helpers
from .abc import AbstractResolver, ResolveResult
from .client_exceptions import (
    ClientConnectionError,
    ClientConnectorCertificateError,
    ClientConnectorDNSError,
    ClientConnectorError,
    ClientConnectorSSLError,
    ClientHttpProxyError,
    ClientProxyConnectionError,
    ServerFingerprintMismatch,
    UnixClientConnectorError,
    cert_errors,
    ssl_errors,
)
from .client_proto import ResponseHandler
from .client_reqrep import SSL_ALLOWED_TYPES, ClientRequest, Fingerprint
from .helpers import (
    _SENTINEL,
    ceil_timeout,
    is_ip_address,
    sentinel,
    set_exception,
    set_result,
)
from .resolver import DefaultResolver

if TYPE_CHECKING:
    import ssl
    from .client import ClientTimeout
    from .client_reqrep import ConnectionKey
    from .tracing import Trace

    SSLContext = ssl.SSLContext
else:
    try:
        import ssl

        SSLContext = ssl.SSLContext
    except ImportError:  # pragma: no cover
        ssl = None  # type: ignore[assignment]
        SSLContext = object  # type: ignore[misc,assignment]

EMPTY_SCHEMA_SET: FrozenSet[str] = frozenset({""})
HTTP_SCHEMA_SET: FrozenSet[str] = frozenset({"http", "https"})
WS_SCHEMA_SET: FrozenSet[str] = frozenset({"ws", "wss"})

HTTP_AND_EMPTY_SCHEMA_SET: FrozenSet[str] = HTTP_SCHEMA_SET | EMPTY_SCHEMA_SET
HIGH_LEVEL_SCHEMA_SET: FrozenSet[str] = HTTP_AND_EMPTY_SCHEMA_SET | WS_SCHEMA_SET

NEEDS_CLEANUP_CLOSED: bool = (3, 13, 0) <= sys.version_info < (
    3,
    13,
    1,
) or sys.version_info < (3, 12, 7)

__all__: Tuple[str, ...] = ("BaseConnector", "TCPConnector", "UnixConnector", "NamedPipeConnector")


class Connection:
    __slots__ = (
        "_key",
        "_connector",
        "_loop",
        "_protocol",
        "_callbacks",
        "_source_traceback",
    )

    def __init__(
        self,
        connector: "BaseConnector",
        key: "ConnectionKey",
        protocol: Optional[ResponseHandler],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._key = key
        self._connector = connector
        self._loop = loop
        self._protocol: Optional[ResponseHandler] = protocol
        self._callbacks: List[Callable[[], None]] = []
        self._source_traceback = (
            traceback.extract_stack(sys._getframe(1)) if loop.get_debug() else None
        )

    def __repr__(self) -> str:
        return f"Connection<{self._key}>"

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._protocol is not None:
            _warnings.warn(
                f"Unclosed connection {self!r}", ResourceWarning, source=self
            )
            if self._loop.is_closed():
                return

            self._connector._release(self._key, self._protocol, should_close=True)

            context = {"client_connection": self, "message": "Unclosed connection"}
            if self._source_traceback is not None:
                context["source_traceback"] = self._source_traceback
            self._loop.call_exception_handler(context)

    def __bool__(self) -> Literal[True]:
        return True

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        if self._protocol is None:
            return None
        return self._protocol.transport

    @property
    def protocol(self) -> Optional[ResponseHandler]:
        return self._protocol

    def add_callback(self, callback: Callable[[], None]) -> None:
        if callback is not None:
            self._callbacks.append(callback)

    def _notify_release(self) -> None:
        callbacks, self._callbacks = self._callbacks[:], []

        for cb in callbacks:
            with suppress(Exception):
                cb()

    def close(self) -> None:
        self._notify_release()

        if self._protocol is not None:
            self._connector._release(self._key, self._protocol, should_close=True)
            self._protocol = None

    def release(self) -> None:
        self._notify_release()

        if self._protocol is not None:
            self._connector._release(self._key, self._protocol)
            self._protocol = None

    @property
    def closed(self) -> bool:
        return self._protocol is None or not self._protocol.is_connected()


class _TransportPlaceholder:
    __slots__ = ("closed",)

    def __init__(self, closed_future: asyncio.Future[Optional[Exception]]) -> None:
        self.closed = closed_future

    def close(self) -> None:
        pass


class BaseConnector:
    _closed: bool = True
    _source_traceback: Optional[traceback.StackSummary] = None
    _cleanup_closed_period: float = 2.0
    allowed_protocol_schema_set: FrozenSet[str] = HIGH_LEVEL_SCHEMA_SET

    def __init__(
        self,
        *,
        keepalive_timeout: Union[_SENTINEL, None, float] = sentinel,
        force_close: bool = False,
        limit: int = 100,
        limit_per_host: int = 0,
        enable_cleanup_closed: bool = False,
        timeout_ceil_threshold: float = 5,
    ) -> None:
        if force_close:
            if keepalive_timeout is not None and keepalive_timeout is not sentinel:
                raise ValueError(
                    "keepalive_timeout cannot be set if force_close is True"
                )
        else:
            if keepalive_timeout is sentinel:
                keepalive_timeout = 15.0

        self._timeout_ceil_threshold: float = timeout_ceil_threshold
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._closed: bool = False
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

        self._conns: DefaultDict[
            "ConnectionKey", Deque[Tuple[ResponseHandler, float]]
        ] = defaultdict(deque)
        self._limit: int = limit
        self._limit_per_host: int = limit_per_host
        self._acquired: Set[ResponseHandler] = set()
        self._acquired_per_host: DefaultDict["ConnectionKey", Set[ResponseHandler]] = (
            defaultdict(set)
        )
        self._keepalive_timeout: float = cast(float, keepalive_timeout)
        self._force_close: bool = force_close

        self._waiters: DefaultDict[
            "ConnectionKey", OrderedDict[asyncio.Future[None], None]
        ] = defaultdict(OrderedDict)

        self._loop: asyncio.AbstractEventLoop = loop
        self._factory: Callable[..., ResponseHandler] = functools.partial(
            ResponseHandler, loop=loop
        )

        self._cleanup_handle: Optional[asyncio.TimerHandle] = None
        self._cleanup_closed_handle: Optional[asyncio.TimerHandle] = None

        if enable_cleanup_closed and not NEEDS_CLEANUP_CLOSED:
            warnings.warn(
                "enable_cleanup_closed ignored because "
                "https://github.com/python/cpython/pull/118960 is fixed "
                f"in Python version {sys.version_info}",
                DeprecationWarning,
                stacklevel=2,
            )
            enable_cleanup_closed = False

        self._cleanup_closed_disabled: bool = not enable_cleanup_closed
        self._cleanup_closed_transports: List[Optional[asyncio.Transport]] = []

        self._placeholder_future: asyncio.Future[Optional[Exception]] = (
            loop.create_future()
        )
        self._placeholder_future.set_result(None)
        self._cleanup_closed()

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._closed:
            return
        if not self._conns:
            return

        conns: List[str] = [repr(c) for c in self._conns.values()]

        self._close_immediately()

        _warnings.warn(f"Unclosed connector {self!r}", ResourceWarning, source=self)
        context: Dict[str, Any] = {
            "connector": self,
            "connections": conns,
            "message": "Unclosed connector",
        }
        if self._source_traceback is not None:
            context["source_traceback"] = self._source_traceback
        self._loop.call_exception_handler(context)

    async def __aenter__(self) -> "BaseConnector":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ) -> None:
        await self.close()

    @property
    def force_close(self) -> bool:
        return self._force_close

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def limit_per_host(self) -> int:
        return self._limit_per_host

    def _cleanup(self) -> None:
        if self._cleanup_handle:
            self._cleanup_handle.cancel()
            self._cleanup_handle = None

        now: float = monotonic()
        timeout: float = self._keepalive_timeout

        if self._conns:
            connections: DefaultDict["ConnectionKey", Deque[Tuple[ResponseHandler, float]]] = defaultdict(deque)
            deadline: float = now - timeout
            for key, conns in self._conns.items():
                alive: Deque[Tuple[ResponseHandler, float]] = deque()
                for proto, use_time in conns:
                    if proto.is_connected() and use_time - deadline >= 0:
                        alive.append((proto, use_time))
                        continue
                    transport = proto.transport
                    proto.close()
                    if not self._cleanup_closed_disabled and key.is_ssl:
                        self._cleanup_closed_transports.append(transport)

                if alive:
                    connections[key] = alive

            self._conns = connections

        if self._conns:
            self._cleanup_handle = helpers.weakref_handle(
                self,
                "_cleanup",
                timeout,
                self._loop,
                timeout_ceil_threshold=self._timeout_ceil_threshold,
            )

    def _cleanup_closed(self) -> None:
        if self._cleanup_closed_handle:
            self._cleanup_closed_handle.cancel()

        for transport in self._cleanup_closed_transports:
            if transport is not None:
                transport.abort()

        self._cleanup_closed_transports = []

        if not self._cleanup_closed_disabled:
            self._cleanup_closed_handle = helpers.weakref_handle(
                self,
                "_cleanup_closed",
                self._cleanup_closed_period,
                self._loop,
                timeout_ceil_threshold=self._timeout_ceil_threshold,
            )

    async def close(self) -> None:
        waiters: List[Awaitable[object]] = self._close_immediately()
        if waiters:
            results: List[Any] = await asyncio.gather(*waiters, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    err_msg = "Error while closing connector: " + repr(res)
                    logging.error(err_msg)

    def _close_immediately(self) -> List[Awaitable[object]]:
        waiters: List[Awaitable[object]] = []

        if self._closed:
            return waiters

        self._closed = True

        try:
            if self._loop.is_closed():
                return waiters

            if self._cleanup_handle:
                self._cleanup_handle.cancel()

            if self._cleanup_closed_handle:
                self._cleanup_closed_handle.cancel()

            for data in self._conns.values():
                for proto, t0 in data:
                    proto.close()
                    waiters.append(proto.closed)

            for proto in self._acquired:
                proto.close()
                waiters.append(proto.closed)

            for transport in self._cleanup_closed_transports:
                if transport is not None:
                    transport.abort()

            return waiters

        finally:
            self._conns.clear()
            self._acquired.clear()
            for keyed_waiters in self._waiters.values():
                for keyed_waiter in keyed_waiters:
                    keyed_waiter.cancel()
            self._waiters.clear()
            self._cleanup_handle = None
            self._cleanup_closed_transports.clear()
            self._cleanup_closed_handle = None

    @property
    def closed(self) -> bool:
        return self._closed

    def _available_connections(self, key: "ConnectionKey") -> int:
        total_remain: int = 1

        if self._limit and (total_remain := self._limit - len(self._acquired)) <= 0:
            return total_remain

        if host_remain := self._limit_per_host:
            if acquired := self._acquired_per_host.get(key):
                host_remain -= len(acquired)
            if total_remain > host_remain:
                return host_remain

        return total_remain

    async def connect(
        self, req: "ClientRequest", traces: List["Trace"], timeout: "ClientTimeout"
    ) -> Connection:
        key = req.connection_key
        if (conn := await self._get(key, traces)) is not None:
            return conn

        async with ceil_timeout(timeout.connect, timeout.ceil_threshold):
            if self._available_connections(key) <= 0:
                await self._wait_for_available_connection(key, traces)
                if (conn := await self._get(key, traces)) is not None:
                    return conn

            placeholder = cast(
                ResponseHandler, _TransportPlaceholder(self._placeholder_future)
            )
            self._acquired.add(placeholder)
            if self._limit_per_host:
                self._acquired_per_host[key].add(placeholder)

            try:
                if traces:
                    for trace in traces:
                        await trace.send_connection_create_start()
                proto = await self._create_connection(req, traces, timeout)
                if traces:
                    for trace in traces:
                        await trace.send_connection_create_end()
            except BaseException:
                self._release_acquired(key, placeholder)
                raise
            else:
                if self._closed:
                    proto.close()
                    raise ClientConnectionError("Connector is closed.")

        self._acquired.remove(placeholder)
        self._acquired.add(proto)
        if self._limit_per_host:
            acquired_per_host = self._acquired_per_host[key]
            acquired_per_host.remove(placeholder)
            acquired_per_host.add(proto)
        return Connection(self, key, proto, self._loop)

    async def _wait_for_available_connection(
        self, key: "ConnectionKey", traces: List["Trace"]
    ) -> None:
        attempts: int = 0
        while True:
            fut: asyncio.Future[None] = self._loop.create_future()
            keyed_waiters = self._waiters[key]
            keyed_waiters[fut] = None
            if attempts:
                keyed_waiters.move_to_end(fut, last=False)

            try:
                if traces:
                    for trace in traces:
                        await trace.send_connection_queued_start()
                await fut
                if traces:
                    for trace in traces:
                        await trace.send_connection_queued_end()
            finally:
                keyed_waiters.pop(fut, None)
                if not self._waiters.get(key, True):
                    del self._waiters[key]

            if self._available_connections(key) > 0:
                break
            attempts += 1

    async def _get(
        self, key: "ConnectionKey", traces: List["Trace"]
    ) -> Optional[Connection]:
        if (conns := self._conns.get(key)) is None:
            return None

        t1: float = monotonic()
        while conns:
            proto, t0 = conns.popleft()
            if proto.is_connected() and t1 - t0 <= self._keepalive_timeout:
                if not conns:
                    del self._conns[key]
                self._acquired.add(proto)
                if self._limit_per_host:
                    self._acquired_per_host[key].add(proto)
                if traces:
                    for trace in traces:
                        try:
                            await trace.send_connection_reuseconn()
                        except BaseException:
                            self._release_acquired(key, proto)
                            raise
                return Connection(self, key, proto, self._loop)

            transport = proto.transport
            proto.close()
            if not self._cleanup_closed_disabled and key.is_ssl:
                self._cleanup_closed_transports.append(transport)

        del self._conns[key]
        return None

   