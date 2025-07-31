#!/usr/bin/env python3
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
from typing import (Any, Awaitable, Callable, DefaultDict, Deque, Dict, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Type, Union, cast)
import aiohappyeyeballs
from . import hdrs, helpers
from .abc import AbstractResolver, ResolveResult
from .client_exceptions import (ClientConnectionError, ClientConnectorCertificateError, ClientConnectorDNSError,
                                ClientConnectorError, ClientConnectorSSLError, ClientHttpProxyError, ClientProxyConnectionError,
                                ServerFingerprintMismatch, UnixClientConnectorError, cert_errors, ssl_errors)
from .client_proto import ResponseHandler
from .client_reqrep import SSL_ALLOWED_TYPES, ClientRequest, Fingerprint
from .helpers import _SENTINEL, ceil_timeout, is_ip_address, sentinel, set_exception, set_result
from .resolver import DefaultResolver
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ssl
    from .client import ClientTimeout
    from .client_reqrep import ConnectionKey
    from .tracing import Trace
    SSLContext = "ssl.SSLContext"
else:
    try:
        import ssl
        SSLContext = ssl.SSLContext
    except ImportError:
        ssl = None  # type: ignore
        SSLContext = object

EMPTY_SCHEMA_SET = frozenset({''})
HTTP_SCHEMA_SET = frozenset({'http', 'https'})
WS_SCHEMA_SET = frozenset({'ws', 'wss'})
HTTP_AND_EMPTY_SCHEMA_SET = HTTP_SCHEMA_SET | EMPTY_SCHEMA_SET
HIGH_LEVEL_SCHEMA_SET = HTTP_AND_EMPTY_SCHEMA_SET | WS_SCHEMA_SET
NEEDS_CLEANUP_CLOSED = (3, 13, 0) <= sys.version_info < (3, 13, 1) or sys.version_info < (3, 12, 7)
__all__ = ('BaseConnector', 'TCPConnector', 'UnixConnector', 'NamedPipeConnector')


class Connection:
    """Represents a single connection."""
    __slots__ = ('_key', '_connector', '_loop', '_protocol', '_callbacks', '_source_traceback')

    def __init__(self, connector: "BaseConnector", key: Any, protocol: ResponseHandler, loop: asyncio.AbstractEventLoop) -> None:
        self._key = key
        self._connector = connector
        self._loop = loop
        self._protocol = protocol
        self._callbacks: List[Callable[[], None]] = []
        self._source_traceback: Optional[List[traceback.FrameSummary]] = (
            traceback.extract_stack(sys._getframe(1)) if loop.get_debug() else None
        )

    def __repr__(self) -> str:
        return f'Connection<{self._key}>'

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._protocol is not None:
            _warnings.warn(f'Unclosed connection {self!r}', ResourceWarning, source=self)
            if self._loop.is_closed():
                return
            self._connector._release(self._key, self._protocol, should_close=True)
            context: Dict[str, Any] = {'client_connection': self, 'message': 'Unclosed connection'}
            if self._source_traceback is not None:
                context['source_traceback'] = self._source_traceback
            self._loop.call_exception_handler(context)

    def __bool__(self) -> bool:
        """Force subclasses to not be falsy, to make checks simpler."""
        return True

    @property
    def transport(self) -> Optional[asyncio.BaseTransport]:
        if self._protocol is None:
            return None
        return self._protocol.transport  # type: ignore

    @property
    def protocol(self) -> ResponseHandler:
        return self._protocol

    def add_callback(self, callback: Optional[Callable[[], None]]) -> None:
        if callback is not None:
            self._callbacks.append(callback)

    def _notify_release(self) -> None:
        callbacks = self._callbacks[:]
        self._callbacks = []
        for cb in callbacks:
            with suppress(Exception):
                cb()

    def close(self) -> None:
        self._notify_release()
        if self._protocol is not None:
            self._connector._release(self._key, self._protocol, should_close=True)
            self._protocol = None  # type: ignore

    def release(self) -> None:
        self._notify_release()
        if self._protocol is not None:
            self._connector._release(self._key, self._protocol)
            self._protocol = None  # type: ignore

    @property
    def closed(self) -> bool:
        return self._protocol is None or not self._protocol.is_connected()


class _TransportPlaceholder:
    """placeholder for BaseConnector.connect function"""
    __slots__ = ('closed',)

    def __init__(self, closed_future: asyncio.Future[Any]) -> None:
        """Initialize a placeholder for a transport."""
        self.closed: asyncio.Future[Any] = closed_future

    def close(self) -> None:
        """Close the placeholder."""
        pass


class BaseConnector:
    """Base connector class.

    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    enable_cleanup_closed - Enables clean-up closed ssl transports.
                            Disabled by default.
    timeout_ceil_threshold - Trigger ceiling of timeout values when
                             it's above timeout_ceil_threshold.
    loop - Optional event loop.
    """
    _closed: bool = True
    _source_traceback: Optional[List[traceback.FrameSummary]] = None
    _cleanup_closed_period: float = 2.0
    allowed_protocol_schema_set = HIGH_LEVEL_SCHEMA_SET

    def __init__(self, *,
                 keepalive_timeout: Union[float, Any] = sentinel,
                 force_close: bool = False,
                 limit: int = 100,
                 limit_per_host: int = 0,
                 enable_cleanup_closed: bool = False,
                 timeout_ceil_threshold: float = 5) -> None:
        if force_close:
            if keepalive_timeout is not None and keepalive_timeout is not sentinel:
                raise ValueError('keepalive_timeout cannot be set if force_close is True')
        elif keepalive_timeout is sentinel:
            keepalive_timeout = 15.0
        self._timeout_ceil_threshold: float = timeout_ceil_threshold
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._closed = False
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))
        self._conns: DefaultDict[Any, Deque[Tuple[ResponseHandler, float]]] = defaultdict(deque)
        self._limit: int = limit
        self._limit_per_host: int = limit_per_host
        self._acquired: Set[ResponseHandler] = set()
        self._acquired_per_host: DefaultDict[Any, Set[ResponseHandler]] = defaultdict(set)
        self._keepalive_timeout: float = cast(float, keepalive_timeout)
        self._force_close: bool = force_close
        self._waiters: DefaultDict[Any, "OrderedDict[asyncio.Future[Any], None]"] = defaultdict(OrderedDict)
        self._loop: asyncio.AbstractEventLoop = loop
        self._factory: Callable[[], ResponseHandler] = functools.partial(ResponseHandler, loop=loop)
        self._cleanup_handle: Optional[asyncio.Handle] = None
        self._cleanup_closed_handle: Optional[asyncio.Handle] = None
        if enable_cleanup_closed and (not NEEDS_CLEANUP_CLOSED):
            warnings.warn(f'enable_cleanup_closed ignored because https://github.com/python/cpython/pull/118960 is fixed in Python version {sys.version_info}', DeprecationWarning, stacklevel=2)
            enable_cleanup_closed = False
        self._cleanup_closed_disabled: bool = not enable_cleanup_closed
        self._cleanup_closed_transports: List[Any] = []
        self._placeholder_future: asyncio.Future[Any] = loop.create_future()
        self._placeholder_future.set_result(None)
        self._cleanup_closed()

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._closed:
            return
        if not self._conns:
            return
        conns = [repr(c) for c in self._conns.values()]
        self._close_immediately()
        _warnings.warn(f'Unclosed connector {self!r}', ResourceWarning, source=self)
        context: Dict[str, Any] = {'connector': self, 'connections': conns, 'message': 'Unclosed connector'}
        if self._source_traceback is not None:
            context['source_traceback'] = self._source_traceback
        self._loop.call_exception_handler(context)

    async def __aenter__(self) -> "BaseConnector":
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]] = None,
                        exc_value: Optional[BaseException] = None,
                        exc_traceback: Optional[TracebackType] = None) -> None:
        await self.close()

    @property
    def force_close(self) -> bool:
        """Ultimately close connection on releasing if True."""
        return self._force_close

    @property
    def limit(self) -> int:
        """The total number for simultaneous connections.

        If limit is 0 the connector has no limit.
        The default limit size is 100.
        """
        return self._limit

    @property
    def limit_per_host(self) -> int:
        """The limit for simultaneous connections to the same endpoint.

        Endpoints are the same if they are have equal
        (host, port, is_ssl) triple.
        """
        return self._limit_per_host

    def _cleanup(self) -> None:
        """Cleanup unused transports."""
        if self._cleanup_handle:
            self._cleanup_handle.cancel()
            self._cleanup_handle = None
        now: float = monotonic()
        timeout: float = self._keepalive_timeout
        if self._conns:
            connections: DefaultDict[Any, Deque[Tuple[ResponseHandler, float]]] = defaultdict(deque)
            deadline: float = now - timeout
            for key, conns in self._conns.items():
                alive: Deque[Tuple[ResponseHandler, float]] = deque()
                for proto, use_time in conns:
                    if proto.is_connected() and use_time - deadline >= 0:
                        alive.append((proto, use_time))
                        continue
                    transport = proto.transport
                    proto.close()
                    if not self._cleanup_closed_disabled and getattr(key, "is_ssl", False):
                        self._cleanup_closed_transports.append(transport)
                if alive:
                    connections[key] = alive
            self._conns = connections
        if self._conns:
            self._cleanup_handle = helpers.weakref_handle(self, '_cleanup', timeout, self._loop, timeout_ceil_threshold=self._timeout_ceil_threshold)

    def _cleanup_closed(self) -> None:
        """Double confirmation for transport close.

        Some broken ssl servers may leave socket open without proper close.
        """
        if self._cleanup_closed_handle:
            self._cleanup_closed_handle.cancel()
        for transport in self._cleanup_closed_transports:
            if transport is not None:
                transport.abort()
        self._cleanup_closed_transports = []
        if not self._cleanup_closed_disabled:
            self._cleanup_closed_handle = helpers.weakref_handle(self, '_cleanup_closed', self._cleanup_closed_period, self._loop, timeout_ceil_threshold=self._timeout_ceil_threshold)

    async def close(self) -> None:
        """Close all opened transports."""
        waiters: List[asyncio.Future[Any]] = self._close_immediately()
        if waiters:
            results = await asyncio.gather(*waiters, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    err_msg: str = 'Error while closing connector: ' + repr(res)
                    logging.error(err_msg)

    def _close_immediately(self) -> List[asyncio.Future[Any]]:
        waiters: List[asyncio.Future[Any]] = []
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
                    waiters.append(proto.closed)  # type: ignore
            for proto in self._acquired:
                proto.close()
                waiters.append(proto.closed)  # type: ignore
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
        """Is connector closed.

        A readonly property.
        """
        return self._closed

    def _available_connections(self, key: Any) -> int:
        """
        Return number of available connections.

        The limit, limit_per_host and the connection key are taken into account.

        If it returns less than 1 means that there are no connections
        available.
        """
        total_remain: int = 1
        if self._limit and (total_remain := (self._limit - len(self._acquired))) <= 0:
            return total_remain
        if (host_remain := self._limit_per_host):
            if (acquired := self._acquired_per_host.get(key)):
                host_remain -= len(acquired)
            if total_remain > host_remain:
                return host_remain
        return total_remain

    async def connect(self, req: ClientRequest, traces: Optional[List["Trace"]], timeout: "ClientTimeout") -> Connection:
        """Get from pool or create new connection."""
        key: Any = req.connection_key
        maybe_conn: Optional[Connection] = await self._get(key, traces)
        if maybe_conn is not None:
            return maybe_conn
        async with ceil_timeout(timeout.connect, timeout.ceil_threshold):
            if self._available_connections(key) <= 0:
                await self._wait_for_available_connection(key, traces)
                maybe_conn = await self._get(key, traces)
                if maybe_conn is not None:
                    return maybe_conn
            placeholder: ResponseHandler = cast(ResponseHandler, _TransportPlaceholder(self._placeholder_future))
            self._acquired.add(placeholder)
            if self._limit_per_host:
                self._acquired_per_host[key].add(placeholder)
            try:
                if traces:
                    for trace in traces:
                        await trace.send_connection_create_start()
                proto: ResponseHandler = await self._create_connection(req, traces, timeout)
                if traces:
                    for trace in traces:
                        await trace.send_connection_create_end()
            except BaseException:
                self._release_acquired(key, placeholder)
                raise
            else:
                if self._closed:
                    proto.close()
                    raise ClientConnectionError('Connector is closed.')
        self._acquired.remove(placeholder)
        self._acquired.add(proto)
        if self._limit_per_host:
            acquired_per_host = self._acquired_per_host[key]
            acquired_per_host.remove(placeholder)
            acquired_per_host.add(proto)
        return Connection(self, key, proto, self._loop)

    async def _wait_for_available_connection(self, key: Any, traces: Optional[List["Trace"]]) -> None:
        """Wait for an available connection slot."""
        attempts: int = 0
        while True:
            fut: asyncio.Future[Any] = self._loop.create_future()
            keyed_waiters: "OrderedDict[asyncio.Future[Any], None]" = self._waiters[key]
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

    async def _get(self, key: Any, traces: Optional[List["Trace"]]) -> Optional[Connection]:
        """Get next reusable connection for the key or None.

        The connection will be marked as acquired.
        """
        conns: Optional[Deque[Tuple[ResponseHandler, float]]] = self._conns.get(key)
        if conns is None:
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
            if not self._cleanup_closed_disabled and getattr(key, "is_ssl", False):
                self._cleanup_closed_transports.append(transport)
        del self._conns[key]
        return None

    def _release_waiter(self) -> None:
        """
        Iterates over all waiters until one to be released is found.

        The one to be released is not finished and
        belongs to a host that has available connections.
        """
        if not self._waiters:
            return
        queues: List[Any] = list(self._waiters)
        random.shuffle(queues)
        for key in queues:
            if self._available_connections(key) < 1:
                continue
            waiters: "OrderedDict[asyncio.Future[Any], None]" = self._waiters[key]
            while waiters:
                waiter, _ = waiters.popitem(last=False)
                if not waiter.done():
                    waiter.set_result(None)
                    return

    def _release_acquired(self, key: Any, proto: ResponseHandler) -> None:
        """Release acquired connection."""
        if self._closed:
            return
        self._acquired.discard(proto)
        if self._limit_per_host and (conns := self._acquired_per_host.get(key)):
            conns.discard(proto)
            if not conns:
                del self._acquired_per_host[key]
        self._release_waiter()

    def _release(self, key: Any, protocol: ResponseHandler, *, should_close: bool = False) -> None:
        if self._closed:
            return
        self._release_acquired(key, protocol)
        if self._force_close or should_close or getattr(protocol, "should_close", False):
            transport = protocol.transport
            protocol.close()
            if getattr(key, "is_ssl", False) and (not self._cleanup_closed_disabled):
                self._cleanup_closed_transports.append(transport)
            return
        self._conns[key].append((protocol, monotonic()))
        if self._cleanup_handle is None:
            self._cleanup_handle = helpers.weakref_handle(self, '_cleanup', self._keepalive_timeout, self._loop, timeout_ceil_threshold=self._timeout_ceil_threshold)

    async def _create_connection(self, req: ClientRequest, traces: Optional[List["Trace"]], timeout: "ClientTimeout") -> ResponseHandler:
        raise NotImplementedError()


class _DNSCacheTable:
    def __init__(self, ttl: Optional[float] = None) -> None:
        self._addrs_rr: Dict[Any, Tuple[Iterator[Any], int]] = {}
        self._timestamps: Dict[Any, float] = {}
        self._ttl: Optional[float] = ttl

    def __contains__(self, host: Any) -> bool:
        return host in self._addrs_rr

    def add(self, key: Any, addrs: List[Any]) -> None:
        self._addrs_rr[key] = (cycle(addrs), len(addrs))
        if self._ttl is not None:
            self._timestamps[key] = monotonic()

    def remove(self, key: Any) -> None:
        self._addrs_rr.pop(key, None)
        if self._ttl is not None:
            self._timestamps.pop(key, None)

    def clear(self) -> None:
        self._addrs_rr.clear()
        self._timestamps.clear()

    def next_addrs(self, key: Any) -> List[Any]:
        loop, length = self._addrs_rr[key]
        addrs: List[Any] = list(islice(loop, length))
        next(loop)
        return addrs

    def expired(self, key: Any) -> bool:
        if self._ttl is None:
            return False
        return self._timestamps[key] + self._ttl < monotonic()


def _make_ssl_context(verified: bool) -> Optional[SSLContext]:
    """Create SSL context.

    This method is not async-friendly and should be called from a thread
    because it will load certificates from disk and do other blocking I/O.
    """
    if ssl is None:
        return None
    if verified:
        sslcontext: SSLContext = ssl.create_default_context()  # type: ignore
    else:
        sslcontext = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        sslcontext.options |= ssl.OP_NO_SSLv2
        sslcontext.options |= ssl.OP_NO_SSLv3
        sslcontext.check_hostname = False
        sslcontext.verify_mode = ssl.CERT_NONE
        sslcontext.options |= ssl.OP_NO_COMPRESSION
        sslcontext.set_default_verify_paths()
    sslcontext.set_alpn_protocols(('http/1.1',))
    return sslcontext

_SSL_CONTEXT_VERIFIED: Optional[SSLContext] = _make_ssl_context(True)
_SSL_CONTEXT_UNVERIFIED: Optional[SSLContext] = _make_ssl_context(False)


class TCPConnector(BaseConnector):
    """TCP connector.

    verify_ssl - Set to True to check ssl certifications.
    fingerprint - Pass the binary sha256
        digest of the expected certificate in DER format to verify
        that the certificate the server presents matches. See also
        https://en.wikipedia.org/wiki/HTTP_Public_Key_Pinning
    resolver - Enable DNS lookups and use this
        resolver
    use_dns_cache - Use memory cache for DNS lookups.
    ttl_dns_cache - Max seconds having cached a DNS entry, None forever.
    family - socket address family
    local_addr - local tuple of (host, port) to bind socket to

    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    enable_cleanup_closed - Enables clean-up closed ssl transports.
                            Disabled by default.
    happy_eyeballs_delay - This is the “Connection Attempt Delay”
                           as defined in RFC 8305. To disable
                           the happy eyeballs algorithm, set to None.
    interleave - “First Address Family Count” as defined in RFC 8305
    loop - Optional event loop.
    """
    allowed_protocol_schema_set = HIGH_LEVEL_SCHEMA_SET | frozenset({'tcp'})

    def __init__(self, *,
                 use_dns_cache: bool = True,
                 ttl_dns_cache: int = 10,
                 family: socket.AddressFamily = socket.AddressFamily.AF_UNSPEC,
                 ssl: Union[SSLContext, Fingerprint, bool] = True,
                 local_addr: Optional[Tuple[str, int]] = None,
                 resolver: Optional[AbstractResolver] = None,
                 keepalive_timeout: Union[float, Any] = sentinel,
                 force_close: bool = False,
                 limit: int = 100,
                 limit_per_host: int = 0,
                 enable_cleanup_closed: bool = False,
                 timeout_ceil_threshold: float = 5,
                 happy_eyeballs_delay: Optional[float] = 0.25,
                 interleave: Optional[int] = None) -> None:
        super().__init__(keepalive_timeout=keepalive_timeout, force_close=force_close,
                         limit=limit, limit_per_host=limit_per_host,
                         enable_cleanup_closed=enable_cleanup_closed, timeout_ceil_threshold=timeout_ceil_threshold)
        if not isinstance(ssl, SSL_ALLOWED_TYPES):
            raise TypeError('ssl should be SSLContext, Fingerprint, or bool, got {!r} instead.'.format(ssl))
        self._ssl: Union[SSLContext, Fingerprint, bool] = ssl
        if resolver is None:
            resolver = DefaultResolver()
        self._resolver: AbstractResolver = resolver
        self._use_dns_cache: bool = use_dns_cache
        self._cached_hosts: _DNSCacheTable = _DNSCacheTable(ttl=ttl_dns_cache)
        self._throttle_dns_futures: Dict[Tuple[str, int], Set[asyncio.Future[Any]]] = {}
        self._family: socket.AddressFamily = family
        self._local_addr_infos: List[Any] = aiohappyeyeballs.addr_to_addr_infos(local_addr)
        self._happy_eyeballs_delay: Optional[float] = happy_eyeballs_delay
        self._interleave: Optional[int] = interleave
        self._resolve_host_tasks: Set[asyncio.Task[Any]] = set()

    def _close_immediately(self) -> List[asyncio.Future[Any]]:
        for futs in self._throttle_dns_futures.values():
            for fut in futs:
                fut.cancel()
        waiters: List[asyncio.Future[Any]] = super()._close_immediately()
        for t in self._resolve_host_tasks:
            t.cancel()
            waiters.append(t)
        return waiters

    @property
    def family(self) -> socket.AddressFamily:
        """Socket family like AF_INET."""
        return self._family

    @property
    def use_dns_cache(self) -> bool:
        """True if local DNS caching is enabled."""
        return self._use_dns_cache

    def clear_dns_cache(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Remove specified host/port or clear all dns local cache."""
        if host is not None and port is not None:
            self._cached_hosts.remove((host, port))
        elif host is not None or port is not None:
            raise ValueError('either both host and port or none of them are allowed')
        else:
            self._cached_hosts.clear()

    async def _resolve_host(self, host: str, port: int, traces: Optional[List["Trace"]] = None) -> List[Dict[str, Any]]:
        """Resolve host and return list of addresses."""
        if is_ip_address(host):
            return [{'hostname': host, 'host': host, 'port': port, 'family': self._family, 'proto': 0, 'flags': 0}]
        if not self._use_dns_cache:
            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_start(host)
            res: List[Dict[str, Any]] = await self._resolver.resolve(host, port, family=self._family)
            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_end(host)
            return res
        key: Tuple[str, int] = (host, port)
        if key in self._cached_hosts and (not self._cached_hosts.expired(key)):
            result: List[Any] = self._cached_hosts.next_addrs(key)
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_hit(host)
            return result
        if key in self._throttle_dns_futures:
            futures: Set[asyncio.Future[Any]] = self._throttle_dns_futures[key]
            future: asyncio.Future[Any] = self._loop.create_future()
            futures.add(future)
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_hit(host)
            try:
                await future
            finally:
                futures.discard(future)
            return self._cached_hosts.next_addrs(key)
        self._throttle_dns_futures[key] = set()
        futures = self._throttle_dns_futures[key]
        coro: Awaitable[List[Any]] = self._resolve_host_with_throttle(key, host, port, futures, traces)
        loop = asyncio.get_running_loop()
        if sys.version_info >= (3, 12):
            resolved_host_task: asyncio.Task[Any] = asyncio.Task(coro, loop=loop, eager_start=True)
        else:
            resolved_host_task = loop.create_task(coro)
        if not resolved_host_task.done():
            self._resolve_host_tasks.add(resolved_host_task)
            resolved_host_task.add_done_callback(self._resolve_host_tasks.discard)
        try:
            return await asyncio.shield(resolved_host_task)
        except asyncio.CancelledError:
            def drop_exception(fut: asyncio.Future[Any]) -> None:
                with suppress(Exception, asyncio.CancelledError):
                    fut.result()
            resolved_host_task.add_done_callback(drop_exception)
            raise

    async def _resolve_host_with_throttle(self, key: Tuple[str, int], host: str, port: int,
                                          futures: Set[asyncio.Future[Any]],
                                          traces: Optional[List["Trace"]]) -> List[Any]:
        """Resolve host and set result for all waiters.

        This method must be run in a task and shielded from cancellation
        to avoid cancelling the underlying lookup.
        """
        if traces:
            for trace in traces:
                await trace.send_dns_cache_miss(host)
        try:
            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_start(host)
            addrs: List[Dict[str, Any]] = await self._resolver.resolve(host, port, family=self._family)
            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_end(host)
            self._cached_hosts.add(key, addrs)
            for fut in futures:
                set_result(fut, None)
        except BaseException as e:
            for fut in futures:
                set_exception(fut, e)
            raise
        finally:
            self._throttle_dns_futures.pop(key)
        return self._cached_hosts.next_addrs(key)

    async def _create_connection(self, req: ClientRequest, traces: Optional[List["Trace"]],
                                 timeout: "ClientTimeout") -> ResponseHandler:
        """Create connection.

        Has same keyword arguments as BaseEventLoop.create_connection.
        """
        if req.proxy:
            _, proto = await self._create_proxy_connection(req, traces, timeout)
        else:
            _, proto = await self._create_direct_connection(req, traces, timeout)
        return proto

    def _get_ssl_context(self, req: ClientRequest) -> Optional[SSLContext]:
        """Logic to get the correct SSL context

        0. if req.ssl is false, return None

        1. if ssl_context is specified in req, use it
        2. if _ssl_context is specified in self, use it
        3. otherwise:
            1. if verify_ssl is not specified in req, use self.ssl_context
               (will generate a default context according to self.verify_ssl)
            2. if verify_ssl is True in req, generate a default SSL context
            3. if verify_ssl is False in req, generate a SSL context that
               won't verify
        """
        if not req.is_ssl():
            return None
        if ssl is None:
            raise RuntimeError('SSL is not supported.')
        sslcontext = req.ssl
        if isinstance(sslcontext, SSLContext):
            return sslcontext
        if sslcontext is not True:
            return _SSL_CONTEXT_UNVERIFIED
        sslcontext = self._ssl
        if isinstance(sslcontext, SSLContext):
            return sslcontext
        if sslcontext is not True:
            return _SSL_CONTEXT_UNVERIFIED
        return _SSL_CONTEXT_VERIFIED

    def _get_fingerprint(self, req: ClientRequest) -> Optional[Fingerprint]:
        ret: Any = req.ssl
        if isinstance(ret, Fingerprint):
            return ret
        ret = self._ssl
        if isinstance(ret, Fingerprint):
            return ret
        return None

    async def _wrap_create_connection(self, *args: Any, addr_infos: List[Any], req: ClientRequest,
                                        timeout: "ClientTimeout", client_error: Type[ClientConnectorError] = ClientConnectorError,
                                        **kwargs: Any) -> Tuple[asyncio.Transport, ResponseHandler]:
        try:
            async with ceil_timeout(timeout.sock_connect, ceil_threshold=timeout.ceil_threshold):
                sock = await aiohappyeyeballs.start_connection(addr_infos=addr_infos, local_addr_infos=self._local_addr_infos,
                                                               happy_eyeballs_delay=self._happy_eyeballs_delay, interleave=self._interleave,
                                                               loop=self._loop)
                return await self._loop.create_connection(*args, **kwargs, sock=sock)
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc

    def _warn_about_tls_in_tls(self, underlying_transport: asyncio.Transport, req: ClientRequest) -> None:
        """Issue a warning if the requested URL has HTTPS scheme."""
        if req.request_info.url.scheme != 'https':
            return
        asyncio_supports_tls_in_tls = getattr(underlying_transport, '_start_tls_compatible', False)
        if asyncio_supports_tls_in_tls:
            return
        warnings.warn(
            "An HTTPS request is being sent through an HTTPS proxy. This support for TLS in TLS is known to be disabled in the stdlib asyncio. This is why you'll probably see an error in the log below.\n\nIt is possible to enable it via monkeypatching. For more details, see:\n* https://bugs.python.org/issue37179\n* https://github.com/python/cpython/pull/28073\n\nYou can temporarily patch this as follows:\n* https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support\n* https://github.com/aio-libs/aiohttp/discussions/6044\n",
            RuntimeWarning,
            source=self,
            stacklevel=3
        )

    async def _start_tls_connection(self, underlying_transport: asyncio.Transport, req: ClientRequest,
                                    timeout: "ClientTimeout", client_error: Type[ClientConnectorError] = ClientConnectorError
                                    ) -> Tuple[asyncio.Transport, ResponseHandler]:
        """Wrap the raw TCP transport with TLS."""
        tls_proto: ResponseHandler = self._factory()
        sslcontext: Optional[SSLContext] = self._get_ssl_context(req)
        if TYPE_CHECKING:
            assert sslcontext is not None
        try:
            async with ceil_timeout(timeout.sock_connect, ceil_threshold=timeout.ceil_threshold):
                try:
                    tls_transport: asyncio.Transport = await self._loop.start_tls(
                        underlying_transport, tls_proto, sslcontext, server_hostname=req.server_hostname or req.host,
                        ssl_handshake_timeout=timeout.total)
                except BaseException:
                    underlying_transport.close()
                    raise
                if isinstance(tls_transport, asyncio.Transport):
                    fingerprint = self._get_fingerprint(req)
                    if fingerprint:
                        try:
                            fingerprint.check(tls_transport)
                        except ServerFingerprintMismatch:
                            tls_transport.close()
                            if not self._cleanup_closed_disabled:
                                self._cleanup_closed_transports.append(tls_transport)
                            raise
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc
        except TypeError as type_err:
            raise ClientConnectionError(f'Cannot initialize a TLS-in-TLS connection to host {req.host!s}:{req.port:d} through an underlying connection to an HTTPS proxy {req.proxy!s} ssl:{req.ssl or "default"} [{type_err!s}]') from type_err
        else:
            if tls_transport is None:
                msg = 'Failed to start TLS (possibly caused by closing transport)'
                raise client_error(req.connection_key, OSError(msg))
            tls_proto.connection_made(tls_transport)
        return (tls_transport, tls_proto)

    def _convert_hosts_to_addr_infos(self, hosts: List[Dict[str, Any]]) -> List[Tuple[int, int, int, str, Any]]:
        """Converts the list of hosts to a list of addr_infos.

        The list of hosts is the result of a DNS lookup. The list of
        addr_infos is the result of a call to `socket.getaddrinfo()`.
        """
        addr_infos: List[Tuple[int, int, int, str, Any]] = []
        for hinfo in hosts:
            host: str = hinfo['host']
            is_ipv6: bool = ':' in host
            family: int = socket.AF_INET6 if is_ipv6 else socket.AF_INET
            if self._family and self._family != family:
                continue
            addr = (host, hinfo['port'], 0, 0) if is_ipv6 else (host, hinfo['port'])
            addr_infos.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', addr))
        return addr_infos

    async def _create_direct_connection(self, req: ClientRequest, traces: Optional[List["Trace"]],
                                          timeout: "ClientTimeout", *,
                                          client_error: Type[ClientConnectorError] = ClientConnectorError
                                          ) -> Tuple[asyncio.Transport, ResponseHandler]:
        sslcontext: Optional[SSLContext] = self._get_ssl_context(req)
        fingerprint: Optional[Fingerprint] = self._get_fingerprint(req)
        host: str = req.url.raw_host  # type: ignore
        assert host is not None
        if host.endswith('..'):
            host = host.rstrip('.') + '.'
        port: int = req.port  # type: ignore
        assert port is not None
        try:
            hosts: List[Dict[str, Any]] = await self._resolve_host(host, port, traces=traces)
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise ClientConnectorDNSError(req.connection_key, exc) from exc
        last_exc: Optional[BaseException] = None
        addr_infos: List[Any] = self._convert_hosts_to_addr_infos(hosts)
        while addr_infos:
            server_hostname: Optional[str] = (req.server_hostname or host).rstrip('.') if sslcontext else None
            try:
                transp, proto = await self._wrap_create_connection(self._factory, timeout=timeout, ssl=sslcontext,
                                                                     addr_infos=addr_infos, server_hostname=server_hostname,
                                                                     req=req, client_error=client_error)
            except (ClientConnectorError, asyncio.TimeoutError) as exc:
                last_exc = exc
                aiohappyeyeballs.pop_addr_infos_interleave(addr_infos, self._interleave)
                continue
            if req.is_ssl() and fingerprint:
                try:
                    fingerprint.check(transp)
                except ServerFingerprintMismatch as exc:
                    transp.close()
                    if not self._cleanup_closed_disabled:
                        self._cleanup_closed_transports.append(transp)
                    last_exc = exc
                    sock = transp.get_extra_info('socket')
                    bad_peer = sock.getpeername()
                    aiohappyeyeballs.remove_addr_infos(addr_infos, bad_peer)
                    continue
            return (transp, proto)
        assert last_exc is not None
        raise last_exc

    async def _create_proxy_connection(self, req: ClientRequest, traces: Optional[List["Trace"]],
                                         timeout: "ClientTimeout") -> Tuple[asyncio.Transport, ResponseHandler]:
        headers: Dict[str, str] = {}
        if req.proxy_headers is not None:
            headers = req.proxy_headers
        headers[hdrs.HOST] = req.headers[hdrs.HOST]
        url = req.proxy
        assert url is not None
        proxy_req = ClientRequest(hdrs.METH_GET, url, headers=headers, auth=req.proxy_auth, loop=self._loop, ssl=req.ssl)
        transport, proto = await self._create_direct_connection(proxy_req, [], timeout, client_error=ClientProxyConnectionError)
        auth = proxy_req.headers.pop(hdrs.AUTHORIZATION, None)
        if auth is not None:
            if not req.is_ssl():
                req.headers[hdrs.PROXY_AUTHORIZATION] = auth
            else:
                proxy_req.headers[hdrs.PROXY_AUTHORIZATION] = auth
        if req.is_ssl():
            self._warn_about_tls_in_tls(transport, req)
            proxy_req.method = hdrs.METH_CONNECT
            proxy_req.url = req.url
            key = req.connection_key._replace(proxy=None, proxy_auth=None, proxy_headers_hash=None)
            conn = Connection(self, key, proto, self._loop)
            proxy_resp = await proxy_req.send(conn)
            try:
                protocol = conn._protocol
                assert protocol is not None
                protocol.set_response_params(read_until_eof=True, timeout_ceil_threshold=self._timeout_ceil_threshold)
                resp = await proxy_resp.start(conn)
            except BaseException:
                proxy_resp.close()
                conn.close()
                raise
            else:
                conn._protocol = None  # type: ignore
                try:
                    if resp.status != 200:
                        message = resp.reason
                        if message is None:
                            message = HTTPStatus(resp.status).phrase
                        raise ClientHttpProxyError(proxy_resp.request_info, resp.history, status=resp.status, message=message, headers=resp.headers)
                except BaseException:
                    transport.close()
                    raise
                return await self._start_tls_connection(transport, req=req, timeout=timeout)
            finally:
                proxy_resp.close()
        return (transport, proto)


class UnixConnector(BaseConnector):
    """Unix socket connector.

    path - Unix socket path.
    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    loop - Optional event loop.
    """
    allowed_protocol_schema_set = HIGH_LEVEL_SCHEMA_SET | frozenset({'unix'})

    def __init__(self, path: str, force_close: bool = False,
                 keepalive_timeout: Union[float, Any] = sentinel, limit: int = 100, limit_per_host: int = 0) -> None:
        super().__init__(force_close=force_close, keepalive_timeout=keepalive_timeout,
                         limit=limit, limit_per_host=limit_per_host)
        self._path: str = path

    @property
    def path(self) -> str:
        """Path to unix socket."""
        return self._path

    async def _create_connection(self, req: ClientRequest, traces: Optional[List["Trace"]],
                                 timeout: "ClientTimeout") -> ResponseHandler:
        try:
            async with ceil_timeout(timeout.sock_connect, ceil_threshold=timeout.ceil_threshold):
                _, proto = await self._loop.create_unix_connection(self._factory, self._path)
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise UnixClientConnectorError(self.path, req.connection_key, exc) from exc
        return proto


class NamedPipeConnector(BaseConnector):
    """Named pipe connector.

    Only supported by the proactor event loop.
    See also: https://docs.python.org/3/library/asyncio-eventloop.html

    path - Windows named pipe path.
    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    loop - Optional event loop.
    """
    allowed_protocol_schema_set = HIGH_LEVEL_SCHEMA_SET | frozenset({'npipe'})

    def __init__(self, path: str, force_close: bool = False,
                 keepalive_timeout: Union[float, Any] = sentinel, limit: int = 100, limit_per_host: int = 0) -> None:
        super().__init__(force_close=force_close, keepalive_timeout=keepalive_timeout,
                         limit=limit, limit_per_host=limit_per_host)
        if not isinstance(self._loop, asyncio.ProactorEventLoop):
            raise RuntimeError('Named Pipes only available in proactor loop under windows')
        self._path: str = path

    @property
    def path(self) -> str:
        """Path to the named pipe."""
        return self._path

    async def _create_connection(self, req: ClientRequest, traces: Optional[List["Trace"]],
                                 timeout: "ClientTimeout") -> ResponseHandler:
        try:
            async with ceil_timeout(timeout.sock_connect, ceil_threshold=timeout.ceil_threshold):
                _, proto = await self._loop.create_pipe_connection(self._factory, self._path)
                await asyncio.sleep(0)
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise ClientConnectorError(req.connection_key, exc) from exc
        return cast(ResponseHandler, proto)