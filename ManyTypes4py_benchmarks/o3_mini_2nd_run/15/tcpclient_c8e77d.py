#!/usr/bin/env python3
"""A non-blocking TCP connection factory.
"""
import functools
import socket
import numbers
import datetime
import ssl
import typing
from typing import Any, Union, Dict, Tuple, List, Callable, Iterator, Optional, Set
from tornado.concurrent import Future, future_add_done_callback
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado import gen
from tornado.netutil import Resolver
from tornado.gen import TimeoutError

if typing.TYPE_CHECKING:
    from typing import NoReturn

_INITIAL_CONNECT_TIMEOUT: float = 0.3

class _Connector:
    def __init__(self, addrinfo: List[Tuple[int, Tuple[Any, ...]]],
                 connect: Callable[[int, Tuple[Any, ...]], Tuple[IOStream, Future[IOStream]]]) -> None:
        self.io_loop: IOLoop = IOLoop.current()
        self.connect: Callable[[int, Tuple[Any, ...]], Tuple[IOStream, Future[IOStream]]] = connect
        self.future: Future[Tuple[int, Tuple[Any, ...], IOStream]] = Future()
        self.timeout: Optional[Any] = None
        self.connect_timeout: Optional[Any] = None
        self.last_error: Optional[BaseException] = None
        self.remaining: int = len(addrinfo)
        self.primary_addrs, self.secondary_addrs = self.split(addrinfo)
        self.streams: Set[IOStream] = set()

    @staticmethod
    def split(addrinfo: List[Tuple[int, Tuple[Any, ...]]]
              ) -> Tuple[List[Tuple[int, Tuple[Any, ...]]], List[Tuple[int, Tuple[Any, ...]]]]:
        primary: List[Tuple[int, Tuple[Any, ...]]] = []
        secondary: List[Tuple[int, Tuple[Any, ...]]] = []
        primary_af: int = addrinfo[0][0]
        for af, addr in addrinfo:
            if af == primary_af:
                primary.append((af, addr))
            else:
                secondary.append((af, addr))
        return (primary, secondary)

    def start(self, timeout: float = _INITIAL_CONNECT_TIMEOUT,
              connect_timeout: Optional[float] = None) -> Future[Tuple[int, Tuple[Any, ...], IOStream]]:
        self.try_connect(iter(self.primary_addrs))
        self.set_timeout(timeout)
        if connect_timeout is not None:
            self.set_connect_timeout(connect_timeout)
        return self.future

    def try_connect(self, addrs: Iterator[Tuple[int, Tuple[Any, ...]]]) -> None:
        try:
            af, addr = next(addrs)
        except StopIteration:
            if self.remaining == 0 and (not self.future.done()):
                self.future.set_exception(self.last_error or IOError('connection failed'))
            return
        stream, future = self.connect(af, addr)
        self.streams.add(stream)
        future_add_done_callback(
            future,
            functools.partial(self.on_connect_done, addrs, af, addr)
        )

    def on_connect_done(self, addrs: Iterator[Tuple[int, Tuple[Any, ...]]],
                        af: int, addr: Tuple[Any, ...],
                        future: Future[IOStream]) -> None:
        self.remaining -= 1
        try:
            stream = future.result()
        except Exception as e:
            if self.future.done():
                return
            self.last_error = e
            self.try_connect(addrs)
            if self.timeout is not None:
                self.io_loop.remove_timeout(self.timeout)
                self.on_timeout()
            return
        self.clear_timeouts()
        if self.future.done():
            stream.close()
        else:
            self.streams.discard(stream)
            self.future.set_result((af, addr, stream))
            self.close_streams()

    def set_timeout(self, timeout: float) -> None:
        self.timeout = self.io_loop.add_timeout(self.io_loop.time() + timeout, self.on_timeout)

    def on_timeout(self) -> None:
        self.timeout = None
        if not self.future.done():
            self.try_connect(iter(self.secondary_addrs))

    def clear_timeout(self) -> None:
        if self.timeout is not None:
            self.io_loop.remove_timeout(self.timeout)

    def set_connect_timeout(self, connect_timeout: float) -> None:
        self.connect_timeout = self.io_loop.add_timeout(connect_timeout, self.on_connect_timeout)

    def on_connect_timeout(self) -> None:
        if not self.future.done():
            self.future.set_exception(TimeoutError())
        self.close_streams()

    def clear_timeouts(self) -> None:
        if self.timeout is not None:
            self.io_loop.remove_timeout(self.timeout)
        if self.connect_timeout is not None:
            self.io_loop.remove_timeout(self.connect_timeout)

    def close_streams(self) -> None:
        for stream in self.streams:
            stream.close()

class TCPClient:
    def __init__(self, resolver: Optional[Resolver] = None) -> None:
        if resolver is not None:
            self.resolver: Resolver = resolver
            self._own_resolver: bool = False
        else:
            self.resolver = Resolver()
            self._own_resolver = True

    def close(self) -> None:
        if self._own_resolver:
            self.resolver.close()

    async def connect(self,
                      host: str,
                      port: int,
                      af: int = socket.AF_UNSPEC,
                      ssl_options: Optional[Union[ssl.SSLContext, Dict[str, Any]]] = None,
                      max_buffer_size: Optional[int] = None,
                      source_ip: Optional[str] = None,
                      source_port: Optional[int] = None,
                      timeout: Optional[Union[float, datetime.timedelta]] = None
                      ) -> IOStream:
        if timeout is not None:
            if isinstance(timeout, numbers.Real):
                timeout = IOLoop.current().time() + float(timeout)
            elif isinstance(timeout, datetime.timedelta):
                timeout = IOLoop.current().time() + timeout.total_seconds()
            else:
                raise TypeError('Unsupported timeout %r' % timeout)
        if timeout is not None:
            addrinfo: List[Tuple[int, Tuple[Any, ...]]] = await gen.with_timeout(
                timeout, self.resolver.resolve(host, port, af))
        else:
            addrinfo = await self.resolver.resolve(host, port, af)
        connector = _Connector(
            addrinfo,
            functools.partial(self._create_stream, max_buffer_size, source_ip=source_ip, source_port=source_port)
        )
        af_result, addr_result, stream = await connector.start(connect_timeout=timeout)  # type: ignore[arg-type]
        if ssl_options is not None:
            if timeout is not None:
                stream = await gen.with_timeout(
                    timeout,
                    stream.start_tls(False, ssl_options=ssl_options, server_hostname=host)
                )
            else:
                stream = await stream.start_tls(False, ssl_options=ssl_options, server_hostname=host)
        return stream

    def _create_stream(self,
                       max_buffer_size: Optional[int],
                       af: int,
                       addr: Tuple[Any, ...],
                       source_ip: Optional[str] = None,
                       source_port: Optional[int] = None
                       ) -> Tuple[IOStream, Future[IOStream]]:
        source_port_bind: int = source_port if isinstance(source_port, int) else 0
        source_ip_bind: Optional[str] = source_ip
        if source_port_bind and (not source_ip):
            source_ip_bind = '::1' if af == socket.AF_INET6 else '127.0.0.1'
        socket_obj: socket.socket = socket.socket(af)
        if source_port_bind or source_ip_bind:
            try:
                socket_obj.bind((source_ip_bind, source_port_bind))  # type: ignore
            except OSError:
                socket_obj.close()
                raise
        try:
            stream: IOStream = IOStream(socket_obj, max_buffer_size=max_buffer_size)
        except OSError as e:
            fu: Future[IOStream] = Future()
            fu.set_exception(e)
            return (stream, fu)  # type: ignore
        else:
            return (stream, stream.connect(addr))
