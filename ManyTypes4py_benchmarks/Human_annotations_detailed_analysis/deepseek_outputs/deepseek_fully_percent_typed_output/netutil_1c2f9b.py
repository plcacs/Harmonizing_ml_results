#
# Copyright 2011 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Miscellaneous network utility code."""

import asyncio
import concurrent.futures
import errno
import os
import sys
import socket
import ssl
import stat

from tornado.concurrent import dummy_executor, run_on_executor
from tornado.ioloop import IOLoop
from tornado.util import Configurable, errno_from_exception

from typing import List, Callable, Any, Type, Dict, Union, Tuple, Awaitable, Optional, Set, cast

# Note that the naming of ssl.Purpose is confusing; the purpose
# of a context is to authenticate the opposite side of the connection.
_client_ssl_defaults: ssl.SSLContext = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
_server_ssl_defaults: ssl.SSLContext = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
if hasattr(ssl, "OP_NO_COMPRESSION"):
    # See netutil.ssl_options_to_context
    _client_ssl_defaults.options |= ssl.OP_NO_COMPRESSION
    _server_ssl_defaults.options |= ssl.OP_NO_COMPRESSION

# ThreadedResolver runs getaddrinfo on a thread. If the hostname is unicode,
# getaddrinfo attempts to import encodings.idna. If this is done at
# module-import time, the import lock is already held by the main thread,
# leading to deadlock. Avoid it by caching the idna encoder on the main
# thread now.
"foo".encode("idna")

# For undiagnosed reasons, 'latin1' codec may also need to be preloaded.
"foo".encode("latin1")

# Default backlog used when calling sock.listen()
_DEFAULT_BACKLOG: int = 128


def bind_sockets(
    port: int,
    address: Optional[str] = None,
    family: socket.AddressFamily = socket.AF_UNSPEC,
    backlog: int = _DEFAULT_BACKLOG,
    flags: Optional[int] = None,
    reuse_port: bool = False,
) -> List[socket.socket]:
    """Creates listening sockets bound to the given port and address."""
    if reuse_port and not hasattr(socket, "SO_REUSEPORT"):
        raise ValueError("the platform doesn't support SO_REUSEPORT")

    sockets: List[socket.socket] = []
    if address == "":
        address = None
    if not socket.has_ipv6 and family == socket.AF_UNSPEC:
        family = socket.AF_INET
    if flags is None:
        flags = socket.AI_PASSIVE
    bound_port: Optional[int] = None
    unique_addresses: Set[Any] = set()
    for res in sorted(
        socket.getaddrinfo(address, port, family, socket.SOCK_STREAM, 0, flags),
        key=lambda x: x[0],
    ):
        if res in unique_addresses:
            continue

        unique_addresses.add(res)

        af, socktype, proto, canonname, sockaddr = res
        if (
            sys.platform == "darwin"
            and address == "localhost"
            and af == socket.AF_INET6
            and sockaddr[3] != 0  # type: ignore
        ):
            continue
        try:
            sock = socket.socket(af, socktype, proto)
        except OSError as e:
            if errno_from_exception(e) == errno.EAFNOSUPPORT:
                continue
            raise
        if os.name != "nt":
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError as e:
                if errno_from_exception(e) != errno.ENOPROTOOPT:
                    raise
        if reuse_port:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if af == socket.AF_INET6:
            if hasattr(socket, "IPPROTO_IPV6"):
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)

        host, requested_port = sockaddr[:2]
        if requested_port == 0 and bound_port is not None:
            sockaddr = tuple([host, bound_port] + list(sockaddr[2:]))

        sock.setblocking(False)
        try:
            sock.bind(sockaddr)
        except OSError as e:
            if (
                errno_from_exception(e) == errno.EADDRNOTAVAIL
                and address == "localhost"
                and sockaddr[0] == "::1"
            ):
                sock.close()
                continue
            else:
                raise
        bound_port = sock.getsockname()[1]
        sock.listen(backlog)
        sockets.append(sock)
    return sockets


if hasattr(socket, "AF_UNIX"):

    def bind_unix_socket(
        file: str, mode: int = 0o600, backlog: int = _DEFAULT_BACKLOG
    ) -> socket.socket:
        """Creates a listening unix socket."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError as e:
            if errno_from_exception(e) != errno.ENOPROTOOPT:
                raise
        sock.setblocking(False)
        if not file.startswith("\0"):
            try:
                st = os.stat(file)
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISSOCK(st.st_mode):
                    os.remove(file)
                else:
                    raise ValueError("File %s exists and is not a socket", file)
            sock.bind(file)
            os.chmod(file, mode)
        else:
            sock.bind(file)
        sock.listen(backlog)
        return sock


def add_accept_handler(
    sock: socket.socket, callback: Callable[[socket.socket, Any], None]
) -> Callable[[], None]:
    """Adds an `.IOLoop` event handler to accept new connections on ``sock``."""
    io_loop = IOLoop.current()
    removed = [False]

    def accept_handler(fd: socket.socket, events: int) -> None:
        for i in range(_DEFAULT_BACKLOG):
            if removed[0]:
                return
            try:
                connection, address = sock.accept()
            except BlockingIOError:
                return
            except ConnectionAbortedError:
                continue
            callback(connection, address)

    def remove_handler() -> None:
        io_loop.remove_handler(sock)
        removed[0] = True

    io_loop.add_handler(sock, accept_handler, IOLoop.READ)
    return remove_handler


def is_valid_ip(ip: str) -> bool:
    """Returns ``True`` if the given string is a well-formed IP address."""
    if not ip or "\x00" in ip:
        return False
    try:
        res = socket.getaddrinfo(
            ip, 0, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST
        )
        return bool(res)
    except socket.gaierror as e:
        if e.args[0] == socket.EAI_NONAME:
            return False
        raise
    except UnicodeError:
        return False
    return True


class Resolver(Configurable):
    """Configurable asynchronous DNS resolver interface."""

    @classmethod
    def configurable_base(cls) -> Type["Resolver"]:
        return Resolver

    @classmethod
    def configurable_default(cls) -> Type["Resolver"]:
        return DefaultLoopResolver

    def resolve(
        self, host: str, port: int, family: socket.AddressFamily = socket.AF_UNSPEC
    ) -> Awaitable[List[Tuple[int, Any]]]:
        raise NotImplementedError()

    def close(self) -> None:
        pass


def _resolve_addr(
    host: str, port: int, family: socket.AddressFamily = socket.AF_UNSPEC
) -> List[Tuple[int, Any]]:
    addrinfo = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)
    results = []
    for fam, socktype, proto, canonname, address in addrinfo:
        results.append((fam, address))
    return cast(List[Tuple[int, Any]], results)


class DefaultExecutorResolver(Resolver):
    """Resolver implementation using `.IOLoop.run_in_executor`."""

    async def resolve(
        self, host: str, port: int, family: socket.AddressFamily = socket.AF_UNSPEC
    ) -> List[Tuple[int, Any]]:
        result = await IOLoop.current().run_in_executor(
            None, _resolve_addr, host, port, family
        )
        return cast(List[Tuple[int, Any]], result)


class DefaultLoopResolver(Resolver):
    """Resolver implementation using `asyncio.loop.getaddrinfo`."""

    async def resolve(
        self, host: str, port: int, family: socket.AddressFamily = socket.AF_UNSPEC
    ) -> List[Tuple[int, Any]]:
        return [
            (fam, address)
            for fam, _, _, _, address in await asyncio.get_running_loop().getaddrinfo(
                host, port, family=family, type=socket.SOCK_STREAM
            )
        ]


class ExecutorResolver(Resolver):
    """Resolver implementation using a `concurrent.futures.Executor`."""

    def initialize(
        self,
        executor: Optional[concurrent.futures.Executor] = None,
        close_executor: bool = True,
    ) -> None:
        if executor is not None:
            self.executor = executor
            self.close_executor = close_executor
        else:
            self.executor = dummy_executor
            self.close_executor = False

    def close(self) -> None:
        if self.close_executor:
            assert self.executor is not None
            self.executor.shutdown()
        self.executor = None  # type: ignore

    @run_on_executor
    def resolve(
        self, host: str, port: int, family: socket.AddressFamily = socket.AF_UNSPEC
    ) -> List[Tuple[int, Any]]:
        return _resolve_addr(host, port, family)


class BlockingResolver(ExecutorResolver):
    """Default `Resolver` implementation, using `socket.getaddrinfo`."""

    def initialize(self) -> None:  # type: ignore
        super().initialize()


class ThreadedResolver(ExecutorResolver):
    """Multithreaded non-blocking `Resolver` implementation."""

    _threadpool: Optional[concurrent.futures.ThreadPoolExecutor] = None
    _threadpool_pid: Optional[int] = None

    def initialize(self, num_threads: int = 10) -> None:  # type: ignore
        threadpool = ThreadedResolver._create_threadpool(num_threads)
        super().initialize(executor=threadpool, close_executor=False)

    @classmethod
    def _create_threadpool(
        cls, num_threads: int
    ) -> concurrent.futures.ThreadPoolExecutor:
        pid = os.getpid()
        if cls._threadpool_pid != pid:
            cls._threadpool = None
        if cls._threadpool is None:
            cls._threadpool = concurrent.futures.ThreadPoolExecutor(num_threads)
            cls._threadpool_pid = pid
        assert cls._threadpool is not None
        return cls._threadpool


class OverrideResolver(Resolver):
    """Wraps a resolver with a mapping of overrides."""

    def initialize(self, resolver: Resolver, mapping: Dict[Any, Any]) -> None:
        self.resolver = resolver
        self.mapping = mapping

    def close(self) -> None:
        self.resolver.close()

    def resolve(
        self, host: str, port: int, family: socket.AddressFamily = socket.AF_UNSPEC
    ) -> Awaitable[List[Tuple[int, Any]]]:
        if (host, port, family) in self.mapping:
            host, port = self.mapping[(host, port, family)]
        elif (host, port) in self.mapping:
            host, port = self.mapping[(host, port)]
        elif host in self.mapping:
            host = self.mapping[host]
        return self.resolver.resolve(host, port, family)


_SSL_CONTEXT_KEYWORDS = frozenset(
    ["ssl_version", "certfile", "keyfile", "cert_reqs", "ca_certs", "ciphers"]
)


def ssl_options_to_context(
    ssl_options: Union[Dict[str, Any], ssl.SSLContext],
    server_side: Optional[bool] = None,
) -> ssl.SSLContext:
    """Try to convert an ``ssl_options`` dictionary to an `~ssl.SSLContext` object."""
    if isinstance(ssl_options, ssl.SSLContext):
        return ssl_options
    assert isinstance(ssl_options, dict)
    assert all(k in _SSL_CONTEXT_KEYWORDS for k in ssl_options), ssl_options
    default_version = ssl.PROTOCOL_TLS
    if server_side:
        default_version = ssl.PROTOCOL_TLS_SERVER
    elif server_side is not None:
        default_version = ssl.PROTOCOL_TLS_CLIENT
    context = ssl.SSLContext(ssl_options.get("ssl_version", default_version))
    if "certfile" in ssl_options:
        context.load_cert_chain(
            ssl_options["certfile"], ssl_options.get("keyfile", None)
        )
    if "cert_reqs" in ssl_options:
        if ssl_options["cert_reqs"] == ssl.CERT_NONE:
            context.check_hostname = False
        context.verify_mode = ssl_options["cert_reqs"]
    if "ca_certs" in ssl_options:
        context.load_verify_locations(ssl_options["ca_certs"])
    if "ciphers" in ssl_options:
        context.set_ciphers(ssl_options["ciphers"])
    if hasattr(ssl, "OP_NO_COMPRESSION"):
        context.options |= ssl.OP_NO_COMPRESSION
    return context


def ssl_wrap_socket(
    socket: socket.socket,
    ssl_options: Union[Dict[str, Any], ssl.SSLContext],
    server_hostname: Optional[str] = None,
    server_side: Optional[bool] = None,
    **kwargs: Any,
) -> ssl.SSLSocket:
    """Returns an ``ssl.SSLSocket`` wrapping the given socket."""
    context = ssl_options_to_context(ssl_options, server_side=server_side)
    if server_side is None:
        server_side = False
    assert ssl.HAS_SNI
    return context.wrap_socket(
        socket, server_hostname=server_hostname, server_side=server_side, **kwargs
    )
