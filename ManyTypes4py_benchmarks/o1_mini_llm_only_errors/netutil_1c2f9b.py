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
from typing import List, Callable, Any, Type, Dict, Union, Tuple, Awaitable, Optional

_client_ssl_defaults: ssl.SSLContext = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
_server_ssl_defaults: ssl.SSLContext = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
if hasattr(ssl, 'OP_NO_COMPRESSION'):
    _client_ssl_defaults.options |= ssl.OP_NO_COMPRESSION
    _server_ssl_defaults.options |= ssl.OP_NO_COMPRESSION
'foo'.encode('idna')
'foo'.encode('latin1')
_DEFAULT_BACKLOG: int = 128

def bind_sockets(
    port: int,
    address: Optional[str] = None,
    family: int = socket.AF_UNSPEC,
    backlog: int = _DEFAULT_BACKLOG,
    flags: Optional[int] = None,
    reuse_port: bool = False
) -> List[socket.socket]:
    """Creates listening sockets bound to the given port and address.

    Returns a list of socket objects (multiple sockets are returned if
    the given address maps to multiple IP addresses, which is most common
    for mixed IPv4 and IPv6 use).

    Address may be either an IP address or hostname.  If it's a hostname,
    the server will listen on all IP addresses associated with the
    name.  Address may be an empty string or None to listen on all
    available interfaces.  Family may be set to either `socket.AF_INET`
    or `socket.AF_INET6` to restrict to IPv4 or IPv6 addresses, otherwise
    both will be used if available.

    The ``backlog`` argument has the same meaning as for
    `socket.listen() <socket.socket.listen>`.

    ``flags`` is a bitmask of AI_* flags to `~socket.getaddrinfo`, like
    ``socket.AI_PASSIVE | socket.AI_NUMERICHOST``.

    ``reuse_port`` option sets ``SO_REUSEPORT`` option for every socket
    in the list. If your platform doesn't support this option ValueError will
    be raised.
    """
    if reuse_port and (not hasattr(socket, 'SO_REUSEPORT')):
        raise ValueError("the platform doesn't support SO_REUSEPORT")
    sockets: List[socket.socket] = []
    if address == '':
        address = None
    if not socket.has_ipv6 and family == socket.AF_UNSPEC:
        family = socket.AF_INET
    if flags is None:
        flags = socket.AI_PASSIVE
    bound_port: Optional[int] = None
    unique_addresses: set = set()
    for res in sorted(socket.getaddrinfo(address, port, family, socket.SOCK_STREAM, 0, flags), key=lambda x: x[0]):
        if res in unique_addresses:
            continue
        unique_addresses.add(res)
        af, socktype, proto, canonname, sockaddr = res
        if sys.platform == 'darwin' and address == 'localhost' and (af == socket.AF_INET6) and (sockaddr[3] != 0):
            continue
        try:
            sock: socket.socket = socket.socket(af, socktype, proto)
        except OSError as e:
            if errno_from_exception(e) == errno.EAFNOSUPPORT:
                continue
            raise
        if os.name != 'nt':
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError as e:
                if errno_from_exception(e) != errno.ENOPROTOOPT:
                    raise
        if reuse_port:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if af == socket.AF_INET6:
            if hasattr(socket, 'IPPROTO_IPV6'):
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        host, requested_port, *rest = sockaddr
        if requested_port == 0 and bound_port is not None:
            sockaddr = tuple([host, bound_port] + rest)
        sock.setblocking(False)
        try:
            sock.bind(sockaddr)
        except OSError as e:
            if errno_from_exception(e) == errno.EADDRNOTAVAIL and address == 'localhost' and (sockaddr[0] == '::1'):
                sock.close()
                continue
            else:
                raise
        bound_port = sock.getsockname()[1]
        sock.listen(backlog)
        sockets.append(sock)
    return sockets

if hasattr(socket, 'AF_UNIX'):

    def bind_unix_socket(file: str, mode: int = 0o600, backlog: int = _DEFAULT_BACKLOG) -> socket.socket:
        """Creates a listening unix socket.

        If a socket with the given name already exists, it will be deleted.
        If any other file with that name exists, an exception will be
        raised.

        Returns a socket object (not a list of socket objects like
        `bind_sockets`)
        """
        sock: socket.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError as e:
            if errno_from_exception(e) != errno.ENOPROTOOPT:
                raise
        sock.setblocking(False)
        if not file.startswith('\x00'):
            try:
                st: os.stat_result = os.stat(file)
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISSOCK(st.st_mode):
                    os.remove(file)
                else:
                    raise ValueError('File %s exists and is not a socket', file)
            sock.bind(file)
            os.chmod(file, mode)
        else:
            sock.bind(file)
        sock.listen(backlog)
        return sock

def add_accept_handler(
    sock: socket.socket,
    callback: Callable[[socket.socket, Tuple[Any, ...]], None]
) -> Callable[[], None]:
    """Adds an `.IOLoop` event handler to accept new connections on ``sock``.

    When a connection is accepted, ``callback(connection, address)`` will
    be run (``connection`` is a socket object, and ``address`` is the
    address of the other end of the connection).  Note that this signature
    is different from the ``callback(fd, events)`` signature used for
    `.IOLoop` handlers.

    A callable is returned which, when called, will remove the `.IOLoop`
    event handler and stop processing further incoming connections.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    .. versionchanged:: 5.0
       A callable is returned (``None`` was returned before).
    """
    io_loop: IOLoop = IOLoop.current()
    removed = [False]

    def accept_handler(fd: int, events: int) -> None:
        for _ in range(_DEFAULT_BACKLOG):
            if removed[0]:
                return
            try:
                connection: socket.socket
                address: Tuple[Any, ...]
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

def is_valid_ip(ip: Optional[str]) -> bool:
    """Returns ``True`` if the given string is a well-formed IP address.

    Supports IPv4 and IPv6.
    """
    if not ip or '\x00' in ip:
        return False
    try:
        res = socket.getaddrinfo(ip, 0, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST)
        return bool(res)
    except socket.gaierror as e:
        if e.args[0] == socket.EAI_NONAME:
            return False
        raise
    except UnicodeError:
        return False
    return True

class Resolver(Configurable):
    """Configurable asynchronous DNS resolver interface.

    By default, a blocking implementation is used (which simply calls
    `socket.getaddrinfo`.  An alternative implementation can be
    chosen with the `Resolver.configure <.Configurable.configure>`
    class method::

        Resolver.configure('tornado.netutil.ThreadedResolver')

    The implementations of this interface included with Tornado are

    * `tornado.netutil.DefaultLoopResolver`
    * `tornado.netutil.DefaultExecutorResolver` (deprecated)
    * `tornado.netutil.BlockingResolver` (deprecated)
    * `tornado.netutil.ThreadedResolver` (deprecated)
    * `tornado.netutil.OverrideResolver`
    * `tornado.platform.caresresolver.CaresResolver` (deprecated)

    .. versionchanged:: 5.0
       The default implementation has changed from `BlockingResolver` to
       `DefaultExecutorResolver`.

    .. versionchanged:: 6.2
       The default implementation has changed from `DefaultExecutorResolver` to
       `DefaultLoopResolver`.
    """

    @classmethod
    def configurable_base(cls) -> Type['Resolver']:
        return Resolver

    @classmethod
    def configurable_default(cls) -> Type['Resolver']:
        return DefaultLoopResolver

    async def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[Any, ...]]]:
        """Resolves an address.

        The ``host`` argument is a string which may be a hostname or a
        literal IP address.

        Returns a `.Future` whose result is a list of (family,
        address) pairs, where address is a tuple suitable to pass to
        `socket.connect <socket.socket.connect>` (i.e. a ``(host,
        port)`` pair for IPv4; additional fields may be present for
        IPv6). If a ``callback`` is passed, it will be run with the
        result as an argument when it is complete.

        :raises IOError: if the address cannot be resolved.

        .. versionchanged:: 4.4
           Standardized all implementations to raise `IOError`.

        .. versionchanged:: 6.0 The ``callback`` argument was removed.
           Use the returned awaitable object instead.

        """
        raise NotImplementedError()

    def close(self) -> None:
        """Closes the `Resolver`, freeing any resources used.

        .. versionadded:: 3.1

        """
        pass

def _resolve_addr(host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[Any, ...]]]:
    addrinfo = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)
    results: List[Tuple[int, Tuple[Any, ...]]] = []
    for fam, socktype, proto, canonname, address in addrinfo:
        results.append((fam, address))
    return results

class DefaultExecutorResolver(Resolver):
    """Resolver implementation using `.IOLoop.run_in_executor`.

    .. versionadded:: 5.0

    .. deprecated:: 6.2

       Use `DefaultLoopResolver` instead.
    """

    async def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[Any, ...]]]:
        result: List[Tuple[int, Tuple[Any, ...]]] = await IOLoop.current().run_in_executor(None, _resolve_addr, host, port, family)
        return result

class DefaultLoopResolver(Resolver):
    """Resolver implementation using `asyncio.loop.getaddrinfo`."""

    async def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[Any, ...]]]:
        addrinfos: List[Tuple[int, int, int, str, Tuple[Any, ...]]] = await asyncio.get_running_loop().getaddrinfo(
            host,
            port,
            family=family,
            type=socket.SOCK_STREAM
        )
        return [(fam, address) for fam, _, _, _, address in addrinfos]

class ExecutorResolver(Resolver):
    """Resolver implementation using a `concurrent.futures.Executor`.

    Use this instead of `ThreadedResolver` when you require additional
    control over the executor being used.

    The executor will be shut down when the resolver is closed unless
    ``close_resolver=False``; use this if you want to reuse the same
    executor elsewhere.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    .. deprecated:: 5.0
       The default `Resolver` now uses `asyncio.loop.getaddrinfo`;
       use that instead of this class.
    """

    def initialize(self, executor: Optional[concurrent.futures.Executor] = None, close_executor: bool = True) -> None:
        if executor is not None:
            self.executor: Optional[concurrent.futures.Executor] = executor
            self.close_executor: bool = close_executor
        else:
            self.executor = dummy_executor
            self.close_executor = False

    def close(self) -> None:
        if self.close_executor and self.executor is not None:
            self.executor.shutdown()
        self.executor = None

    @run_on_executor
    def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[Any, ...]]]:
        return _resolve_addr(host, port, family)

class BlockingResolver(ExecutorResolver):
    """Default `Resolver` implementation, using `socket.getaddrinfo`.

    The `.IOLoop` will be blocked during the resolution, although the
    callback will not be run until the next `.IOLoop` iteration.

    .. deprecated:: 5.0
       The default `Resolver` now uses `.IOLoop.run_in_executor`; use that instead
       of this class.
    """

    def initialize(self) -> None:
        super().initialize()

class ThreadedResolver(ExecutorResolver):
    """Multithreaded non-blocking `Resolver` implementation.

    The thread pool size can be configured with::

        Resolver.configure('tornado.netutil.ThreadedResolver',
                           num_threads=10)

    .. versionchanged:: 3.1
       All ``ThreadedResolvers`` share a single thread pool, whose
       size is set by the first one to be created.

    .. deprecated:: 5.0
       The default `Resolver` now uses `.IOLoop.run_in_executor`; use that instead
       of this class.
    """
    _threadpool: Optional[concurrent.futures.ThreadPoolExecutor] = None
    _threadpool_pid: Optional[int] = None

    def initialize(self, num_threads: int = 10) -> None:
        threadpool: concurrent.futures.ThreadPoolExecutor = ThreadedResolver._create_threadpool(num_threads)
        super().initialize(executor=threadpool, close_executor=False)

    @classmethod
    def _create_threadpool(cls, num_threads: int) -> concurrent.futures.ThreadPoolExecutor:
        pid: int = os.getpid()
        if cls._threadpool_pid != pid:
            cls._threadpool = None
        if cls._threadpool is None:
            cls._threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
            cls._threadpool_pid = pid
        return cls._threadpool

class OverrideResolver(Resolver):
    """Wraps a resolver with a mapping of overrides.

    This can be used to make local DNS changes (e.g. for testing)
    without modifying system-wide settings.

    The mapping can be in three formats::

        {
            # Hostname to host or ip
            "example.com": "127.0.1.1",

            # Host+port to host+port
            ("login.example.com", 443): ("localhost", 1443),

            # Host+port+address family to host+port
            ("login.example.com", 443, socket.AF_INET6): ("::1", 1443),
        }

    .. versionchanged:: 5.0
       Added support for host-port-family triplets.
    """
    mapping: Dict[Union[str, Tuple[str, int], Tuple[str, int, int]], Union[Tuple[str, int], str]]

    def initialize(
        self,
        resolver: Resolver,
        mapping: Dict[Union[str, Tuple[str, int], Tuple[str, int, int]], Union[Tuple[str, int], str]]
    ) -> None:
        self.resolver: Resolver = resolver
        self.mapping: Dict[Union[str, Tuple[str, int], Tuple[str, int, int]], Union[Tuple[str, int], str]] = mapping

    def close(self) -> None:
        self.resolver.close()

    async def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[Any, ...]]]:
        key3: Tuple[str, int, int] = (host, port, family)
        key2: Tuple[str, int] = (host, port)
        key1: str = host
        if key3 in self.mapping:
            new_host, new_port = self.mapping[key3]
            host, port = new_host, new_port
        elif key2 in self.mapping:
            new_host, new_port = self.mapping[key2]
            host, port = new_host, new_port
        elif key1 in self.mapping:
            host = self.mapping[key1]
        return await self.resolver.resolve(host, port, family)

_SSL_CONTEXT_KEYWORDS: frozenset = frozenset(['ssl_version', 'certfile', 'keyfile', 'cert_reqs', 'ca_certs', 'ciphers'])

def ssl_options_to_context(
    ssl_options: Union[ssl.SSLContext, Dict[str, Any]],
    server_side: Optional[bool] = None
) -> ssl.SSLContext:
    """Try to convert an ``ssl_options`` dictionary to an
    `~ssl.SSLContext` object.

    The ``ssl_options`` argument may be either an `ssl.SSLContext` object or a dictionary containing
    keywords to be passed to ``ssl.SSLContext.wrap_socket``.  This function converts the dict form
    to its `~ssl.SSLContext` equivalent, and may be used when a component which accepts both forms
    needs to upgrade to the `~ssl.SSLContext` version to use features like SNI or ALPN.

    .. versionchanged:: 6.2

       Added server_side argument. Omitting this argument will result in a DeprecationWarning on
       Python 3.10.

    """
    if isinstance(ssl_options, ssl.SSLContext):
        return ssl_options
    assert isinstance(ssl_options, dict)
    assert all((k in _SSL_CONTEXT_KEYWORDS for k in ssl_options)), ssl_options
    default_version: int = ssl.PROTOCOL_TLS
    if server_side:
        default_version = ssl.PROTOCOL_TLS_SERVER
    elif server_side is not None:
        default_version = ssl.PROTOCOL_TLS_CLIENT
    context: ssl.SSLContext = ssl.SSLContext(ssl_options.get('ssl_version', default_version))
    if 'certfile' in ssl_options:
        context.load_cert_chain(ssl_options['certfile'], ssl_options.get('keyfile', None))
    if 'cert_reqs' in ssl_options:
        if ssl_options['cert_reqs'] == ssl.CERT_NONE:
            context.check_hostname = False
        context.verify_mode = ssl_options['cert_reqs']
    if 'ca_certs' in ssl_options:
        context.load_verify_locations(ssl_options['ca_certs'])
    if 'ciphers' in ssl_options:
        context.set_ciphers(ssl_options['ciphers'])
    if hasattr(ssl, 'OP_NO_COMPRESSION'):
        context.options |= ssl.OP_NO_COMPRESSION
    return context

def ssl_wrap_socket(
    sock: socket.socket,
    ssl_options: Union[ssl.SSLContext, Dict[str, Any]],
    server_hostname: Optional[str] = None,
    server_side: Optional[bool] = None,
    **kwargs: Any
) -> ssl.SSLSocket:
    """Returns an ``ssl.SSLSocket`` wrapping the given socket.

    ``ssl_options`` may be either an `ssl.SSLContext` object or a
    dictionary (as accepted by `ssl_options_to_context`).  Additional
    keyword arguments are passed to `ssl.SSLContext.wrap_socket`.

    .. versionchanged:: 6.2

       Added server_side argument. Omitting this argument will
       result in a DeprecationWarning on Python 3.10.
    """
    context: ssl.SSLContext = ssl_options_to_context(ssl_options, server_side=server_side)
    if server_side is None:
        server_side = False
    assert ssl.HAS_SNI
    return context.wrap_socket(
        sock,
        server_hostname=server_hostname,
        server_side=server_side,
        **kwargs
    )
