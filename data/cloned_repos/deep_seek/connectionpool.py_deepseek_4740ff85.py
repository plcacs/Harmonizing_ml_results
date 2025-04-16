```python
from __future__ import absolute_import
import errno
import logging
import sys
import warnings
from socket import error as SocketError, timeout as SocketTimeout
import socket
import h11
from typing import (
    Optional,
    ClassVar,
    Dict,
    Any,
    Union,
    List,
    Type,
    cast,
    Callable,
    Awaitable,
    TYPE_CHECKING,
)
from queue import LifoQueue
import ssl

from ..base import Request, DEFAULT_PORTS
from ..exceptions import (
    ClosedPoolError,
    ProtocolError,
    EmptyPoolError,
    LocationValueError,
    MaxRetryError,
    ProxyError,
    ReadTimeoutError,
    SSLError,
    TimeoutError,
    InsecureRequestWarning,
    NewConnectionError,
)
from ..packages.ssl_match_hostname import CertificateError
from ..packages import six
from ..packages.six.moves import queue
from ..request import RequestMethods
from .response import HTTPResponse
from .connection import HTTP1Connection
from ..util.connection import is_connection_dropped
from ..util.request import set_file_position
from ..util.retry import Retry
from ..util.ssl_ import (
    create_urllib3_context,
    merge_context_settings,
    resolve_ssl_version,
    resolve_cert_reqs,
    BaseSSLError,
)
from ..util.timeout import Timeout
from ..util.url import get_host, Url

if six.PY2:
    import Queue as _unused_module_Queue  # noqa: F401
xrange = six.moves.xrange
log = logging.getLogger(__name__)
_Default = object()


def _add_transport_headers(headers: Dict[str, str]) -> None:
    transfer_headers = ("content-length", "transfer-encoding")
    for header_name in headers:
        if header_name.lower() in transfer_headers:
            return
    headers["transfer-encoding"] = "chunked"


def _build_context(
    context: Optional[ssl.SSLContext],
    keyfile: Optional[str],
    certfile: Optional[str],
    cert_reqs: Optional[int],
    ca_certs: Optional[str],
    ca_cert_dir: Optional[str],
    ssl_version: Optional[int],
) -> ssl.SSLContext:
    if context is None:
        context = create_urllib3_context(
            ssl_version=resolve_ssl_version(ssl_version),
            cert_reqs=resolve_cert_reqs(cert_reqs),
        )
    context = merge_context_settings(
        context,
        keyfile=keyfile,
        certfile=certfile,
        cert_reqs=cert_reqs,
        ca_certs=ca_certs,
        ca_cert_dir=ca_cert_dir,
    )
    return context


class ConnectionPool(object):
    scheme: ClassVar[Optional[str]] = None
    QueueCls: ClassVar[Type[LifoQueue]] = queue.LifoQueue

    def __init__(self, host: str, port: Optional[int] = None) -> None:
        if not host:
            raise LocationValueError("No host specified.")
        self.host = _ipv6_host(host).lower()
        self.port = port

    def __str__(self) -> str:
        return f"{type(self).__name__}(host={self.host!r}, port={self.port!r})"

    def __enter__(self) -> "ConnectionPool":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        self.close()
        return False

    def close(self) -> None:
        pass


_blocking_errnos: Set[int] = {errno.EAGAIN, errno.EWOULDBLOCK}


class HTTPConnectionPool(ConnectionPool, RequestMethods):
    scheme: ClassVar[str] = "http"
    ConnectionCls: ClassVar[Type[HTTP1Connection]] = HTTP1Connection
    ResponseCls: ClassVar[Type[HTTPResponse]] = HTTPResponse

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        timeout: Union[Timeout, float] = Timeout.DEFAULT_TIMEOUT,
        maxsize: int = 1,
        block: bool = False,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[Union[Retry, bool, int]] = None,
        _proxy: Optional[Url] = None,
        _proxy_headers: Optional[Dict[str, str]] = None,
        **conn_kw: Any,
    ) -> None:
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers or {})
        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)
        if retries is None:
            retries = Retry.DEFAULT
        self.timeout: Timeout = timeout
        self.retries: Retry = retries  # type: ignore
        self.pool: LifoQueue[Optional[HTTP1Connection]] = self.QueueCls(maxsize)
        self.block: bool = block
        self.proxy: Optional[Url] = _proxy
        self.proxy_headers: Dict[str, str] = _proxy_headers or {}
        for _ in xrange(maxsize):
            self.pool.put(None)
        self.num_connections: int = 0
        self.num_requests: int = 0
        self.conn_kw: Dict[str, Any] = conn_kw
        if self.proxy:
            self.conn_kw.setdefault("socket_options", [])

    def _new_conn(self) -> HTTP1Connection:
        self.num_connections += 1
        for kw in ("strict",):
            self.conn_kw.pop(kw, None)
        log.debug(
            "Starting new HTTP connection (%d): %s:%s",
            self.num_connections,
            self.host,
            self.port or "80",
        )
        return self.ConnectionCls(host=self.host, port=self.port, **self.conn_kw)

    async def _get_conn(self, timeout: Optional[float] = None) -> HTTP1Connection:
        try:
            conn: Optional[HTTP1Connection] = self.pool.get(block=self.block, timeout=timeout)
        except AttributeError:
            raise ClosedPoolError(self, "Pool is closed.")
        except queue.Empty:
            if self.block:
                raise EmptyPoolError(self, "Pool reached maximum size and no more connections are allowed.")
            conn = None
        if conn and is_connection_dropped(conn):
            log.debug("Resetting dropped connection: %s", self.host)
            conn.close()
        return conn or self._new_conn()

    async def _put_conn(self, conn: Optional[HTTP1Connection]) -> None:
        try:
            self.pool.put(conn, block=False)
        except AttributeError:
            pass
        except queue.Full:
            log.warning("Connection pool is full, discarding connection: %s", self.host)
        if conn:
            conn.close()

    async def _start_conn(self, conn: HTTP1Connection, connect_timeout: float) -> None:
        await conn.connect(connect_timeout=connect_timeout)

    def _get_timeout(self, timeout: Union[Timeout, float, object]) -> Timeout:
        if timeout is _Default:
            return self.timeout.clone()
        if isinstance(timeout, Timeout):
            return timeout.clone()
        return Timeout.from_float(cast(float, timeout))

    def _raise_timeout(self, err: Exception, url: str, timeout_value: float) -> None:
        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(self, url, f"Read timed out. (read timeout={timeout_value})")
        if hasattr(err, "errno") and err.errno in _blocking_errnos:
            raise ReadTimeoutError(self, url, f"Read timed out. (read timeout={timeout_value})")
        if "timed out" in str(err) or "did not complete (read)" in str(err):
            raise ReadTimeoutError(self, url, f"Read timed out. (read timeout={timeout_value})")

    async def _make_request(
        self,
        conn: HTTP1Connection,
        method: str,
        url: str,
        timeout: Any = _Default,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HTTPResponse:
        self.num_requests += 1
        timeout_obj: Timeout = self._get_timeout(timeout)
        timeout_obj.start_connect()
        try:
            await self._start_conn(conn, timeout_obj.connect_timeout)
        except (SocketTimeout, BaseSSLError) as e:
            self._raise_timeout(e, url, conn.timeout)
            raise
        request = Request(method=method, target=url, headers=headers or {}, body=body)
        host = self.host
        port = self.port
        scheme = self.scheme
        request.add_host(host, port, scheme)
        read_timeout = timeout_obj.read_timeout
        if read_timeout == 0:
            raise ReadTimeoutError(self, url, f"Read timed out. (read timeout={read_timeout})")
        if read_timeout is Timeout.DEFAULT_TIMEOUT:
            read_timeout = socket.getdefaulttimeout()
        try:
            response: HTTPResponse = await conn.send_request(request, read_timeout=read_timeout)
        except (SocketTimeout, BaseSSLError, SocketError) as e:
            self._raise_timeout(e, url, read_timeout)
            raise
        http_version = getattr(conn, "_http_vsn_str", "HTTP/?")
        log.debug(
            '%s://%s:%s "%s %s %s" %s',
            self.scheme,
            self.host,
            self.port,
            method,
            url,
            http_version,
            response.status_code,
        )
        return response

    def _absolute_url(self, path: str) -> str:
        return Url(scheme=self.scheme, host=self.host, port=self.port, path=path).url

    def close(self) -> None:
        if self.pool is None:
            return
        old_pool, self.pool = self.pool, None
        try:
            while True:
                conn: Optional[HTTP1Connection] = old_pool.get(block=False)
                if conn:
                    conn.close()
        except queue.Empty:
            pass

    def is_same_host(self, url: str) -> bool:
        if url.startswith("/"):
            return True
        scheme, host, port = get_host(url)
        host = _ipv6_host(host).lower()
        if self.port and not port:
            port = DEFAULT_PORTS.get(scheme)
        elif not self.port and port == DEFAULT_PORTS.get(scheme):
            port = None
        return (scheme, host, port) == (self.scheme, self.host, self.port)

    async def urlopen(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[Union[Retry, bool, int]] = None,
        timeout: Any = _Default,
        pool_timeout: Optional[float] = None,
        body_pos: Optional[int] = None,
        **response_kw: Any,
    ) -> HTTPResponse:
        headers = headers.copy() if headers else self.headers.copy()
        if self.scheme == "http":
            headers.update(self.proxy_headers)
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, default=self.retries, redirect=False)
        conn: Optional[HTTP1Connection] = None
        release_this_conn: bool = False
        clean_exit: bool = False
        body_pos = set_file_position(body, body_pos)
        if body is not None:
            _add_transport_headers(headers)
        try:
            timeout_obj: Timeout = self._get_timeout(timeout)
            conn = await self._get_conn(timeout=pool_timeout)
            conn.timeout = timeout_obj.connect_timeout
            base_response = await self._make_request(
                conn, method, url, timeout=timeout_obj, body=body, headers=headers
            )
            response = self.ResponseCls.from_base(
                base_response, pool=self, retries=retries, **response_kw
            )
            clean_exit = True
        except queue.Empty:
            raise EmptyPoolError(self, "No pool connections are available.")
        except (TimeoutError, SocketError, ProtocolError, h11.ProtocolError, BaseSSLError, SSLError, CertificateError) as e:
            clean_exit = False
            if isinstance(e, (BaseSSLError, CertificateError)):
                e = SSLError(e)
            elif isinstance(e, (SocketError, NewConnectionError)) and self.proxy:
                e = ProxyError("Cannot connect to proxy.", e)
            elif isinstance(e, (SocketError, h11.ProtocolError)):
                e = ProtocolError("Connection aborted.", e)
            retries = retries.increment(method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2])
            retries.sleep()
            err = e
        finally:
            if not clean_exit:
                conn = conn and conn.close()
                release_this_conn = True
            if release_this_conn:
                await self._put_conn(conn)
        if not conn:
            log.warning("Retrying (%r) after connection broken by '%r': %s", retries, err, url)
            return await self.urlopen(
                method,
                url,
                body,
                headers,
                retries,
                timeout=timeout,
                pool_timeout=pool_timeout,
                body_pos=body_pos,
                **response_kw
            )
        def drain_and_release_conn(response: HTTPResponse) -> None:
            try:
                response.read()
            except (TimeoutError, SocketError, ProtocolError, BaseSSLError, SSLError):
                pass
        has_retry_after = bool(response.getheader("Retry-After"))
        if retries.is_retry(method, response.status, has_retry_after):
            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_status:
                    drain_and_release_conn(response)
                    raise
                return response
            drain_and_release_conn(response)
            retries.sleep(response)
            log.debug("Retry: %s", url)
            return await self.urlopen(
                method,
                url,
                body,
                headers,
                retries=retries,
                timeout=timeout,
                pool_timeout=pool_timeout,
                body_pos=body_pos,
                **response_kw
            )
        return response


class HTTPSConnectionPool(HTTPConnectionPool):
    scheme: ClassVar[str] = "https"

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        timeout: Union[Timeout, float] = Timeout.DEFAULT_TIMEOUT,
        maxsize: int = 1,
        block: bool = False,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[Union[Retry, bool, int]] = None,
        _proxy: Optional[Url] = None,
        _proxy_headers: Optional[Dict[str, str]] = None,
        key_file: Optional[str] = None,
        cert_file: Optional[str] = None,
        cert_reqs: Optional[int] = None,
        ca_certs: Optional[str] = None,
        ssl_version: Optional[int] = None,
        assert_hostname: Optional[str] = None,
        assert_fingerprint: Optional[str] = None,
        ca_cert_dir: Optional[str] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        **conn_kw: Any,
    ) -> None:
        HTTPConnectionPool.__init__(
            self,
            host,
            port,
            timeout,
            maxsize,
            block,
            headers,
            retries,
            _proxy,
            _proxy_headers,
            **conn_kw
        )
        if ssl is None:
            raise SSLError("SSL module is not available")
        if ca_certs and cert_reqs is None:
            cert_reqs = ssl.CERT_REQUIRED
        self.ssl_context: ssl.SSLContext = _build_context(
            ssl_context,
            keyfile=key_file,
            certfile=cert_file,
            cert_reqs=cert_reqs,
            ca_certs=ca_certs,
            ca_cert_dir=ca_cert_dir,
            ssl_version=ssl_version,
        )
        self.assert_hostname: Optional[str] = assert_hostname
        self.assert_fingerprint: Optional[str] = assert_fingerprint

    def _new_conn(self) -> HTTP1Connection:
        self.num_connections += 1
        log.debug(
            "Starting new HTTPS connection (%d): %s:%s",
            self.num_connections,
            self.host,
            self.port or "443",
        )
        actual_host = self.host
        actual_port = self.port
        tunnel_host = None
        tunnel_port = None
        tunnel_headers = None
        if self.proxy is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port
            tunnel_host = self.host
            tunnel_port = self.port
            tunnel_headers = self.proxy_headers
        for kw in ("strict", "redirect"):
            self.conn_kw.pop(kw, None)
        return self.ConnectionCls(
            host=actual_host,
            port=actual_port,
            tunnel_host=tunnel_host,
            tunnel_port=tunnel_port,
            tunnel_headers=tunnel_headers,
            **self.conn_kw
        )

    async def _start_conn(self, conn: HTTP1Connection, connect_timeout: float) -> None:
        await conn.connect(
            ssl_context=self.ssl_context,
            fingerprint=self.assert_fingerprint,
            assert_hostname=self.assert_hostname,
            connect_timeout=connect_timeout,
        )
        if not conn.is_