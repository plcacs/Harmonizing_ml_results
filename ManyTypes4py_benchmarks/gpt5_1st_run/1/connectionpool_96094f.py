from __future__ import absolute_import, annotations
import errno
import logging
import sys
import warnings
from socket import error as SocketError, timeout as SocketTimeout
import socket
import h11
from ..base import Request, DEFAULT_PORTS
from ..exceptions import ClosedPoolError, ProtocolError, EmptyPoolError, LocationValueError, MaxRetryError, ProxyError, ReadTimeoutError, SSLError, TimeoutError, InsecureRequestWarning, NewConnectionError
from ..packages.ssl_match_hostname import CertificateError
from ..packages import six
from ..packages.six.moves import queue
from ..request import RequestMethods
from .response import HTTPResponse
from .connection import HTTP1Connection
from ..util.connection import is_connection_dropped
from ..util.request import set_file_position
from ..util.retry import Retry
from ..util.ssl_ import create_urllib3_context, merge_context_settings, resolve_ssl_version, resolve_cert_reqs, BaseSSLError
from ..util.timeout import Timeout
from ..util.url import get_host, Url
try:
    import ssl
except ImportError:
    ssl = None  # type: ignore[assignment]
if six.PY2:
    import Queue as _unused_module_Queue  # type: ignore[import-not-found]
xrange = six.moves.xrange
log = logging.getLogger(__name__)
from typing import Any, Dict, Optional, Set, Type, Union, IO

Headers = Dict[str, str]
TimeoutValue = Union[Timeout, float, int]
BodyType = Optional[Union[bytes, bytearray, memoryview]]

_Default: object = object()


def _add_transport_headers(headers: Headers) -> None:
    """
    Adds the transport framing headers, if needed. Naturally, this method
    cannot add a content-length header, so if there is no content-length header
    then it will add Transfer-Encoding: chunked instead. Should only be called
    if there is a body to upload.

    This should be a bit smarter: in particular, it should allow for bad or
    unexpected versions of these headers, particularly transfer-encoding.
    """
    transfer_headers = ('content-length', 'transfer-encoding')
    for header_name in headers:
        if header_name.lower() in transfer_headers:
            return
    headers['transfer-encoding'] = 'chunked'


def _build_context(
    context: Optional["ssl.SSLContext"],
    keyfile: Optional[str],
    certfile: Optional[str],
    cert_reqs: Optional[Union[str, int]],
    ca_certs: Optional[str],
    ca_cert_dir: Optional[str],
    ssl_version: Optional[int],
) -> "ssl.SSLContext":
    """
    Creates a urllib3 context suitable for a given request based on a
    collection of possible properties of that context.
    """
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
    """
    Base class for all connection pools, such as
    :class:`.HTTPConnectionPool` and :class:`.HTTPSConnectionPool`.
    """
    scheme: Optional[str] = None
    QueueCls: Type[Any] = queue.LifoQueue  # type: ignore[assignment]

    def __init__(self, host: str, port: Optional[int] = None) -> None:
        if not host:
            raise LocationValueError('No host specified.')
        self.host: str = _ipv6_host(host).lower()
        self.port: Optional[int] = port

    def __str__(self) -> str:
        return '%s(host=%r, port=%r)' % (type(self).__name__, self.host, self.port)

    def __enter__(self) -> "ConnectionPool":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """
        Close all pooled connections and disable the pool.
        """
        pass


_blocking_errnos: Set[int] = set([errno.EAGAIN, errno.EWOULDBLOCK])


class HTTPConnectionPool(ConnectionPool, RequestMethods):
    """
    Thread-safe connection pool for one host.

    :param host:
        Host used for this HTTP Connection (e.g. "localhost"), passed into
        :class:`httplib.HTTPConnection`.

    :param port:
        Port used for this HTTP Connection (None is equivalent to 80), passed
        into :class:`httplib.HTTPConnection`.

    :param strict:
        Causes BadStatusLine to be raised if the status line can't be parsed
        as a valid HTTP/1.0 or 1.1 status line, passed into
        :class:`httplib.HTTPConnection`.

        .. note::
           Only works in Python 2. This parameter is ignored in Python 3.

    :param timeout:
        Socket timeout in seconds for each individual connection. This can
        be a float or integer, which sets the timeout for the HTTP request,
        or an instance of :class:`urllib3.util.Timeout` which gives you more
        fine-grained control over request timeouts. After the constructor has
        been parsed, this is always a `urllib3.util.Timeout` object.

    :param maxsize:
        Number of connections to save that can be reused. More than 1 is useful
        in multithreaded situations. If ``block`` is set to False, more
        connections will be created but they will not be saved once they've
        been used.

    :param block:
        If set to True, no more than ``maxsize`` connections will be used at
        a time. When no free connections are available, the call will block
        until a connection has been released. This is a useful side effect for
        particular multithreaded situations where one does not want to use more
        than maxsize connections per host to prevent flooding.

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.

    :param retries:
        Retry configuration to use by default with requests in this pool.

    :param _proxy:
        Parsed proxy URL, should not be used directly, instead, see
        :class:`urllib3.connectionpool.ProxyManager`"

    :param _proxy_headers:
        A dictionary with proxy headers, should not be used directly,
        instead, see :class:`urllib3.connectionpool.ProxyManager`"

    :param \\**conn_kw:
        Additional parameters are used to create fresh :class:`urllib3.connection.HTTPConnection`,
        :class:`urllib3.connection.HTTPSConnection` instances.
    """
    scheme: str = 'http'
    ConnectionCls: Type[HTTP1Connection] = HTTP1Connection
    ResponseCls: Type[HTTPResponse] = HTTPResponse

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        timeout: TimeoutValue = Timeout.DEFAULT_TIMEOUT,
        maxsize: int = 1,
        block: bool = False,
        headers: Optional[Headers] = None,
        retries: Optional[Union[Retry, bool, int]] = None,
        _proxy: Optional[Url] = None,
        _proxy_headers: Optional[Headers] = None,
        **conn_kw: Any
    ) -> None:
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers)
        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)  # type: ignore[arg-type]
        if retries is None:
            retries = Retry.DEFAULT
        self.timeout: Timeout = timeout
        self.retries: Retry = retries if isinstance(retries, Retry) else Retry.from_int(retries, default=Retry.DEFAULT, redirect=False)
        self.pool: Optional[Any] = self.QueueCls(maxsize)
        self.block: bool = block
        self.proxy: Optional[Url] = _proxy
        self.proxy_headers: Headers = _proxy_headers or {}
        for _ in xrange(maxsize):
            self.pool.put(None)  # type: ignore[union-attr]
        self.num_connections: int = 0
        self.num_requests: int = 0
        self.conn_kw: Dict[str, Any] = dict(conn_kw)
        if self.proxy:
            self.conn_kw.setdefault('socket_options', [])

    def _new_conn(self) -> HTTP1Connection:
        """
        Return a fresh connection.
        """
        self.num_connections += 1
        for kw in ('strict',):
            if kw in self.conn_kw:
                self.conn_kw.pop(kw)
        log.debug('Starting new HTTP connection (%d): %s:%s', self.num_connections, self.host, self.port or '80')
        conn = self.ConnectionCls(host=self.host, port=self.port, **self.conn_kw)
        return conn

    def _get_conn(self, timeout: Optional[float] = None) -> HTTP1Connection:
        """
        Get a connection. Will return a pooled connection if one is available.

        If no connections are available and :prop:`.block` is ``False``, then a
        fresh connection is returned.

        :param timeout:
            Seconds to wait before giving up and raising
            :class:`urllib3.exceptions.EmptyPoolError` if the pool is empty and
            :prop:`.block` is ``True``.
        """
        conn: Optional[HTTP1Connection] = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)  # type: ignore[union-attr]
        except AttributeError:
            raise ClosedPoolError(self, 'Pool is closed.')
        except queue.Empty:
            if self.block:
                raise EmptyPoolError(self, 'Pool reached maximum size and no more connections are allowed.')
            pass
        if conn and is_connection_dropped(conn):
            log.debug('Resetting dropped connection: %s', self.host)
            conn.close()
        return conn or self._new_conn()

    def _put_conn(self, conn: Optional[HTTP1Connection]) -> None:
        """
        Put a connection back into the pool.

        :param conn:
            Connection object for the current host and port as returned by
            :meth:`._new_conn` or :meth:`._get_conn`.

        If the pool is already full, the connection is closed and discarded
        because we exceeded maxsize. If connections are discarded frequently,
        then maxsize should be increased.

        If the pool is closed, then the connection will be closed and discarded.
        """
        try:
            self.pool.put(conn, block=False)  # type: ignore[union-attr]
            return
        except AttributeError:
            pass
        except queue.Full:
            log.warning('Connection pool is full, discarding connection: %s', self.host)
        if conn:
            conn.close()

    def _start_conn(self, conn: HTTP1Connection, connect_timeout: Optional[float]) -> None:
        """
        Called right before a request is made, after the socket is created.
        """
        conn.connect(connect_timeout=connect_timeout)

    def _get_timeout(self, timeout: Union[TimeoutValue, object]) -> Timeout:
        """ Helper that always returns a :class:`urllib3.util.Timeout` """
        if timeout is _Default:
            return self.timeout.clone()
        if isinstance(timeout, Timeout):
            return timeout.clone()
        else:
            return Timeout.from_float(timeout)  # type: ignore[arg-type]

    def _raise_timeout(self, err: BaseException, url: str, timeout_value: Optional[float]) -> None:
        """Is the error actually a timeout? Will raise a ReadTimeout or pass"""
        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if hasattr(err, 'errno') and getattr(err, 'errno') in _blocking_errnos:
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if 'timed out' in str(err) or 'did not complete (read)' in str(err):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)

    def _make_request(
        self,
        conn: HTTP1Connection,
        method: str,
        url: str,
        timeout: Union[TimeoutValue, object] = _Default,
        body: BodyType = None,
        headers: Optional[Headers] = None
    ) -> Any:
        """
        Perform a request on a given urllib connection object taken from our
        pool.

        :param conn:
            a connection from one of our connection pools

        :param timeout:
            Socket timeout in seconds for the request. This can be a
            float or integer, which will set the same timeout value for
            the socket connect and the socket read, or an instance of
            :class:`urllib3.util.Timeout`, which gives you more fine-grained
            control over your timeouts.
        """
        self.num_requests += 1
        timeout_obj = self._get_timeout(timeout)
        timeout_obj.start_connect()
        try:
            self._start_conn(conn, timeout_obj.connect_timeout)
        except (SocketTimeout, BaseSSLError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)  # type: ignore[arg-type]
            raise
        request = Request(method=method, target=url, headers=headers, body=body)
        host = self.host
        port = self.port
        scheme = self.scheme
        request.add_host(host, port, scheme)
        read_timeout = timeout_obj.read_timeout
        if read_timeout == 0:
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % read_timeout)
        if read_timeout is Timeout.DEFAULT_TIMEOUT:
            read_timeout = socket.getdefaulttimeout()
        try:
            response = conn.send_request(request, read_timeout=read_timeout)
        except (SocketTimeout, BaseSSLError, SocketError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=read_timeout if isinstance(read_timeout, (int, float)) else None)
            raise
        http_version = getattr(conn, '_http_vsn_str', 'HTTP/?')
        log.debug('%s://%s:%s "%s %s %s" %s', self.scheme, self.host, self.port, method, url, http_version, response.status_code)
        return response

    def _absolute_url(self, path: str) -> str:
        return Url(scheme=self.scheme, host=self.host, port=self.port, path=path).url

    def close(self) -> None:
        """
        Close all pooled connections and disable the pool.
        """
        if self.pool is None:
            return
        old_pool, self.pool = (self.pool, None)
        try:
            while True:
                conn = old_pool.get(block=False)  # type: ignore[attr-defined]
                if conn:
                    conn.close()
        except queue.Empty:
            pass

    def is_same_host(self, url: str) -> bool:
        """
        Check if the given ``url`` is a member of the same host as this
        connection pool.
        """
        if url.startswith('/'):
            return True
        scheme, host, port = get_host(url)
        host = _ipv6_host(host).lower()
        if self.port and (not port):
            port = DEFAULT_PORTS.get(scheme)
        elif not self.port and port == DEFAULT_PORTS.get(scheme):
            port = None
        return (scheme, host, port) == (self.scheme, self.host, self.port)

    def urlopen(
        self,
        method: str,
        url: str,
        body: BodyType = None,
        headers: Optional[Headers] = None,
        retries: Optional[Union[Retry, bool, int]] = None,
        timeout: Union[TimeoutValue, object] = _Default,
        pool_timeout: Optional[float] = None,
        body_pos: Optional[int] = None,
        **response_kw: Any
    ) -> HTTPResponse:
        """
        Get a connection from the pool and perform an HTTP request. This is the
        lowest level call for making a request, so you'll need to specify all
        the raw details.

        .. note::

           More commonly, it's appropriate to use a convenience method provided
           by :class:`.RequestMethods`, such as :meth:`request`.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param body:
            Data to send in the request body (useful for creating
            POST requests, see HTTPConnectionPool.post_url for
            more convenience).

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param retries:
            Configure the number of retries to allow before raising a
            :class:`~urllib3.exceptions.MaxRetryError` exception.

            Pass ``None`` to retry until you receive a response. Pass a
            :class:`~urllib3.util.retry.Retry` object for fine-grained control
            over different types of retries.
            Pass an integer number to retry connection errors that many times,
            but no other types of errors. Pass zero to never retry.

            If ``False``, then retries are disabled and any exception is raised
            immediately. Also, instead of raising a MaxRetryError on redirects,
            the redirect response will be returned.

        :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

        :param timeout:
            If specified, overrides the default timeout for this one
            request. It may be a float (in seconds) or an instance of
            :class:`urllib3.util.Timeout`.

        :param pool_timeout:
            If set and the pool is set to block=True, then this method will
            block for ``pool_timeout`` seconds and raise EmptyPoolError if no
            connection is available within the time period.

        :param int body_pos:
            Position to seek to in file-like body in the event of a retry or
            redirect. Typically this won't need to be set because urllib3 will
            auto-populate the value when needed.

        :param \\**response_kw:
            Additional parameters are passed to
            :meth:`urllib3.response.HTTPResponse.from_httplib`
        """
        if headers is None:
            headers = self.headers  # type: ignore[assignment]
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, default=self.retries, redirect=False)  # type: ignore[arg-type]
        conn: Optional[HTTP1Connection] = None
        release_this_conn = False
        if self.scheme == 'http':
            headers = headers.copy()
            headers.update(self.proxy_headers)
        err: Optional[BaseException] = None
        clean_exit = False
        body_pos = set_file_position(body, body_pos)
        if body is not None:
            _add_transport_headers(headers)
        try:
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)
            conn.timeout = timeout_obj.connect_timeout  # type: ignore[attr-defined]
            base_response = self._make_request(conn, method, url, timeout=timeout_obj, body=body, headers=headers)
            response_kw['request_method'] = method
            response = self.ResponseCls.from_base(base_response, pool=self, retries=retries, **response_kw)  # type: ignore[arg-type]
            clean_exit = True
        except queue.Empty:
            raise EmptyPoolError(self, 'No pool connections are available.')
        except (TimeoutError, SocketError, ProtocolError, h11.ProtocolError, BaseSSLError, SSLError, CertificateError) as e:
            clean_exit = False
            if isinstance(e, (BaseSSLError, CertificateError)):
                e = SSLError(e)
            elif isinstance(e, (SocketError, NewConnectionError)) and self.proxy:
                e = ProxyError('Cannot connect to proxy.', e)  # type: ignore[assignment]
            elif isinstance(e, (SocketError, h11.ProtocolError)):
                e = ProtocolError('Connection aborted.', e)
            retries = retries.increment(method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2])  # type: ignore[union-attr]
            retries.sleep()  # type: ignore[union-attr]
            err = e
        finally:
            if not clean_exit:
                conn = conn and conn.close()  # type: ignore[assignment]
                release_this_conn = True
            if release_this_conn:
                self._put_conn(conn)
        if not conn:
            log.warning("Retrying (%r) after connection broken by '%r': %s", retries, err, url)
            return self.urlopen(method, url, body, headers, retries, timeout=timeout, pool_timeout=pool_timeout, body_pos=body_pos, **response_kw)

        def drain_and_release_conn(response: HTTPResponse) -> None:
            try:
                response.read()
            except (TimeoutError, SocketError, ProtocolError, BaseSSLError, SSLError) as e:
                pass
        has_retry_after = bool(response.getheader('Retry-After'))
        if retries.is_retry(method, response.status, has_retry_after):  # type: ignore[union-attr]
            try:
                retries = retries.increment(method, url, response=response, _pool=self)  # type: ignore[union-attr]
            except MaxRetryError:
                if retries.raise_on_status:  # type: ignore[union-attr]
                    drain_and_release_conn(response)
                    raise
                return response
            drain_and_release_conn(response)
            retries.sleep(response)  # type: ignore[union-attr]
            log.debug('Retry: %s', url)
            return self.urlopen(method, url, body, headers, retries=retries, timeout=timeout, pool_timeout=pool_timeout, body_pos=body_pos, **response_kw)
        return response


class HTTPSConnectionPool(HTTPConnectionPool):
    """
    Same as :class:`.HTTPConnectionPool`, but HTTPS.

    When Python is compiled with the :mod:`ssl` module, then
    :class:`.VerifiedHTTPSConnection` is used, which *can* verify certificates,
    instead of :class:`.HTTPSConnection`.

    :class:`.VerifiedHTTPSConnection` uses one of ``assert_fingerprint``,
    ``assert_hostname`` and ``host`` in this order to verify connections.
    If ``assert_hostname`` is False, no verification is done.

    The ``key_file``, ``cert_file``, ``cert_reqs``,
    ``ca_certs``, ``ca_cert_dir``, and ``ssl_version`` are only used if :mod:`ssl` is
    available and are fed into :meth:`urllib3.util.ssl_wrap_socket` to upgrade
    the connection socket into an SSL socket.
    """
    scheme: str = 'https'

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        timeout: TimeoutValue = Timeout.DEFAULT_TIMEOUT,
        maxsize: int = 1,
        block: bool = False,
        headers: Optional[Headers] = None,
        retries: Optional[Union[Retry, bool, int]] = None,
        _proxy: Optional[Url] = None,
        _proxy_headers: Optional[Headers] = None,
        key_file: Optional[str] = None,
        cert_file: Optional[str] = None,
        cert_reqs: Optional[Union[str, int]] = None,
        ca_certs: Optional[str] = None,
        ssl_version: Optional[int] = None,
        assert_hostname: Optional[Union[str, bool]] = None,
        assert_fingerprint: Optional[str] = None,
        ca_cert_dir: Optional[str] = None,
        ssl_context: Optional["ssl.SSLContext"] = None,
        **conn_kw: Any
    ) -> None:
        HTTPConnectionPool.__init__(self, host, port, timeout, maxsize, block, headers, retries, _proxy, _proxy_headers, **conn_kw)
        if ssl is None:
            raise SSLError('SSL module is not available')
        if ca_certs and cert_reqs is None:
            cert_reqs = 'CERT_REQUIRED'
        self.ssl_context: "ssl.SSLContext" = _build_context(
            ssl_context,
            keyfile=key_file,
            certfile=cert_file,
            cert_reqs=cert_reqs,
            ca_certs=ca_certs,
            ca_cert_dir=ca_cert_dir,
            ssl_version=ssl_version,
        )
        self.assert_hostname: Optional[Union[str, bool]] = assert_hostname
        self.assert_fingerprint: Optional[str] = assert_fingerprint

    def _new_conn(self) -> HTTP1Connection:
        """
        Return a fresh connection.
        """
        self.num_connections += 1
        log.debug('Starting new HTTPS connection (%d): %s:%s', self.num_connections, self.host, self.port or '443')
        actual_host: str = self.host
        actual_port: Optional[int] = self.port
        tunnel_host: Optional[str] = None
        tunnel_port: Optional[int] = None
        tunnel_headers: Optional[Headers] = None
        if self.proxy is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port
            tunnel_host = self.host
            tunnel_port = self.port
            tunnel_headers = self.proxy_headers
        for kw in ('strict', 'redirect'):
            if kw in self.conn_kw:
                self.conn_kw.pop(kw)
        conn = self.ConnectionCls(
            host=actual_host,
            port=actual_port,
            tunnel_host=tunnel_host,
            tunnel_port=tunnel_port,
            tunnel_headers=tunnel_headers,
            **self.conn_kw
        )
        return conn

    def _start_conn(self, conn: HTTP1Connection, connect_timeout: Optional[float]) -> None:
        """
        Called right before a request is made, after the socket is created.
        """
        conn.connect(
            ssl_context=self.ssl_context,
            fingerprint=self.assert_fingerprint,
            assert_hostname=self.assert_hostname,
            connect_timeout=connect_timeout,
        )
        if not conn.is_verified:  # type: ignore[attr-defined]
            warnings.warn('Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings', InsecureRequestWarning)


def connection_from_url(url: str, **kw: Any) -> ConnectionPool:
    """
    Given a url, return an :class:`.ConnectionPool` instance of its host.

    This is a shortcut for not having to parse out the scheme, host, and port
    of the url before creating an :class:`.ConnectionPool` instance.

    :param url:
        Absolute URL string that must include the scheme. Port is optional.

    :param \\**kw:
        Passes additional parameters to the constructor of the appropriate
        :class:`.ConnectionPool`. Useful for specifying things like
        timeout, maxsize, headers, etc.

    Example::

        >>> conn = connection_from_url('http://google.com/')
        >>> r = conn.request('GET', '/')
    """
    scheme, host, port = get_host(url)
    port = port or DEFAULT_PORTS.get(scheme, 80)
    if scheme == 'https':
        return HTTPSConnectionPool(host, port=port, **kw)
    else:
        return HTTPConnectionPool(host, port=port, **kw)


def _ipv6_host(host: str) -> str:
    """
    Process IPv6 address literals
    """
    if host.startswith('[') and host.endswith(']'):
        host = host.replace('%25', '%').strip('[]')
    return host