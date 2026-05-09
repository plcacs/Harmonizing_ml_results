from __future__ import absolute_import
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

def _add_transport_headers(headers: dict) -> None:
    """
    Adds the transport framing headers, if needed. Naturally, this method
    cannot add a content-length header, so if there is no content-length header
    then it will add Transfer-Encoding: chunked instead. Should only be called
    if there is a body to upload.
    """
    transfer_headers = ('content-length', 'transfer-encoding')
    for header_name in headers:
        if header_name.lower() in transfer_headers:
            return
    headers['transfer-encoding'] = 'chunked'

def _build_context(context: object, keyfile: str, certfile: str, cert_reqs: str, ca_certs: str, ca_cert_dir: str, ssl_version: str) -> object:
    """
    Creates a urllib3 context suitable for a given request based on a
    collection of possible properties of that context.
    """
    if context is None:
        context = create_urllib3_context(ssl_version=resolve_ssl_version(ssl_version), cert_reqs=resolve_cert_reqs(cert_reqs))
    context = merge_context_settings(context, keyfile=keyfile, certfile=certfile, cert_reqs=cert_reqs, ca_certs=ca_certs, ca_cert_dir=ca_cert_dir)
    return context

class ConnectionPool(object):
    """
    Base class for all connection pools, such as
    :class:`.HTTPConnectionPool` and :class:`.HTTPSConnectionPool`.
    """
    scheme: str
    QueueCls = queue.LifoQueue

    def __init__(self, host: str, port: int = None) -> None:
        if not host:
            raise LocationValueError('No host specified.')
        self.host = _ipv6_host(host).lower()
        self.port = port

    def __str__(self) -> str:
        return '%s(host=%r, port=%r)' % (type(self).__name__, self.host, self.port)

    def __enter__(self) -> 'ConnectionPool':
        return self

    def __exit__(self, exc_type: type, exc_val: object, exc_tb: type) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """
        Close all pooled connections and disable the pool.
        """
        pass

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
    scheme = 'http'
    ConnectionCls = HTTP1Connection
    ResponseCls = HTTPResponse

    def __init__(self, host: str, port: int = None, timeout: Timeout = Timeout.DEFAULT_TIMEOUT, maxsize: int = 1, block: bool = False, headers: dict = None, retries: Retry = None, _proxy: object = None, _proxy_headers: dict = None, **conn_kw: dict) -> None:
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers)
        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)
        if retries is None:
            retries = Retry.DEFAULT
        self.timeout = timeout
        self.retries = retries
        self.pool = self.QueueCls(maxsize)
        self.block = block
        self.proxy = _proxy
        self.proxy_headers = _proxy_headers or {}
        for _ in range(maxsize):
            self.pool.put(None)
        self.num_connections = 0
        self.num_requests = 0
        self.conn_kw = conn_kw
        if self.proxy:
            self.conn_kw.setdefault('socket_options', [])

    def _new_conn(self) -> HTTP1Connection:
        """
        Return a fresh connection.
        """
        self.num_connections += 1
        log.debug('Starting new HTTP connection (%d): %s:%s', self.num_connections, self.host, self.port or '80')
        conn = self.ConnectionCls(host=self.host, port=self.port, **self.conn_kw)
        return conn

    async def _get_conn(self, timeout: Timeout = None) -> HTTP1Connection:
        """
        Get a connection. Will return a pooled connection if one is available.

        If no connections are available and :prop:`.block` is ``False``, then a
        fresh connection is returned.

        :param timeout:
            Seconds to wait before giving up and raising
            :class:`urllib3.exceptions.EmptyPoolError` if the pool is empty and
            :prop:`.block` is ``True``.
        """
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)
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

    async def _put_conn(self, conn: HTTP1Connection) -> None:
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
            self.pool.put(conn, block=False)
            return
        except AttributeError:
            pass
        except queue.Full:
            log.warning('Connection pool is full, discarding connection: %s', self.host)
        if conn:
            conn.close()

    async def _start_conn(self, conn: HTTP1Connection, connect_timeout: float) -> None:
        """
        Called right before a request is made, after the socket is created.
        """
        await conn.connect(connect_timeout=connect_timeout)

    def _get_timeout(self, timeout: Timeout) -> Timeout:
        """ Helper that always returns a :class:`urllib3.util.Timeout` """
        if timeout is ConnectionPool._Default:
            return self.timeout.clone()
        if isinstance(timeout, Timeout):
            return timeout.clone()
        else:
            return Timeout.from_float(timeout)

    def _raise_timeout(self, err: Exception, url: str, timeout_value: float) -> None:
        """Is the error actually a timeout? Will raise a ReadTimeout or pass"""
        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if hasattr(err, 'errno') and err.errno in self._blocking_errnos:
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if 'timed out' in str(err) or 'did not complete (read)' in str(err):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)

    async def _make_request(self, conn: HTTP1Connection, method: str, url: str, timeout: Timeout = ConnectionPool._Default, body: bytes = None, headers: dict = None) -> HTTPResponse:
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
            await self._start_conn(conn, timeout_obj.connect_timeout)
        except (SocketTimeout, BaseSSLError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
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
            response = await conn.send_request(request, read_timeout=read_timeout)
        except (SocketTimeout, BaseSSLError, SocketError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
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
                conn = old_pool.get(block=False)
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

    async def urlopen(self, method: str, url: str, body: bytes = None, headers: dict = None, retries: Retry = None, timeout: Timeout = ConnectionPool._Default, pool_timeout: float = None, body_pos: int = None, **response_kw: dict) -> HTTPResponse:
        """
        Get a connection from the pool and perform an HTTP request. This is the
        lowest level call for making a request, so you'll need to specify all
        the raw details.

        .. note::

           More commonly, it's appropriate to use a convenience method provided
           by :class:`.RequestMethods`, such as :meth:`request`.

        :