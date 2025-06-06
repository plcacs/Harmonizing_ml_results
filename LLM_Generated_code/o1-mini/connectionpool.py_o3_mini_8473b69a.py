from __future__ import absolute_import
import errno
import logging
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Type, Union

from socket import error as SocketError, timeout as SocketTimeout
import socket

import h11

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

try:
    import ssl
    from ssl import SSLContext
except ImportError:
    ssl = None  # type: ignore
    SSLContext = Any

if six.PY2:
    # Queue is imported for side effects on MS Windows
    import Queue as _unused_module_Queue  # noqa: F401
xrange = six.moves.xrange
log = logging.getLogger(__name__)
_Default = object()


def _add_transport_headers(headers: Dict[str, Any]) -> None:
    """
    Adds the transport framing headers, if needed. Naturally, this method
    cannot add a content-length header, so if there is no content-length header
    then it will add Transfer-Encoding: chunked instead. Should only be called
    if there is a body to upload.

    This should be a bit smarter: in particular, it should allow for bad or
    unexpected versions of these headers, particularly transfer-encoding.
    """
    transfer_headers = ("content-length", "transfer-encoding")
    for header_name in headers:
        if header_name.lower() in transfer_headers:
            return

    headers["transfer-encoding"] = "chunked"


def _build_context(
    context: Optional[SSLContext],
    keyfile: Optional[str],
    certfile: Optional[str],
    cert_reqs: Optional[str],
    ca_certs: Optional[str],
    ca_cert_dir: Optional[str],
    ssl_version: Optional[int],
) -> SSLContext:
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


# Pool objects
class ConnectionPool(object):
    """
    Base class for all connection pools, such as
    :class:`.HTTPConnectionPool` and :class:`.HTTPSConnectionPool`.
    """

    scheme: Optional[str] = None
    QueueCls = queue.LifoQueue

    def __init__(self, host: str, port: Optional[int] = None) -> None:
        if not host:
            raise LocationValueError("No host specified.")

        self.host: str = _ipv6_host(host).lower()
        self.port: Optional[int] = port

    def __str__(self) -> str:
        return "%s(host=%r, port=%r)" % (type(self).__name__, self.host, self.port)

    def __enter__(self) -> "ConnectionPool":
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> bool:
        self.close()
        # Return False to re-raise any potential exceptions
        return False

    def close(self) -> None:
        """
        Close all pooled connections and disable the pool.
        """
        pass


# This is taken from http://hg.python.org/cpython/file/7aaba721ebc0/Lib/socket.py#l252
_blocking_errnos = {errno.EAGAIN, errno.EWOULDBLOCK}


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

    scheme: str = "http"
    ConnectionCls: Type[HTTP1Connection] = HTTP1Connection
    ResponseCls: Type[HTTPResponse] = HTTPResponse

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        timeout: Union[Timeout, float] = Timeout.DEFAULT_TIMEOUT,
        maxsize: int = 1,
        block: bool = False,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[Union[Retry, int, bool]] = None,
        _proxy: Optional[Any] = None,
        _proxy_headers: Optional[Dict[str, str]] = None,
        **conn_kw: Any,
    ) -> None:
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers)
        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)  # type: ignore
        if retries is None:
            retries = Retry.DEFAULT
        self.timeout: Timeout = timeout
        self.retries: Union[Retry, int, bool] = retries
        self.pool: queue.LifoQueue = self.QueueCls(maxsize)
        self.block: bool = block
        self.proxy: Optional[Any] = _proxy
        self.proxy_headers: Dict[str, str] = _proxy_headers or {}
        # Fill the queue up so that doing get() on it will block properly
        for _ in xrange(maxsize):
            self.pool.put(None)
        # These are mostly for testing and debugging purposes.
        self.num_connections: int = 0
        self.num_requests: int = 0
        self.conn_kw: Dict[str, Any] = conn_kw
        if self.proxy:
            # Enable Nagle's algorithm for proxies, to avoid packet fragmentation.
            # We cannot know if the user has added default socket options, so we cannot replace the
            # list.
            self.conn_kw.setdefault("socket_options", [])

    def _new_conn(self) -> HTTP1Connection:
        """
        Return a fresh connection.
        """
        self.num_connections += 1

        # TODO: Huge hack.
        for kw in ("strict",):
            if kw in self.conn_kw:
                self.conn_kw.pop(kw)

        log.debug(
            "Starting new HTTP connection (%d): %s:%s",
            self.num_connections,
            self.host,
            self.port or "80",
        )
        conn: HTTP1Connection = self.ConnectionCls(host=self.host, port=self.port, **self.conn_kw)
        return conn

    def _get_conn(self, timeout: Optional[float] = None) -> Optional[HTTP1Connection]:
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
            conn = self.pool.get(block=self.block, timeout=timeout)
        except AttributeError:  # self.pool is None
            raise ClosedPoolError(self, "Pool is closed.")

        except queue.Empty:
            if self.block:
                raise EmptyPoolError(
                    self,
                    "Pool reached maximum size and no more " "connections are allowed.",
                )

            pass  # Oh well, we'll create a new connection then
        # If this is a persistent connection, check if it got disconnected
        if conn and is_connection_dropped(conn):
            log.debug("Resetting dropped connection: %s", self.host)
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
            self.pool.put(conn, block=False)
            return  # Everything is dandy, done.

        except AttributeError:
            # self.pool is None.
            pass
        except queue.Full:
            # This should never happen if self.block == True
            log.warning("Connection pool is full, discarding connection: %s", self.host)
        # Connection never got put back into the pool, close it.
        if conn:
            conn.close()

    def _start_conn(self, conn: HTTP1Connection, connect_timeout: float) -> None:
        """
        Called right before a request is made, after the socket is created.
        """
        conn.connect(connect_timeout=connect_timeout)

    def _get_timeout(self, timeout: Any) -> Timeout:
        """ Helper that always returns a :class:`urllib3.util.Timeout` """
        if timeout is _Default:
            return self.timeout.clone()

        if isinstance(timeout, Timeout):
            return timeout.clone()

        else:
            # User passed us an int/float. This is for backwards compatibility,
            # can be removed later
            return Timeout.from_float(timeout)

    def _raise_timeout(self, err: BaseException, url: str, timeout_value: float) -> None:
        """Is the error actually a timeout? Will raise a ReadTimeout or pass"""
        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(
                self, url, "Read timed out. (read timeout=%s)" % timeout_value
            )

        # See the above comment about EAGAIN in Python 3. In Python 2 we have
        # to specifically catch it and throw the timeout error
        if hasattr(err, "errno") and err.errno in _blocking_errnos:
            raise ReadTimeoutError(
                self, url, "Read timed out. (read timeout=%s)" % timeout_value
            )

        # Catch possible read timeouts thrown as SSL errors. If not the
        # case, rethrow the original. We need to do this because of:
        # http://bugs.python.org/issue10272
        # TODO: Can we remove this?
        if "timed out" in str(err) or "did not complete (read)" in str(
            err
        ):  # Python 2.6
            raise ReadTimeoutError(
                self, url, "Read timed out. (read timeout=%s)" % timeout_value
            )

    def _make_request(
        self,
        conn: HTTP1Connection,
        method: str,
        url: str,
        timeout: Union[Timeout, float] = _Default,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HTTPResponse:
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
        # Trigger any extra validation we need to do.
        try:
            self._start_conn(conn, timeout_obj.connect_timeout)
        except (SocketTimeout, BaseSSLError) as e:
            # Py2 raises this as a BaseSSLError, Py3 raises it as socket timeout.
            self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
            raise

        # TODO: We need to encapsulate our proxy logic in here somewhere.
        request = Request(method=method, target=url, headers=headers, body=body)
        host = self.host
        port = self.port
        scheme = self.scheme
        request.add_host(host, port, scheme)
        # Reset the timeout for the recv() on the socket
        read_timeout = timeout_obj.read_timeout
        # In Python 3 socket.py will catch EAGAIN and return None when you
        # try and read into the file pointer created by http.client, which
        # instead raises a BadStatusLine exception. Instead of catching
        # the exception and assuming all BadStatusLine exceptions are read
        # timeouts, check for a zero timeout before making the request.
        if read_timeout == 0:
            raise ReadTimeoutError(
                self, url, "Read timed out. (read timeout=%s)" % read_timeout
            )

        if read_timeout is Timeout.DEFAULT_TIMEOUT:
            read_timeout = socket.getdefaulttimeout()
        # Receive the response from the server
        try:
            response = conn.send_request(request, read_timeout=read_timeout)
        except (SocketTimeout, BaseSSLError, SocketError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
            raise

        # AppEngine doesn't have a version attr.
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
        """
        Close all pooled connections and disable the pool.
        """
        if self.pool is None:
            return

        # Disable access to the pool
        old_pool, self.pool = self.pool, None
        try:
            while True:
                conn = old_pool.get(block=False)
                if conn:
                    conn.close()
        except queue.Empty:
            pass  # Done.

    def is_same_host(self, url: str) -> bool:
        """
        Check if the given ``url`` is a member of the same host as this
        connection pool.
        """
        if url.startswith("/"):
            return True

        # TODO: Add optional support for socket.gethostbyname checking.
        scheme, host, port = get_host(url)
        host = _ipv6_host(host).lower()
        # Use explicit default port for comparison when none is given
        if self.port and not port:
            port = DEFAULT_PORTS.get(scheme)
        elif not self.port and port == DEFAULT_PORTS.get(scheme):
            port = None
        return (scheme, host, port) == (self.scheme, self.host, self.port)

    def urlopen(
        self,
        method: str,
        url: str,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[Union[Retry, int, bool]] = None,
        timeout: Any = _Default,
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
            Retry configuration to use by default with requests in this pool.

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
            headers = self.headers
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, default=self.retries, redirect=False)
        conn: Optional[HTTP1Connection] = None
        # Track whether `conn` needs to be released before
        # returning/raising/recursing.
        release_this_conn: bool = False
        # Merge the proxy headers. Only do this in HTTP. We have to copy the
        # headers dict so we can safely change it without those changes being
        # reflected in anyone else's copy.
        if self.scheme == "http":
            headers = headers.copy() if headers else {}
            headers.update(self.proxy_headers)
        # Must keep the exception bound to a separate variable or else Python 3
        # complains about UnboundLocalError.
        err: Optional[BaseException] = None
        # Keep track of whether we cleanly exited the except block. This
        # ensures we do proper cleanup in finally.
        clean_exit: bool = False
        # Rewind body position, if needed. Record current position
        # for future rewinds in the event of a redirect/retry.
        body_pos = set_file_position(body, body_pos)
        if body is not None:
            _add_transport_headers(headers)
        try:
            # Request a connection from the queue.
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)
            conn.timeout = timeout_obj.connect_timeout
            # Make the request on the base connection object.
            base_response = self._make_request(
                conn, method, url, timeout=timeout_obj, body=body, headers=headers
            )
            # Pass method to Response for length checking
            response_kw["request_method"] = method
            # Import httplib's response into our own wrapper object
            response = self.ResponseCls.from_base(
                base_response, pool=self, retries=retries, **response_kw
            )
            # Everything went great!
            clean_exit = True
        except queue.Empty:
            # Timed out by queue.
            raise EmptyPoolError(self, "No pool connections are available.")

        except (
            TimeoutError,
            SocketError,
            ProtocolError,
            h11.ProtocolError,
            BaseSSLError,
            SSLError,
            CertificateError,
        ) as e:
            # Discard the connection for these exceptions. It will be
            # replaced during the next _get_conn() call.
            clean_exit = False
            if isinstance(e, (BaseSSLError, CertificateError)):
                e = SSLError(e)
            elif isinstance(e, (SocketError, NewConnectionError)) and self.proxy:
                e = ProxyError("Cannot connect to proxy.", e)
            elif isinstance(e, (SocketError, h11.ProtocolError)):
                e = ProtocolError("Connection aborted.", e)
            retries = retries.increment(
                method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2]
            )
            retries.sleep()
            # Keep track of the error for the retry warning.
            err = e
        finally:
            if not clean_exit:
                # We hit some kind of exception, handled or otherwise. We need
                # to throw the connection away unless explicitly told not to.
                # Close the connection, set the variable to None, and make sure
                # we put the None back in the pool to avoid leaking it.
                conn = conn and conn.close()
                release_this_conn = True
            if release_this_conn:
                # Put the connection back to be reused. If the connection is
                # expired then it will be None, which will get replaced with a
                # fresh connection during _get_conn.
                self._put_conn(conn)
        if not conn:
            # Try again
            log.warning(
                "Retrying (%r) after connection " "broken by '%r': %s",
                retries,
                err,
                url,
            )
            return self.urlopen(
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
                # discard any remaining response body, the connection will be
                # released back to the pool once the entire response is read
                response.read()
            except (
                TimeoutError,
                SocketError,
                ProtocolError,
                BaseSSLError,
                SSLError,
            ):
                pass

        # Check if we should retry the HTTP response.
        has_retry_after = bool(response.getheader("Retry-After"))
        if retries.is_retry(method, response.status, has_retry_after):
            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_status:
                    # Drain and release the connection for this response, since
                    # we're not returning it to be released manually.
                    drain_and_release_conn(response)
                    raise

                return response

            # drain and return the connection to the pool before recursing
            drain_and_release_conn(response)
            retries.sleep(response)
            log.debug("Retry: %s", url)
            return self.urlopen(
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
    """
    Same as :class:`.HTTPConnectionPool`, but HTTPS.

    When Python is compiled with the :mod:`ssl` module, then
    :class:`.VerifiedHTTPSConnection` is used, which *can* verify certificates,
    instead of :class:`.HTTPSConnection`.

    :class:`.VerifiedHTTPSConnection` uses one of ``assert_fingerprint``,
    ``assert_hostname`` and ``host`` in this order to verify connections.
    If ``assert_hostname`` is False, no verification is done.

    The ``key_file``, ``cert_file``, ``cert_reqs``, ``ca_certs``,
    ``ca_cert_dir``, and ``ssl_version`` are only used if :mod:`ssl` is
    available and are fed into :meth:`urllib3.util.ssl_wrap_socket` to upgrade
    the connection socket into an SSL socket.
    """

    scheme: str = "https"

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        timeout: Union[Timeout, float] = Timeout.DEFAULT_TIMEOUT,
        maxsize: int = 1,
        block: bool = False,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[Union[Retry, int, bool]] = None,
        _proxy: Optional[Any] = None,
        _proxy_headers: Optional[Dict[str, str]] = None,
        key_file: Optional[str] = None,
        cert_file: Optional[str] = None,
        cert_reqs: Optional[str] = None,
        ca_certs: Optional[str] = None,
        ssl_version: Optional[int] = None,
        assert_hostname: Optional[bool] = None,
        assert_fingerprint: Optional[str] = None,
        ca_cert_dir: Optional[str] = None,
        ssl_context: Optional[SSLContext] = None,
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
            cert_reqs = "CERT_REQUIRED"
        self.ssl_context: SSLContext = _build_context(
            ssl_context,
            keyfile=key_file,
            certfile=cert_file,
            cert_reqs=cert_reqs,
            ca_certs=ca_certs,
            ca_cert_dir=ca_cert_dir,
            ssl_version=ssl_version,
        )
        self.assert_hostname: Optional[bool] = assert_hostname
        self.assert_fingerprint: Optional[str] = assert_fingerprint

    def _new_conn(self) -> HTTP1Connection:
        """
        Return a fresh connection.
        """
        self.num_connections += 1
        log.debug(
            "Starting new HTTPS connection (%d): %s:%s",
            self.num_connections,
            self.host,
            self.port or "443",
        )
        actual_host: str = self.host
        actual_port: Optional[int] = self.port
        tunnel_host: Optional[str] = None
        tunnel_port: Optional[int] = None
        tunnel_headers: Optional[Dict[str, str]] = None
        if self.proxy is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port
            tunnel_host = self.host
            tunnel_port = self.port
            tunnel_headers = self.proxy_headers

        # TODO: Huge hack.
        for kw in ("strict", "redirect"):
            if kw in self.conn_kw:
                self.conn_kw.pop(kw)

        conn: HTTP1Connection = self.ConnectionCls(
            host=actual_host,
            port=actual_port,
            tunnel_host=tunnel_host,
            tunnel_port=tunnel_port,
            tunnel_headers=tunnel_headers,
            **self.conn_kw
        )
        return conn

    def _start_conn(self, conn: HTTP1Connection, connect_timeout: float) -> None:
        """
        Called right before a request is made, after the socket is created.
        """
        conn.connect(
            ssl_context=self.ssl_context,
            fingerprint=self.assert_fingerprint,
            assert_hostname=self.assert_hostname,
            connect_timeout=connect_timeout,
        )
        if not conn.is_verified:
            warnings.warn(
                (
                    "Unverified HTTPS request is being made. "
                    "Adding certificate verification is strongly advised. See: "
                    "https://urllib3.readthedocs.io/en/latest/advanced-usage.html"
                    "#ssl-warnings"
                ),
                InsecureRequestWarning,
            )


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
    if scheme == "https":
        return HTTPSConnectionPool(host, port=port, **kw)

    else:
        return HTTPConnectionPool(host, port=port, **kw)


def _ipv6_host(host: str) -> str:
    """
    Process IPv6 address literals
    """
    # httplib doesn't like it when we include brackets in IPv6 addresses
    # Specifically, if we include brackets but also pass the port then
    # httplib crazily doubles up the square brackets on the Host header.
    # Instead, we need to make sure we never pass ``None`` as the port.
    # However, for backward compatibility reasons we can't actually
    # *assert* that.  See http://bugs.python.org/issue28539
    #
    # Also if an IPv6 address literal has a zone identifier, the
    # percent sign might be URIencoded, convert it back into ASCII
    if host.startswith("[") and host.endswith("]"):
        host = host.replace("%25", "%").strip("[]")
    return host
