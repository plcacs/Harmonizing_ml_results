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

class ConnectionPool(object):
    """
    Base class for all connection pools, such as
    :class:`.HTTPConnectionPool` and :class:`.HTTPSConnectionPool`.
    """
    scheme: str
    QueueCls: type

    def __init__(self, host: str, port: int | None) -> None:
        if not host:
            raise LocationValueError('No host specified.')
        self.host: str = _ipv6_host(host).lower()
        self.port: int | None = port

    # ...

class HTTPConnectionPool(ConnectionPool, RequestMethods):
    """
    Thread-safe connection pool for one host.

    :param host:
        Host used for this HTTP Connection (e.g. "localhost"), passed into
        :class:`httplib.HTTPConnection`.

    :param port:
        Port used for this HTTP Connection (None is equivalent to 80), passed
        into :class:`httplib.HTTPConnection`.

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
        a time. When no free connections are available and :prop:`.block` is
        ``False``, then a fresh connection is returned.

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

    :param **conn_kw:
        Additional parameters are used to create fresh :class:`urllib3.connection.HTTPConnection`,
        :class:`urllib3.connection.HTTPSConnection` instances.
    """
    scheme: str = 'http'
    ConnectionCls: type
    ResponseCls: type

    # ...

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
    scheme: str = 'https'

    def __init__(self, host: str, port: int | None = None, timeout: Timeout | float | int = Timeout.DEFAULT_TIMEOUT, maxsize: int = 1, block: bool = False, headers: dict | None = None, retries: Retry | int | bool | None = None, _proxy: str | None = None, _proxy_headers: dict | None = None, key_file: str | None = None, cert_file: str | None = None, cert_reqs: str | None = None, ca_certs: str | None = None, ssl_version: str | None = None, assert_hostname: bool | None = None, assert_fingerprint: str | None = None, ca_cert_dir: str | None = None, ssl_context: ssl.SSLContext | None = None, **conn_kw: dict) -> None:
        HTTPConnectionPool.__init__(self, host, port, timeout, maxsize, block, headers, retries, _proxy, _proxy_headers, **conn_kw)
        if ssl is None:
            raise SSLError('SSL module is not available')
        if ca_certs and cert_reqs is None:
            cert_reqs = 'CERT_REQUIRED'
        self.ssl_context: ssl.SSLContext = _build_context(ssl_context, keyfile=key_file, certfile=cert_file, cert_reqs=resolve_cert_reqs(cert_reqs), ca_certs=ca_certs, ca_cert_dir=ca_cert_dir, ssl_version=resolve_ssl_version(ssl_version))
        self.assert_hostname: bool | None = assert_hostname
        self.assert_fingerprint: str | None = assert_fingerprint

    # ...

def connection_from_url(url: str, **kw: dict) -> ConnectionPool:
    """
    Given a url, return an :class:`.ConnectionPool` instance of its host.

    This is a shortcut for not having to parse out the scheme, host, and port
    of the url before creating an :class:`.ConnectionPool` instance.

    :param url:
        Absolute URL string that must include the scheme. Port is optional.

    :param **kw:
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
