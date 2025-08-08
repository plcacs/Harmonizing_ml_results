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
try:
    import ssl
except ImportError:
    ssl = None
if six.PY2:
    import Queue as _unused_module_Queue
xrange = six.moves.xrange
log = logging.getLogger(__name__)
_Default = object()

def _add_transport_headers(headers: dict) -> None:
    ...

def _build_context(context: ssl.SSLContext, keyfile: str, certfile: str, cert_reqs: str, ca_certs: str, ca_cert_dir: str, ssl_version: str) -> ssl.SSLContext:
    ...

class ConnectionPool(object):
    scheme: str = None
    QueueCls = queue.LifoQueue

    def __init__(self, host: str, port: int = None) -> None:
        ...

    def __str__(self) -> str:
        ...

    def __enter__(self) -> 'ConnectionPool':
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        ...

    def close(self) -> None:
        ...

_blocking_errnos: set = set([errno.EAGAIN, errno.EWOULDBLOCK])

class HTTPConnectionPool(ConnectionPool, RequestMethods):
    scheme: str = 'http'
    ConnectionCls = HTTP1Connection
    ResponseCls = HTTPResponse

    def __init__(self, host: str, port: int = None, timeout: Union[float, Timeout] = Timeout.DEFAULT_TIMEOUT, maxsize: int = 1, block: bool = False, headers: dict = None, retries: Retry = None, _proxy: Any = None, _proxy_headers: dict = None, **conn_kw: Any) -> None:
        ...

    def _new_conn(self) -> HTTP1Connection:
        ...

    def _get_conn(self, timeout: Optional[float] = None) -> HTTP1Connection:
        ...

    def _put_conn(self, conn: HTTP1Connection) -> None:
        ...

    def _start_conn(self, conn: HTTP1Connection, connect_timeout: float) -> None:
        ...

    def _get_timeout(self, timeout: Union[float, Timeout]) -> Timeout:
        ...

    def _raise_timeout(self, err: Union[SocketTimeout, SocketError], url: str, timeout_value: float) -> None:
        ...

    def _make_request(self, conn: HTTP1Connection, method: str, url: str, timeout: Union[float, Timeout], body: Optional[bytes], headers: dict) -> HTTPResponse:
        ...

    def _absolute_url(self, path: str) -> str:
        ...

    def close(self) -> None:
        ...

    def is_same_host(self, url: str) -> bool:
        ...

    def urlopen(self, method: str, url: str, body: Optional[bytes] = None, headers: dict = None, retries: Union[Retry, int] = None, timeout: Union[float, Timeout] = _Default, pool_timeout: Optional[float] = None, body_pos: Optional[int] = None, **response_kw: Any) -> HTTPResponse:
        ...

class HTTPSConnectionPool(HTTPConnectionPool):
    scheme: str = 'https'

    def __init__(self, host: str, port: int = None, timeout: Union[float, Timeout] = Timeout.DEFAULT_TIMEOUT, maxsize: int = 1, block: bool = False, headers: dict = None, retries: Retry = None, _proxy: Any = None, _proxy_headers: dict = None, key_file: str = None, cert_file: str = None, cert_reqs: str = None, ca_certs: str = None, ssl_version: str = None, assert_hostname: bool = None, assert_fingerprint: str = None, ca_cert_dir: str = None, ssl_context: ssl.SSLContext = None, **conn_kw: Any) -> None:
        ...

    def _new_conn(self) -> HTTP1Connection:
        ...

    def _start_conn(self, conn: HTTP1Connection, connect_timeout: float) -> None:
        ...

def connection_from_url(url: str, **kw: Any) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
    ...
