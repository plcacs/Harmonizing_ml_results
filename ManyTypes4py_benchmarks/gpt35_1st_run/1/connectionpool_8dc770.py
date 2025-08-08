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
    ...

class HTTPConnectionPool(ConnectionPool, RequestMethods):
    ...

class HTTPSConnectionPool(HTTPConnectionPool):
    ...

def connection_from_url(url: str, **kw) -> ConnectionPool:
    ...
