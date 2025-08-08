from __future__ import absolute_import
import collections
import datetime
import socket
import warnings
import h11
from ..base import Request, Response
from ..exceptions import ConnectTimeoutError, NewConnectionError, SubjectAltNameWarning, SystemTimeWarning, BadVersionError, FailedTunnelError, InvalidBodyError, ProtocolError
from ..packages import six
from ..util import ssl_ as ssl_util
from .._backends import SyncBackend
from .._backends._common import LoopAbort
try:
    import ssl
except ImportError:
    ssl = None
RECENT_DATE: datetime.date = datetime.date(2016, 1, 1)
_SUPPORTED_VERSIONS: frozenset = frozenset([b'1.0', b'1.1'])
_EAGAIN: object = object()

def _headers_to_native_string(headers: collections.Iterable) -> collections.Iterable:
    ...

def _stringify_headers(headers: collections.Iterable) -> collections.Iterable:
    ...

def _read_readable(readable: socket.socket) -> collections.Iterable:
    ...

def _make_body_iterable(body: Any) -> collections.Iterable:
    ...

def _request_bytes_iterable(request: Request, state_machine: h11.Connection) -> collections.Iterable:
    ...

def _response_from_h11(h11_response: h11.Response, body_object: Any) -> Response:
    ...

def _build_tunnel_request(host: str, port: int, headers: collections.Iterable) -> Request:
    ...

def _start_http_request(request: Request, state_machine: h11.Connection, conn: socket.socket) -> h11.Response:
    ...

def _read_until_event(state_machine: h11.Connection, conn: socket.socket) -> h11.Event:
    ...

_DEFAULT_SOCKET_OPTIONS: object = object()

class HTTP1Connection:
    default_socket_options: List[Tuple[int, int, int]] = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]

    def __init__(self, host: str, port: int, backend: SyncBackend = None, socket_options: Union[List[Tuple[int, int, int]], object] = _DEFAULT_SOCKET_OPTIONS, source_address: Optional[Tuple[str, int]] = None, tunnel_host: Optional[str] = None, tunnel_port: Optional[int] = None, tunnel_headers: Optional[collections.Iterable] = None):
        ...

    def _wrap_socket(self, conn: socket.socket, ssl_context: ssl.SSLContext, fingerprint: Optional[str], assert_hostname: Optional[bool]) -> socket.socket:
        ...

    def send_request(self, request: Request, read_timeout: int) -> Response:
        ...

    def _tunnel(self, conn: socket.socket) -> None:
        ...

    def connect(self, ssl_context: Optional[ssl.SSLContext], fingerprint: Optional[str], assert_hostname: Optional[bool], connect_timeout: Optional[int]) -> None:
        ...

    def close(self) -> None:
        ...

    def is_dropped(self) -> bool:
        ...

    def _reset(self) -> None:
        ...

    @property
    def complete(self) -> bool:
        ...

    def __iter__(self) -> 'HTTP1Connection':
        ...

    def next(self) -> bytes:
        ...

    def __next__(self) -> bytes:
        ...
