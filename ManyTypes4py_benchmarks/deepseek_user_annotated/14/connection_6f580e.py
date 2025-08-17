# -*- coding: utf-8 -*-
"""
This module implements the connection management logic.
"""
from __future__ import absolute_import

import collections
import datetime
import socket
import warnings
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import h11

from ..base import Request, Response
from ..exceptions import (
    ConnectTimeoutError,
    NewConnectionError,
    SubjectAltNameWarning,
    SystemTimeWarning,
    BadVersionError,
    FailedTunnelError,
    InvalidBodyError,
    ProtocolError,
)
from ..packages import six
from ..util import ssl_ as ssl_util
from .._backends import SyncBackend
from .._backends._common import LoopAbort

try:
    import ssl
except ImportError:
    ssl = None  # type: ignore

RECENT_DATE = datetime.date(2016, 1, 1)
_SUPPORTED_VERSIONS = frozenset([b"1.0", b"1.1"])
_EAGAIN = object()

def _headers_to_native_string(
    headers: Iterable[Tuple[bytes, bytes]]
) -> Generator[Tuple[str, str], None, None]:
    for n, v in headers:
        if not isinstance(n, str):
            n = n.decode("latin1")
        if not isinstance(v, str):
            v = v.decode("latin1")
        yield (n, v)

def _stringify_headers(
    headers: Iterable[Tuple[Union[str, bytes], Union[str, bytes, int]]]
) -> Generator[Tuple[bytes, bytes], None, None]:
    for name, value in headers:
        if isinstance(name, six.text_type):
            name = name.encode("ascii")
        if isinstance(value, six.text_type):
            value = value.encode("latin-1")
        elif isinstance(value, int):
            value = str(value).encode("ascii")
        yield (name, value)

def _read_readable(readable: Any) -> Generator[bytes, None, None]:
    blocksize = 8192
    while True:
        datablock = readable.read(blocksize)
        if not datablock:
            break
        yield datablock

def _make_body_iterable(
    body: Optional[Union[bytes, Iterable[bytes], Any]]
) -> Iterable[bytes]:
    if body is None:
        return []
    elif isinstance(body, six.binary_type):
        return [body]
    elif hasattr(body, "read"):
        return _read_readable(body)
    elif isinstance(body, collections.Iterable) and not isinstance(body, six.text_type):
        return body
    else:
        raise InvalidBodyError("Unacceptable body type: %s" % type(body))

def _request_bytes_iterable(
    request: Request, state_machine: h11.Connection
) -> Generator[bytes, None, None]:
    h11_request = h11.Request(
        method=request.method,
        target=request.target,
        headers=_stringify_headers(request.headers.items()),
    )
    yield state_machine.send(h11_request)

    for chunk in _make_body_iterable(request.body):
        yield state_machine.send(h11.Data(data=chunk))

    yield state_machine.send(h11.EndOfMessage())

def _response_from_h11(
    h11_response: h11.Response, body_object: Any
) -> Response:
    if h11_response.http_version not in _SUPPORTED_VERSIONS:
        raise BadVersionError(h11_response.http_version)

    version = b"HTTP/" + h11_response.http_version
    our_response = Response(
        status_code=h11_response.status_code,
        headers=_headers_to_native_string(h11_response.headers),
        body=body_object,
        version=version,
    )
    return our_response

def _build_tunnel_request(
    host: str, port: int, headers: Optional[Dict[str, str]]
) -> Request:
    target = "%s:%d" % (host, port)
    if not isinstance(target, bytes):
        target = target.encode("latin1")
    tunnel_request = Request(method=b"CONNECT", target=target, headers=headers)
    tunnel_request.add_host(host=host, port=port, scheme="http")
    return tunnel_request

def _start_http_request(
    request: Request, state_machine: h11.Connection, conn: Any
) -> h11.Response:
    if (
        state_machine.our_state is not h11.IDLE
        or state_machine.their_state is not h11.IDLE
    ):
        raise ProtocolError("Invalid internal state transition")

    request_bytes_iterable = _request_bytes_iterable(request, state_machine)
    context = {"send_aborted": True, "h11_response": None}  # type: Dict[str, Any]

    def next_bytes_to_send() -> Optional[bytes]:
        try:
            return next(request_bytes_iterable)
        except StopIteration:
            context["send_aborted"] = False
            return None

    def consume_bytes(data: bytes) -> None:
        state_machine.receive_data(data)
        while True:
            event = state_machine.next_event()
            if event is h11.NEED_DATA:
                break
            elif isinstance(event, h11.InformationalResponse):
                continue
            elif isinstance(event, h11.Response):
                context["h11_response"] = event
                raise LoopAbort
            else:
                raise RuntimeError("Unexpected h11 event {}".format(event))

    conn.send_and_receive_for_a_while(next_bytes_to_send, consume_bytes)
    assert context["h11_response"] is not None
    if context["send_aborted"]:
        state_machine._cstate.process_error(state_machine.our_role)
    return cast(h11.Response, context["h11_response"])

def _read_until_event(
    state_machine: h11.Connection, conn: Any
) -> Union[h11.Data, h11.EndOfMessage]:
    while True:
        event = state_machine.next_event()
        if event is not h11.NEED_DATA:
            return event
        state_machine.receive_data(conn.receive_some())

_DEFAULT_SOCKET_OPTIONS = object()

class HTTP1Connection:
    default_socket_options = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]

    def __init__(
        self,
        host: str,
        port: int,
        backend: Optional[SyncBackend] = None,
        socket_options: Any = _DEFAULT_SOCKET_OPTIONS,
        source_address: Optional[str] = None,
        tunnel_host: Optional[str] = None,
        tunnel_port: Optional[int] = None,
        tunnel_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.is_verified = False
        self._backend = backend or SyncBackend()
        self._host = host
        self._port = port
        self._socket_options = (
            socket_options
            if socket_options is not _DEFAULT_SOCKET_OPTIONS
            else self.default_socket_options
        )
        self._source_address = source_address
        self._tunnel_host = tunnel_host
        self._tunnel_port = tunnel_port
        self._tunnel_headers = tunnel_headers
        self._sock = None  # type: Optional[Any]
        self._state_machine = h11.Connection(our_role=h11.CLIENT)

    def _wrap_socket(
        self,
        conn: Any,
        ssl_context: Optional[ssl.SSLContext],
        fingerprint: Optional[str],
        assert_hostname: Optional[Union[str, bool]],
    ) -> Any:
        is_time_off = datetime.date.today() < RECENT_DATE
        if is_time_off:
            warnings.warn(
                (
                    "System time is way off (before {0}). This will probably "
                    "lead to SSL verification errors"
                ).format(RECENT_DATE),
                SystemTimeWarning,
            )
        check_host = assert_hostname or self._tunnel_host or self._host
        check_host = check_host.rstrip(".")
        conn = conn.start_tls(check_host, ssl_context)
        if fingerprint:
            ssl_util.assert_fingerprint(conn.getpeercert(binary_form=True), fingerprint)
        elif ssl_context.verify_mode != ssl.CERT_NONE and assert_hostname is not False:
            cert = conn.getpeercert()
            if not cert.get("subjectAltName", ()):
                warnings.warn(
                    (
                        "Certificate for {0} has no `subjectAltName`, falling "
                        "back to check for a `commonName` for now. This "
                        "feature is being removed by major browsers and "
                        "deprecated by RFC 2818. (See "
                        "https://github.com/shazow/urllib3/issues/497 for "
                        "details.)".format(self._host)
                    ),
                    SubjectAltNameWarning,
                )
            ssl_util.match_hostname(cert, check_host)
        self.is_verified = ssl_context.verify_mode == ssl.CERT_REQUIRED and (
            assert_hostname is not False or fingerprint
        )
        return conn

    def send_request(
        self, request: Request, read_timeout: Optional[float]
    ) -> Response:
        h11_response = _start_http_request(request, self._state_machine, self._sock)
        return _response_from_h11(h11_response, self)

    def _tunnel(self, conn: Any) -> None:
        assert self._state_machine.our_state is h11.IDLE
        tunnel_request = _build_tunnel_request(
            self._tunnel_host, self._tunnel_port, self._tunnel_headers
        )
        tunnel_state_machine = h11.Connection(our_role=h11.CLIENT)
        h11_response = _start_http_request(tunnel_request, tunnel_state_machine, conn)
        tunnel_response = _response_from_h11(h11_response, self)
        if h11_response.status_code != 200:
            conn.forceful_close()
            raise FailedTunnelError(
                "Unable to establish CONNECT tunnel", tunnel_response
            )

    def connect(
        self,
        ssl_context: Optional[ssl.SSLContext] = None,
        fingerprint: Optional[str] = None,
        assert_hostname: Optional[Union[str, bool]] = None,
        connect_timeout: Optional[float] = None,
    ) -> None:
        if self._sock is not None:
            self._sock.set_readable_watch_state(False)
            return

        extra_kw = {}  # type: Dict[str, Any]
        if self._source_address:
            extra_kw["source_address"] = self._source_address
        if self._socket_options:
            extra_kw["socket_options"] = self._socket_options

        try:
            conn = self._backend.connect(self._host, self._port, **extra_kw)
        except socket.timeout:
            raise ConnectTimeoutError(
                self,
                "Connection to %s timed out. (connect timeout=%s)"
                % (self._host, connect_timeout),
            )
        except socket.error as e:
            raise NewConnectionError(
                self, "Failed to establish a new connection: %s" % e
            )

        if ssl_context is not None:
            if self._tunnel_host is not None:
                self._tunnel(conn)
            conn = self._wrap_socket(conn, ssl_context, fingerprint, assert_hostname)
        self._sock = conn

    def close(self) -> None:
        if self._sock is not None:
            sock, self._sock = self._sock, None
            sock.forceful_close()

    def is_dropped(self) -> bool:
        if self._sock is None:
            return True
        return self._sock.is_readable()

    def _reset(self) -> None:
        try:
            self._state_machine.start_next_cycle()
        except h11.LocalProtocolError:
            self.close()
        else:
            self._sock.set_readable_watch_state(True)

    @property
    def complete(self) -> bool:
        our_state = self._state_machine.our_state
        their_state = self._state_machine.their_state
        return our_state is h11.IDLE and their_state is h11.IDLE

    def __iter__(self) -> Iterator[bytes]:
        return self

    def next(self) -> bytes:
        return self.__next__()

    def __next__(self) -> bytes:
        event = _read_until_event(self._state_machine, self._sock)
        if isinstance(event, h11.Data):
            return bytes(event.data)
        elif isinstance(event, h11.EndOfMessage):
            self._reset()
            raise StopIteration
        else:
            raise RuntimeError("Unexpected h11 event {}".format(event))
