"""
This module implements the connection management logic.

Unlike in http.client, the connection here is an object that is responsible
for a very small number of tasks:

    1. Serializing/deserializing data to/from the network.
    2. Being able to do basic parsing of HTTP and maintaining the framing.
    3. Understanding connection state.

This object knows very little about the semantics of HTTP in terms of how to
construct HTTP requests and responses. It mostly manages the socket itself.
"""
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
RECENT_DATE = datetime.date(2016, 1, 1)
_SUPPORTED_VERSIONS = frozenset([b'1.0', b'1.1'])
_EAGAIN = object()

def _headers_to_native_string(headers: str) -> typing.Generator[tuple[typing.Union[bytes,str]]]:
    """
    A temporary shim to convert received headers to native strings, to match
    the behaviour of httplib. We will reconsider this later in the process.
    """
    for n, v in headers:
        if not isinstance(n, str):
            n = n.decode('latin1')
        if not isinstance(v, str):
            v = v.decode('latin1')
        yield (n, v)

def _stringify_headers(headers: str) -> typing.Generator[tuple[typing.Union[str,tuple[str],bytes,int]]]:
    """
    A generator that transforms headers so they're suitable for sending by h11.
    """
    for name, value in headers:
        if isinstance(name, six.text_type):
            name = name.encode('ascii')
        if isinstance(value, six.text_type):
            value = value.encode('latin-1')
        elif isinstance(value, int):
            value = str(value).encode('ascii')
        yield (name, value)

def _read_readable(readable: Any) -> typing.Generator:
    blocksize = 8192
    while True:
        datablock = readable.read(blocksize)
        if not datablock:
            break
        yield datablock

def _make_body_iterable(body: Union[bytes, typing.Any, None, str]) -> Union[list, list[six_@_binary_type], str, bytes, typing.Any]:
    """
    This function turns all possible body types that urllib3 supports into an
    iterable of bytes. The goal is to expose a uniform structure to request
    bodies so that they all appear to be identical to the low-level code.

    The basic logic here is:
        - byte strings are turned into single-element lists
        - readables are wrapped in an iterable that repeatedly calls read until
          nothing is returned anymore
        - other iterables are used directly
        - anything else is not acceptable

    In particular, note that we do not support *text* data of any kind. This
    is deliberate: users must make choices about the encoding of the data they
    use.
    """
    if body is None:
        return []
    elif isinstance(body, six.binary_type):
        return [body]
    elif hasattr(body, 'read'):
        return _read_readable(body)
    elif isinstance(body, collections.Iterable) and (not isinstance(body, six.text_type)):
        return body
    else:
        raise InvalidBodyError('Unacceptable body type: %s' % type(body))

def _request_bytes_iterable(request: _models.Response, state_machine: Any) -> typing.Generator:
    """
    An iterable that serialises a set of bytes for the body.
    """
    h11_request = h11.Request(method=request.method, target=request.target, headers=_stringify_headers(request.headers.items()))
    yield state_machine.send(h11_request)
    for chunk in _make_body_iterable(request.body):
        yield state_machine.send(h11.Data(data=chunk))
    yield state_machine.send(h11.EndOfMessage())

def _response_from_h11(h11_response: Union[int, dict, str], body_object: Union[dict, int, str]) -> Response:
    """
    Given a h11 Response object, build a urllib3 response object and return it.
    """
    if h11_response.http_version not in _SUPPORTED_VERSIONS:
        raise BadVersionError(h11_response.http_version)
    version = b'HTTP/' + h11_response.http_version
    our_response = Response(status_code=h11_response.status_code, headers=_headers_to_native_string(h11_response.headers), body=body_object, version=version)
    return our_response

def _build_tunnel_request(host: Union[str, int, None], port: Union[str, int, None], headers: str) -> Request:
    """
    Builds a urllib3 Request object that is set up correctly to request a proxy
    to establish a TCP tunnel to the remote host.
    """
    target = '%s:%d' % (host, port)
    if not isinstance(target, bytes):
        target = target.encode('latin1')
    tunnel_request = Request(method=b'CONNECT', target=target, headers=headers)
    tunnel_request.add_host(host=host, port=port, scheme='http')
    return tunnel_request

def _start_http_request(request: dict, state_machine: Union[dict[str, typing.Any], typing.Mapping, dict], conn: Union[typing.Mapping, dict]) -> Union[bool, None]:
    """
    Send the request using the given state machine and connection, wait
    for the response headers, and return them.

    If we get response headers early, then we stop sending and return
    immediately, poisoning the state machine along the way so that we know
    it can't be re-used.

    This is a standalone function because we use it both to set up both
    CONNECT requests and real requests.
    """
    if state_machine.our_state is not h11.IDLE or state_machine.their_state is not h11.IDLE:
        raise ProtocolError('Invalid internal state transition')
    request_bytes_iterable = _request_bytes_iterable(request, state_machine)
    context = {'send_aborted': True, 'h11_response': None}

    def next_bytes_to_send() -> None:
        try:
            return next(request_bytes_iterable)
        except StopIteration:
            context['send_aborted'] = False
            return None

    def consume_bytes(data: Any) -> None:
        state_machine.receive_data(data)
        while True:
            event = state_machine.next_event()
            if event is h11.NEED_DATA:
                break
            elif isinstance(event, h11.InformationalResponse):
                continue
            elif isinstance(event, h11.Response):
                context['h11_response'] = event
                raise LoopAbort
            else:
                raise RuntimeError('Unexpected h11 event {}'.format(event))
    conn.send_and_receive_for_a_while(next_bytes_to_send, consume_bytes)
    assert context['h11_response'] is not None
    if context['send_aborted']:
        state_machine._cstate.process_error(state_machine.our_role)
    return context['h11_response']

def _read_until_event(state_machine: int, conn: Union[util.freefocus.sql.Task, str]) -> Union[dict[str, str], list[dict[str, typing.Any]]]:
    """
    A loop that keeps issuing reads and feeding the data into h11 and
    checking whether h11 has an event for us. The moment there is an event
    other than h11.NEED_DATA, this function returns that event.
    """
    while True:
        event = state_machine.next_event()
        if event is not h11.NEED_DATA:
            return event
        state_machine.receive_data(conn.receive_some())
_DEFAULT_SOCKET_OPTIONS = object()

class HTTP1Connection(object):
    """
    A wrapper around a single HTTP/1.1 connection.

    This wrapper manages connection state, ensuring that connections are
    appropriately managed throughout the lifetime of a HTTP transaction. In
    particular, this object understands the conditions in which connections
    should be torn down, and also manages sending data and handling early
    responses.

    This object can be iterated over to return the response body. When iterated
    over it will return all of the data that is currently buffered, and if no
    data is buffered it will issue one read syscall and return all of that
    data. Buffering of response data must happen at a higher layer.
    """
    default_socket_options = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]

    def __init__(self, host: Union[str, None], port: Union[str, None, int, tuple[typing.Union[str,int]]], backend: Union[None, bool, str]=None, socket_options: Union[dict[str, dict[str, typing.Any]], dict[str, list[str]]]=_DEFAULT_SOCKET_OPTIONS, source_address: Union[None, str, bool]=None, tunnel_host: Union[None, str, typing.Sequence[str]]=None, tunnel_port: Union[None, int, str]=None, tunnel_headers: Union[None, typing.AbstractSet, str, bool]=None) -> None:
        self.is_verified = False
        self._backend = backend or SyncBackend()
        self._host = host
        self._port = port
        self._socket_options = socket_options if socket_options is not _DEFAULT_SOCKET_OPTIONS else self.default_socket_options
        self._source_address = source_address
        self._tunnel_host = tunnel_host
        self._tunnel_port = tunnel_port
        self._tunnel_headers = tunnel_headers
        self._sock = None
        self._state_machine = h11.Connection(our_role=h11.CLIENT)

    def _wrap_socket(self, conn: Any, ssl_context: Union[ssl.SSLContext, None, Frame], fingerprint: Union[str, None], assert_hostname: Union[core.base.types.TorrentClient, None]):
        """
        Handles extra logic to wrap the socket in TLS magic.
        """
        is_time_off = datetime.date.today() < RECENT_DATE
        if is_time_off:
            warnings.warn('System time is way off (before {0}). This will probably lead to SSL verification errors'.format(RECENT_DATE), SystemTimeWarning)
        check_host = assert_hostname or self._tunnel_host or self._host
        check_host = check_host.rstrip('.')
        conn = conn.start_tls(check_host, ssl_context)
        if fingerprint:
            ssl_util.assert_fingerprint(conn.getpeercert(binary_form=True), fingerprint)
        elif ssl_context.verify_mode != ssl.CERT_NONE and assert_hostname is not False:
            cert = conn.getpeercert()
            if not cert.get('subjectAltName', ()):
                warnings.warn('Certificate for {0} has no `subjectAltName`, falling back to check for a `commonName` for now. This feature is being removed by major browsers and deprecated by RFC 2818. (See https://github.com/shazow/urllib3/issues/497 for details.)'.format(self._host), SubjectAltNameWarning)
            ssl_util.match_hostname(cert, check_host)
        self.is_verified = ssl_context.verify_mode == ssl.CERT_REQUIRED and (assert_hostname is not False or fingerprint)
        return conn

    def send_request(self, request: Union[int, str, typing.Callable[..., collections.abc.Coroutine]], read_timeout: Union[bool, list[int]]):
        """
        Given a Request object, performs the logic required to get a response.
        """
        h11_response = _start_http_request(request, self._state_machine, self._sock)
        return _response_from_h11(h11_response, self)

    def _tunnel(self, conn: grouper.models.base.session.Session) -> None:
        """
        This method establishes a CONNECT tunnel shortly after connection.
        """
        assert self._state_machine.our_state is h11.IDLE
        tunnel_request = _build_tunnel_request(self._tunnel_host, self._tunnel_port, self._tunnel_headers)
        tunnel_state_machine = h11.Connection(our_role=h11.CLIENT)
        h11_response = _start_http_request(tunnel_request, tunnel_state_machine, conn)
        tunnel_response = _response_from_h11(h11_response, self)
        if h11_response.status_code != 200:
            conn.forceful_close()
            raise FailedTunnelError('Unable to establish CONNECT tunnel', tunnel_response)

    def connect(self, ssl_context: Union[None, ssl.SSLContext, str, int]=None, fingerprint: Union[None, int, ssl.SSLContext, str]=None, assert_hostname: Union[None, int, ssl.SSLContext, str]=None, connect_timeout: Union[None, int, str, float]=None) -> None:
        """
        Connect this socket to the server, applying the source address, any
        relevant socket options, and the relevant connection timeout.
        """
        if self._sock is not None:
            self._sock.set_readable_watch_state(False)
            return
        extra_kw = {}
        if self._source_address:
            extra_kw['source_address'] = self._source_address
        if self._socket_options:
            extra_kw['socket_options'] = self._socket_options
        try:
            conn = self._backend.connect(self._host, self._port, **extra_kw)
        except socket.timeout:
            raise ConnectTimeoutError(self, 'Connection to %s timed out. (connect timeout=%s)' % (self._host, connect_timeout))
        except socket.error as e:
            raise NewConnectionError(self, 'Failed to establish a new connection: %s' % e)
        if ssl_context is not None:
            if self._tunnel_host is not None:
                self._tunnel(conn)
            conn = self._wrap_socket(conn, ssl_context, fingerprint, assert_hostname)
        self._sock = conn

    def close(self) -> None:
        """
        Close this connection.
        """
        if self._sock is not None:
            sock, self._sock = (self._sock, None)
            sock.forceful_close()

    def is_dropped(self) -> bool:
        """
        Returns True if the connection is closed: returns False otherwise. This
        includes closures that do not mark the FD as closed, such as when the
        remote peer has sent EOF but we haven't read it yet.

        Pre-condition: _reset must have been called.
        """
        if self._sock is None:
            return True
        return self._sock.is_readable()

    def _reset(self) -> None:
        """
        Called once we hit EndOfMessage, and checks whether we can re-use this
        state machine and connection or not, and if not, closes the socket and
        state machine.
        """
        try:
            self._state_machine.start_next_cycle()
        except h11.LocalProtocolError:
            self.close()
        else:
            self._sock.set_readable_watch_state(True)

    @property
    def complete(self) -> bool:
        """
        XX what is this supposed to do? check if the response has been fully
        iterated over? check for that + the connection being reusable?
        """
        our_state = self._state_machine.our_state
        their_state = self._state_machine.their_state
        return our_state is h11.IDLE and their_state is h11.IDLE

    def __iter__(self) -> HTTP1Connection:
        return self

    def next(self) -> Union[int, str, dict]:
        return self.__next__()

    def __next__(self) -> bytes:
        """
        Iterate over the body bytes of the response until end of message.
        """
        event = _read_until_event(self._state_machine, self._sock)
        if isinstance(event, h11.Data):
            return bytes(event.data)
        elif isinstance(event, h11.EndOfMessage):
            self._reset()
            raise StopIteration
        else:
            raise RuntimeError('Unexpected h11 event {}'.format(event))