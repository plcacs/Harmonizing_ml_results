from __future__ import absolute_import
import datetime
import logging
import os
import sys
import socket
from socket import error as SocketError, timeout as SocketTimeout
import warnings
from typing import Any, Optional, Tuple, Union, Iterable, Mapping, Dict, List
from .packages import six
from .packages.six.moves.http_client import HTTPConnection as _HTTPConnection
from .packages.six.moves.http_client import HTTPException
try:
    import ssl
    BaseSSLError = ssl.SSLError
except (ImportError, AttributeError):
    ssl = None

    class BaseSSLError(BaseException):
        pass
try:
    ConnectionError = ConnectionError
except NameError:

    class ConnectionError(Exception):
        pass
from .exceptions import NewConnectionError, ConnectTimeoutError, SubjectAltNameWarning, SystemTimeWarning
from .packages.ssl_match_hostname import match_hostname, CertificateError
from .util.ssl_ import resolve_cert_reqs, resolve_ssl_version, assert_fingerprint, create_urllib3_context, ssl_wrap_socket
from .util import connection
from ._collections import HTTPHeaderDict

log = logging.getLogger(__name__)
port_by_scheme: Dict[str, int] = {'http': 80, 'https': 443}
RECENT_DATE: datetime.date = datetime.date(2017, 6, 30)


class DummyConnection(object):
    """Used to detect a failed ConnectionCls import."""
    pass


class HTTPConnection(_HTTPConnection, object):
    """
    Based on httplib.HTTPConnection but provides an extra constructor
    backwards-compatibility layer between older and newer Pythons.

    Additional keyword parameters are used to configure attributes of the connection.
    Accepted parameters include:

      - ``strict``: See the documentation on :class:`urllib3.connectionpool.HTTPConnectionPool`
      - ``source_address``: Set the source address for the current connection.

        .. note:: This is ignored for Python 2.6. It is only applied for 2.7 and 3.x

      - ``socket_options``: Set specific options on the underlying socket. If not specified, then
        defaults are loaded from ``HTTPConnection.default_socket_options`` which includes disabling
        Nagle's algorithm (sets TCP_NODELAY to 1) unless the connection is behind a proxy.

        For example, if you wish to enable TCP Keep Alive in addition to the defaults,
        you might pass::

            HTTPConnection.default_socket_options + [
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            ]

        Or you may want to disable the defaults by passing an empty list (e.g., ``[]``).
    """
    default_port: int = port_by_scheme['http']
    default_socket_options: List[Tuple[int, int, int]] = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
    is_verified: bool = False

    def __init__(self, *args: Any, **kw: Any) -> None:
        if six.PY3:
            kw.pop('strict', None)
        self.source_address: Optional[Tuple[str, int]] = kw.get('source_address')
        if sys.version_info < (2, 7):
            kw.pop('source_address', None)
        self.socket_options: List[Tuple[int, int, int]] = kw.pop('socket_options', self.default_socket_options)  # type: ignore
        _HTTPConnection.__init__(self, *args, **kw)

    @property
    def host(self) -> str:
        """
        Getter method to remove any trailing dots that indicate the hostname is an FQDN.
        """
        return self._dns_host.rstrip('.')

    @host.setter
    def host(self, value: str) -> None:
        """
        Setter for the `host` property.
        """
        self._dns_host = value

    def _new_conn(self) -> socket.socket:
        """Establish a socket connection and set nodelay settings on it.

        :return: New socket connection.
        """
        extra_kw: Dict[str, Any] = {}
        if self.source_address:
            extra_kw['source_address'] = self.source_address
        if self.socket_options:
            extra_kw['socket_options'] = self.socket_options
        try:
            conn: socket.socket = connection.create_connection((self._dns_host, self.port), self.timeout, **extra_kw)
        except SocketTimeout as e:
            raise ConnectTimeoutError(self, 'Connection to %s timed out. (connect timeout=%s)' % (self.host, self.timeout))
        except SocketError as e:
            raise NewConnectionError(self, 'Failed to establish a new connection: %s' % e)
        return conn

    def _prepare_conn(self, conn: socket.socket) -> None:
        self.sock = conn
        if getattr(self, '_tunnel_host', None):
            self._tunnel()
            self.auto_open = 0

    def connect(self) -> None:
        conn: socket.socket = self._new_conn()
        self._prepare_conn(conn)

    def request_chunked(self, method: str, url: str, body: Optional[Union[str, bytes, Iterable[Union[str, bytes]]]] = None, headers: Optional[Mapping[str, str]] = None) -> None:
        """
        Alternative to the common request method, which sends the
        body with chunked encoding and not as one block
        """
        headers = HTTPHeaderDict(headers if headers is not None else {})
        skip_accept_encoding: bool = 'accept-encoding' in headers
        skip_host: bool = 'host' in headers
        self.putrequest(method, url, skip_accept_encoding=skip_accept_encoding, skip_host=skip_host)
        for header, value in headers.items():
            self.putheader(header, value)
        if 'transfer-encoding' not in headers:
            self.putheader('Transfer-Encoding', 'chunked')
        self.endheaders()
        if body is not None:
            stringish_types = six.string_types + (six.binary_type,)
            if isinstance(body, stringish_types):  # type: ignore
                body = (body,)  # type: ignore
            for chunk in body:  # type: ignore
                if not chunk:
                    continue
                if not isinstance(chunk, six.binary_type):  # type: ignore
                    chunk = chunk.encode('utf8')  # type: ignore
                len_str: str = hex(len(chunk))[2:]
                self.send(len_str.encode('utf-8'))
                self.send(b'\r\n')
                self.send(chunk)
                self.send(b'\r\n')
        self.send(b'0\r\n\r\n')


class HTTPSConnection(HTTPConnection):
    default_port: int = port_by_scheme['https']
    ssl_version: Optional[Any] = None

    def __init__(self, host: str, port: Optional[int] = None, key_file: Optional[str] = None, cert_file: Optional[str] = None, strict: Optional[Any] = None, timeout: Union[float, Any] = socket._GLOBAL_DEFAULT_TIMEOUT, ssl_context: Optional[Any] = None, **kw: Any) -> None:
        HTTPConnection.__init__(self, host, port, strict=strict, timeout=timeout, **kw)
        self.key_file: Optional[str] = key_file
        self.cert_file: Optional[str] = cert_file
        self.ssl_context: Optional[Any] = ssl_context
        self._protocol: str = 'https'

    def connect(self) -> None:
        conn: socket.socket = self._new_conn()
        self._prepare_conn(conn)
        if self.ssl_context is None:
            self.ssl_context = create_urllib3_context(ssl_version=resolve_ssl_version(None), cert_reqs=resolve_cert_reqs(None))
        self.sock = ssl_wrap_socket(sock=conn, keyfile=self.key_file, certfile=self.cert_file, ssl_context=self.ssl_context)


class VerifiedHTTPSConnection(HTTPSConnection):
    """
    Based on httplib.HTTPSConnection but wraps the socket with
    SSL certification.
    """
    cert_reqs: Optional[Union[str, int]] = None
    ca_certs: Optional[str] = None
    ca_cert_dir: Optional[str] = None
    ssl_version: Optional[Any] = None
    assert_fingerprint: Optional[str] = None

    def set_cert(self, key_file: Optional[str] = None, cert_file: Optional[str] = None, cert_reqs: Optional[Union[str, int]] = None, ca_certs: Optional[str] = None, assert_hostname: Optional[Union[str, bool]] = None, assert_fingerprint: Optional[str] = None, ca_cert_dir: Optional[str] = None) -> None:
        """
        This method should only be called once, before the connection is used.
        """
        if cert_reqs is None:
            if ca_certs or ca_cert_dir:
                cert_reqs = 'CERT_REQUIRED'
            elif self.ssl_context is not None:
                cert_reqs = self.ssl_context.verify_mode
        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)

    def connect(self) -> None:
        conn: socket.socket = self._new_conn()
        hostname: str = self.host
        if getattr(self, '_tunnel_host', None):
            self.sock = conn
            self._tunnel()
            self.auto_open = 0
            hostname = self._tunnel_host  # type: ignore
        is_time_off: bool = datetime.date.today() < RECENT_DATE
        if is_time_off:
            warnings.warn('System time is way off (before {0}). This will probably lead to SSL verification errors'.format(RECENT_DATE), SystemTimeWarning)
        if self.ssl_context is None:
            self.ssl_context = create_urllib3_context(ssl_version=resolve_ssl_version(self.ssl_version), cert_reqs=resolve_cert_reqs(self.cert_reqs))
        context: Any = self.ssl_context
        context.verify_mode = resolve_cert_reqs(self.cert_reqs)
        self.sock = ssl_wrap_socket(sock=conn, keyfile=self.key_file, certfile=self.cert_file, ca_certs=self.ca_certs, ca_cert_dir=self.ca_cert_dir, server_hostname=hostname, ssl_context=context)
        if self.assert_fingerprint:
            assert_fingerprint(self.sock.getpeercert(binary_form=True), self.assert_fingerprint)
        elif context.verify_mode != ssl.CERT_NONE and (not getattr(context, 'check_hostname', False)) and (self.assert_hostname is not False):
            cert: Dict[str, Any] = self.sock.getpeercert()
            if not cert.get('subjectAltName', ()):
                warnings.warn('Certificate for {0} has no `subjectAltName`, falling back to check for a `commonName` for now. This feature is being removed by major browsers and deprecated by RFC 2818. (See https://github.com/shazow/urllib3/issues/497 for details.)'.format(hostname), SubjectAltNameWarning)
            _match_hostname(cert, self.assert_hostname or hostname)  # type: ignore
        self.is_verified = context.verify_mode == ssl.CERT_REQUIRED or self.assert_fingerprint is not None


def _match_hostname(cert: Dict[str, Any], asserted_hostname: str) -> None:
    try:
        match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        log.error('Certificate did not match expected hostname: %s. Certificate: %s', asserted_hostname, cert)
        e._peer_cert = cert
        raise


if ssl:
    UnverifiedHTTPSConnection = HTTPSConnection
    HTTPSConnection = VerifiedHTTPSConnection
else:
    HTTPSConnection = DummyConnection
