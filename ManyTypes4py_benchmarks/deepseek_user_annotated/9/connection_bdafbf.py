from __future__ import absolute_import
import datetime
import logging
import os
import sys
import socket
from socket import error as SocketError, timeout as SocketTimeout
import warnings
from typing import Optional, Dict, List, Tuple, Any, Union, Type, cast
from .packages import six
from .packages.six.moves.http_client import HTTPConnection as _HTTPConnection
from .packages.six.moves.http_client import HTTPException  # noqa: F401

try:  # Compiled with SSL?
    import ssl
    BaseSSLError = ssl.SSLError
except (ImportError, AttributeError):  # Platform-specific: No SSL.
    ssl = None

    class BaseSSLError(BaseException):
        pass


try:  # Python 3:
    # Not a no-op, we're adding this to the namespace so it can be imported.
    ConnectionError = ConnectionError
except NameError:  # Python 2:

    class ConnectionError(Exception):
        pass


from .exceptions import (
    NewConnectionError,
    ConnectTimeoutError,
    SubjectAltNameWarning,
    SystemTimeWarning,
)
from .packages.ssl_match_hostname import match_hostname, CertificateError

from .util.ssl_ import (
    resolve_cert_reqs,
    resolve_ssl_version,
    assert_fingerprint,
    create_urllib3_context,
    ssl_wrap_socket,
)

from .util import connection
from ._collections import HTTPHeaderDict

log = logging.getLogger(__name__)
port_by_scheme = {"http": 80, "https": 443}  # type: Dict[str, int]
RECENT_DATE = datetime.date(2017, 6, 30)


class DummyConnection(object):
    """Used to detect a failed ConnectionCls import."""
    pass


class HTTPConnection(_HTTPConnection, object):
    default_port = port_by_scheme["http"]  # type: int
    default_socket_options = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]  # type: List[Tuple[int, int, int]]
    is_verified = False  # type: bool

    def __init__(self, *args: Any, **kw: Any) -> None:
        if six.PY3:  # Python 3
            kw.pop("strict", None)
        self.source_address = kw.get("source_address")  # type: Optional[Tuple[str, int]]
        if sys.version_info < (2, 7):  # Python 2.6
            kw.pop("source_address", None)
        self.socket_options = kw.pop("socket_options", self.default_socket_options)  # type: List[Tuple[int, int, int]]
        _HTTPConnection.__init__(self, *args, **kw)

    @property
    def host(self) -> str:
        return self._dns_host.rstrip(".")

    @host.setter
    def host(self, value: str) -> None:
        self._dns_host = value

    def _new_conn(self) -> socket.socket:
        extra_kw = {}  # type: Dict[str, Any]
        if self.source_address:
            extra_kw["source_address"] = self.source_address
        if self.socket_options:
            extra_kw["socket_options"] = self.socket_options
        try:
            conn = connection.create_connection(
                (self._dns_host, self.port), self.timeout, **extra_kw
            )
        except SocketTimeout as e:
            raise ConnectTimeoutError(
                self,
                "Connection to %s timed out. (connect timeout=%s)"
                % (self.host, self.timeout),
            )

        except SocketError as e:
            raise NewConnectionError(
                self, "Failed to establish a new connection: %s" % e
            )

        return conn

    def _prepare_conn(self, conn: socket.socket) -> None:
        self.sock = conn
        if getattr(self, "_tunnel_host", None):
            self._tunnel()
            self.auto_open = 0

    def connect(self) -> None:
        conn = self._new_conn()
        self._prepare_conn(conn)

    def request_chunked(self, method: str, url: str, body: Optional[Union[str, bytes, List[Union[str, bytes]]] = None, headers: Optional[Dict[str, str]] = None) -> None:
        headers = HTTPHeaderDict(headers if headers is not None else {})
        skip_accept_encoding = "accept-encoding" in headers
        skip_host = "host" in headers
        self.putrequest(
            method, url, skip_accept_encoding=skip_accept_encoding, skip_host=skip_host
        )
        for header, value in headers.items():
            self.putheader(header, value)
        if "transfer-encoding" not in headers:
            self.putheader("Transfer-Encoding", "chunked")
        self.endheaders()
        if body is not None:
            stringish_types = six.string_types + (six.binary_type,)
            if isinstance(body, stringish_types):
                body = (body,)
            for chunk in body:
                if not chunk:
                    continue

                if not isinstance(chunk, six.binary_type):
                    chunk = chunk.encode("utf8")
                len_str = hex(len(chunk))[2:]
                self.send(len_str.encode("utf-8"))
                self.send(b"\r\n")
                self.send(chunk)
                self.send(b"\r\n")
        self.send(b"0\r\n\r\n")


class HTTPSConnection(HTTPConnection):
    default_port = port_by_scheme["https"]  # type: int
    ssl_version = None  # type: Optional[int]

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        key_file: Optional[str] = None,
        cert_file: Optional[str] = None,
        strict: Optional[bool] = None,
        timeout: float = socket._GLOBAL_DEFAULT_TIMEOUT,
        ssl_context: Optional[ssl.SSLContext] = None,
        **kw: Any
    ) -> None:
        HTTPConnection.__init__(self, host, port, strict=strict, timeout=timeout, **kw)
        self.key_file = key_file  # type: Optional[str]
        self.cert_file = cert_file  # type: Optional[str]
        self.ssl_context = ssl_context  # type: Optional[ssl.SSLContext]
        self._protocol = "https"  # type: str

    def connect(self) -> None:
        conn = self._new_conn()
        self._prepare_conn(conn)
        if self.ssl_context is None:
            self.ssl_context = create_urllib3_context(
                ssl_version=resolve_ssl_version(None), cert_reqs=resolve_cert_reqs(None)
            )
        self.sock = ssl_wrap_socket(
            sock=conn,
            keyfile=self.key_file,
            certfile=self.cert_file,
            ssl_context=self.ssl_context,
        )


class VerifiedHTTPSConnection(HTTPSConnection):
    cert_reqs = None  # type: Optional[str]
    ca_certs = None  # type: Optional[str]
    ca_cert_dir = None  # type: Optional[str]
    ssl_version = None  # type: Optional[int]
    assert_fingerprint = None  # type: Optional[str]

    def set_cert(
        self,
        key_file: Optional[str] = None,
        cert_file: Optional[str] = None,
        cert_reqs: Optional[str] = None,
        ca_certs: Optional[str] = None,
        assert_hostname: Optional[str] = None,
        assert_fingerprint: Optional[str] = None,
        ca_cert_dir: Optional[str] = None,
    ) -> None:
        if cert_reqs is None:
            if ca_certs or ca_cert_dir:
                cert_reqs = "CERT_REQUIRED"
            elif self.ssl_context is not None:
                cert_reqs = self.ssl_context.verify_mode
        self.key_file = key_file  # type: Optional[str]
        self.cert_file = cert_file  # type: Optional[str]
        self.cert_reqs = cert_reqs  # type: Optional[str]
        self.assert_hostname = assert_hostname  # type: Optional[str]
        self.assert_fingerprint = assert_fingerprint  # type: Optional[str]
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)  # type: Optional[str]
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)  # type: Optional[str]

    def connect(self) -> None:
        conn = self._new_conn()
        hostname = self.host  # type: str
        if getattr(self, "_tunnel_host", None):
            self.sock = conn
            self._tunnel()
            self.auto_open = 0
            hostname = self._tunnel_host
        is_time_off = datetime.date.today() < RECENT_DATE  # type: bool
        if is_time_off:
            warnings.warn(
                (
                    "System time is way off (before {0}). This will probably "
                    "lead to SSL verification errors"
                ).format(RECENT_DATE),
                SystemTimeWarning,
            )
        if self.ssl_context is None:
            self.ssl_context = create_urllib3_context(
                ssl_version=resolve_ssl_version(self.ssl_version),
                cert_reqs=resolve_cert_reqs(self.cert_reqs),
            )
        context = self.ssl_context  # type: ssl.SSLContext
        context.verify_mode = resolve_cert_reqs(self.cert_reqs)
        self.sock = ssl_wrap_socket(
            sock=conn,
            keyfile=self.key_file,
            certfile=self.cert_file,
            ca_certs=self.ca_certs,
            ca_cert_dir=self.ca_cert_dir,
            server_hostname=hostname,
            ssl_context=context,
        )
        if self.assert_fingerprint:
            assert_fingerprint(
                self.sock.getpeercert(binary_form=True), self.assert_fingerprint
            )
        elif (
            context.verify_mode != ssl.CERT_NONE
            and not getattr(context, "check_hostname", False)
            and self.assert_hostname is not False
        ):
            cert = self.sock.getpeercert()  # type: Dict[str, Any]
            if not cert.get("subjectAltName", ()):
                warnings.warn(
                    (
                        "Certificate for {0} has no `subjectAltName`, falling back to check for a "
                        "`commonName` for now. This feature is being removed by major browsers and "
                        "deprecated by RFC 2818. (See https://github.com/shazow/urllib3/issues/497 "
                        "for details.)".format(hostname)
                    ),
                    SubjectAltNameWarning,
                )
            _match_hostname(cert, self.assert_hostname or hostname)
        self.is_verified = (
            context.verify_mode == ssl.CERT_REQUIRED
            or self.assert_fingerprint is not None
        )


def _match_hostname(cert: Dict[str, Any], asserted_hostname: str) -> None:
    try:
        match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        log.error(
            "Certificate did not match expected hostname: %s. " "Certificate: %s",
            asserted_hostname,
            cert,
        )
        e._peer_cert = cert
        raise


if ssl:
    UnverifiedHTTPSConnection = HTTPSConnection
    HTTPSConnection = VerifiedHTTPSConnection
else:
    HTTPSConnection = DummyConnection
