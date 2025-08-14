"""HTTP related errors."""

import asyncio
from typing import TYPE_CHECKING, Optional, Tuple, Union, TypeVar, Any, Type, Sequence

from multidict import MultiMapping

from .typedefs import StrOrURL

if TYPE_CHECKING:
    import ssl
    SSLContext = ssl.SSLContext
else:
    try:
        import ssl
        SSLContext = ssl.SSLContext
    except ImportError:  # pragma: no cover
        ssl = SSLContext = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .client_reqrep import ClientResponse, ConnectionKey, Fingerprint, RequestInfo
    from .http_parser import RawResponseMessage
else:
    RequestInfo = ClientResponse = ConnectionKey = RawResponseMessage = None

__all__ = (
    "ClientError",
    "ClientConnectionError",
    "ClientConnectionResetError",
    "ClientOSError",
    "ClientConnectorError",
    "ClientProxyConnectionError",
    "ClientSSLError",
    "ClientConnectorDNSError",
    "ClientConnectorSSLError",
    "ClientConnectorCertificateError",
    "ConnectionTimeoutError",
    "SocketTimeoutError",
    "ServerConnectionError",
    "ServerTimeoutError",
    "ServerDisconnectedError",
    "ServerFingerprintMismatch",
    "ClientResponseError",
    "ClientHttpProxyError",
    "WSServerHandshakeError",
    "ContentTypeError",
    "ClientPayloadError",
    "InvalidURL",
    "InvalidUrlClientError",
    "RedirectClientError",
    "NonHttpUrlClientError",
    "InvalidUrlRedirectClientError",
    "NonHttpUrlRedirectClientError",
    "WSMessageTypeError",
)

T = TypeVar('T')

class ClientError(Exception):
    """Base class for client connection errors."""

class ClientResponseError(ClientError):
    """Base class for exceptions that occur after getting a response."""

    def __init__(
        self,
        request_info: RequestInfo,
        history: Tuple[ClientResponse, ...],
        *,
        status: Optional[int] = None,
        message: str = "",
        headers: Optional[MultiMapping[str]] = None,
    ) -> None:
        self.request_info: RequestInfo = request_info
        self.status: int = status if status is not None else 0
        self.message: str = message
        self.headers: Optional[MultiMapping[str]] = headers
        self.history: Tuple[ClientResponse, ...] = history
        self.args: Tuple[RequestInfo, Tuple[ClientResponse, ...]] = (request_info, history)

    def __str__(self) -> str:
        return "{}, message={!r}, url={!r}".format(
            self.status,
            self.message,
            str(self.request_info.real_url),
        )

    def __repr__(self) -> str:
        args = f"{self.request_info!r}, {self.history!r}"
        if self.status != 0:
            args += f", status={self.status!r}"
        if self.message != "":
            args += f", message={self.message!r}"
        if self.headers is not None:
            args += f", headers={self.headers!r}"
        return f"{type(self).__name__}({args})"

class ContentTypeError(ClientResponseError):
    """ContentType found is not valid."""

class WSServerHandshakeError(ClientResponseError):
    """websocket server handshake error."""

class ClientHttpProxyError(ClientResponseError):
    """HTTP proxy error."""

class TooManyRedirects(ClientResponseError):
    """Client was redirected too many times."""

class ClientConnectionError(ClientError):
    """Base class for client socket errors."""

class ClientConnectionResetError(ClientConnectionError, ConnectionResetError):
    """ConnectionResetError"""

class ClientOSError(ClientConnectionError, OSError):
    """OSError error."""

class ClientConnectorError(ClientOSError):
    """Client connector error."""

    def __init__(self, connection_key: ConnectionKey, os_error: OSError) -> None:
        self._conn_key: ConnectionKey = connection_key
        self._os_error: OSError = os_error
        super().__init__(os_error.errno, os_error.strerror)
        self.args: Tuple[ConnectionKey, OSError] = (connection_key, os_error)

    @property
    def os_error(self) -> OSError:
        return self._os_error

    @property
    def host(self) -> str:
        return self._conn_key.host

    @property
    def port(self) -> Optional[int]:
        return self._conn_key.port

    @property
    def ssl(self) -> Union[SSLContext, bool, "Fingerprint"]:
        return self._conn_key.ssl

    def __str__(self) -> str:
        return "Cannot connect to host {0.host}:{0.port} ssl:{1} [{2}]".format(
            self, "default" if self.ssl is True else self.ssl, self.strerror
        )

    __reduce__ = BaseException.__reduce__

class ClientConnectorDNSError(ClientConnectorError):
    """DNS resolution failed during client connection."""

class ClientProxyConnectionError(ClientConnectorError):
    """Proxy connection error."""

class UnixClientConnectorError(ClientConnectorError):
    """Unix connector error."""

    def __init__(
        self, path: str, connection_key: ConnectionKey, os_error: OSError
    ) -> None:
        self._path: str = path
        super().__init__(connection_key, os_error)

    @property
    def path(self) -> str:
        return self._path

    def __str__(self) -> str:
        return "Cannot connect to unix socket {0.path} ssl:{1} [{2}]".format(
            self, "default" if self.ssl is True else self.ssl, self.strerror
        )

class ServerConnectionError(ClientConnectionError):
    """Server connection errors."""

class ServerDisconnectedError(ServerConnectionError):
    """Server disconnected."""

    def __init__(self, message: Union[RawResponseMessage, str, None] = None) -> None:
        if message is None:
            message = "Server disconnected"

        self.args: Tuple[Union[RawResponseMessage, str, None], ...] = (message,)
        self.message: Union[RawResponseMessage, str, None] = message

class ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError):
    """Server timeout error."""

class ConnectionTimeoutError(ServerTimeoutError):
    """Connection timeout error."""

class SocketTimeoutError(ServerTimeoutError):
    """Socket timeout error."""

class ServerFingerprintMismatch(ServerConnectionError):
    """SSL certificate does not match expected fingerprint."""

    def __init__(self, expected: bytes, got: bytes, host: str, port: int) -> None:
        self.expected: bytes = expected
        self.got: bytes = got
        self.host: str = host
        self.port: int = port
        self.args: Tuple[bytes, bytes, str, int] = (expected, got, host, port)

    def __repr__(self) -> str:
        return "<{} expected={!r} got={!r} host={!r} port={!r}>".format(
            self.__class__.__name__, self.expected, self.got, self.host, self.port
        )

class ClientPayloadError(ClientError):
    """Response payload error."""

class InvalidURL(ClientError, ValueError):
    """Invalid URL."""

    def __init__(self, url: StrOrURL, description: Optional[str] = None) -> None:
        self._url: StrOrURL = url
        self._description: Optional[str] = description

        if description:
            super().__init__(url, description)
        else:
            super().__init__(url)

    @property
    def url(self) -> StrOrURL:
        return self._url

    @property
    def description(self) -> Optional[str]:
        return self._description

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self}>"

    def __str__(self) -> str:
        if self._description:
            return f"{self._url} - {self._description}"
        return str(self._url)

class InvalidUrlClientError(InvalidURL):
    """Invalid URL client error."""

class RedirectClientError(ClientError):
    """Client redirect error."""

class NonHttpUrlClientError(ClientError):
    """Non http URL client error."""

class InvalidUrlRedirectClientError(InvalidUrlClientError, RedirectClientError):
    """Invalid URL redirect client error."""

class NonHttpUrlRedirectClientError(NonHttpUrlClientError, RedirectClientError):
    """Non http URL redirect client error."""

class ClientSSLError(ClientConnectorError):
    """Base error for ssl.*Errors."""

if ssl is not None:
    cert_errors: Tuple[Type[Exception], ...] = (ssl.CertificateError,)
    cert_errors_bases: Tuple[Type[Any], ...] = (
        ClientSSLError,
        ssl.CertificateError,
    )

    ssl_errors: Tuple[Type[Exception], ...] = (ssl.SSLError,)
    ssl_error_bases: Tuple[Type[Any], ...] = (ClientSSLError, ssl.SSLError)
else:  # pragma: no cover
    cert_errors: Tuple[Type[Exception], ...] = tuple()
    cert_errors_bases: Tuple[Type[Any], ...] = (
        ClientSSLError,
        ValueError,
    )

    ssl_errors: Tuple[Type[Exception], ...] = tuple()
    ssl_error_bases: Tuple[Type[Any], ...] = (ClientSSLError,)

class ClientConnectorSSLError(*ssl_error_bases):  # type: ignore[misc]
    """Response ssl error."""

class ClientConnectorCertificateError(*cert_errors_bases):  # type: ignore[misc]
    """Response certificate error."""

    def __init__(
        self, connection_key: ConnectionKey, certificate_error: Exception
    ) -> None:
        self._conn_key: ConnectionKey = connection_key
        self._certificate_error: Exception = certificate_error
        self.args: Tuple[ConnectionKey, Exception] = (connection_key, certificate_error)

    @property
    def certificate_error(self) -> Exception:
        return self._certificate_error

    @property
    def host(self) -> str:
        return self._conn_key.host

    @property
    def port(self) -> Optional[int]:
        return self._conn_key.port

    @property
    def ssl(self) -> bool:
        return self._conn_key.is_ssl

    def __str__(self) -> str:
        return (
            "Cannot connect to host {0.host}:{0.port} ssl:{0.ssl} "
            "[{0.certificate_error.__class__.__name__}: "
            "{0.certificate_error.args}]".format(self)
        )

class WSMessageTypeError(TypeError):
    """WebSocket message type is not valid."""
