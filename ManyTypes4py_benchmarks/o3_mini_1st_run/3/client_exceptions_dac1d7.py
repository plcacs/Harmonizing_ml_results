"""HTTP related errors."""
import asyncio
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any
from multidict import MultiMapping
from .typedefs import StrOrURL

if TYPE_CHECKING:
    import ssl
    SSLContext = ssl.SSLContext
else:
    try:
        import ssl
        SSLContext = ssl.SSLContext
    except ImportError:
        ssl = SSLContext = None

if TYPE_CHECKING:
    from .client_reqrep import ClientResponse, ConnectionKey, Fingerprint, RequestInfo
    from .http_parser import RawResponseMessage
else:
    RequestInfo = ClientResponse = ConnectionKey = RawResponseMessage = None

__all__ = (
    'ClientError', 'ClientConnectionError', 'ClientConnectionResetError', 'ClientOSError', 'ClientConnectorError',
    'ClientProxyConnectionError', 'ClientSSLError', 'ClientConnectorDNSError', 'ClientConnectorSSLError',
    'ClientConnectorCertificateError', 'ConnectionTimeoutError', 'SocketTimeoutError', 'ServerConnectionError',
    'ServerTimeoutError', 'ServerDisconnectedError', 'ServerFingerprintMismatch', 'ClientResponseError',
    'ClientHttpProxyError', 'WSServerHandshakeError', 'ContentTypeError', 'ClientPayloadError', 'InvalidURL',
    'InvalidUrlClientError', 'RedirectClientError', 'NonHttpUrlClientError', 'InvalidUrlRedirectClientError',
    'NonHttpUrlRedirectClientError', 'WSMessageTypeError'
)

class ClientError(Exception):
    """Base class for client connection errors."""
    pass

class ClientResponseError(ClientError):
    """Base class for exceptions that occur after getting a response.

    request_info: An instance of RequestInfo.
    history: A sequence of responses, if redirects occurred.
    status: HTTP status code.
    message: Error message.
    headers: Response headers.
    """

    def __init__(
        self,
        request_info: Any,
        history: Tuple[Any, ...],
        *,
        status: Optional[int] = None,
        message: str = '',
        headers: Optional[MultiMapping[str, str]] = None
    ) -> None:
        self.request_info = request_info
        if status is not None:
            self.status = status
        else:
            self.status = 0
        self.message = message
        self.headers = headers
        self.history = history
        self.args = (request_info, history)

    def __str__(self) -> str:
        return '{}, message={!r}, url={!r}'.format(self.status, self.message, str(self.request_info.real_url))

    def __repr__(self) -> str:
        args = f'{self.request_info!r}, {self.history!r}'
        if self.status != 0:
            args += f', status={self.status!r}'
        if self.message != '':
            args += f', message={self.message!r}'
        if self.headers is not None:
            args += f', headers={self.headers!r}'
        return f'{type(self).__name__}({args})'

class ContentTypeError(ClientResponseError):
    """ContentType found is not valid."""
    pass

class WSServerHandshakeError(ClientResponseError):
    """websocket server handshake error."""
    pass

class ClientHttpProxyError(ClientResponseError):
    """HTTP proxy error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
    proxy responds with status other than ``200 OK``
    on ``CONNECT`` request.
    """
    pass

class TooManyRedirects(ClientResponseError):
    """Client was redirected too many times."""
    pass

class ClientConnectionError(ClientError):
    """Base class for client socket errors."""
    pass

class ClientConnectionResetError(ClientConnectionError, ConnectionResetError):
    """ConnectionResetError"""
    pass

class ClientOSError(ClientConnectionError, OSError):
    """OSError error."""
    pass

class ClientConnectorError(ClientOSError):
    """Client connector error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        a connection can not be established.
    """

    def __init__(self, connection_key: Any, os_error: OSError) -> None:
        self._conn_key = connection_key
        self._os_error = os_error
        super().__init__(os_error.errno, os_error.strerror)
        self.args = (connection_key, os_error)

    @property
    def os_error(self) -> OSError:
        return self._os_error

    @property
    def host(self) -> Any:
        return self._conn_key.host

    @property
    def port(self) -> Any:
        return self._conn_key.port

    @property
    def ssl(self) -> Any:
        return self._conn_key.ssl

    def __str__(self) -> str:
        return 'Cannot connect to host {0.host}:{0.port} ssl:{1} [{2}]'.format(
            self, 'default' if self.ssl is True else self.ssl, self.strerror
        )
    __reduce__ = BaseException.__reduce__

class ClientConnectorDNSError(ClientConnectorError):
    """DNS resolution failed during client connection.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        DNS resolution fails.
    """
    pass

class ClientProxyConnectionError(ClientConnectorError):
    """Proxy connection error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        connection to proxy can not be established.
    """
    pass

class UnixClientConnectorError(ClientConnectorError):
    """Unix connector error.

    Raised in :py:class:`aiohttp.connector.UnixConnector`
    if connection to unix socket can not be established.
    """

    def __init__(self, path: str, connection_key: Any, os_error: OSError) -> None:
        self._path = path
        super().__init__(connection_key, os_error)

    @property
    def path(self) -> str:
        return self._path

    def __str__(self) -> str:
        return 'Cannot connect to unix socket {0.path} ssl:{1} [{2}]'.format(
            self, 'default' if self.ssl is True else self.ssl, self.strerror
        )

class ServerConnectionError(ClientConnectionError):
    """Server connection errors."""
    pass

class ServerDisconnectedError(ServerConnectionError):
    """Server disconnected."""

    def __init__(self, message: Optional[str] = None) -> None:
        if message is None:
            message = 'Server disconnected'
        self.args = (message,)
        self.message = message

class ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError):
    """Server timeout error."""
    pass

class ConnectionTimeoutError(ServerTimeoutError):
    """Connection timeout error."""
    pass

class SocketTimeoutError(ServerTimeoutError):
    """Socket timeout error."""
    pass

class ServerFingerprintMismatch(ServerConnectionError):
    """SSL certificate does not match expected fingerprint."""

    def __init__(self, expected: Any, got: Any, host: str, port: int) -> None:
        self.expected = expected
        self.got = got
        self.host = host
        self.port = port
        self.args = (expected, got, host, port)

    def __repr__(self) -> str:
        return '<{} expected={!r} got={!r} host={!r} port={!r}>'.format(
            self.__class__.__name__, self.expected, self.got, self.host, self.port
        )

class ClientPayloadError(ClientError):
    """Response payload error."""
    pass

class InvalidURL(ClientError, ValueError):
    """Invalid URL.

    URL used for fetching is malformed, e.g. it doesn't contains host
    part.
    """

    def __init__(self, url: StrOrURL, description: Optional[str] = None) -> None:
        self._url = url
        self._description = description
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
        return f'<{self.__class__.__name__} {self}>'

    def __str__(self) -> str:
        if self._description:
            return f'{self._url} - {self._description}'
        return str(self._url)

class InvalidUrlClientError(InvalidURL):
    """Invalid URL client error."""
    pass

class RedirectClientError(ClientError):
    """Client redirect error."""
    pass

class NonHttpUrlClientError(ClientError):
    """Non http URL client error."""
    pass

class InvalidUrlRedirectClientError(InvalidUrlClientError, RedirectClientError):
    """Invalid URL redirect client error."""
    pass

class NonHttpUrlRedirectClientError(NonHttpUrlClientError, RedirectClientError):
    """Non http URL redirect client error."""
    pass

class ClientSSLError(ClientConnectorError):
    """Base error for ssl.*Errors."""
    pass

if ssl is not None:
    cert_errors = (ssl.CertificateError,)
    cert_errors_bases = (ClientSSLError, ssl.CertificateError)
    ssl_errors = (ssl.SSLError,)
    ssl_error_bases = (ClientSSLError, ssl.SSLError)
else:
    cert_errors = tuple()
    cert_errors_bases = (ClientSSLError, ValueError)
    ssl_errors = tuple()
    ssl_error_bases = (ClientSSLError,)

class ClientConnectorSSLError(*ssl_error_bases):  # type: ignore
    """Response ssl error."""
    pass

class ClientConnectorCertificateError(*cert_errors_bases):  # type: ignore
    """Response certificate error."""

    def __init__(self, connection_key: Any, certificate_error: Exception) -> None:
        self._conn_key = connection_key
        self._certificate_error = certificate_error
        self.args = (connection_key, certificate_error)

    @property
    def certificate_error(self) -> Exception:
        return self._certificate_error

    @property
    def host(self) -> Any:
        return self._conn_key.host

    @property
    def port(self) -> Any:
        return self._conn_key.port

    @property
    def ssl(self) -> Any:
        return self._conn_key.is_ssl

    def __str__(self) -> str:
        return 'Cannot connect to host {0.host}:{0.port} ssl:{0.ssl} [{0.certificate_error.__class__.__name__}: {0.certificate_error.args}]'.format(self)

class WSMessageTypeError(TypeError):
    """WebSocket message type is not valid."""
    pass
