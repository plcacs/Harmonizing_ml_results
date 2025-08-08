from typing import Optional, Tuple, Union
from multidict import MultiMapping

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

class ClientResponseError(ClientError):
    def __init__(self, request_info: RequestInfo, history: Tuple[ClientResponse], *, status: Optional[int] = None, message: str = '', headers: Optional[MultiMapping] = None):
        ...

class ContentTypeError(ClientResponseError):
    ...

class WSServerHandshakeError(ClientResponseError):
    ...

class ClientHttpProxyError(ClientResponseError):
    ...

class TooManyRedirects(ClientResponseError):
    ...

class ClientConnectionError(ClientError):
    ...

class ClientConnectionResetError(ClientConnectionError, ConnectionResetError):
    ...

class ClientOSError(ClientConnectionError, OSError):
    ...

class ClientConnectorError(ClientOSError):
    def __init__(self, connection_key, os_error):
        ...

class ClientConnectorDNSError(ClientConnectorError):
    ...

class ClientProxyConnectionError(ClientConnectorError):
    ...

class UnixClientConnectorError(ClientConnectorError):
    ...

class ServerConnectionError(ClientConnectionError):
    ...

class ServerDisconnectedError(ServerConnectionError):
    def __init__(self, message: Optional[str] = None):
        ...

class ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError):
    ...

class ConnectionTimeoutError(ServerTimeoutError):
    ...

class SocketTimeoutError(ServerTimeoutError):
    ...

class ServerFingerprintMismatch(ServerConnectionError):
    ...

class ClientPayloadError(ClientError):
    ...

class InvalidURL(ClientError, ValueError):
    ...

class InvalidUrlClientError(InvalidURL):
    ...

class RedirectClientError(ClientError):
    ...

class NonHttpUrlClientError(ClientError):
    ...

class InvalidUrlRedirectClientError(InvalidUrlClientError, RedirectClientError):
    ...

class NonHttpUrlRedirectClientError(NonHttpUrlClientError, RedirectClientError):
    ...

class ClientSSLError(ClientConnectorError):
    ...

class ClientConnectorSSLError(*ssl_error_bases):
    ...

class ClientConnectorCertificateError(*cert_errors_bases):
    ...
