from __future__ import annotations
import os
import typing
from typing import Optional, Tuple, Union, Dict, Any
from ._models import Headers
from ._types import CertTypes, HeaderTypes, TimeoutTypes
from ._urls import URL
if typing.TYPE_CHECKING:
    import ssl
    from ssl import SSLContext
    from typing_extensions import Literal

__all__ = ['Limits', 'Proxy', 'Timeout', 'create_ssl_context']

class UnsetType:
    pass
UNSET: UnsetType = UnsetType()

def create_ssl_context(
    verify: Union[bool, str, SSLContext] = True,
    cert: Optional[CertTypes] = None,
    trust_env: bool = True
) -> SSLContext:
    import ssl
    import warnings
    import certifi
    if verify is True:
        if trust_env and os.environ.get('SSL_CERT_FILE'):
            ctx = ssl.create_default_context(cafile=os.environ['SSL_CERT_FILE'])
        elif trust_env and os.environ.get('SSL_CERT_DIR'):
            ctx = ssl.create_default_context(capath=os.environ['SSL_CERT_DIR'])
        else:
            ctx = ssl.create_default_context(cafile=certifi.where())
    elif verify is False:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    elif isinstance(verify, str):
        message = '`verify=<str>` is deprecated. Use `verify=ssl.create_default_context(cafile=...)` or `verify=ssl.create_default_context(capath=...)` instead.'
        warnings.warn(message, DeprecationWarning)
        if os.path.isdir(verify):
            return ssl.create_default_context(capath=verify)
        return ssl.create_default_context(cafile=verify)
    else:
        ctx = verify
    if cert:
        message = '`cert=...` is deprecated. Use `verify=<ssl_context>` instead,with `.load_cert_chain()` to configure the certificate chain.'
        warnings.warn(message, DeprecationWarning)
        if isinstance(cert, str):
            ctx.load_cert_chain(cert)
        else:
            ctx.load_cert_chain(*cert)
    return ctx

class Timeout:
    def __init__(
        self,
        timeout: Union[Timeout, Tuple[Optional[float], ...], Optional[float], UnsetType] = UNSET,
        *,
        connect: Union[Optional[float], UnsetType] = UNSET,
        read: Union[Optional[float], UnsetType] = UNSET,
        write: Union[Optional[float], UnsetType] = UNSET,
        pool: Union[Optional[float], UnsetType] = UNSET
    ) -> None:
        if isinstance(timeout, Timeout):
            assert connect is UNSET
            assert read is UNSET
            assert write is UNSET
            assert pool is UNSET
            self.connect = timeout.connect
            self.read = timeout.read
            self.write = timeout.write
            self.pool = timeout.pool
        elif isinstance(timeout, tuple):
            self.connect = timeout[0]
            self.read = timeout[1]
            self.write = None if len(timeout) < 3 else timeout[2]
            self.pool = None if len(timeout) < 4 else timeout[3]
        elif not (isinstance(connect, UnsetType) or isinstance(read, UnsetType) or isinstance(write, UnsetType) or isinstance(pool, UnsetType)):
            self.connect = connect
            self.read = read
            self.write = write
            self.pool = pool
        else:
            if isinstance(timeout, UnsetType):
                raise ValueError('httpx.Timeout must either include a default, or set all four parameters explicitly.')
            self.connect = timeout if isinstance(connect, UnsetType) else connect
            self.read = timeout if isinstance(read, UnsetType) else read
            self.write = timeout if isinstance(write, UnsetType) else write
            self.pool = timeout if isinstance(pool, UnsetType) else pool

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {'connect': self.connect, 'read': self.read, 'write': self.write, 'pool': self.pool}

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.connect == other.connect and (self.read == other.read) and (self.write == other.write) and (self.pool == other.pool)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if len({self.connect, self.read, self.write, self.pool}) == 1:
            return f'{class_name}(timeout={self.connect})'
        return f'{class_name}(connect={self.connect}, read={self.read}, write={self.write}, pool={self.pool})'

class Limits:
    def __init__(
        self,
        *,
        max_connections: Optional[int] = None,
        max_keepalive_connections: Optional[int] = None,
        keepalive_expiry: float = 5.0
    ) -> None:
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.max_connections == other.max_connections and (self.max_keepalive_connections == other.max_keepalive_connections) and (self.keepalive_expiry == other.keepalive_expiry)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(max_connections={self.max_connections}, max_keepalive_connections={self.max_keepalive_connections}, keepalive_expiry={self.keepalive_expiry})'

class Proxy:
    def __init__(
        self,
        url: Union[str, URL],
        *,
        ssl_context: Optional[SSLContext] = None,
        auth: Optional[Tuple[str, str]] = None,
        headers: Optional[HeaderTypes] = None
    ) -> None:
        url = URL(url)
        headers = Headers(headers)
        if url.scheme not in ('http', 'https', 'socks5', 'socks5h'):
            raise ValueError(f'Unknown scheme for proxy URL {url!r}')
        if url.username or url.password:
            auth = (url.username, url.password)
            url = url.copy_with(username=None, password=None)
        self.url = url
        self.auth = auth
        self.headers = headers
        self.ssl_context = ssl_context

    @property
    def raw_auth(self) -> Optional[Tuple[bytes, bytes]]:
        return None if self.auth is None else (self.auth[0].encode('utf-8'), self.auth[1].encode('utf-8'))

    def __repr__(self) -> str:
        auth = (self.auth[0], '********') if self.auth else None
        url_str = f'{str(self.url)!r}'
        auth_str = f', auth={auth!r}' if auth else ''
        headers_str = f', headers={dict(self.headers)!r}' if self.headers else ''
        return f'Proxy({url_str}{auth_str}{headers_str})'

DEFAULT_TIMEOUT_CONFIG: Timeout = Timeout(timeout=5.0)
DEFAULT_LIMITS: Limits = Limits(max_connections=100, max_keepalive_connections=20)
DEFAULT_MAX_REDIRECTS: int = 20
