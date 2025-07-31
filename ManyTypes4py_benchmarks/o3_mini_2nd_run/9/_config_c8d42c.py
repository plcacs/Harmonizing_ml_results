from __future__ import annotations
import os
import typing
from ._models import Headers
from ._types import CertTypes, HeaderTypes, TimeoutTypes
from ._urls import URL
if typing.TYPE_CHECKING:
    import ssl

__all__ = ['Limits', 'Proxy', 'Timeout', 'create_ssl_context']


class UnsetType:
    pass


UNSET: UnsetType = UnsetType()


def create_ssl_context(
    verify: bool | str | ssl.SSLContext = True,
    cert: None | str | tuple[str, ...] = None,
    trust_env: bool = True,
) -> ssl.SSLContext:
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
        message = ('`verify=<str>` is deprecated. Use `verify=ssl.create_default_context('
                   'cafile=...)` or `verify=ssl.create_default_context(capath=...)` instead.')
        warnings.warn(message, DeprecationWarning)
        if os.path.isdir(verify):
            return ssl.create_default_context(capath=verify)
        return ssl.create_default_context(cafile=verify)
    else:
        ctx = verify
    if cert:
        message = ('`cert=...` is deprecated. Use `verify=<ssl_context>` instead,with '
                   "`.load_cert_chain()` to configure the certificate chain.")
        warnings.warn(message, DeprecationWarning)
        if isinstance(cert, str):
            ctx.load_cert_chain(cert)
        else:
            ctx.load_cert_chain(*cert)
    return ctx


class Timeout:
    """
    Timeout configuration.

    **Usage**:

    Timeout(None)               # No timeouts.
    Timeout(5.0)                # 5s timeout on all operations.
    Timeout(None, connect=5.0)  # 5s timeout on connect, no other timeouts.
    Timeout(5.0, connect=10.0)  # 10s timeout on connect. 5s timeout elsewhere.
    Timeout(5.0, pool=None)     # No timeout on acquiring connection from pool.
                                # 5s timeout elsewhere.
    """

    def __init__(
        self,
        timeout: float | Timeout | tuple[float, ...] | UnsetType = UNSET,
        *,
        connect: UnsetType | float | None = UNSET,
        read: UnsetType | float | None = UNSET,
        write: UnsetType | float | None = UNSET,
        pool: UnsetType | float | None = UNSET,
    ) -> None:
        if isinstance(timeout, Timeout):
            assert isinstance(connect, UnsetType)
            assert isinstance(read, UnsetType)
            assert isinstance(write, UnsetType)
            assert isinstance(pool, UnsetType)
            self.connect: float | None = timeout.connect
            self.read: float | None = timeout.read
            self.write: float | None = timeout.write
            self.pool: float | None = timeout.pool
        elif isinstance(timeout, tuple):
            self.connect = timeout[0]
            self.read = timeout[1]
            self.write = None if len(timeout) < 3 else timeout[2]
            self.pool = None if len(timeout) < 4 else timeout[3]
        elif not (
            isinstance(connect, UnsetType)
            or isinstance(read, UnsetType)
            or isinstance(write, UnsetType)
            or isinstance(pool, UnsetType)
        ):
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

    def as_dict(self) -> dict[str, float | None]:
        return {'connect': self.connect, 'read': self.read, 'write': self.write, 'pool': self.pool}

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.connect == other.connect
            and self.read == other.read
            and self.write == other.write
            and self.pool == other.pool
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if len({self.connect, self.read, self.write, self.pool}) == 1:
            return f'{class_name}(timeout={self.connect})'
        return f'{class_name}(connect={self.connect}, read={self.read}, write={self.write}, pool={self.pool})'


class Limits:
    """
    Configuration for limits to various client behaviors.

    **Parameters:**

    * **max_connections** - The maximum number of concurrent connections that may be
            established.
    * **max_keepalive_connections** - Allow the connection pool to maintain
            keep-alive connections below this point. Should be less than or equal
            to `max_connections`.
    * **keepalive_expiry** - Time limit on idle keep-alive connections in seconds.
    """

    def __init__(
        self,
        *,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float = 5.0,
    ) -> None:
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.max_connections == other.max_connections
            and self.max_keepalive_connections == other.max_keepalive_connections
            and self.keepalive_expiry == other.keepalive_expiry
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f'{class_name}(max_connections={self.max_connections}, '
            f'max_keepalive_connections={self.max_keepalive_connections}, '
            f'keepalive_expiry={self.keepalive_expiry})'
        )


class Proxy:
    def __init__(
        self,
        url: str,
        *,
        ssl_context: typing.Optional[ssl.SSLContext] = None,
        auth: typing.Optional[tuple[str, str]] = None,
        headers: typing.Optional[HeaderTypes] = None,
    ) -> None:
        url_obj = URL(url)
        headers_obj = Headers(headers)
        if url_obj.scheme not in ('http', 'https', 'socks5', 'socks5h'):
            raise ValueError(f'Unknown scheme for proxy URL {url_obj!r}')
        if url_obj.username or url_obj.password:
            auth = (url_obj.username, url_obj.password)  # type: ignore
            url_obj = url_obj.copy_with(username=None, password=None)
        self.url: URL = url_obj
        self.auth: typing.Optional[tuple[str, str]] = auth
        self.headers: Headers = headers_obj
        self.ssl_context: typing.Optional[ssl.SSLContext] = ssl_context

    @property
    def raw_auth(self) -> typing.Optional[tuple[bytes, bytes]]:
        return None if self.auth is None else (self.auth[0].encode('utf-8'), self.auth[1].encode('utf-8'))

    def __repr__(self) -> str:
        auth_repr = (self.auth[0], '********') if self.auth else None
        url_str: str = f'{str(self.url)!r}'
        auth_str: str = f', auth={auth_repr!r}' if auth_repr else ''
        headers_str: str = f', headers={dict(self.headers)!r}' if self.headers else ''
        return f'Proxy({url_str}{auth_str}{headers_str})'


DEFAULT_TIMEOUT_CONFIG: Timeout = Timeout(timeout=5.0)
DEFAULT_LIMITS: Limits = Limits(max_connections=100, max_keepalive_connections=20)
DEFAULT_MAX_REDIRECTS: int = 20