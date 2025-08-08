from __future__ import absolute_import
import collections
import functools
import logging
from .._collections import RecentlyUsedContainer
from ..base import DEFAULT_PORTS
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import LocationValueError, MaxRetryError, ProxySchemeUnknown
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..util.url import parse_url
from ..util.request import set_file_position
from ..util.retry import Retry
from typing import Any, Dict, Tuple, Union

__all__: Tuple[str] = ['PoolManager', 'ProxyManager', 'proxy_from_url']
log: logging.Logger = logging.getLogger(__name__)
SSL_KEYWORDS: Tuple[str] = ('key_file', 'cert_file', 'cert_reqs', 'ca_certs', 'ssl_version', 'ca_cert_dir', 'ssl_context')
_key_fields: Tuple[str] = ('key_scheme', 'key_host', 'key_strict', 'key_port', 'key_timeout', 'key_retries', 'key_block', 'key_source_address', 'key_key_file', 'key_cert_file', 'key_cert_reqs', 'key_ca_certs', 'key_ssl_version', 'key_ca_cert_dir', 'key_ssl_context', 'key_maxsize', 'key_headers', 'key__proxy', 'key__proxy_headers', 'key_socket_options', 'key__socks_options', 'key_assert_hostname', 'key_assert_fingerprint')
PoolKey: collections.namedtuple = collections.namedtuple('PoolKey', _key_fields)

def _default_key_normalizer(key_class: Any, request_context: Dict[str, Any]) -> PoolKey:
    ...

def _merge_pool_kwargs(override: Dict[str, Any]) -> Dict[str, Any]:
    ...

class PoolManager(RequestMethods):
    def __init__(self, num_pools: int = 10, headers: Dict[str, str] = None, backend: Any = None, **connection_pool_kw: Any) -> None:
        ...

    def _new_pool(self, scheme: str, host: str, port: int, request_context: Dict[str, Any] = None) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        ...

    def clear(self) -> None:
        ...

    def connection_from_host(self, host: str, port: int = None, scheme: str = 'http', pool_kwargs: Dict[str, Any] = None) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        ...

    def connection_from_context(self, request_context: Dict[str, Any]) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        ...

    def connection_from_pool_key(self, pool_key: PoolKey, request_context: Dict[str, Any] = None) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        ...

    def connection_from_url(self, url: str, pool_kwargs: Dict[str, Any] = None) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        ...

    async def urlopen(self, method: str, url: str, redirect: bool = True, **kw: Any) -> Any:
        ...

class ProxyManager(PoolManager):
    def __init__(self, proxy_url: str, num_pools: int = 10, headers: Dict[str, str] = None, proxy_headers: Dict[str, str] = None, **connection_pool_kw: Any) -> None:
        ...

    def connection_from_host(self, host: str, port: int = None, scheme: str = 'http', pool_kwargs: Dict[str, Any] = None) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        ...

    def _set_proxy_headers(self, url: str, headers: Dict[str, str] = None) -> Dict[str, str]:
        ...

    def urlopen(self, method: str, url: str, redirect: bool = True, **kw: Any) -> Any:
        ...

def proxy_from_url(url: str, **kw: Any) -> ProxyManager:
    ...
