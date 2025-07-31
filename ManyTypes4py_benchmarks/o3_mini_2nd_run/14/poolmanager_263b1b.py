from __future__ import absolute_import
import collections
import functools
import logging
from typing import Any, Dict, Optional, Type, TypeVar, Callable, FrozenSet, Tuple
from .._collections import RecentlyUsedContainer
from ..base import DEFAULT_PORTS
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import LocationValueError, MaxRetryError, ProxySchemeUnknown
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..util.url import parse_url, Url
from ..util.request import set_file_position
from ..util.retry import Retry

T = TypeVar('T')
__all__ = ['PoolManager', 'ProxyManager', 'proxy_from_url']
log = logging.getLogger(__name__)
SSL_KEYWORDS = ('key_file', 'cert_file', 'cert_reqs', 'ca_certs', 'ssl_version', 'ca_cert_dir', 'ssl_context')
_key_fields = (
    'key_scheme', 'key_host', 'key_strict', 'key_port', 'key_timeout', 'key_retries', 'key_block',
    'key_source_address', 'key_key_file', 'key_cert_file', 'key_cert_reqs', 'key_ca_certs',
    'key_ssl_version', 'key_ca_cert_dir', 'key_ssl_context', 'key_maxsize', 'key_headers',
    'key__proxy', 'key__proxy_headers', 'key_socket_options', 'key__socks_options',
    'key_assert_hostname', 'key_assert_fingerprint'
)
PoolKey = collections.namedtuple('PoolKey', _key_fields)

def _default_key_normalizer(key_class: Type[T], request_context: Dict[str, Any]) -> T:
    """
    Create a pool key out of a request context dictionary.

    According to RFC 3986, both the scheme and host are case-insensitive.
    Therefore, this function normalizes both before constructing the pool
    key for an HTTPS request. If you wish to change this behaviour, provide
    alternate callables to ``key_fn_by_scheme``.

    :param key_class:
        The class to use when constructing the key. This should be a namedtuple
        with the ``scheme`` and ``host`` keys at a minimum.
    :type  key_class: namedtuple
    :param request_context:
        A dictionary-like object that contain the context for a request.
    :type  request_context: dict

    :return: A namedtuple that can be used as a connection pool key.
    :rtype:  PoolKey
    """
    context: Dict[str, Any] = request_context.copy()
    context['scheme'] = context['scheme'].lower()
    context['host'] = context['host'].lower()
    for key in ('headers', '_proxy_headers', '_socks_options'):
        if key in context and context[key] is not None:
            context[key] = frozenset(context[key].items())  # type: FrozenSet[Any]
    socket_opts = context.get('socket_options')
    if socket_opts is not None:
        context['socket_options'] = tuple(socket_opts)  # type: Tuple[Any, ...]
    for key in list(context.keys()):
        context['key_' + key] = context.pop(key)
    for field in key_class._fields:
        if field not in context:
            context[field] = None
    return key_class(**context)

key_fn_by_scheme: Dict[str, Callable[[Dict[str, Any]], PoolKey]] = {
    'http': functools.partial(_default_key_normalizer, PoolKey),
    'https': functools.partial(_default_key_normalizer, PoolKey)
}
pool_classes_by_scheme: Dict[str, Any] = {'http': HTTPConnectionPool, 'https': HTTPSConnectionPool}

class PoolManager(RequestMethods):
    """
    Allows for arbitrary requests while transparently keeping track of
    necessary connection pools for you.

    :param num_pools:
        Number of connection pools to cache before discarding the least
        recently used pool.
    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.
    :param backend:
        Optional backend for connection pools.
    :param connection_pool_kw:
        Additional parameters used to create new
        :class:`urllib3.connectionpool.ConnectionPool` instances.
    """
    proxy: Optional[Url] = None

    def __init__(self, num_pools: int = 10, headers: Optional[Dict[str, str]] = None, backend: Optional[Any] = None, **connection_pool_kw: Any) -> None:
        RequestMethods.__init__(self, headers)
        self.connection_pool_kw: Dict[str, Any] = connection_pool_kw
        self.pools: RecentlyUsedContainer = RecentlyUsedContainer(num_pools, dispose_func=lambda p: p.close())
        self.pool_classes_by_scheme: Dict[str, Any] = pool_classes_by_scheme
        self.key_fn_by_scheme: Dict[str, Callable[[Dict[str, Any]], PoolKey]] = key_fn_by_scheme.copy()
        self.backend: Optional[Any] = backend

    def __enter__(self) -> "PoolManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.clear()
        return False

    def _new_pool(self, scheme: str, host: str, port: int, request_context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a new :class:`ConnectionPool` based on host, port, scheme, and
        any additional pool keyword arguments.
        """
        pool_cls = self.pool_classes_by_scheme[scheme]
        if request_context is None:
            request_context = self.connection_pool_kw.copy()
        for key in ('scheme', 'host', 'port'):
            request_context.pop(key, None)
        if scheme == 'http':
            for kw in SSL_KEYWORDS:
                request_context.pop(kw, None)
        return pool_cls(host, port, backend=self.backend, **request_context)

    def clear(self) -> None:
        """
        Empty our store of pools and direct them all to close.

        This will not affect in-flight connections, but they will not be
        re-used after completion.
        """
        self.pools.clear()

    def connection_from_host(self, host: str, port: Optional[int] = None, scheme: str = 'http', pool_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a :class:`ConnectionPool` based on the host, port, and scheme.
        """
        if not host:
            raise LocationValueError('No host specified.')
        request_context: Dict[str, Any] = self._merge_pool_kwargs(pool_kwargs)
        request_context['scheme'] = scheme or 'http'
        if not port:
            port = DEFAULT_PORTS.get(request_context['scheme'].lower(), 80)
        request_context['port'] = port
        request_context['host'] = host
        return self.connection_from_context(request_context)

    def connection_from_context(self, request_context: Dict[str, Any]) -> Any:
        """
        Get a :class:`ConnectionPool` based on the request context.
        """
        scheme: str = request_context['scheme'].lower()
        pool_key_constructor: Callable[[Dict[str, Any]], PoolKey] = self.key_fn_by_scheme[scheme]
        pool_key: PoolKey = pool_key_constructor(request_context)
        return self.connection_from_pool_key(pool_key, request_context=request_context)

    def connection_from_pool_key(self, pool_key: PoolKey, request_context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a :class:`ConnectionPool` based on the provided pool key.
        """
        with self.pools.lock:
            pool = self.pools.get(pool_key)
            if pool:
                return pool
            if request_context is None:
                raise ValueError("request_context must be provided when pool key is not found.")
            scheme: str = request_context['scheme']
            host: str = request_context['host']
            port: int = request_context['port']
            pool = self._new_pool(scheme, host, port, request_context=request_context)
            self.pools[pool_key] = pool
        return pool

    def connection_from_url(self, url: str, pool_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Similar to :func:`urllib3.connectionpool.connection_from_url`.
        """
        u: Url = parse_url(url)
        return self.connection_from_host(u.host, port=u.port, scheme=u.scheme, pool_kwargs=pool_kwargs)

    def _merge_pool_kwargs(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a dictionary of override values for self.connection_pool_kw.
        """
        base_pool_kwargs: Dict[str, Any] = self.connection_pool_kw.copy()
        if override:
            for key, value in override.items():
                if value is None:
                    try:
                        del base_pool_kwargs[key]
                    except KeyError:
                        pass
                else:
                    base_pool_kwargs[key] = value
        return base_pool_kwargs

    async def urlopen(self, method: str, url: str, redirect: bool = True, **kw: Any) -> Any:
        """
        Same as :meth:`urllib3.connectionpool.HTTPConnectionPool.urlopen`
        with redirect logic and only sends the request-uri portion of the
        ``url``.
        """
        u: Url = parse_url(url)
        conn = self.connection_from_host(u.host, port=u.port, scheme=u.scheme)
        body: Any = kw.get('body')
        body_pos: Any = kw.get('body_pos')
        kw['body_pos'] = set_file_position(body, body_pos)
        if 'headers' not in kw:
            kw['headers'] = self.headers
        if self.proxy is not None and u.scheme == 'http':
            response: Any = await conn.urlopen(method, url, **kw)
        else:
            response = await conn.urlopen(method, u.request_uri, **kw)
        redirect_location = redirect and response.get_redirect_location()
        if not redirect_location:
            return response
        redirect_location = urljoin(url, redirect_location)
        if response.status == 303:
            method = 'GET'
        retries = kw.get('retries')
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect)
        try:
            retries = retries.increment(method, url, response=response, _pool=conn)
        except MaxRetryError:
            if retries.raise_on_redirect:
                raise
            return response
        kw['retries'] = retries
        kw['redirect'] = redirect
        retries.sleep_for_retry(response)
        log.info('Redirecting %s -> %s', url, redirect_location)
        return await self.urlopen(method, redirect_location, **kw)

class ProxyManager(PoolManager):
    """
    Behaves just like :class:`PoolManager`, but sends all requests through
    the defined proxy, using the CONNECT method for HTTPS URLs.
    """
    def __init__(self, proxy_url: Any, num_pools: int = 10, headers: Optional[Dict[str, str]] = None, proxy_headers: Optional[Dict[str, str]] = None, **connection_pool_kw: Any) -> None:
        if isinstance(proxy_url, HTTPConnectionPool):
            proxy_url = '%s://%s:%i' % (proxy_url.scheme, proxy_url.host, proxy_url.port)
        proxy: Url = parse_url(proxy_url)
        if not proxy.port:
            port: int = DEFAULT_PORTS.get(proxy.scheme, 80)
            proxy = proxy._replace(port=port)
        if proxy.scheme not in ('http', 'https'):
            raise ProxySchemeUnknown(proxy.scheme)
        self.proxy: Url = proxy
        self.proxy_headers: Dict[str, str] = proxy_headers or {}
        connection_pool_kw['_proxy'] = self.proxy
        connection_pool_kw['_proxy_headers'] = self.proxy_headers
        super(ProxyManager, self).__init__(num_pools, headers, **connection_pool_kw)

    def connection_from_host(self, host: str, port: Optional[int] = None, scheme: str = 'http', pool_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if scheme == 'https':
            return super(ProxyManager, self).connection_from_host(host, port, scheme, pool_kwargs=pool_kwargs)
        return super(ProxyManager, self).connection_from_host(self.proxy.host, self.proxy.port, self.proxy.scheme, pool_kwargs=pool_kwargs)

    def _set_proxy_headers(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Sets headers needed by proxies: specifically, the Accept and Host
        headers. Only sets headers not provided by the user.
        """
        headers_: Dict[str, str] = {'Accept': '*/*'}
        netloc = parse_url(url).netloc
        if netloc:
            headers_['Host'] = netloc
        if headers:
            headers_.update(headers)
        return headers_

    async def urlopen(self, method: str, url: str, redirect: bool = True, **kw: Any) -> Any:
        """Same as HTTP(S)ConnectionPool.urlopen, ``url`` must be absolute."""
        u: Url = parse_url(url)
        if u.scheme == 'http':
            headers: Optional[Dict[str, str]] = kw.get('headers', self.headers)
            kw['headers'] = self._set_proxy_headers(url, headers)
        return await super(ProxyManager, self).urlopen(method, url, redirect=redirect, **kw)

def proxy_from_url(url: str, **kw: Any) -> ProxyManager:
    return ProxyManager(proxy_url=url, **kw)