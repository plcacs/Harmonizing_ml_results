from __future__ import absolute_import
import collections
import functools
import logging
from typing import Any, Dict, Optional, Tuple, Type, Union, cast, Callable, List
from .._collections import RecentlyUsedContainer
from ..base import DEFAULT_PORTS
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import LocationValueError, MaxRetryError, ProxySchemeUnknown
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..util.url import Url, parse_url
from ..util.request import set_file_position
from ..util.retry import Retry

__all__ = ['PoolManager', 'ProxyManager', 'proxy_from_url']
log = logging.getLogger(__name__)

SSL_KEYWORDS = ('key_file', 'cert_file', 'cert_reqs', 'ca_certs', 'ssl_version', 'ca_cert_dir', 'ssl_context')
_key_fields = ('key_scheme', 'key_host', 'key_strict', 'key_port', 'key_timeout', 'key_retries', 'key_block', 'key_source_address', 'key_key_file', 'key_cert_file', 'key_cert_reqs', 'key_ca_certs', 'key_ssl_version', 'key_ca_cert_dir', 'key_ssl_context', 'key_maxsize', 'key_headers', 'key__proxy', 'key__proxy_headers', 'key_socket_options', 'key__socks_options', 'key_assert_hostname', 'key_assert_fingerprint')

PoolKey = collections.namedtuple('PoolKey', _key_fields)

def _default_key_normalizer(key_class: Type[PoolKey], request_context: Dict[str, Any]) -> PoolKey:
    context = request_context.copy()
    context['scheme'] = context['scheme'].lower()
    context['host'] = context['host'].lower()
    for key in ('headers', '_proxy_headers', '_socks_options'):
        if key in context and context[key] is not None:
            context[key] = frozenset(context[key].items())
    socket_opts = context.get('socket_options')
    if socket_opts is not None:
        context['socket_options'] = tuple(socket_opts)
    for key in list(context.keys()):
        context['key_' + key] = context.pop(key)
    for field in key_class._fields:
        if field not in context:
            context[field] = None
    return key_class(**context)

key_fn_by_scheme = {
    'http': functools.partial(_default_key_normalizer, PoolKey),
    'https': functools.partial(_default_key_normalizer, PoolKey)
}

pool_classes_by_scheme = {
    'http': HTTPConnectionPool,
    'https': HTTPSConnectionPool
}

class PoolManager(RequestMethods):
    proxy: Optional[Url] = None

    def __init__(
        self,
        num_pools: int = 10,
        headers: Optional[Dict[str, str]] = None,
        backend: Any = None,
        **connection_pool_kw: Any
    ) -> None:
        RequestMethods.__init__(self, headers)
        self.connection_pool_kw = connection_pool_kw
        self.pools = RecentlyUsedContainer(num_pools, dispose_func=lambda p: p.close())
        self.pool_classes_by_scheme = pool_classes_by_scheme
        self.key_fn_by_scheme = key_fn_by_scheme.copy()
        self.backend = backend

    def __enter__(self) -> 'PoolManager':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.clear()
        return False

    def _new_pool(
        self,
        scheme: str,
        host: str,
        port: Optional[int],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
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
        self.pools.clear()

    def connection_from_host(
        self,
        host: str,
        port: Optional[int] = None,
        scheme: str = 'http',
        pool_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        if not host:
            raise LocationValueError('No host specified.')
        request_context = self._merge_pool_kwargs(pool_kwargs)
        request_context['scheme'] = scheme or 'http'
        if not port:
            port = DEFAULT_PORTS.get(request_context['scheme'].lower(), 80)
        request_context['port'] = port
        request_context['host'] = host
        return self.connection_from_context(request_context)

    def connection_from_context(self, request_context: Dict[str, Any]) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        scheme = request_context['scheme'].lower()
        pool_key_constructor = self.key_fn_by_scheme[scheme]
        pool_key = pool_key_constructor(request_context)
        return self.connection_from_pool_key(pool_key, request_context=request_context)

    def connection_from_pool_key(
        self,
        pool_key: PoolKey,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        with self.pools.lock:
            pool = self.pools.get(pool_key)
            if pool:
                return pool
            scheme = request_context['scheme']
            host = request_context['host']
            port = request_context['port']
            pool = self._new_pool(scheme, host, port, request_context=request_context)
            self.pools[pool_key] = pool
        return pool

    def connection_from_url(
        self,
        url: str,
        pool_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        u = parse_url(url)
        return self.connection_from_host(u.host, port=u.port, scheme=u.scheme, pool_kwargs=pool_kwargs)

    def _merge_pool_kwargs(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base_pool_kwargs = self.connection_pool_kw.copy()
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

    def urlopen(
        self,
        method: str,
        url: str,
        redirect: bool = True,
        **kw: Any
    ) -> Any:
        u = parse_url(url)
        conn = self.connection_from_host(u.host, port=u.port, scheme=u.scheme)
        body = kw.get('body')
        body_pos = kw.get('body_pos')
        kw['body_pos'] = set_file_position(body, body_pos)
        if 'headers' not in kw:
            kw['headers'] = self.headers
        if self.proxy is not None and u.scheme == 'http':
            response = conn.urlopen(method, url, **kw)
        else:
            response = conn.urlopen(method, u.request_uri, **kw)
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
        return self.urlopen(method, redirect_location, **kw)

class ProxyManager(PoolManager):
    def __init__(
        self,
        proxy_url: Union[str, HTTPConnectionPool],
        num_pools: int = 10,
        headers: Optional[Dict[str, str]] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        **connection_pool_kw: Any
    ) -> None:
        if isinstance(proxy_url, HTTPConnectionPool):
            proxy_url = '%s://%s:%i' % (proxy_url.scheme, proxy_url.host, proxy_url.port)
        proxy = parse_url(proxy_url)
        if not proxy.port:
            port = DEFAULT_PORTS.get(proxy.scheme, 80)
            proxy = proxy._replace(port=port)
        if proxy.scheme not in ('http', 'https'):
            raise ProxySchemeUnknown(proxy.scheme)
        self.proxy = proxy
        self.proxy_headers = proxy_headers or {}
        connection_pool_kw['_proxy'] = self.proxy
        connection_pool_kw['_proxy_headers'] = self.proxy_headers
        super(ProxyManager, self).__init__(num_pools, headers, **connection_pool_kw)

    def connection_from_host(
        self,
        host: str,
        port: Optional[int] = None,
        scheme: str = 'http',
        pool_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        if scheme == 'https':
            return super(ProxyManager, self).connection_from_host(host, port, scheme, pool_kwargs=pool_kwargs)
        return super(ProxyManager, self).connection_from_host(
            self.proxy.host, self.proxy.port, self.proxy.scheme, pool_kwargs=pool_kwargs
        )

    def _set_proxy_headers(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        headers_ = {'Accept': '*/*'}
        netloc = parse_url(url).netloc
        if netloc:
            headers_['Host'] = netloc
        if headers:
            headers_.update(headers)
        return headers_

    def urlopen(
        self,
        method: str,
        url: str,
        redirect: bool = True,
        **kw: Any
    ) -> Any:
        u = parse_url(url)
        if u.scheme == 'http':
            headers = kw.get('headers', self.headers)
            kw['headers'] = self._set_proxy_headers(url, headers)
        return super(ProxyManager, self).urlopen(method, url, redirect=redirect, **kw)

def proxy_from_url(url: str, **kw: Any) -> ProxyManager:
    return ProxyManager(proxy_url=url, **kw)
