from __future__ import absolute_import
import collections
import functools
import logging
from typing import Any, Callable, Dict, Optional, Tuple, List, FrozenSet, Type, Union

from .._collections import RecentlyUsedContainer
from ..base import DEFAULT_PORTS
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import LocationValueError, MaxRetryError, ProxySchemeUnknown
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..util.url import parse_url
from ..util.request import set_file_position
from ..util.retry import Retry

__all__ = ["PoolManager", "ProxyManager", "proxy_from_url"]

SSL_KEYWORDS: Tuple[str, ...] = (
    "key_file",
    "cert_file",
    "cert_reqs",
    "ca_certs",
    "ssl_version",
    "ca_cert_dir",
    "ssl_context",
)

_key_fields: Tuple[str, ...] = (
    "key_scheme",  # str
    "key_host",  # str
    "key_strict",
    "key_port",  # int
    "key_timeout",  # int or float or Timeout
    "key_retries",  # int or Retry
    "key_block",  # bool
    "key_source_address",  # str
    "key_key_file",  # str
    "key_cert_file",  # str
    "key_cert_reqs",  # str
    "key_ca_certs",  # str
    "key_ssl_version",  # str
    "key_ca_cert_dir",  # str
    "key_ssl_context",  # instance of ssl.SSLContext or urllib3.util.ssl_.SSLContext
    "key_maxsize",  # int
    "key_headers",  # dict
    "key__proxy",  # parsed proxy url
    "key__proxy_headers",  # dict
    "key_socket_options",  # list of (level (int), optname (int), value (int or str)) tuples
    "key__socks_options",  # dict
    "key_assert_hostname",  # bool or string
    "key_assert_fingerprint",  # str
)
PoolKey = collections.namedtuple("PoolKey", _key_fields)

def _default_key_normalizer(
    key_class: Callable[..., Any], request_context: Dict[str, Any]
) -> Any:
    context = request_context.copy()
    context["scheme"] = context["scheme"].lower()
    context["host"] = context["host"].lower()
    for key in ("headers", "_proxy_headers", "_socks_options"):
        if key in context and context[key] is not None:
            context[key] = frozenset(context[key].items())  # type: FrozenSet[Any]
    socket_opts = context.get("socket_options")
    if socket_opts is not None:
        context["socket_options"] = tuple(socket_opts)
    for key in list(context.keys()):
        context["key_" + key] = context.pop(key)
    for field in key_class._fields:
        if field not in context:
            context[field] = None
    return key_class(**context)

key_fn_by_scheme: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "http": functools.partial(_default_key_normalizer, PoolKey),
    "https": functools.partial(_default_key_normalizer, PoolKey),
}
pool_classes_by_scheme: Dict[str, Type[HTTPConnectionPool]] = {
    "http": HTTPConnectionPool,
    "https": HTTPSConnectionPool,
}

class PoolManager(RequestMethods):
    proxy: Optional[Any] = None

    def __init__(
        self,
        num_pools: int = 10,
        headers: Optional[Dict[str, str]] = None,
        backend: Optional[Any] = None,
        **connection_pool_kw: Any
    ) -> None:
        RequestMethods.__init__(self, headers)
        self.connection_pool_kw: Dict[str, Any] = connection_pool_kw
        self.pools: RecentlyUsedContainer = RecentlyUsedContainer(num_pools, dispose_func=lambda p: p.close())
        self.pool_classes_by_scheme: Dict[str, Type[HTTPConnectionPool]] = pool_classes_by_scheme
        self.key_fn_by_scheme: Dict[str, Callable[[Dict[str, Any]], Any]] = key_fn_by_scheme.copy()
        self.backend: Optional[Any] = backend

    def __enter__(self) -> "PoolManager":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> bool:
        self.clear()
        return False

    def _new_pool(self, scheme: str, host: str, port: int, request_context: Optional[Dict[str, Any]] = None) -> HTTPConnectionPool:
        pool_cls: Type[HTTPConnectionPool] = self.pool_classes_by_scheme[scheme]
        if request_context is None:
            request_context = self.connection_pool_kw.copy()
        for key in ("scheme", "host", "port"):
            request_context.pop(key, None)
        if scheme == "http":
            for kw in SSL_KEYWORDS:
                request_context.pop(kw, None)
        return pool_cls(host, port, backend=self.backend, **request_context)

    def clear(self) -> None:
        self.pools.clear()

    def connection_from_host(
        self, host: str, port: Optional[int] = None, scheme: str = "http", pool_kwargs: Optional[Dict[str, Any]] = None
    ) -> HTTPConnectionPool:
        if not host:
            raise LocationValueError("No host specified.")
        request_context: Dict[str, Any] = self._merge_pool_kwargs(pool_kwargs)
        request_context["scheme"] = scheme or "http"
        if not port:
            port = DEFAULT_PORTS.get(request_context["scheme"].lower(), 80)
        request_context["port"] = port
        request_context["host"] = host
        return self.connection_from_context(request_context)

    def connection_from_context(self, request_context: Dict[str, Any]) -> HTTPConnectionPool:
        scheme: str = request_context["scheme"].lower()
        pool_key_constructor: Callable[[Dict[str, Any]], Any] = self.key_fn_by_scheme[scheme]
        pool_key: Any = pool_key_constructor(request_context)
        return self.connection_from_pool_key(pool_key, request_context=request_context)

    def connection_from_pool_key(self, pool_key: Any, request_context: Optional[Dict[str, Any]] = None) -> HTTPConnectionPool:
        with self.pools.lock:
            pool: Optional[HTTPConnectionPool] = self.pools.get(pool_key)
            if pool:
                return pool
            scheme: str = request_context["scheme"]  # type: ignore
            host: str = request_context["host"]  # type: ignore
            port: int = request_context["port"]  # type: ignore
            pool = self._new_pool(scheme, host, port, request_context=request_context)
            self.pools[pool_key] = pool
        return pool

    def connection_from_url(self, url: str, pool_kwargs: Optional[Dict[str, Any]] = None) -> HTTPConnectionPool:
        u = parse_url(url)
        return self.connection_from_host(
            u.host, port=u.port, scheme=u.scheme, pool_kwargs=pool_kwargs
        )

    def _merge_pool_kwargs(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
        u = parse_url(url)
        conn: HTTPConnectionPool = self.connection_from_host(u.host, port=u.port, scheme=u.scheme)
        body: Any = kw.get("body")
        body_pos: Any = kw.get("body_pos")
        kw["body_pos"] = set_file_position(body, body_pos)
        if "headers" not in kw:
            kw["headers"] = self.headers
        if self.proxy is not None and u.scheme == "http":
            response: Any = await conn.urlopen(method, url, **kw)
        else:
            response = await conn.urlopen(method, u.request_uri, **kw)
        redirect_location: Optional[str] = redirect and response.get_redirect_location()
        if not redirect_location:
            return response
        redirect_location = urljoin(url, redirect_location)
        if response.status == 303:
            method = "GET"
        retries: Union[Retry, Any] = kw.get("retries")
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect)
        try:
            retries = retries.increment(method, url, response=response, _pool=conn)
        except MaxRetryError:
            if retries.raise_on_redirect:
                raise
            return response
        kw["retries"] = retries
        kw["redirect"] = redirect
        retries.sleep_for_retry(response)
        log.info("Redirecting %s -> %s", url, redirect_location)
        return await self.urlopen(method, redirect_location, **kw)

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
            proxy_url = "%s://%s:%i" % (
                proxy_url.scheme,
                proxy_url.host,
                proxy_url.port,
            )
        proxy = parse_url(proxy_url)
        if not proxy.port:
            port = DEFAULT_PORTS.get(proxy.scheme, 80)
            proxy = proxy._replace(port=port)
        if proxy.scheme not in ("http", "https"):
            raise ProxySchemeUnknown(proxy.scheme)
        self.proxy = proxy
        self.proxy_headers: Dict[str, str] = proxy_headers or {}
        connection_pool_kw["_proxy"] = self.proxy
        connection_pool_kw["_proxy_headers"] = self.proxy_headers
        super(ProxyManager, self).__init__(num_pools, headers, **connection_pool_kw)

    def connection_from_host(
        self, host: str, port: Optional[int] = None, scheme: str = "http", pool_kwargs: Optional[Dict[str, Any]] = None
    ) -> HTTPConnectionPool:
        if scheme == "https":
            return super(ProxyManager, self).connection_from_host(host, port, scheme, pool_kwargs=pool_kwargs)
        return super(ProxyManager, self).connection_from_host(self.proxy.host, self.proxy.port, self.proxy.scheme, pool_kwargs=pool_kwargs)

    def _set_proxy_headers(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers_: Dict[str, str] = {"Accept": "*/*"}
        netloc: str = parse_url(url).netloc
        if netloc:
            headers_["Host"] = netloc
        if headers:
            headers_.update(headers)
        return headers_

    async def urlopen(self, method: str, url: str, redirect: bool = True, **kw: Any) -> Any:
        u = parse_url(url)
        if u.scheme == "http":
            headers: Dict[str, str] = kw.get("headers", self.headers)  # type: ignore
            kw["headers"] = self._set_proxy_headers(url, headers)
        return await super(ProxyManager, self).urlopen(method, url, redirect=redirect, **kw)

def proxy_from_url(url: str, **kw: Any) -> ProxyManager:
    return ProxyManager(proxy_url=url, **kw)