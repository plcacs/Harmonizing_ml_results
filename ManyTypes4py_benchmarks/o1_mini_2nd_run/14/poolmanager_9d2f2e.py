from __future__ import absolute_import
import collections
import functools
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .._collections import RecentlyUsedContainer
from ..base import DEFAULT_PORTS
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import LocationValueError, MaxRetryError, ProxySchemeUnknown
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..util.url import parse_url
from ..util.request import set_file_position
from ..util.retry import Retry

__all__ = ['PoolManager', 'ProxyManager', 'proxy_from_url']
log = logging.getLogger(__name__)
SSL_KEYWORDS: Tuple[str, ...] = (
    'key_file',
    'cert_file',
    'cert_reqs',
    'ca_certs',
    'ssl_version',
    'ca_cert_dir',
    'ssl_context',
)
_key_fields: Tuple[str, ...] = (
    'key_scheme',
    'key_host',
    'key_strict',
    'key_port',
    'key_timeout',
    'key_retries',
    'key_block',
    'key_source_address',
    'key_key_file',
    'key_cert_file',
    'key_cert_reqs',
    'key_ca_certs',
    'key_ssl_version',
    'key_ca_cert_dir',
    'key_ssl_context',
    'key_maxsize',
    'key_headers',
    'key__proxy',
    'key__proxy_headers',
    'key_socket_options',
    'key__socks_options',
    'key_assert_hostname',
    'key_assert_fingerprint',
)
PoolKey = collections.namedtuple('PoolKey', _key_fields)


def _default_key_normalizer(
    key_class: Callable[..., collections.namedtuple],
    request_context: Dict[str, Any],
) -> PoolKey:
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
            context[key] = frozenset(context[key].items())
    socket_opts = context.get('socket_options')
    if socket_opts is not None:
        context['socket_options'] = tuple(socket_opts)
    for key in list(context.keys()):
        context['key_' + key] = context.pop(key)
    for field in key_class._fields:
        if field not in context:
            context[field] = None
    return key_class(**context)  # type: ignore


key_fn_by_scheme: Dict[str, Callable[[Dict[str, Any]], PoolKey]] = {
    'http': functools.partial(_default_key_normalizer, PoolKey),
    'https': functools.partial(_default_key_normalizer, PoolKey),
}
pool_classes_by_scheme: Dict[str, Callable[..., Union[HTTPConnectionPool, HTTPSConnectionPool]]] = {
    'http': HTTPConnectionPool,
    'https': HTTPSConnectionPool,
}


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

    :param \\**connection_pool_kw:
        Additional parameters are used to create fresh
        :class:`urllib3.connectionpool.ConnectionPool` instances.

    Example::

        >>> manager = PoolManager(num_pools=2)
        >>> r = manager.request('GET', 'http://google.com/')
        >>> r = manager.request('GET', 'http://google.com/mail')
        >>> r = manager.request('GET', 'http://yahoo.com/')
        >>> len(manager.pools)
        2

    """

    proxy: Optional[Any] = None

    def __init__(
        self,
        num_pools: int = 10,
        headers: Optional[Dict[str, str]] = None,
        backend: Optional[Any] = None,
        **connection_pool_kw: Any,
    ) -> None:
        super(PoolManager, self).__init__(headers)
        self.connection_pool_kw: Dict[str, Any] = connection_pool_kw
        self.pools: RecentlyUsedContainer = RecentlyUsedContainer(
            num_pools, dispose_func=lambda p: p.close()
        )
        self.pool_classes_by_scheme = pool_classes_by_scheme
        self.key_fn_by_scheme = key_fn_by_scheme.copy()
        self.backend = backend

    def __enter__(self) -> 'PoolManager':
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        self.clear()
        return False

    def _new_pool(
        self,
        scheme: str,
        host: str,
        port: int,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        """
        Create a new :class:`ConnectionPool` based on host, port, scheme, and
        any additional pool keyword arguments.

        If ``request_context`` is provided, it is provided as keyword arguments
        to the pool class used. This method is used to actually create the
        connection pools handed out by :meth:`connection_from_url` and
        companion methods. It is intended to be overridden for customization.
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

    def connection_from_host(
        self,
        host: str,
        port: Optional[int] = None,
        scheme: str = 'http',
        pool_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        """
        Get a :class:`ConnectionPool` based on the host, port, and scheme.

        If ``port`` isn't given, it will be derived from the ``scheme`` using
        ``urllib3.connectionpool.port_by_scheme``. If ``pool_kwargs`` is
        provided, it is merged with the instance's ``connection_pool_kw``
        variable and used to create the new connection pool, if one is
        needed.
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

    def connection_from_context(
        self, request_context: Dict[str, Any]
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        """
        Get a :class:`ConnectionPool` based on the request context.

        ``request_context`` must at least contain the ``scheme`` key and its
        value must be a key in ``key_fn_by_scheme`` instance variable.
        """
        scheme = request_context['scheme'].lower()
        pool_key_constructor = self.key_fn_by_scheme[scheme]
        pool_key: PoolKey = pool_key_constructor(request_context)
        return self.connection_from_pool_key(pool_key, request_context=request_context)

    def connection_from_pool_key(
        self,
        pool_key: PoolKey,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        """
        Get a :class:`ConnectionPool` based on the provided pool key.

        ``pool_key`` should be a namedtuple that only contains immutable
        objects. At a minimum it must have the ``scheme``, ``host``, and
        ``port`` fields.
        """
        with self.pools.lock:
            pool = self.pools.get(pool_key)
            if pool:
                return pool
            if request_context is None:
                raise ValueError("request_context must be provided if pool is not found.")
            scheme = request_context['scheme']
            host = request_context['host']
            port = request_context['port']
            pool = self._new_pool(scheme, host, port, request_context=request_context)
            self.pools[pool_key] = pool
        return pool

    def connection_from_url(
        self, url: str, pool_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        """
        Similar to :func:`urllib3.connectionpool.connection_from_url`.

        If ``pool_kwargs`` is not provided and a new pool needs to be
        constructed, ``self.connection_pool_kw`` is used to initialize
        the :class:`urllib3.connectionpool.ConnectionPool`. If ``pool_kwargs``
        is provided, it is used instead. Note that if a new pool does not
        need to be created for the request, the provided ``pool_kwargs`` are
        not used.
        """
        u = parse_url(url)
        return self.connection_from_host(
            u.host, port=u.port, scheme=u.scheme, pool_kwargs=pool_kwargs
        )

    def _merge_pool_kwargs(
        self, override: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge a dictionary of override values for self.connection_pool_kw.

        This does not modify self.connection_pool_kw and returns a new dict.
        Any keys in the override dictionary with a value of ``None`` are
        removed from the merged dictionary.
        """
        base_pool_kwargs: Dict[str, Any] = self.connection_pool_kw.copy()
        if override:
            for key, value in override.items():
                if value is None:
                    base_pool_kwargs.pop(key, None)
                else:
                    base_pool_kwargs[key] = value
        return base_pool_kwargs

    def urlopen(
        self,
        method: str,
        url: str,
        redirect: bool = True,
        **kw: Any,
    ) -> Any:
        """
        Same as :meth:`urllib3.connectionpool.HTTPConnectionPool.urlopen`
        with redirect logic and only sends the request-uri portion of the
        ``url``.

        The given ``url`` parameter must be absolute, such that an appropriate
        :class:`urllib3.connectionpool.ConnectionPool` can be chosen for it.
        """
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
        retries: Union[int, Retry] = kw.get('retries', Retry.DEFAULT)
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect)
        try:
            retries = retries.increment(
                method, url, response=response, _pool=conn
            )
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
    """
    Behaves just like :class:`PoolManager`, but sends all requests through
    the defined proxy, using the CONNECT method for HTTPS URLs.

    :param proxy_url:
        The URL of the proxy to be used.

    :param proxy_headers:
        A dictionary contaning headers that will be sent to the proxy. In case
        of HTTP they are being sent with each request, while in the
        HTTPS/CONNECT case they are sent only once. Could be used for proxy
        authentication.

    Example:
        >>> proxy = urllib3.ProxyManager('http://localhost:3128/')
        >>> r1 = proxy.request('GET', 'http://google.com/')
        >>> r2 = proxy.request('GET', 'http://httpbin.org/')
        >>> len(proxy.pools)
        1
        >>> r3 = proxy.request('GET', 'https://httpbin.org/')
        >>> r4 = proxy.request('GET', 'https://twitter.com/')
        >>> len(proxy.pools)
        3

    """

    def __init__(
        self,
        proxy_url: Union[str, HTTPConnectionPool],
        num_pools: int = 10,
        headers: Optional[Dict[str, str]] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        **connection_pool_kw: Any,
    ) -> None:
        if isinstance(proxy_url, HTTPConnectionPool):
            proxy_url = f"{proxy_url.scheme}://{proxy_url.host}:{proxy_url.port}"
        proxy = parse_url(proxy_url)
        if not proxy.port:
            port = DEFAULT_PORTS.get(proxy.scheme, 80)
            proxy = proxy._replace(port=port)
        if proxy.scheme not in ('http', 'https'):
            raise ProxySchemeUnknown(proxy.scheme)
        self.proxy: Any = proxy
        self.proxy_headers: Dict[str, str] = proxy_headers or {}
        connection_pool_kw['_proxy'] = self.proxy
        connection_pool_kw['_proxy_headers'] = self.proxy_headers
        super(ProxyManager, self).__init__(
            num_pools, headers, **connection_pool_kw
        )

    def connection_from_host(
        self,
        host: str,
        port: Optional[int] = None,
        scheme: str = 'http',
        pool_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[HTTPConnectionPool, HTTPSConnectionPool]:
        if scheme == 'https':
            return super(ProxyManager, self).connection_from_host(
                host, port, scheme, pool_kwargs=pool_kwargs
            )
        return super(ProxyManager, self).connection_from_host(
            self.proxy.host, self.proxy.port, self.proxy.scheme, pool_kwargs=pool_kwargs
        )

    def _set_proxy_headers(
        self, url: str, headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
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

    def urlopen(
        self,
        method: str,
        url: str,
        redirect: bool = True,
        **kw: Any,
    ) -> Any:
        """Same as HTTP(S)ConnectionPool.urlopen, ``url`` must be absolute."""
        u = parse_url(url)
        if u.scheme == 'http':
            headers = kw.get('headers', self.headers)
            kw['headers'] = self._set_proxy_headers(url, headers)
        return super(ProxyManager, self).urlopen(method, url, redirect=redirect, **kw)


def proxy_from_url(url: str, **kw: Any) -> ProxyManager:
    return ProxyManager(proxy_url=url, **kw)
