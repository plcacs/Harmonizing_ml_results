from typing import Optional, Dict, Any, List, Tuple, Union

def url_to_origin(url: str) -> httpcore.URL:
    """
    Given a URL string, return the origin in the raw tuple format that
    `httpcore` uses for it's representation.
    """
    u = httpx.URL(url)
    return httpcore.URL(scheme=u.raw_scheme, host=u.raw_host, port=u.port, target='/')

def test_socks_proxy() -> None:
    url: httpx.URL = httpx.URL('http://www.example.com')
    for proxy in ('socks5://localhost/', 'socks5h://localhost/'):
        client: httpx.Client = httpx.Client(proxy=proxy)
        transport: httpx.HTTPTransport = client._transport_for_url(url)
        assert isinstance(transport, httpx.HTTPTransport)
        assert isinstance(transport._pool, httpcore.SOCKSProxy)
        async_client: httpx.AsyncClient = httpx.AsyncClient(proxy=proxy)
        async_transport: httpx.AsyncHTTPTransport = async_client._transport_for_url(url)
        assert isinstance(async_transport, httpx.AsyncHTTPTransport)
        assert isinstance(async_transport._pool, httpcore.AsyncSOCKSProxy)

PROXY_URL: str = 'http://[::1]'

def test_transport_for_request(url: str, proxies: Dict[str, str], expected: Optional[str]) -> None:
    mounts: Dict[str, httpx.HTTPTransport] = {key: httpx.HTTPTransport(proxy=value) for key, value in proxies.items()}
    client: httpx.Client = httpx.Client(mounts=mounts)
    transport: httpx.HTTPTransport = client._transport_for_url(httpx.URL(url))
    if expected is None:
        assert transport is client._transport
    else:
        assert isinstance(transport, httpx.HTTPTransport)
        assert isinstance(transport._pool, httpcore.HTTPProxy)
        assert transport._pool._proxy_url == url_to_origin(expected)

async def test_async_proxy_close() -> None:
    try:
        transport: httpx.AsyncHTTPTransport = httpx.AsyncHTTPTransport(proxy=PROXY_URL)
        client: httpx.AsyncClient = httpx.AsyncClient(mounts={'https://': transport})
        await client.get('http://example.com')
    finally:
        await client.aclose()

def test_sync_proxy_close() -> None:
    try:
        transport: httpx.HTTPTransport = httpx.HTTPTransport(proxy=PROXY_URL)
        client: httpx.Client = httpx.Client(mounts={'https://': transport})
        client.get('http://example.com')
    finally:
        client.close()

def test_unsupported_proxy_scheme() -> None:
    with pytest.raises(ValueError):
        httpx.Client(proxy='ftp://127.0.0.1')

def test_proxies_environ(monkeypatch, client_class, url, env, expected) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    client: client_class = client_class()
    transport: httpx.HTTPTransport = client._transport_for_url(httpx.URL(url))
    if expected is None:
        assert transport == client._transport
    else:
        assert transport._pool._proxy_url == url_to_origin(expected)

def test_for_deprecated_proxy_params(proxies: Dict[str, str], is_valid: bool) -> None:
    mounts: Dict[str, httpx.HTTPTransport] = {key: httpx.HTTPTransport(proxy=value) for key, value in proxies.items()}
    if not is_valid:
        with pytest.raises(ValueError):
            httpx.Client(mounts=mounts)
    else:
        httpx.Client(mounts=mounts)

def test_proxy_with_mounts() -> None:
    proxy_transport: httpx.HTTPTransport = httpx.HTTPTransport(proxy='http://127.0.0.1')
    client: httpx.Client = httpx.Client(mounts={'http://': proxy_transport})
    transport: httpx.HTTPTransport = client._transport_for_url(httpx.URL('http://example.com'))
    assert transport == proxy_transport
