import httpcore
import pytest
import httpx
from typing import Dict, Optional, Type, Union


def url_to_origin(url: str) -> httpcore.URL:
    """
    Given a URL string, return the origin in the raw tuple format that
    `httpcore` uses for its representation.
    """
    u: httpx.URL = httpx.URL(url)
    return httpcore.URL(scheme=u.raw_scheme, host=u.raw_host, port=u.port, target='/')


def test_socks_proxy() -> None:
    url: httpx.URL = httpx.URL('http://www.example.com')
    for proxy in ('socks5://localhost/', 'socks5h://localhost/'):
        client: httpx.Client = httpx.Client(proxy=proxy)
        transport = client._transport_for_url(url)
        assert isinstance(transport, httpx.HTTPTransport)
        assert isinstance(transport._pool, httpcore.SOCKSProxy)
        async_client: httpx.AsyncClient = httpx.AsyncClient(proxy=proxy)
        async_transport = async_client._transport_for_url(url)
        assert isinstance(async_transport, httpx.AsyncHTTPTransport)
        assert isinstance(async_transport._pool, httpcore.AsyncSOCKSProxy)


PROXY_URL: str = 'http://[::1]'


@pytest.mark.parametrize(
    ['url', 'proxies', 'expected'],
    [
        ('http://example.com', {}, None),
        ('http://example.com', {'https://': PROXY_URL}, None),
        ('http://example.com', {'http://example.net': PROXY_URL}, None),
        ('http://example.com', {'http://*': PROXY_URL}, PROXY_URL),
        ('https://example.com', {'http://*': PROXY_URL}, None),
        ('http://example.com', {'http://example.com': PROXY_URL}, PROXY_URL),
        ('http://www.example.com', {'http://example.com': PROXY_URL}, None),
        ('http://example.com', {'http://*.example.com': PROXY_URL}, None),
        ('http://www.example.com', {'http://*.example.com': PROXY_URL}, PROXY_URL),
        ('http://example.com', {'http://*example.com': PROXY_URL}, PROXY_URL),
        ('http://www.example.com', {'http://*example.com': PROXY_URL}, PROXY_URL),
        ('http://wwwexample.com', {'http://*example.com': PROXY_URL}, None),
        ('http://example.com:443', {'http://example.com': PROXY_URL}, PROXY_URL),
        ('http://example.com', {'all://': PROXY_URL}, PROXY_URL),
        ('http://example.com', {'http://': PROXY_URL}, PROXY_URL),
        ('http://example.com', {'all://example.com': PROXY_URL}, PROXY_URL),
        ('http://example.com', {'http://example.com': PROXY_URL}, PROXY_URL),
        ('http://example.com', {'http://example.com:80': PROXY_URL}, PROXY_URL),
        ('http://example.com:8080', {'http://example.com:8080': PROXY_URL}, PROXY_URL),
        ('http://example.com:8080', {'http://example.com': PROXY_URL}, PROXY_URL),
        (
            'http://example.com',
            {
                'all://': PROXY_URL + ':1',
                'http://': PROXY_URL + ':2',
                'all://example.com': PROXY_URL + ':3',
                'http://example.com': PROXY_URL + ':4',
            },
            PROXY_URL + ':4',
        ),
        (
            'http://example.com',
            {
                'all://': PROXY_URL + ':1',
                'http://': PROXY_URL + ':2',
                'all://example.com': PROXY_URL + ':3',
            },
            PROXY_URL + ':3',
        ),
        (
            'http://example.com',
            {
                'all://': PROXY_URL + ':1',
                'http://': PROXY_URL + ':2',
            },
            PROXY_URL + ':2',
        ),
    ]
)
def test_transport_for_request(
    url: str,
    proxies: Dict[str, str],
    expected: Optional[str],
) -> None:
    mounts: Dict[str, httpx.HTTPTransport] = {key: httpx.HTTPTransport(proxy=value) for key, value in proxies.items()}
    client: httpx.Client = httpx.Client(mounts=mounts)
    transport = client._transport_for_url(httpx.URL(url))
    if expected is None:
        assert transport is client._transport
    else:
        assert isinstance(transport, httpx.HTTPTransport)
        assert isinstance(transport._pool, httpcore.HTTPProxy)
        assert transport._pool._proxy_url == url_to_origin(expected)


@pytest.mark.anyio
@pytest.mark.network
async def test_async_proxy_close() -> None:
    try:
        transport: httpx.AsyncHTTPTransport = httpx.AsyncHTTPTransport(proxy=PROXY_URL)
        client: httpx.AsyncClient = httpx.AsyncClient(mounts={'https://': transport})
        await client.get('http://example.com')
    finally:
        await client.aclose()


@pytest.mark.network
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


@pytest.mark.parametrize(
    ['url', 'env', 'expected'],
    [
        ('http://google.com', {}, None),
        ('http://google.com', {'HTTP_PROXY': 'http://example.com'}, 'http://example.com'),
        ('http://google.com', {'HTTP_PROXY': 'example.com'}, 'http://example.com'),
        ('http://google.com', {'HTTP_PROXY': 'http://example.com', 'NO_PROXY': 'google.com'}, None),
        ('http://127.0.0.1', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': ''}, 'http://localhost:123'),
        ('http://127.0.0.1', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': '127.0.0.1'}, None),
        (
            'http://127.0.0.1',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'https://127.0.0.1'},
            'http://localhost:123',
        ),
        (
            'http://127.0.0.1',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': '1.1.1.1'},
            'http://localhost:123',
        ),
        (
            'http://courses.mit.edu',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu'},
            None,
        ),
        (
            'https://mit.edu.info',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu'},
            'http://localhost:123',
        ),
        (
            'https://mit.edu.info',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu,edu.info'},
            None,
        ),
        (
            'https://mit.edu.info',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu, edu.info'},
            None,
        ),
        (
            'https://mit.edu.info',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu,mit.info'},
            'http://localhost:123',
        ),
        (
            'https://foo.example.com',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'www.example.com'},
            'http://localhost:123',
        ),
        (
            'https://www.example1.com',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': '.example1.com'},
            None,
        ),
        (
            'https://www.example2.com',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'ample2.com'},
            'http://localhost:123',
        ),
        (
            'https://www.example3.com',
            {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': '*'},
            None,
        ),
    ],
)
@pytest.mark.parametrize('client_class', [httpx.Client, httpx.AsyncClient])
def test_proxies_environ(
    monkeypatch: pytest.MonkeyPatch,
    client_class: Union[Type[httpx.Client], Type[httpx.AsyncClient]],
    url: str,
    env: Dict[str, str],
    expected: Optional[str],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    client: Union[httpx.Client, httpx.AsyncClient] = client_class()
    transport = client._transport_for_url(httpx.URL(url))
    if expected is None:
        assert transport == client._transport
    else:
        assert transport._pool._proxy_url == url_to_origin(expected)


@pytest.mark.parametrize(
    ['proxies', 'is_valid'],
    [
        ({'http': 'http://127.0.0.1'}, False),
        ({'https': 'http://127.0.0.1'}, False),
        ({'all': 'http://127.0.0.1'}, False),
        ({'http://': 'http://127.0.0.1'}, True),
        ({'https://': 'http://127.0.0.1'}, True),
        ({'all://': 'http://127.0.0.1'}, True),
    ],
)
def test_for_deprecated_proxy_params(
    proxies: Dict[str, str],
    is_valid: bool,
) -> None:
    mounts: Dict[str, httpx.HTTPTransport] = {key: httpx.HTTPTransport(proxy=value) for key, value in proxies.items()}
    if not is_valid:
        with pytest.raises(ValueError):
            httpx.Client(mounts=mounts)
    else:
        httpx.Client(mounts=mounts)


def test_proxy_with_mounts() -> None:
    proxy_transport: httpx.HTTPTransport = httpx.HTTPTransport(proxy='http://127.0.0.1')
    client: httpx.Client = httpx.Client(mounts={'http://': proxy_transport})
    transport = client._transport_for_url(httpx.URL('http://example.com'))
    assert transport == proxy_transport
