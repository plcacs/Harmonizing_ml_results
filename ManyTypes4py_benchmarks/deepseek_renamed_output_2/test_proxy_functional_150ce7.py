import asyncio
import os
import pathlib
import ssl
import sys
from re import match as match_regex
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple, TypedDict, Union, cast
from unittest import mock
from uuid import UUID, uuid4
import proxy
import pytest
from pytest_mock import MockerFixture
from yarl import URL
import aiohttp
from aiohttp import ClientResponse, web
from aiohttp.client import _RequestContextManager, _RequestOptions
from aiohttp.client_exceptions import ClientConnectionError, ClientConnectorError, ClientHttpProxyError
from aiohttp.pytest_plugin import AiohttpRawServer, AiohttpServer
from aiohttp.typedefs import LooseHeaders
from aiohttp.http_websocket import WSMessage

ASYNCIO_SUPPORTS_TLS_IN_TLS = sys.version_info >= (3, 11)


class _ResponseArgs(TypedDict):
    status: int
    headers: Optional[LooseHeaders]
    body: Optional[Union[str, bytes]]


if sys.version_info >= (3, 11) and TYPE_CHECKING:
    from typing import Unpack

    async def func_7plw6tuz(
        method: str = 'GET',
        *,
        url: Union[str, URL],
        trust_env: bool = False,
        **kwargs: Unpack[_RequestOptions]
    ) -> ClientResponse:
        ...
else:
    from typing import Any

    async def func_7plw6tuz(
        method: str = 'GET',
        *,
        url: Union[str, URL],
        trust_env: bool = False,
        **kwargs: Any
    ) -> ClientResponse:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector, trust_env=trust_env) as client:
            async with client.request(method, url, **kwargs) as resp:
                return resp


@pytest.fixture
def func_e0k8yhts(tls_certificate_pem_path: str) -> Iterator[URL]:
    """Return the URL of an instance of a running secure proxy.

    This fixture also spawns that instance and tears it down after the test.
    """
    proxypy_args = [
        '--threaded' if os.name == 'nt' else '--threadless',
        '--num-workers', '1', '--hostname', '127.0.0.1', '--port', '0',
        '--cert-file', tls_certificate_pem_path, '--key-file',
        tls_certificate_pem_path
    ]
    with proxy.Proxy(input_args=proxypy_args) as proxy_instance:
        yield URL.build(
            scheme='https',
            host=str(proxy_instance.flags.hostname),
            port=proxy_instance.flags.port
        )


@pytest.fixture
def func_bic9bwv8() -> str:
    return str(uuid4())


@pytest.fixture(params=('http', 'https'))
def func_tlsl13ng(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture
async def func_4u4w7iq7(
    aiohttp_server: Callable[..., Awaitable[AiohttpServer]],
    ssl_ctx: Optional[ssl.SSLContext],
    web_server_endpoint_payload: str,
    web_server_endpoint_type: str
) -> URL:
    async def func_8dxfn7hv(request: web.Request) -> web.Response:
        return web.Response(text=web_server_endpoint_payload)
    app = web.Application()
    app.router.add_route('GET', '/', func_8dxfn7hv)
    if web_server_endpoint_type == 'https':
        server = await aiohttp_server(app, ssl=ssl_ctx)
    else:
        server = await aiohttp_server(app)
    return URL.build(
        scheme=web_server_endpoint_type,
        host=server.host,
        port=server.port
    )


@pytest.mark.skipif(
    not ASYNCIO_SUPPORTS_TLS_IN_TLS,
    reason='asyncio on this python does not support TLS in TLS'
)
@pytest.mark.parametrize('web_server_endpoint_type', ('http', 'https'))
@pytest.mark.filterwarnings('ignore:.*ssl.OP_NO_SSL*')
@pytest.mark.usefixtures('loop')
async def func_pj05lpvm(
    client_ssl_ctx: ssl.SSLContext,
    secure_proxy_url: URL,
    web_server_endpoint_url: URL,
    web_server_endpoint_payload: str
) -> None:
    """Ensure HTTP(S) sites are accessible through a secure proxy."""
    conn = aiohttp.TCPConnector()
    sess = aiohttp.ClientSession(connector=conn)
    async with sess.get(
        web_server_endpoint_url,
        proxy=secure_proxy_url,
        ssl=client_ssl_ctx
    ) as response:
        assert response.status == 200
        assert await response.text() == web_server_endpoint_payload
    await sess.close()
    await conn.close()
    await asyncio.sleep(0.1)


@pytest.mark.parametrize('web_server_endpoint_type', ('https',))
@pytest.mark.usefixtures('loop')
@pytest.mark.skipif(
    ASYNCIO_SUPPORTS_TLS_IN_TLS,
    reason='asyncio on this python supports TLS in TLS'
)
@pytest.mark.filterwarnings('ignore:.*ssl.OP_NO_SSL*')
async def func_d49rxnvl(
    client_ssl_ctx: ssl.SSLContext,
    secure_proxy_url: URL,
    web_server_endpoint_type: str
) -> None:
    """Ensure connecting to TLS endpoints w/ HTTPS proxy needs patching.

    This also checks that a helpful warning on how to patch the env
    is displayed.
    """
    url = URL.build(scheme=web_server_endpoint_type, host='python.org')
    assert url.host is not None
    escaped_host_port = ':'.join((url.host.replace('.', '\\.'), str(url.port)))
    escaped_proxy_url = str(secure_proxy_url).replace('.', '\\.')
    conn = aiohttp.TCPConnector()
    sess = aiohttp.ClientSession(connector=conn)
    expected_warning_text = (
        "^An HTTPS request is being sent through an HTTPS proxy\\. This support for TLS in TLS is known to be disabled in the stdlib asyncio\\. This is why you'll probably see an error in the log below\\.\\n\\nIt is possible to enable it via monkeypatching\\. For more details, see:\\n\\* https://bugs\\.python\\.org/issue37179\\n\\* https://github\\.com/python/cpython/pull/28073\\n\\nYou can temporarily patch this as follows:\\n\\* https://docs\\.aiohttp\\.org/en/stable/client_advanced\\.html#proxy-support\\n\\* https://github\\.com/aio-libs/aiohttp/discussions/6044\\n$"
    )
    type_err = (
        'transport <asyncio\\.sslproto\\._SSLProtocolTransport object at 0x[\\d\\w]+> is not supported by start_tls\\(\\)'
    )
    expected_exception_reason = (
        f'^Cannot initialize a TLS-in-TLS connection to host {escaped_host_port!s} through an underlying connection to an HTTPS proxy {escaped_proxy_url!s} ssl:{client_ssl_ctx!s} [{type_err!s}]$'
    )
    with pytest.warns(RuntimeWarning, match=expected_warning_text), \
         pytest.raises(ClientConnectionError, match=expected_exception_reason) as conn_err:
        async with sess.get(url, proxy=secure_proxy_url, ssl=client_ssl_ctx):
            pass
    assert isinstance(conn_err.value.__cause__, TypeError)
    assert match_regex(f'^{type_err!s}$', str(conn_err.value.__cause__))
    await sess.close()
    await conn.close()
    await asyncio.sleep(0.1)


@pytest.fixture
def func_5ysx6lkz(
    aiohttp_raw_server: Callable[..., Awaitable[AiohttpRawServer]],
    loop: asyncio.AbstractEventLoop,
    monkeypatch: pytest.MonkeyPatch
) -> Callable[[], Awaitable[mock.Mock]]:
    def _patch_ssl_transport(monkeypatch: pytest.MonkeyPatch) -> None:
        def _make_ssl_transport_dummy(
            self: asyncio.AbstractEventLoop,
            rawsock: socket.socket,
            protocol: asyncio.protocols.Protocol,
            sslcontext: ssl.SSLContext,
            waiter: Optional[asyncio.Future[None]] = None,
            **kwargs: Any
        ) -> asyncio.transports.Transport:
            return self._make_socket_transport(
                rawsock, protocol, waiter,
                extra=kwargs.get('extra'),
                server=kwargs.get('server')
            )
        monkeypatch.setattr(
            'asyncio.selector_events.BaseSelectorEventLoop._make_ssl_transport',
            _make_ssl_transport_dummy
        )

    _patch_ssl_transport(monkeypatch)
    default_response: _ResponseArgs = {
        'status': 200,
        'headers': None,
        'body': None
    }
    proxy_mock = mock.Mock()

    async def func_wki57u2z(request: web.Request) -> web.Response:
        proxy_mock.request = request
        proxy_mock.requests_list.append(request)
        response = default_response.copy()
        if isinstance(proxy_mock.return_value, dict):
            response.update(proxy_mock.return_value)
        headers = response['headers']
        if not headers:
            headers = {}
        if request.method == 'CONNECT':
            response['body'] = None
        response['headers'] = headers
        resp = web.Response(**response)
        await resp.prepare(request)
        await resp.write_eof()
        return resp

    async def func_4j0jx9qy() -> mock.Mock:
        proxy_mock.request = None
        proxy_mock.auth = None
        proxy_mock.requests_list = []
        server = await aiohttp_raw_server(func_wki57u2z)
        proxy_mock.server = server
        proxy_mock.url = server.make_url('/')
        return proxy_mock
    return func_4j0jx9qy


async def func_ip8gkzx2(func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]]) -> None:
    url = 'http://aiohttp.io/path?query=yes'
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    assert len(proxy.requests_list) == 1
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path?query=yes'


async def func_tamans5s(func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]]) -> None:
    url = 'http://aiohttp.io:2561/space sheep?q=can:fly'
    raw_url = '/space%20sheep?q=can:fly'
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == raw_url


async def func_eh2mcqh3(func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]]) -> None:
    url = 'http://éé.com/'
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    assert proxy.request.host == 'éé.com'
    assert proxy.request.path_qs == '/'


async def func_k0hlwd5t() -> None:
    url = 'http://aiohttp.io/path'
    proxy_url = 'http://localhost:2242/'
    with pytest.raises(ClientConnectorError):
        await func_7plw6tuz(url=url, proxy=proxy_url)


async def func_5r8uzbq0(func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]]) -> None:
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    proxy.return_value = {
        'status': 502,
        'headers': {'Proxy-Agent': 'TestProxy'}
    }
    resp = await func_7plw6tuz(url=url, proxy=proxy.url)
    assert resp.status == 502
    assert resp.headers['Proxy-Agent'] == 'TestProxy'


async def func_rcfw3uie(func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]]) -> None:
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    assert 'Authorization' not in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers
    auth = aiohttp.BasicAuth('user', 'pass')
    await func_7plw6tuz(url=url, auth=auth, proxy=proxy.url)
    assert 'Authorization' in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers
    await func_7plw6tuz(url=url, proxy_auth=auth, proxy=proxy.url)
    assert 'Authorization' not in proxy.request.headers
    assert 'Proxy-Authorization' in proxy.request.headers
    await func_7plw6tuz(url=url, auth=auth, proxy_auth=auth, proxy=proxy.url)
    assert 'Authorization' in proxy.request.headers
    assert 'Proxy-Authorization' in proxy.request.headers


async def func_59jqnhqn(func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]]) -> None:
    url = 'http://aiohttp.io/path'
    auth = aiohttp.BasicAuth('юзер', 'пасс', 'utf-8')
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, auth=auth, proxy=proxy.url)
    assert 'Authorization' in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers


async def func_dthrebwa(func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]]) -> None:
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    auth_url = URL(url).with_user('user').with_password('pass')
    await func_7plw6tuz(url=auth_url, proxy=proxy.url)
    assert 'Authorization' in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers
    proxy_url = URL(proxy.url).with_user('user').with_password('pass')
    await func_7plw6tuz(url=url, proxy=proxy_url)
    assert 'Authorization' not in proxy.request.headers
    assert 'Proxy-Authorization' in proxy.request.headers


async def func_vgm1uc7v(
    func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]],
    loop: asyncio.AbstractEventLoop
) -> None:
    url = 'http://aiohttp.io/path'
    conn = aiohttp.TCPConnector()
    sess = aiohttp.ClientSession(connector=conn)
    proxy = await func_5ysx6lkz()
    assert 0 == len(conn._acquired)
    async with sess.get(url, proxy=proxy.url) as resp:
        pass
    assert resp.closed
    assert 0 == len(conn._acquired)
    await sess.close()


@pytest.mark.skip('we need to reconsider how we test this')
async def func_rvn4xsd7(
    func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]],
    loop: asyncio.AbstractEventLoop
) -> None:
    url = 'http://aiohttp.io/path'
    conn = aiohttp.TCPConnector(force_close=True)
    sess = aiohttp.ClientSession(connector=conn)
    proxy = await func_5ysx6lkz()
    assert 0 == len(conn._acquired)

    async def func_i3c3guta() -> None:
        async with sess.get(url, proxy=proxy.url):
            assert 1 == len(conn._acquired)
    await func_i3c3guta()
    assert 0 == len(conn._acquired)
    await sess.close()


@pytest.mark.skip('we need to reconsider how we test this')
async def func_91sauqg5(
    func_5ysx6lkz: Callable[[], Awaitable[mock.Mock]],
    loop: asyncio.AbstractEventLoop
) -> None:
    url = 'http://aiohttp.io/path'
    limit, multi_conn_num = 1, 5
    conn = aiohttp.TCPConnector(limit=limit)
    sess = aiohttp.ClientSession(connector=conn)
    proxy = await func_5ysx6lkz()
    current_pid: Optional[int] = None

    async def func_i3c3guta(pid: int) -> ClientResponse:
        nonlocal current_pid
        async with sess.get(url, proxy=proxy.url) as resp:
            current_pid = pid
            await asyncio.sleep(0.2)
            assert current_pid == pid
        return resp
    requests = [func_i3c3guta(pid) for