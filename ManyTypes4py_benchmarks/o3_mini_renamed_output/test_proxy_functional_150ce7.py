import asyncio
import os
import pathlib
import ssl
import sys
from re import match as match_regex
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterator, Optional, TypedDict, Union
from unittest import mock
from uuid import uuid4

import proxy
import pytest
from pytest_mock import MockerFixture
from yarl import URL
import aiohttp
from aiohttp import ClientResponse, web
from aiohttp.client import _RequestOptions  # noqa: F401
from aiohttp.client_exceptions import ClientConnectionError
from aiohttp.pytest_plugin import AiohttpRawServer, AiohttpServer

ASYNCIO_SUPPORTS_TLS_IN_TLS: bool = sys.version_info >= (3, 11)


class _ResponseArgs(TypedDict):
    pass


if sys.version_info >= (3, 11) and TYPE_CHECKING:
    from typing import Unpack

    async def func_7plw6tuz(method: str = 'GET', *, url: Union[str, URL], trust_env: bool = False, **kwargs: Any) -> Any:
        ...
else:
    async def func_7plw6tuz(method: str = 'GET', *, url: Union[str, URL], trust_env: bool = False, **kwargs: Any) -> ClientResponse:
        connector: aiohttp.TCPConnector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector, trust_env=trust_env) as client:
            async with client.request(method, url, **kwargs) as resp:
                return resp


@pytest.fixture
def func_e0k8yhts(tls_certificate_pem_path: str) -> Iterator[URL]:
    """Return the URL of an instance of a running secure proxy.

    This fixture also spawns that instance and tears it down after the test.
    """
    proxypy_args: list[str] = [
        '--threaded' if os.name == 'nt' else '--threadless',
        '--num-workers', '1',
        '--hostname', '127.0.0.1',
        '--port', '0',
        '--cert-file', tls_certificate_pem_path,
        '--key-file', tls_certificate_pem_path
    ]
    with proxy.Proxy(input_args=proxypy_args) as proxy_instance:
        yield URL.build(scheme='https', host=str(proxy_instance.flags.hostname), port=proxy_instance.flags.port)


@pytest.fixture
def func_bic9bwv8() -> str:
    return str(uuid4())


@pytest.fixture(params=('http', 'https'))
def func_tlsl13ng(request: Any) -> str:
    return request.param


@pytest.fixture
async def func_4u4w7iq7(aiohttp_server: Any, ssl_ctx: ssl.SSLContext, web_server_endpoint_payload: str, web_server_endpoint_type: str) -> URL:

    async def func_8dxfn7hv(request: web.Request) -> web.Response:
        return web.Response(text=web_server_endpoint_payload)

    app: web.Application = web.Application()
    # Use the correct handler name.
    app.router.add_route('GET', '/', func_8dxfn7hv)
    if web_server_endpoint_type == 'https':
        server = await aiohttp_server(app, ssl=ssl_ctx)
    else:
        server = await aiohttp_server(app)
    return URL.build(scheme=web_server_endpoint_type, host=server.host, port=server.port)


@pytest.mark.skipif(not ASYNCIO_SUPPORTS_TLS_IN_TLS, reason='asyncio on this python does not support TLS in TLS')
@pytest.mark.parametrize('web_server_endpoint_type', ('http', 'https'))
@pytest.mark.filterwarnings('ignore:.*ssl.OP_NO_SSL*')
@pytest.mark.usefixtures('loop')
async def func_pj05lpvm(client_ssl_ctx: ssl.SSLContext, secure_proxy_url: URL, web_server_endpoint_url: URL, web_server_endpoint_payload: str) -> None:
    """Ensure HTTP(S) sites are accessible through a secure proxy."""
    conn: aiohttp.TCPConnector = aiohttp.TCPConnector()
    sess: aiohttp.ClientSession = aiohttp.ClientSession(connector=conn)
    async with sess.get(web_server_endpoint_url, proxy=secure_proxy_url, ssl=client_ssl_ctx) as response:
        assert response.status == 200
        text: str = await response.text()
        assert text == web_server_endpoint_payload
    await sess.close()
    await conn.close()
    await asyncio.sleep(0.1)


@pytest.mark.parametrize('web_server_endpoint_type', ('https',))
@pytest.mark.usefixtures('loop')
@pytest.mark.skipif(ASYNCIO_SUPPORTS_TLS_IN_TLS, reason='asyncio on this python supports TLS in TLS')
@pytest.mark.filterwarnings('ignore:.*ssl.OP_NO_SSL*')
async def func_d49rxnvl(client_ssl_ctx: ssl.SSLContext, secure_proxy_url: URL, web_server_endpoint_type: str) -> None:
    """Ensure connecting to TLS endpoints w/ HTTPS proxy needs patching.

    This also checks that a helpful warning on how to patch the env
    is displayed.
    """
    url: URL = URL.build(scheme=web_server_endpoint_type, host='python.org')
    assert url.host is not None
    escaped_host_port: str = ':'.join((url.host.replace('.', '\\.'), str(url.port)))
    escaped_proxy_url: str = str(secure_proxy_url).replace('.', '\\.')
    conn: aiohttp.TCPConnector = aiohttp.TCPConnector()
    sess: aiohttp.ClientSession = aiohttp.ClientSession(connector=conn)
    expected_warning_text: str = (
        "^An HTTPS request is being sent through an HTTPS proxy\\. This support for TLS in TLS is known to be disabled in the stdlib asyncio\\. This is why you'll probably see an error in the log below\\.\\n\\nIt is possible to enable it via monkeypatching\\. For more details, see:\\n\\* https://bugs\\.python\\.org/issue37179\\n\\* https://github\\.com/python/cpython/pull/28073\\n\\nYou can temporarily patch this as follows:\\n\\* https://docs\\.aiohttp\\.org/en/stable/client_advanced\\.html#proxy-support\\n\\* https://github\\.com/aio-libs/aiohttp/discussions/6044\\n$"
    )
    type_err: str = (
        'transport <asyncio\\.sslproto\\._SSLProtocolTransport object at 0x[\\d\\w]+> is not supported by start_tls\\(\\)'
    )
    expected_exception_reason: str = (
        f'^Cannot initialize a TLS-in-TLS connection to host {escaped_host_port!s} through an underlying connection to an HTTPS proxy {escaped_proxy_url!s} ssl:{client_ssl_ctx!s} [{type_err!s}]$'
    )
    with pytest.warns(RuntimeWarning, match=expected_warning_text), pytest.raises(ClientConnectionError, match=expected_exception_reason) as conn_err:
        async with sess.get(url, proxy=secure_proxy_url, ssl=client_ssl_ctx):
            pass
    assert isinstance(conn_err.value.__cause__, TypeError)
    assert match_regex(f'^{type_err!s}$', str(conn_err.value.__cause__))
    await sess.close()
    await conn.close()
    await asyncio.sleep(0.1)


@pytest.fixture
async def func_5ysx6lkz(aiohttp_raw_server: Any, loop: Any, monkeypatch: Any) -> Any:
    # Assume _patch_ssl_transport is defined elsewhere.
    _patch_ssl_transport(monkeypatch)
    default_response: _ResponseArgs = {'status': 200, 'headers': None, 'body': None}
    proxy_mock: Any = mock.Mock()

    async def func_wki57u2z(request: web.Request) -> web.Response:
        proxy_mock.request = request
        if not hasattr(proxy_mock, 'requests_list'):
            proxy_mock.requests_list = []
        proxy_mock.requests_list.append(request)
        response: Dict[str, Any] = default_response.copy()
        if isinstance(proxy_mock.return_value, dict):
            response.update(proxy_mock.return_value)
        headers: Optional[Dict[str, str]] = response['headers']
        if not headers:
            headers = {}
        if request.method == 'CONNECT':
            response['body'] = None
        response['headers'] = headers
        resp: web.Response = web.Response(**response)
        await resp.prepare(request)
        await resp.write_eof()
        return resp

    async def func_4j0jx9qy() -> Any:
        proxy_mock.request = None
        proxy_mock.auth = None
        proxy_mock.requests_list = []
        server = await aiohttp_raw_server(func_wki57u2z)
        proxy_mock.server = server
        proxy_mock.url = server.make_url('/')
        return proxy_mock

    proxy_server: Any = await func_4j0jx9qy()
    return proxy_server


async def func_ip8gkzx2(proxy_test_server: Any) -> None:
    url: str = 'http://aiohttp.io/path?query=yes'
    proxy_obj: Any = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    assert len(proxy_obj.requests_list) == 1
    assert proxy_obj.request.method == 'GET'
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == '/path?query=yes'


async def func_tamans5s(proxy_test_server: Any) -> None:
    url: str = 'http://aiohttp.io:2561/space sheep?q=can:fly'
    raw_url: str = '/space%20sheep?q=can:fly'
    proxy_obj: Any = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == raw_url


async def func_eh2mcqh3(proxy_test_server: Any) -> None:
    url: str = 'http://éé.com/'
    proxy_obj: Any = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    assert proxy_obj.request.host == 'éé.com'
    assert proxy_obj.request.path_qs == '/'


async def func_k0hlwd5t() -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_url: str = 'http://localhost:2242/'
    with pytest.raises(aiohttp.ClientConnectorError):
        await func_7plw6tuz(url=url, proxy=proxy_url)


async def func_5r8uzbq0(proxy_test_server: Any) -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    proxy_obj.return_value = dict(status=502, headers={'Proxy-Agent': 'TestProxy'})
    resp: ClientResponse = await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    assert resp.status == 502
    assert resp.headers['Proxy-Agent'] == 'TestProxy'


async def func_rcfw3uie(proxy_test_server: Any) -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    assert 'Authorization' not in proxy_obj.request.headers
    assert 'Proxy-Authorization' not in proxy_obj.request.headers
    auth: aiohttp.BasicAuth = aiohttp.BasicAuth('user', 'pass')
    await func_7plw6tuz(url=url, auth=auth, proxy=proxy_obj.url)
    assert 'Authorization' in proxy_obj.request.headers
    assert 'Proxy-Authorization' not in proxy_obj.request.headers
    await func_7plw6tuz(url=url, proxy_auth=auth, proxy=proxy_obj.url)
    assert 'Authorization' not in proxy_obj.request.headers
    assert 'Proxy-Authorization' in proxy_obj.request.headers
    await func_7plw6tuz(url=url, auth=auth, proxy_auth=auth, proxy=proxy_obj.url)
    assert 'Authorization' in proxy_obj.request.headers
    assert 'Proxy-Authorization' in proxy_obj.request.headers


async def func_59jqnhqn(proxy_test_server: Any) -> None:
    url: str = 'http://aiohttp.io/path'
    auth: aiohttp.BasicAuth = aiohttp.BasicAuth('юзер', 'пасс', 'utf-8')
    proxy_obj: Any = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, auth=auth, proxy=proxy_obj.url)
    assert 'Authorization' in proxy_obj.request.headers
    assert 'Proxy-Authorization' not in proxy_obj.request.headers


async def func_dthrebwa(proxy_test_server: Any) -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    auth_url: URL = URL(url).with_user('user').with_password('pass')
    await func_7plw6tuz(url=auth_url, proxy=proxy_obj.url)
    assert 'Authorization' in proxy_obj.request.headers
    assert 'Proxy-Authorization' not in proxy_obj.request.headers
    proxy_url: URL = URL(proxy_obj.url).with_user('user').with_password('pass')
    await func_7plw6tuz(url=url, proxy=proxy_url)
    assert 'Authorization' not in proxy_obj.request.headers
    assert 'Proxy-Authorization' in proxy_obj.request.headers


async def func_vgm1uc7v(proxy_test_server: Any, loop: Any) -> None:
    url: str = 'http://aiohttp.io/path'
    conn: aiohttp.TCPConnector = aiohttp.TCPConnector()
    sess: aiohttp.ClientSession = aiohttp.ClientSession(connector=conn)
    proxy_obj: Any = await func_5ysx6lkz()
    assert len(conn._acquired) == 0
    async with sess.get(url, proxy=proxy_obj.url) as resp:
        pass
    assert resp.closed
    assert len(conn._acquired) == 0
    await sess.close()


@pytest.mark.skip('we need to reconsider how we test this')
async def func_rvn4xsd7(proxy_test_server: Any, loop: Any) -> None:
    url: str = 'http://aiohttp.io/path'
    conn: aiohttp.TCPConnector = aiohttp.TCPConnector(force_close=True)
    sess: aiohttp.ClientSession = aiohttp.ClientSession(connector=conn)
    proxy_obj: Any = await func_5ysx6lkz()
    assert len(conn._acquired) == 0

    async def func_i3c3guta() -> None:
        async with sess.get(url, proxy=proxy_obj.url):
            assert len(conn._acquired) == 1
    await func_i3c3guta()
    assert len(conn._acquired) == 0
    await sess.close()


@pytest.mark.skip('we need to reconsider how we test this')
async def func_91sauqg5(proxy_test_server: Any, loop: Any) -> None:
    url: str = 'http://aiohttp.io/path'
    limit: int = 1
    multi_conn_num: int = 5
    conn: aiohttp.TCPConnector = aiohttp.TCPConnector(limit=limit)
    sess: aiohttp.ClientSession = aiohttp.ClientSession(connector=conn)
    proxy_obj: Any = await func_5ysx6lkz()
    current_pid: Optional[int] = None

    async def func_i3c3guta(pid: int) -> ClientResponse:
        nonlocal current_pid
        async with sess.get(url, proxy=proxy_obj.url) as resp:
            current_pid = pid
            await asyncio.sleep(0.2)
            assert current_pid == pid
        return resp

    requests = [func_i3c3guta(pid) for pid in range(multi_conn_num)]
    responses = await asyncio.gather(*requests)
    assert len(responses) == multi_conn_num
    assert {resp.status for resp in responses} == {200}
    await sess.close()


@pytest.mark.xfail
async def func_91746m52(proxy_test_server: Any) -> None:
    proxy_obj: Any = await func_5ysx6lkz()
    url: str = 'https://www.google.com.ua/search?q=aiohttp proxy'
    await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    connect = proxy_obj.requests_list[0]
    assert connect.method == 'CONNECT'
    assert connect.path == 'www.google.com.ua:443'
    assert connect.host == 'www.google.com.ua'
    assert proxy_obj.request.host == 'www.google.com.ua'
    assert proxy_obj.request.path_qs == '/search?q=aiohttp+proxy'


@pytest.mark.xfail
async def func_p4ffujhq(proxy_test_server: Any) -> None:
    proxy_obj: Any = await func_5ysx6lkz()
    url: str = 'https://secure.aiohttp.io:2242/path'
    await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    connect = proxy_obj.requests_list[0]
    assert connect.method == 'CONNECT'
    assert connect.path == 'secure.aiohttp.io:2242'
    assert connect.host == 'secure.aiohttp.io:2242'
    assert proxy_obj.request.host == 'secure.aiohttp.io:2242'
    assert proxy_obj.request.path_qs == '/path'


@pytest.mark.xfail
async def func_0l4yme68(proxy_test_server: Any, loop: Any) -> None:
    sess: aiohttp.ClientSession = aiohttp.ClientSession()
    proxy_obj: Any = await func_5ysx6lkz()
    proxy_obj.return_value = {'status': 200, 'body': b'1' * 2 ** 20}
    url: str = 'https://www.google.com.ua/search?q=aiohttp proxy'
    async with sess.get(url, proxy=proxy_obj.url) as resp:
        body: bytes = await resp.read()
    await sess.close()
    assert body == b'1' * 2 ** 20


@pytest.mark.xfail
async def func_f1pyl7qj(proxy_test_server: Any) -> None:
    url: str = 'https://éé.com/'
    proxy_obj: Any = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    connect = proxy_obj.requests_list[0]
    assert connect.method == 'CONNECT'
    assert connect.path == 'xn--9caa.com:443'
    assert connect.host == 'xn--9caa.com'


async def func_13galeuq() -> None:
    url: str = 'https://secure.aiohttp.io/path'
    proxy_url: str = 'http://localhost:2242/'
    with pytest.raises(aiohttp.ClientConnectorError):
        await func_7plw6tuz(url=url, proxy=proxy_url)


async def func_x3tv0lvq(proxy_test_server: Any) -> None:
    url: str = 'https://secure.aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    proxy_obj.return_value = dict(status=502, headers={'Proxy-Agent': 'TestProxy'})
    with pytest.raises(aiohttp.ClientHttpProxyError):
        await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    assert len(proxy_obj.requests_list) == 1
    assert proxy_obj.request.method == 'CONNECT'


@pytest.mark.xfail
async def func_shr8sdeo(proxy_test_server: Any) -> None:
    url: str = 'https://secure.aiohttp.io/path'
    auth: aiohttp.BasicAuth = aiohttp.BasicAuth('user', 'pass')
    proxy_obj: Any = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy_obj.url)
    connect = proxy_obj.requests_list[0]
    assert 'Authorization' not in connect.headers
    assert 'Proxy-Authorization' not in connect.headers
    assert 'Authorization' not in proxy_obj.request.headers
    assert 'Proxy-Authorization' not in proxy_obj.request.headers
    proxy_obj = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, auth=auth, proxy=proxy_obj.url)
    connect = proxy_obj.requests_list[0]
    assert 'Authorization' not in connect.headers
    assert 'Proxy-Authorization' not in connect.headers
    assert 'Authorization' in proxy_obj.request.headers
    assert 'Proxy-Authorization' not in proxy_obj.request.headers
    proxy_obj = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy_auth=auth, proxy=proxy_obj.url)
    connect = proxy_obj.requests_list[0]
    assert 'Authorization' not in connect.headers
    assert 'Proxy-Authorization' in connect.headers
    assert 'Authorization' not in proxy_obj.request.headers
    assert 'Proxy-Authorization' not in proxy_obj.request.headers
    proxy_obj = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, auth=auth, proxy_auth=auth, proxy=proxy_obj.url)
    connect = proxy_obj.requests_list[0]
    assert 'Authorization' not in connect.headers
    assert 'Proxy-Authorization' in connect.headers
    assert 'Authorization' in proxy_obj.request.headers
    assert 'Proxy-Authorization' not in proxy_obj.request.headers


@pytest.mark.xfail
async def func_34nj69kb(proxy_test_server: Any, loop: Any) -> None:
    url: str = 'https://secure.aiohttp.io/path'
    conn: aiohttp.TCPConnector = aiohttp.TCPConnector()
    sess: aiohttp.ClientSession = aiohttp.ClientSession(connector=conn)
    proxy_obj: Any = await func_5ysx6lkz()
    assert len(conn._acquired) == 0

    async def func_i3c3guta() -> None:
        async with sess.get(url, proxy=proxy_obj.url):
            assert len(conn._acquired) == 1
    await func_i3c3guta()
    assert len(conn._acquired) == 0
    await sess.close()


@pytest.mark.xfail
async def func_jui49i11(proxy_test_server: Any, loop: Any) -> None:
    url: str = 'https://secure.aiohttp.io/path'
    conn: aiohttp.TCPConnector = aiohttp.TCPConnector(force_close=True)
    sess: aiohttp.ClientSession = aiohttp.ClientSession(connector=conn)
    proxy_obj: Any = await func_5ysx6lkz()
    assert len(conn._acquired) == 0

    async def func_i3c3guta() -> None:
        async with sess.get(url, proxy=proxy_obj.url):
            assert len(conn._acquired) == 1
    await func_i3c3guta()
    assert len(conn._acquired) == 0
    await sess.close()


@pytest.mark.xfail
async def func_yn66l674(proxy_test_server: Any, loop: Any) -> None:
    url: str = 'https://secure.aiohttp.io/path'
    limit: int = 1
    multi_conn_num: int = 5
    conn: aiohttp.TCPConnector = aiohttp.TCPConnector(limit=limit)
    sess: aiohttp.ClientSession = aiohttp.ClientSession(connector=conn)
    proxy_obj: Any = await func_5ysx6lkz()
    current_pid: Optional[int] = None

    async def func_i3c3guta(pid: int) -> ClientResponse:
        nonlocal current_pid
        async with sess.get(url, proxy=proxy_obj.url) as resp:
            current_pid = pid
            await asyncio.sleep(0.2)
            assert current_pid == pid
        return resp

    requests = [func_i3c3guta(pid) for pid in range(multi_conn_num)]
    responses = await asyncio.gather(*requests)
    assert len(responses) == multi_conn_num
    assert {resp.status for resp in responses} == {200}
    await sess.close()


def func_r832wwkq(monkeypatch: Any) -> None:
    def func_r6xug8zr(self: Any, rawsock: Any, protocol: Any, sslcontext: ssl.SSLContext, waiter: Optional[Any] = None, **kwargs: Any) -> Any:
        return self._make_socket_transport(rawsock, protocol, waiter, extra=kwargs.get('extra'), server=kwargs.get('server'))
    monkeypatch.setattr('asyncio.selector_events.BaseSelectorEventLoop._make_ssl_transport', func_r6xug8zr)


original_is_file = pathlib.Path.is_file


def func_jbohtdnp(self: pathlib.Path) -> bool:
    if self.name in ['_netrc', '.netrc'] and self.parent == self.home():
        return False
    else:
        return original_is_file(self)


async def func_4wild0hw(proxy_test_server: Any, mocker: MockerFixture) -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy_obj.url)})
    mocker.patch('pathlib.Path.is_file', func_jbohtdnp)
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy_obj.requests_list) == 1
    assert proxy_obj.request.method == 'GET'
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy_obj.request.headers


async def func_oumath28(proxy_test_server: Any, mocker: MockerFixture) -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    auth: aiohttp.BasicAuth = aiohttp.BasicAuth('user', 'pass')
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy_obj.url.with_user(auth.login).with_password(auth.password))})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy_obj.requests_list) == 1
    assert proxy_obj.request.method == 'GET'
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == '/path'
    assert proxy_obj.request.headers['Proxy-Authorization'] == auth.encode()


async def func_96dzxo4t(proxy_test_server: Any, tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    auth: aiohttp.BasicAuth = aiohttp.BasicAuth('user', 'pass')
    netrc_file: pathlib.Path = tmp_path / 'test_netrc'
    netrc_file_data: str = 'machine 127.0.0.1 login {} password {}'.format(auth.login, auth.password)
    with netrc_file.open('w') as f:
        f.write(netrc_file_data)
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy_obj.url), 'NETRC': str(netrc_file)})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy_obj.requests_list) == 1
    assert proxy_obj.request.method == 'GET'
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == '/path'
    assert proxy_obj.request.headers['Proxy-Authorization'] == auth.encode()


async def func_5kwxr7x2(proxy_test_server: Any, tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    auth: aiohttp.BasicAuth = aiohttp.BasicAuth('user', 'pass')
    netrc_file: pathlib.Path = tmp_path / 'test_netrc'
    netrc_file_data: str = 'machine 127.0.0.2 login {} password {}'.format(auth.login, auth.password)
    with netrc_file.open('w') as f:
        f.write(netrc_file_data)
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy_obj.url), 'NETRC': str(netrc_file)})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy_obj.requests_list) == 1
    assert proxy_obj.request.method == 'GET'
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy_obj.request.headers


async def func_3obn83o4(proxy_test_server: Any, tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    url: str = 'http://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    auth: aiohttp.BasicAuth = aiohttp.BasicAuth('user', 'pass')
    netrc_file: pathlib.Path = tmp_path / 'test_netrc'
    invalid_data: str = f'machine 127.0.0.1 {auth.login} pass {auth.password}'
    with netrc_file.open('w') as f:
        f.write(invalid_data)
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy_obj.url), 'NETRC': str(netrc_file)})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy_obj.requests_list) == 1
    assert proxy_obj.request.method == 'GET'
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy_obj.request.headers


@pytest.mark.xfail
async def func_6npheszz(proxy_test_server: Any, mocker: MockerFixture) -> None:
    url: str = 'https://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    mocker.patch.dict(os.environ, {'https_proxy': str(proxy_obj.url)})
    mock.patch('pathlib.Path.is_file', func_jbohtdnp)
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy_obj.requests_list) == 2
    assert proxy_obj.request.method == 'GET'
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy_obj.request.headers


@pytest.mark.xfail
async def func_6tax1ftm(proxy_test_server: Any, mocker: MockerFixture) -> None:
    url: str = 'https://aiohttp.io/path'
    proxy_obj: Any = await func_5ysx6lkz()
    auth: aiohttp.BasicAuth = aiohttp.BasicAuth('user', 'pass')
    mocker.patch.dict(os.environ, {'https_proxy': str(proxy_obj.url.with_user(auth.login).with_password(auth.password))})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy_obj.requests_list) == 2
    assert proxy_obj.request.method == 'GET'
    assert proxy_obj.request.host == 'aiohttp.io'
    assert proxy_obj.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy_obj.request.headers
    r2 = proxy_obj.requests_list[0]
    assert r2.method == 'CONNECT'
    assert r2.host == 'aiohttp.io'
    assert r2.path_qs == '/path'
    assert r2.headers['Proxy-Authorization'] == auth.encode()


async def func_byvrka4k() -> None:
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match='proxy_auth must be None or BasicAuth\\(\\) tuple'):
            async with session.get('http://python.org', proxy='http://proxy.example.com', proxy_auth=('user', 'pass')):
                pass
