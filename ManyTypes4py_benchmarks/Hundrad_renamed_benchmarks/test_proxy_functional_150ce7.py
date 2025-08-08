import asyncio
import os
import pathlib
import ssl
import sys
from re import match as match_regex
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Iterator, Optional, TypedDict, Union
from unittest import mock
from uuid import uuid4
import proxy
import pytest
from pytest_mock import MockerFixture
from yarl import URL
import aiohttp
from aiohttp import ClientResponse, web
from aiohttp.client import _RequestOptions
from aiohttp.client_exceptions import ClientConnectionError
from aiohttp.pytest_plugin import AiohttpRawServer, AiohttpServer
ASYNCIO_SUPPORTS_TLS_IN_TLS = sys.version_info >= (3, 11)


class _ResponseArgs(TypedDict):
    pass


if sys.version_info >= (3, 11) and TYPE_CHECKING:
    from typing import Unpack

    async def func_7plw6tuz(method='GET', *, url, trust_env=False, **kwargs):
        ...
else:
    from typing import Any

    async def func_7plw6tuz(method='GET', *, url, trust_env=False, **kwargs):
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector, trust_env=
            trust_env) as client:
            async with client.request(method, url, **kwargs) as resp:
                return resp


@pytest.fixture
def func_e0k8yhts(tls_certificate_pem_path):
    """Return the URL of an instance of a running secure proxy.

    This fixture also spawns that instance and tears it down after the test.
    """
    proxypy_args = ['--threaded' if os.name == 'nt' else '--threadless',
        '--num-workers', '1', '--hostname', '127.0.0.1', '--port', '0',
        '--cert-file', tls_certificate_pem_path, '--key-file',
        tls_certificate_pem_path]
    with proxy.Proxy(input_args=proxypy_args) as proxy_instance:
        yield URL.build(scheme='https', host=str(proxy_instance.flags.
            hostname), port=proxy_instance.flags.port)


@pytest.fixture
def func_bic9bwv8():
    return str(uuid4())


@pytest.fixture(params=('http', 'https'))
def func_tlsl13ng(request):
    return request.param


@pytest.fixture
async def func_4u4w7iq7(aiohttp_server, ssl_ctx,
    web_server_endpoint_payload, web_server_endpoint_type):

    async def func_8dxfn7hv(request):
        return web.Response(text=web_server_endpoint_payload)
    app = web.Application()
    app.router.add_route('GET', '/', handler)
    if web_server_endpoint_type == 'https':
        server = await aiohttp_server(app, ssl=ssl_ctx)
    else:
        server = await aiohttp_server(app)
    return URL.build(scheme=web_server_endpoint_type, host=server.host,
        port=server.port)


@pytest.mark.skipif(not ASYNCIO_SUPPORTS_TLS_IN_TLS, reason=
    'asyncio on this python does not support TLS in TLS')
@pytest.mark.parametrize('web_server_endpoint_type', ('http', 'https'))
@pytest.mark.filterwarnings('ignore:.*ssl.OP_NO_SSL*')
@pytest.mark.usefixtures('loop')
async def func_pj05lpvm(client_ssl_ctx, secure_proxy_url,
    web_server_endpoint_url, web_server_endpoint_payload):
    """Ensure HTTP(S) sites are accessible through a secure proxy."""
    conn = aiohttp.TCPConnector()
    sess = aiohttp.ClientSession(connector=conn)
    async with sess.get(web_server_endpoint_url, proxy=secure_proxy_url,
        ssl=client_ssl_ctx) as response:
        assert response.status == 200
        assert await response.text() == web_server_endpoint_payload
    await sess.close()
    await conn.close()
    await asyncio.sleep(0.1)


@pytest.mark.parametrize('web_server_endpoint_type', ('https',))
@pytest.mark.usefixtures('loop')
@pytest.mark.skipif(ASYNCIO_SUPPORTS_TLS_IN_TLS, reason=
    'asyncio on this python supports TLS in TLS')
@pytest.mark.filterwarnings('ignore:.*ssl.OP_NO_SSL*')
async def func_d49rxnvl(client_ssl_ctx, secure_proxy_url,
    web_server_endpoint_type):
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
    with pytest.warns(RuntimeWarning, match=expected_warning_text
        ), pytest.raises(ClientConnectionError, match=expected_exception_reason
        ) as conn_err:
        async with sess.get(url, proxy=secure_proxy_url, ssl=client_ssl_ctx):
            pass
    assert isinstance(conn_err.value.__cause__, TypeError)
    assert match_regex(f'^{type_err!s}$', str(conn_err.value.__cause__))
    await sess.close()
    await conn.close()
    await asyncio.sleep(0.1)


@pytest.fixture
def func_5ysx6lkz(aiohttp_raw_server, loop, monkeypatch):
    _patch_ssl_transport(monkeypatch)
    default_response = _ResponseArgs(status=200, headers=None, body=None)
    proxy_mock = mock.Mock()

    async def func_wki57u2z(request):
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

    async def func_4j0jx9qy():
        proxy_mock.request = None
        proxy_mock.auth = None
        proxy_mock.requests_list = []
        server = await aiohttp_raw_server(proxy_handler)
        proxy_mock.server = server
        proxy_mock.url = server.make_url('/')
        return proxy_mock
    return proxy_server


async def func_ip8gkzx2(proxy_test_server):
    url = 'http://aiohttp.io/path?query=yes'
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    assert len(proxy.requests_list) == 1
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path?query=yes'


async def func_tamans5s(proxy_test_server):
    url = 'http://aiohttp.io:2561/space sheep?q=can:fly'
    raw_url = '/space%20sheep?q=can:fly'
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == raw_url


async def func_eh2mcqh3(proxy_test_server):
    url = 'http://éé.com/'
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    assert proxy.request.host == 'éé.com'
    assert proxy.request.path_qs == '/'


async def func_k0hlwd5t():
    url = 'http://aiohttp.io/path'
    proxy_url = 'http://localhost:2242/'
    with pytest.raises(aiohttp.ClientConnectorError):
        await func_7plw6tuz(url=url, proxy=proxy_url)


async def func_5r8uzbq0(proxy_test_server):
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    proxy.return_value = dict(status=502, headers={'Proxy-Agent': 'TestProxy'})
    resp = await func_7plw6tuz(url=url, proxy=proxy.url)
    assert resp.status == 502
    assert resp.headers['Proxy-Agent'] == 'TestProxy'


async def func_rcfw3uie(proxy_test_server):
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


async def func_59jqnhqn(proxy_test_server):
    url = 'http://aiohttp.io/path'
    auth = aiohttp.BasicAuth('юзер', 'пасс', 'utf-8')
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, auth=auth, proxy=proxy.url)
    assert 'Authorization' in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers


async def func_dthrebwa(proxy_test_server):
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


async def func_vgm1uc7v(proxy_test_server, loop):
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
async def func_rvn4xsd7(proxy_test_server, loop):
    url = 'http://aiohttp.io/path'
    conn = aiohttp.TCPConnector(force_close=True)
    sess = aiohttp.ClientSession(connector=conn)
    proxy = await func_5ysx6lkz()
    assert 0 == len(conn._acquired)

    async def func_i3c3guta():
        async with sess.get(url, proxy=proxy.url):
            assert 1 == len(conn._acquired)
    await func_i3c3guta()
    assert 0 == len(conn._acquired)
    await sess.close()


@pytest.mark.skip('we need to reconsider how we test this')
async def func_91sauqg5(proxy_test_server, loop):
    url = 'http://aiohttp.io/path'
    limit, multi_conn_num = 1, 5
    conn = aiohttp.TCPConnector(limit=limit)
    sess = aiohttp.ClientSession(connector=conn)
    proxy = await func_5ysx6lkz()
    current_pid = None

    async def func_i3c3guta(pid):
        nonlocal current_pid
        async with sess.get(url, proxy=proxy.url) as resp:
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
async def func_91746m52(proxy_test_server):
    proxy = await func_5ysx6lkz()
    url = 'https://www.google.com.ua/search?q=aiohttp proxy'
    await func_7plw6tuz(url=url, proxy=proxy.url)
    connect = proxy.requests_list[0]
    assert connect.method == 'CONNECT'
    assert connect.path == 'www.google.com.ua:443'
    assert connect.host == 'www.google.com.ua'
    assert proxy.request.host == 'www.google.com.ua'
    assert proxy.request.path_qs == '/search?q=aiohttp+proxy'


@pytest.mark.xfail
async def func_p4ffujhq(proxy_test_server):
    proxy = await func_5ysx6lkz()
    url = 'https://secure.aiohttp.io:2242/path'
    await func_7plw6tuz(url=url, proxy=proxy.url)
    connect = proxy.requests_list[0]
    assert connect.method == 'CONNECT'
    assert connect.path == 'secure.aiohttp.io:2242'
    assert connect.host == 'secure.aiohttp.io:2242'
    assert proxy.request.host == 'secure.aiohttp.io:2242'
    assert proxy.request.path_qs == '/path'


@pytest.mark.xfail
async def func_0l4yme68(proxy_test_server, loop):
    sess = aiohttp.ClientSession()
    proxy = await func_5ysx6lkz()
    proxy.return_value = {'status': 200, 'body': b'1' * 2 ** 20}
    url = 'https://www.google.com.ua/search?q=aiohttp proxy'
    async with sess.get(url, proxy=proxy.url) as resp:
        body = await resp.read()
    await sess.close()
    assert body == b'1' * 2 ** 20


@pytest.mark.xfail
async def func_f1pyl7qj(proxy_test_server):
    url = 'https://éé.com/'
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    connect = proxy.requests_list[0]
    assert connect.method == 'CONNECT'
    assert connect.path == 'xn--9caa.com:443'
    assert connect.host == 'xn--9caa.com'


async def func_13galeuq():
    url = 'https://secure.aiohttp.io/path'
    proxy_url = 'http://localhost:2242/'
    with pytest.raises(aiohttp.ClientConnectorError):
        await func_7plw6tuz(url=url, proxy=proxy_url)


async def func_x3tv0lvq(proxy_test_server):
    url = 'https://secure.aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    proxy.return_value = dict(status=502, headers={'Proxy-Agent': 'TestProxy'})
    with pytest.raises(aiohttp.ClientHttpProxyError):
        await func_7plw6tuz(url=url, proxy=proxy.url)
    assert len(proxy.requests_list) == 1
    assert proxy.request.method == 'CONNECT'


@pytest.mark.xfail
async def func_shr8sdeo(proxy_test_server):
    url = 'https://secure.aiohttp.io/path'
    auth = aiohttp.BasicAuth('user', 'pass')
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy=proxy.url)
    connect = proxy.requests_list[0]
    assert 'Authorization' not in connect.headers
    assert 'Proxy-Authorization' not in connect.headers
    assert 'Authorization' not in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, auth=auth, proxy=proxy.url)
    connect = proxy.requests_list[0]
    assert 'Authorization' not in connect.headers
    assert 'Proxy-Authorization' not in connect.headers
    assert 'Authorization' in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, proxy_auth=auth, proxy=proxy.url)
    connect = proxy.requests_list[0]
    assert 'Authorization' not in connect.headers
    assert 'Proxy-Authorization' in connect.headers
    assert 'Authorization' not in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers
    proxy = await func_5ysx6lkz()
    await func_7plw6tuz(url=url, auth=auth, proxy_auth=auth, proxy=proxy.url)
    connect = proxy.requests_list[0]
    assert 'Authorization' not in connect.headers
    assert 'Proxy-Authorization' in connect.headers
    assert 'Authorization' in proxy.request.headers
    assert 'Proxy-Authorization' not in proxy.request.headers


@pytest.mark.xfail
async def func_34nj69kb(proxy_test_server, loop):
    url = 'https://secure.aiohttp.io/path'
    conn = aiohttp.TCPConnector()
    sess = aiohttp.ClientSession(connector=conn)
    proxy = await func_5ysx6lkz()
    assert 0 == len(conn._acquired)

    async def func_i3c3guta():
        async with sess.get(url, proxy=proxy.url):
            assert 1 == len(conn._acquired)
    await func_i3c3guta()
    assert 0 == len(conn._acquired)
    await sess.close()


@pytest.mark.xfail
async def func_jui49i11(proxy_test_server, loop):
    url = 'https://secure.aiohttp.io/path'
    conn = aiohttp.TCPConnector(force_close=True)
    sess = aiohttp.ClientSession(connector=conn)
    proxy = await func_5ysx6lkz()
    assert 0 == len(conn._acquired)

    async def func_i3c3guta():
        async with sess.get(url, proxy=proxy.url):
            assert 1 == len(conn._acquired)
    await func_i3c3guta()
    assert 0 == len(conn._acquired)
    await sess.close()


@pytest.mark.xfail
async def func_yn66l674(proxy_test_server, loop):
    url = 'https://secure.aiohttp.io/path'
    limit, multi_conn_num = 1, 5
    conn = aiohttp.TCPConnector(limit=limit)
    sess = aiohttp.ClientSession(connector=conn)
    proxy = await func_5ysx6lkz()
    current_pid = None

    async def func_i3c3guta(pid):
        nonlocal current_pid
        async with sess.get(url, proxy=proxy.url) as resp:
            current_pid = pid
            await asyncio.sleep(0.2)
            assert current_pid == pid
        return resp
    requests = [func_i3c3guta(pid) for pid in range(multi_conn_num)]
    responses = await asyncio.gather(*requests)
    assert len(responses) == multi_conn_num
    assert {resp.status for resp in responses} == {200}
    await sess.close()


def func_r832wwkq(monkeypatch):

    def func_r6xug8zr(self, rawsock, protocol, sslcontext, waiter=None, **
        kwargs):
        return self._make_socket_transport(rawsock, protocol, waiter, extra
            =kwargs.get('extra'), server=kwargs.get('server'))
    monkeypatch.setattr(
        'asyncio.selector_events.BaseSelectorEventLoop._make_ssl_transport',
        _make_ssl_transport_dummy)


original_is_file = pathlib.Path.is_file


def func_jbohtdnp(self):
    if self.name in ['_netrc', '.netrc'] and self.parent == self.home():
        return False
    else:
        return original_is_file(self)


async def func_4wild0hw(proxy_test_server, mocker):
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy.url)})
    mocker.patch('pathlib.Path.is_file', mock_is_file)
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy.requests_list) == 1
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy.request.headers


async def func_oumath28(proxy_test_server, mocker):
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    auth = aiohttp.BasicAuth('user', 'pass')
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy.url.with_user(
        auth.login).with_password(auth.password))})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy.requests_list) == 1
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path'
    assert proxy.request.headers['Proxy-Authorization'] == auth.encode()


async def func_96dzxo4t(proxy_test_server, tmp_path, mocker):
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    auth = aiohttp.BasicAuth('user', 'pass')
    netrc_file = tmp_path / 'test_netrc'
    netrc_file_data = 'machine 127.0.0.1 login {} password {}'.format(auth.
        login, auth.password)
    with netrc_file.open('w') as f:
        f.write(netrc_file_data)
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy.url), 'NETRC':
        str(netrc_file)})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy.requests_list) == 1
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path'
    assert proxy.request.headers['Proxy-Authorization'] == auth.encode()


async def func_5kwxr7x2(proxy_test_server, tmp_path, mocker):
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    auth = aiohttp.BasicAuth('user', 'pass')
    netrc_file = tmp_path / 'test_netrc'
    netrc_file_data = 'machine 127.0.0.2 login {} password {}'.format(auth.
        login, auth.password)
    with netrc_file.open('w') as f:
        f.write(netrc_file_data)
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy.url), 'NETRC':
        str(netrc_file)})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy.requests_list) == 1
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy.request.headers


async def func_3obn83o4(proxy_test_server, tmp_path, mocker):
    url = 'http://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    auth = aiohttp.BasicAuth('user', 'pass')
    netrc_file = tmp_path / 'test_netrc'
    invalid_data = f'machine 127.0.0.1 {auth.login} pass {auth.password}'
    with netrc_file.open('w') as f:
        f.write(invalid_data)
    mocker.patch.dict(os.environ, {'http_proxy': str(proxy.url), 'NETRC':
        str(netrc_file)})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy.requests_list) == 1
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy.request.headers


@pytest.mark.xfail
async def func_6npheszz(proxy_test_server, mocker):
    url = 'https://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    mocker.patch.dict(os.environ, {'https_proxy': str(proxy.url)})
    mock.patch('pathlib.Path.is_file', mock_is_file)
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy.requests_list) == 2
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy.request.headers


@pytest.mark.xfail
async def func_6tax1ftm(proxy_test_server, mocker):
    url = 'https://aiohttp.io/path'
    proxy = await func_5ysx6lkz()
    auth = aiohttp.BasicAuth('user', 'pass')
    mocker.patch.dict(os.environ, {'https_proxy': str(proxy.url.with_user(
        auth.login).with_password(auth.password))})
    await func_7plw6tuz(url=url, trust_env=True)
    assert len(proxy.requests_list) == 2
    assert proxy.request.method == 'GET'
    assert proxy.request.host == 'aiohttp.io'
    assert proxy.request.path_qs == '/path'
    assert 'Proxy-Authorization' not in proxy.request.headers
    r2 = proxy.requests_list[0]
    assert r2.method == 'CONNECT'
    assert r2.host == 'aiohttp.io'
    assert r2.path_qs == '/path'
    assert r2.headers['Proxy-Authorization'] == auth.encode()


async def func_byvrka4k():
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match=
            'proxy_auth must be None or BasicAuth\\(\\) tuple'):
            async with session.get('http://python.org', proxy=
                'http://proxy.example.com', proxy_auth=('user', 'pass')):
                pass
