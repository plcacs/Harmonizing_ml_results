#!/usr/bin/env python3
"""
Integration tests for authentication.

Unit tests for auth classes also exist in tests/test_auth.py
"""
import hashlib
import netrc
import os
import sys
import threading
import typing
from urllib.request import parse_keqv_list

import anyio
import pytest
import httpx
from httpx import Response, Request
from ..common import FIXTURES_DIR


class App:
    """
    A mock app to test auth credentials.
    """

    def __init__(self, auth_header: str = '', status_code: int = 200) -> None:
        self.auth_header: str = auth_header
        self.status_code: int = status_code

    def __call__(self, request: Request) -> Response:
        headers: dict[str, str] = {'www-authenticate': self.auth_header} if self.auth_header else {}
        data: dict[str, typing.Optional[str]] = {'auth': request.headers.get('Authorization')}
        return httpx.Response(self.status_code, headers=headers, json=data)


class DigestApp:
    def __init__(
        self,
        algorithm: str = 'SHA-256',
        send_response_after_attempt: int = 1,
        qop: str = 'auth',
        regenerate_nonce: bool = True,
    ) -> None:
        self.algorithm: str = algorithm
        self.send_response_after_attempt: int = send_response_after_attempt
        self.qop: str = qop
        self._regenerate_nonce: bool = regenerate_nonce
        self._response_count: int = 0

    def __call__(self, request: Request) -> Response:
        if self._response_count < self.send_response_after_attempt:
            return self.challenge_send(request)
        data: dict[str, typing.Optional[str]] = {'auth': request.headers.get('Authorization')}
        return httpx.Response(200, json=data)

    def challenge_send(self, request: Request) -> Response:
        self._response_count += 1
        nonce: str = (
            hashlib.sha256(os.urandom(8)).hexdigest()
            if self._regenerate_nonce
            else 'ee96edced2a0b43e4869e96ebe27563f369c1205a049d06419bb51d8aeddf3d3'
        )
        challenge_data: dict[str, str] = {
            'nonce': nonce,
            'qop': self.qop,
            'opaque': 'ee6378f3ee14ebfd2fff54b70a91a7c9390518047f242ab2271380db0e14bda1',
            'algorithm': self.algorithm,
            'stale': 'FALSE',
        }
        challenge_str: str = ', '.join(
            ('{}="{}"'.format(key, value) for key, value in challenge_data.items() if value)
        )
        headers: dict[str, str] = {'www-authenticate': f'Digest realm="httpx@example.org", {challenge_str}'}
        return httpx.Response(401, headers=headers)


class RepeatAuth(httpx.Auth):
    """
    A mock authentication scheme that requires clients to send
    the request a fixed number of times, and then send a last request containing
    an aggregation of nonces that the server sent in 'WWW-Authenticate' headers
    of intermediate responses.
    """
    requires_request_body: bool = True

    def __init__(self, repeat: int) -> None:
        self.repeat: int = repeat

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        nonces: list[str] = []
        for index in range(self.repeat):
            request.headers['Authorization'] = f'Repeat {index}'
            response: Response = yield request
            nonces.append(response.headers['www-authenticate'])
        key: str = '.'.join(nonces)
        request.headers['Authorization'] = f'Repeat {key}'
        yield request


class ResponseBodyAuth(httpx.Auth):
    """
    A mock authentication scheme that requires clients to send an 'Authorization'
    header, then send back the contents of the response in the 'Authorization'
    header.
    """
    requires_response_body: bool = True

    def __init__(self, token: str) -> None:
        self.token: str = token

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        request.headers['Authorization'] = self.token
        response: Response = yield request
        data: str = response.text
        request.headers['Authorization'] = data
        yield request


class SyncOrAsyncAuth(httpx.Auth):
    """
    A mock authentication scheme that uses a different implementation for the
    sync and async cases.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._async_lock: anyio.Lock = anyio.Lock()

    def sync_auth_flow(self, request: Request) -> typing.Generator[Request, None, None]:
        with self._lock:
            request.headers['Authorization'] = 'sync-auth'
        yield request

    async def async_auth_flow(self, request: Request) -> typing.AsyncGenerator[Request, None]:
        async with self._async_lock:
            request.headers['Authorization'] = 'async-auth'
        yield request


@pytest.mark.anyio
async def test_basic_auth() -> None:
    url: str = 'https://example.org/'
    auth: tuple[str, str] = ('user', 'password123')
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}


@pytest.mark.anyio
async def test_basic_auth_with_stream() -> None:
    """
    See: https://github.com/encode/httpx/pull/1312
    """
    url: str = 'https://example.org/'
    auth: tuple[str, str] = ('user', 'password123')
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app), auth=auth) as client:
        async with client.stream('GET', url) as response:
            await response.aread()
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}


@pytest.mark.anyio
async def test_basic_auth_in_url() -> None:
    url: str = 'https://user:password123@example.org/'
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}


@pytest.mark.anyio
async def test_basic_auth_on_session() -> None:
    url: str = 'https://example.org/'
    auth: tuple[str, str] = ('user', 'password123')
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app), auth=auth) as client:
        response: httpx.Response = await client.get(url)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}


@pytest.mark.anyio
async def test_custom_auth() -> None:
    url: str = 'https://example.org/'
    app: App = App()

    def auth(request: Request) -> Request:
        request.headers['Authorization'] = 'Token 123'
        return request

    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Token 123'}


def test_netrc_auth_credentials_exist() -> None:
    """
    When netrc auth is being used and a request is made to a host that is
    in the netrc file, then the relevant credentials should be applied.
    """
    netrc_file: str = str(FIXTURES_DIR / '.netrc')
    url: str = 'http://netrcexample.org'
    app: App = App()
    auth: httpx.NetRCAuth = httpx.NetRCAuth(netrc_file)
    with httpx.Client(transport=httpx.MockTransport(app), auth=auth) as client:
        response: httpx.Response = client.get(url)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic ZXhhbXBsZS11c2VybmFtZTpleGFtcGxlLXBhc3N3b3Jk'}


def test_netrc_auth_credentials_do_not_exist() -> None:
    """
    When netrc auth is being used and a request is made to a host that is
    not in the netrc file, then no credentials should be applied.
    """
    netrc_file: str = str(FIXTURES_DIR / '.netrc')
    url: str = 'http://example.org'
    app: App = App()
    auth: httpx.NetRCAuth = httpx.NetRCAuth(netrc_file)
    with httpx.Client(transport=httpx.MockTransport(app), auth=auth) as client:
        response: httpx.Response = client.get(url)
    assert response.status_code == 200
    assert response.json() == {'auth': None}


@pytest.mark.skipif(sys.version_info >= (3, 11), reason='netrc files without a password are valid from Python >= 3.11')
def test_netrc_auth_nopassword_parse_error() -> None:
    """
    Python has different netrc parsing behaviours with different versions.
    For Python < 3.11 a netrc file with no password is invalid. In this case
    we want to allow the parse error to be raised.
    """
    netrc_file: str = str(FIXTURES_DIR / '.netrc-nopassword')
    with pytest.raises(netrc.NetrcParseError):
        httpx.NetRCAuth(netrc_file)


@pytest.mark.anyio
async def test_auth_disable_per_request() -> None:
    url: str = 'https://example.org/'
    auth: tuple[str, str] = ('user', 'password123')
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app), auth=auth) as client:
        response: httpx.Response = await client.get(url, auth=None)
    assert response.status_code == 200
    assert response.json() == {'auth': None}


def test_auth_hidden_url() -> None:
    url: str = 'http://example-username:example-password@example.org/'
    expected: str = "URL('http://example-username:[secure]@example.org/')"
    assert url == httpx.URL(url)
    assert expected == repr(httpx.URL(url))


@pytest.mark.anyio
async def test_auth_hidden_header() -> None:
    url: str = 'https://example.org/'
    auth: tuple[str, str] = ('example-username', 'example-password')
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert "'authorization': '[secure]'" in str(response.request.headers)


@pytest.mark.anyio
async def test_auth_property() -> None:
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        assert client.auth is None
        client.auth = ('user', 'password123')
        assert isinstance(client.auth, httpx.BasicAuth)
        url: str = 'https://example.org/'
        response: httpx.Response = await client.get(url)
        assert response.status_code == 200
        assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}


@pytest.mark.anyio
async def test_auth_invalid_type() -> None:
    app: App = App()
    with pytest.raises(TypeError):
        httpx.AsyncClient(transport=httpx.MockTransport(app), auth='not a tuple, not a callable')
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(TypeError):
            await client.get(auth='not a tuple, not a callable')
        with pytest.raises(TypeError):
            client.auth = 'not a tuple, not a callable'


@pytest.mark.anyio
async def test_digest_auth_returns_no_auth_if_no_digest_header_in_response() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': None}
    assert len(response.history) == 0


def test_digest_auth_returns_no_auth_if_alternate_auth_scheme() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    auth_header: str = 'Token ...'
    app: App = App(auth_header=auth_header, status_code=401)
    client: httpx.Client = httpx.Client(transport=httpx.MockTransport(app))
    response: httpx.Response = client.get(url, auth=auth)
    assert response.status_code == 401
    assert response.json() == {'auth': None}
    assert len(response.history) == 0


@pytest.mark.anyio
async def test_digest_auth_200_response_including_digest_auth_header() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    auth_header: str = 'Digest realm="realm@host.com",qop="auth",nonce="abc",opaque="xyz"'
    app: App = App(auth_header=auth_header, status_code=200)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': None}
    assert len(response.history) == 0


@pytest.mark.anyio
async def test_digest_auth_401_response_without_digest_auth_header() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: App = App(auth_header='', status_code=401)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 401
    assert response.json() == {'auth': None}
    assert len(response.history) == 0


@pytest.mark.parametrize(
    'algorithm,expected_hash_length,expected_response_length',
    [
        ('MD5', 64, 32),
        ('MD5-SESS', 64, 32),
        ('SHA', 64, 40),
        ('SHA-SESS', 64, 40),
        ('SHA-256', 64, 64),
        ('SHA-256-SESS', 64, 64),
        ('SHA-512', 64, 128),
        ('SHA-512-SESS', 64, 128),
    ],
)
@pytest.mark.anyio
async def test_digest_auth(
    algorithm: str, expected_hash_length: int, expected_response_length: int
) -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp(algorithm=algorithm)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert len(response.history) == 1
    authorization: str = typing.cast(typing.Dict[str, typing.Any], response.json())['auth']
    scheme, _, fields = authorization.partition(' ')
    assert scheme == 'Digest'
    response_fields: list[str] = [field.strip() for field in fields.split(',')]
    digest_data: dict[str, str] = dict((field.split('=') for field in response_fields))
    assert digest_data['username'] == '"user"'
    assert digest_data['realm'] == '"httpx@example.org"'
    assert 'nonce' in digest_data
    assert digest_data['uri'] == '"/"'
    assert len(digest_data['response']) == expected_response_length + 2
    assert len(digest_data['opaque']) == expected_hash_length + 2
    assert digest_data['algorithm'] == algorithm
    assert digest_data['qop'] == 'auth'
    assert digest_data['nc'] == '00000001'
    assert len(digest_data['cnonce']) == 16 + 2


@pytest.mark.anyio
async def test_digest_auth_no_specified_qop() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp(qop='')
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert len(response.history) == 1
    authorization: str = typing.cast(typing.Dict[str, typing.Any], response.json())['auth']
    scheme, _, fields = authorization.partition(' ')
    assert scheme == 'Digest'
    response_fields: list[str] = [field.strip() for field in fields.split(',')]
    digest_data: dict[str, str] = dict((field.split('=') for field in response_fields))
    assert 'qop' not in digest_data
    assert 'nc' not in digest_data
    assert 'cnonce' not in digest_data
    assert digest_data['username'] == '"user"'
    assert digest_data['realm'] == '"httpx@example.org"'
    assert len(digest_data['nonce']) == 64 + 2
    assert digest_data['uri'] == '"/"'
    assert len(digest_data['response']) == 64 + 2
    assert len(digest_data['opaque']) == 64 + 2
    assert digest_data['algorithm'] == 'SHA-256'


@pytest.mark.parametrize('qop', ('auth, auth-int', 'auth,auth-int', 'unknown,auth'))
@pytest.mark.anyio
async def test_digest_auth_qop_including_spaces_and_auth_returns_auth(qop: str) -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp(qop=qop)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert len(response.history) == 1


@pytest.mark.anyio
async def test_digest_auth_qop_auth_int_not_implemented() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp(qop='auth-int')
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(NotImplementedError):
            await client.get(url, auth=auth)


@pytest.mark.anyio
async def test_digest_auth_qop_must_be_auth_or_auth_int() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp(qop='not-auth')
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(httpx.ProtocolError):
            await client.get(url, auth=auth)


@pytest.mark.anyio
async def test_digest_auth_incorrect_credentials() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp(send_response_after_attempt=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 401
    assert len(response.history) == 1


@pytest.mark.anyio
async def test_digest_auth_reuses_challenge() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response_1: httpx.Response = await client.get(url, auth=auth)
        response_2: httpx.Response = await client.get(url, auth=auth)
        assert response_1.status_code == 200
        assert response_2.status_code == 200
        assert len(response_1.history) == 1
        assert len(response_2.history) == 0


@pytest.mark.anyio
async def test_digest_auth_resets_nonce_count_after_401() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response_1: httpx.Response = await client.get(url, auth=auth)
        assert response_1.status_code == 200
        assert len(response_1.history) == 1
        auth_header_parts = response_1.request.headers['Authorization'].split(', ')
        first_nonce: str = parse_keqv_list(auth_header_parts)['nonce']
        first_nc: str = parse_keqv_list(auth_header_parts)['nc']
        app.send_response_after_attempt = 2
        response_2: httpx.Response = await client.get(url, auth=auth)
        assert response_2.status_code == 200
        assert len(response_2.history) == 1
        auth_header_parts_2 = response_2.request.headers['Authorization'].split(', ')
        second_nonce: str = parse_keqv_list(auth_header_parts_2)['nonce']
        second_nc: str = parse_keqv_list(auth_header_parts_2)['nc']
    assert first_nonce != second_nonce
    assert first_nc == second_nc


@pytest.mark.parametrize('auth_header', ['Digest realm="httpx@example.org", qop="auth"', 'Digest realm="httpx@example.org", qop="auth,au'])
@pytest.mark.anyio
async def test_async_digest_auth_raises_protocol_error_on_malformed_header(auth_header: str) -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: App = App(auth_header=auth_header, status_code=401)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(httpx.ProtocolError):
            await client.get(url, auth=auth)


@pytest.mark.parametrize('auth_header', ['Digest realm="httpx@example.org", qop="auth"', 'Digest realm="httpx@example.org", qop="auth,au'])
def test_sync_digest_auth_raises_protocol_error_on_malformed_header(auth_header: str) -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: App = App(auth_header=auth_header, status_code=401)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(httpx.ProtocolError):
            client.get(url, auth=auth)


@pytest.mark.anyio
async def test_async_auth_history() -> None:
    """
    Test that intermediate requests sent as part of an authentication flow
    are recorded in the response history.
    """
    url: str = 'https://example.org/'
    auth: RepeatAuth = RepeatAuth(repeat=2)
    app: App = App(auth_header='abc')
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Repeat abc.abc'}
    assert len(response.history) == 2
    resp1, resp2 = response.history
    assert resp1.json() == {'auth': 'Repeat 0'}
    assert resp2.json() == {'auth': 'Repeat 1'}
    assert len(resp2.history) == 1
    assert resp2.history == [resp1]
    assert len(resp1.history) == 0


def test_sync_auth_history() -> None:
    """
    Test that intermediate requests sent as part of an authentication flow
    are recorded in the response history.
    """
    url: str = 'https://example.org/'
    auth: RepeatAuth = RepeatAuth(repeat=2)
    app: App = App(auth_header='abc')
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Repeat abc.abc'}
    assert len(response.history) == 2
    resp1, resp2 = response.history
    assert resp1.json() == {'auth': 'Repeat 0'}
    assert resp2.json() == {'auth': 'Repeat 1'}
    assert len(resp2.history) == 1
    assert resp2.history == [resp1]
    assert len(resp1.history) == 0


class ConsumeBodyTransport(httpx.MockTransport):
    async def handle_async_request(self, request: Request) -> Response:
        assert isinstance(request.stream, httpx.AsyncByteStream)
        async for _ in request.stream:
            pass
        return self.handler(request)


@pytest.mark.anyio
async def test_digest_auth_unavailable_streaming_body() -> None:
    url: str = 'https://example.org/'
    auth: httpx.DigestAuth = httpx.DigestAuth(username='user', password='password123')
    app: DigestApp = DigestApp()

    async def streaming_body() -> typing.AsyncGenerator[bytes, None]:
        yield b'Example request body'

    async with httpx.AsyncClient(transport=ConsumeBodyTransport(app)) as client:
        with pytest.raises(httpx.StreamConsumed):
            await client.post(url, content=streaming_body(), auth=auth)


@pytest.mark.anyio
async def test_async_auth_reads_response_body() -> None:
    """
    Test that we can read the response body in an auth flow if `requires_response_body`
    is set.
    """
    url: str = 'https://example.org/'
    auth: ResponseBodyAuth = ResponseBodyAuth('xyz')
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': '{"auth":"xyz"}'}


def test_sync_auth_reads_response_body() -> None:
    """
    Test that we can read the response body in an auth flow if `requires_response_body`
    is set.
    """
    url: str = 'https://example.org/'
    auth: ResponseBodyAuth = ResponseBodyAuth('xyz')
    app: App = App()
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': '{"auth":"xyz"}'}


@pytest.mark.anyio
async def test_async_auth() -> None:
    """
    Test that we can use an auth implementation specific to the async case, to
    support cases that require performing I/O or using concurrency primitives (such
    as checking a disk-based cache or fetching a token from a remote auth server).
    """
    url: str = 'https://example.org/'
    auth: SyncOrAsyncAuth = SyncOrAsyncAuth()
    app: App = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': 'async-auth'}


def test_sync_auth() -> None:
    """
    Test that we can use an auth implementation specific to the sync case.
    """
    url: str = 'https://example.org/'
    auth: SyncOrAsyncAuth = SyncOrAsyncAuth()
    app: App = App()
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response: httpx.Response = client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': 'sync-auth'}