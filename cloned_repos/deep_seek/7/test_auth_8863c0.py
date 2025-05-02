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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import anyio
import pytest
import httpx
from ..common import FIXTURES_DIR

class App:
    """
    A mock app to test auth credentials.
    """

    def __init__(self, auth_header: str = '', status_code: int = 200) -> None:
        self.auth_header = auth_header
        self.status_code = status_code

    def __call__(self, request: httpx.Request) -> httpx.Response:
        headers = {'www-authenticate': self.auth_header} if self.auth_header else {}
        data = {'auth': request.headers.get('Authorization')}
        return httpx.Response(self.status_code, headers=headers, json=data)

class DigestApp:
    def __init__(
        self,
        algorithm: str = 'SHA-256',
        send_response_after_attempt: int = 1,
        qop: str = 'auth',
        regenerate_nonce: bool = True
    ) -> None:
        self.algorithm = algorithm
        self.send_response_after_attempt = send_response_after_attempt
        self.qop = qop
        self._regenerate_nonce = regenerate_nonce
        self._response_count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self._response_count < self.send_response_after_attempt:
            return self.challenge_send(request)
        data = {'auth': request.headers.get('Authorization')}
        return httpx.Response(200, json=data)

    def challenge_send(self, request: httpx.Request) -> httpx.Response:
        self._response_count += 1
        nonce = hashlib.sha256(os.urandom(8)).hexdigest() if self._regenerate_nonce else 'ee96edced2a0b43e4869e96ebe27563f369c1205a049d06419bb51d8aeddf3d3'
        challenge_data = {
            'nonce': nonce,
            'qop': self.qop,
            'opaque': 'ee6378f3ee14ebfd2fff54b70a91a7c9390518047f242ab2271380db0e14bda1',
            'algorithm': self.algorithm,
            'stale': 'FALSE'
        }
        challenge_str = ', '.join(('{}="{}"'.format(key, value) for key, value in challenge_data.items() if value))
        headers = {'www-authenticate': f'Digest realm="httpx@example.org", {challenge_str}'}
        return httpx.Response(401, headers=headers)

class RepeatAuth(httpx.Auth):
    """
    A mock authentication scheme that requires clients to send
    the request a fixed number of times, and then send a last request containing
    an aggregation of nonces that the server sent in 'WWW-Authenticate' headers
    of intermediate responses.
    """
    requires_request_body = True

    def __init__(self, repeat: int) -> None:
        self.repeat = repeat

    def auth_flow(self, request: httpx.Request) -> Iterator[httpx.Request]:
        nonces: List[str] = []
        for index in range(self.repeat):
            request.headers['Authorization'] = f'Repeat {index}'
            response = yield request
            nonces.append(response.headers['www-authenticate'])
        key = '.'.join(nonces)
        request.headers['Authorization'] = f'Repeat {key}'
        yield request

class ResponseBodyAuth(httpx.Auth):
    """
    A mock authentication scheme that requires clients to send an 'Authorization'
    header, then send back the contents of the response in the 'Authorization'
    header.
    """
    requires_response_body = True

    def __init__(self, token: str) -> None:
        self.token = token

    def auth_flow(self, request: httpx.Request) -> Iterator[httpx.Request]:
        request.headers['Authorization'] = self.token
        response = yield request
        data = response.text
        request.headers['Authorization'] = data
        yield request

class SyncOrAsyncAuth(httpx.Auth):
    """
    A mock authentication scheme that uses a different implementation for the
    sync and async cases.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._async_lock = anyio.Lock()

    def sync_auth_flow(self, request: httpx.Request) -> Iterator[httpx.Request]:
        with self._lock:
            request.headers['Authorization'] = 'sync-auth'
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> typing.AsyncIterator[httpx.Request]:
        async with self._async_lock:
            request.headers['Authorization'] = 'async-auth'
        yield request

@pytest.mark.anyio
async def test_basic_auth() -> None:
    url = 'https://example.org/'
    auth: Tuple[str, str] = ('user', 'password123')
    app = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}

@pytest.mark.anyio
async def test_basic_auth_with_stream() -> None:
    """
    See: https://github.com/encode/httpx/pull/1312
    """
    url = 'https://example.org/'
    auth: Tuple[str, str] = ('user', 'password123')
    app = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app), auth=auth) as client:
        async with client.stream('GET', url) as response:
            await response.aread()
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}

@pytest.mark.anyio
async def test_basic_auth_in_url() -> None:
    url = 'https://user:password123@example.org/'
    app = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}

@pytest.mark.anyio
async def test_basic_auth_on_session() -> None:
    url = 'https://example.org/'
    auth: Tuple[str, str] = ('user', 'password123')
    app = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app), auth=auth) as client:
        response = await client.get(url)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}

@pytest.mark.anyio
async def test_custom_auth() -> None:
    url = 'https://example.org/'
    app = App()

    def auth(request: httpx.Request) -> httpx.Request:
        request.headers['Authorization'] = 'Token 123'
        return request
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Token 123'}

def test_netrc_auth_credentials_exist() -> None:
    """
    When netrc auth is being used and a request is made to a host that is
    in the netrc file, then the relevant credentials should be applied.
    """
    netrc_file = str(FIXTURES_DIR / '.netrc')
    url = 'http://netrcexample.org'
    app = App()
    auth = httpx.NetRCAuth(netrc_file)
    with httpx.Client(transport=httpx.MockTransport(app), auth=auth) as client:
        response = client.get(url)
    assert response.status_code == 200
    assert response.json() == {'auth': 'Basic ZXhhbXBsZS11c2VybmFtZTpleGFtcGxlLXBhc3N3b3Jk'}

def test_netrc_auth_credentials_do_not_exist() -> None:
    """
    When netrc auth is being used and a request is made to a host that is
    not in the netrc file, then no credentials should be applied.
    """
    netrc_file = str(FIXTURES_DIR / '.netrc')
    url = 'http://example.org'
    app = App()
    auth = httpx.NetRCAuth(netrc_file)
    with httpx.Client(transport=httpx.MockTransport(app), auth=auth) as client:
        response = client.get(url)
    assert response.status_code == 200
    assert response.json() == {'auth': None}

@pytest.mark.skipif(sys.version_info >= (3, 11), reason='netrc files without a password are valid from Python >= 3.11')
def test_netrc_auth_nopassword_parse_error() -> None:
    """
    Python has different netrc parsing behaviours with different versions.
    For Python < 3.11 a netrc file with no password is invalid. In this case
    we want to allow the parse error to be raised.
    """
    netrc_file = str(FIXTURES_DIR / '.netrc-nopassword')
    with pytest.raises(netrc.NetrcParseError):
        httpx.NetRCAuth(netrc_file)

@pytest.mark.anyio
async def test_auth_disable_per_request() -> None:
    url = 'https://example.org/'
    auth: Tuple[str, str] = ('user', 'password123')
    app = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app), auth=auth) as client:
        response = await client.get(url, auth=None)
    assert response.status_code == 200
    assert response.json() == {'auth': None}

def test_auth_hidden_url() -> None:
    url = 'http://example-username:example-password@example.org/'
    expected = "URL('http://example-username:[secure]@example.org/')"
    assert url == httpx.URL(url)
    assert expected == repr(httpx.URL(url))

@pytest.mark.anyio
async def test_auth_hidden_header() -> None:
    url = 'https://example.org/'
    auth: Tuple[str, str] = ('example-username', 'example-password')
    app = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url, auth=auth)
    assert "'authorization': '[secure]'" in str(response.request.headers)

@pytest.mark.anyio
async def test_auth_property() -> None:
    app = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        assert client.auth is None
        client.auth = ('user', 'password123')
        assert isinstance(client.auth, httpx.BasicAuth)
        url = 'https://example.org/'
        response = await client.get(url)
        assert response.status_code == 200
        assert response.json() == {'auth': 'Basic dXNlcjpwYXNzd29yZDEyMw=='}

@pytest.mark.anyio
async def test_auth_invalid_type() -> None:
    app = App()
    with pytest.raises(TypeError):
        client = httpx.AsyncClient(transport=httpx.MockTransport(app), auth='not a tuple, not a callable')
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(TypeError):
            await client.get(auth='not a tuple, not a callable')
        with pytest.raises(TypeError):
            client.auth = 'not a tuple, not a callable'

@pytest.mark.anyio
async def test_digest_auth_returns_no_auth_if_no_digest_header_in_response() -> None:
    url = 'https://example.org/'
    auth = httpx.DigestAuth(username='user', password='password123')
    app = App()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': None}
    assert len(response.history) == 0

def test_digest_auth_returns_no_auth_if_alternate_auth_scheme() -> None:
    url = 'https://example.org/'
    auth = httpx.DigestAuth(username='user', password='password123')
    auth_header = 'Token ...'
    app = App(auth_header=auth_header, status_code=401)
    client = httpx.Client(transport=httpx.MockTransport(app))
    response = client.get(url, auth=auth)
    assert response.status_code == 401
    assert response.json() == {'auth': None}
    assert len(response.history) == 0

@pytest.mark.anyio
async def test_digest_auth_200_response_including_digest_auth_header() -> None:
    url = 'https://example.org/'
    auth = httpx.DigestAuth(username='user', password='password123')
    auth_header = 'Digest realm="realm@host.com",qop="auth",nonce="abc",opaque="xyz"'
    app = App(auth_header=auth_header, status_code=200)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert response.json() == {'auth': None}
    assert len(response.history) == 0

@pytest.mark.anyio
async def test_digest_auth_401_response_without_digest_auth_header() -> None:
    url = 'https://example.org/'
    auth = httpx.DigestAuth(username='user', password='password123')
    app = App(auth_header='', status_code=401)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url, auth=auth)
    assert response.status_code == 401
    assert response.json() == {'auth': None}
    assert len(response.history) == 0

@pytest.mark.parametrize('algorithm,expected_hash_length,expected_response_length', [
    ('MD5', 64, 32),
    ('MD5-SESS', 64, 32),
    ('SHA', 64, 40),
    ('SHA-SESS', 64, 40),
    ('SHA-256', 64, 64),
    ('SHA-256-SESS', 64, 64),
    ('SHA-512', 64, 128),
    ('SHA-512-SESS', 64, 128)
])
@pytest.mark.anyio
async def test_digest_auth(
    algorithm: str,
    expected_hash_length: int,
    expected_response_length: int
) -> None:
    url = 'https://example.org/'
    auth = httpx.DigestAuth(username='user', password='password123')
    app = DigestApp(algorithm=algorithm)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert len(response.history) == 1
    authorization = cast(Dict[str, Any], response.json())['auth']
    scheme, _, fields = authorization.partition(' ')
    assert scheme == 'Digest'
    response_fields = [field.strip() for field in fields.split(',')]
    digest_data = dict((field.split('=') for field in response_fields))
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
    url = 'https://example.org/'
    auth = httpx.DigestAuth(username='user', password='password123')
    app = DigestApp(qop='')
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get(url, auth=auth)
    assert response.status_code == 200
    assert len(response.history) == 1
    authorization = cast(Dict[str, Any], response.json())['auth']
    scheme, _, fields = authorization.partition(' ')
    assert scheme == 'Digest'
    response_fields = [field.strip() for field in fields.split(',')]
    digest_data = dict((field.split('=') for field in response_fields))
    assert 'qop' not in digest_data
    assert 'nc' not in digest_data
    assert 'cnonce' not in digest_data
    assert digest_data['username'] == '"user"'
    assert digest_data['realm'] == '"httpx@example.org"'
    assert len(digest_data['nonce']) == 64 + 2
    assert digest_data['uri'] == '"/"'
    assert len(digest_data['response']) == 64 + 2
    assert len(digest_data['opaque']) ==