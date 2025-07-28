"""The module containing the code for GuessAuth."""
from typing import Any, Dict, Optional, Union
from requests import auth, Response, PreparedRequest
from requests import cookies
from . import _digest_auth_compat as auth_compat, http_proxy_digest

class GuessAuth(auth.AuthBase):
    """Guesses the auth type by the WWW-Authentication header."""

    def __init__(self, username: str, password: str) -> None:
        self.username: str = username
        self.password: str = password
        self.auth: Optional[auth.AuthBase] = None
        self.pos: Optional[int] = None

    def _handle_basic_auth_401(self, r: Response, kwargs: Dict[str, Any]) -> Response:
        if self.pos is not None:
            r.request.body.seek(self.pos)
        r.content
        r.raw.release_conn()
        prep: PreparedRequest = r.request.copy()
        if not hasattr(prep, '_cookies'):
            prep._cookies = cookies.RequestsCookieJar()
        cookies.extract_cookies_to_jar(prep._cookies, r.request, r.raw)
        prep.prepare_cookies(prep._cookies)
        self.auth = auth.HTTPBasicAuth(self.username, self.password)
        prep = self.auth(prep)
        _r: Response = r.connection.send(prep, **kwargs)
        _r.history.append(r)
        _r.request = prep
        return _r

    def _handle_digest_auth_401(self, r: Response, kwargs: Dict[str, Any]) -> Response:
        self.auth = auth_compat.HTTPDigestAuth(self.username, self.password)
        try:
            self.auth.init_per_thread_state()
        except AttributeError:
            pass
        if hasattr(self.auth, 'num_401_calls') and self.auth.num_401_calls is None:
            self.auth.num_401_calls = 1
        return self.auth.handle_401(r, **kwargs)

    def handle_401(self, r: Response, **kwargs: Any) -> Response:
        """Resends a request with auth headers, if needed."""
        www_authenticate: str = r.headers.get('www-authenticate', '').lower()
        if 'basic' in www_authenticate:
            return self._handle_basic_auth_401(r, kwargs)
        if 'digest' in www_authenticate:
            return self._handle_digest_auth_401(r, kwargs)
        return r

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        if self.auth is not None:
            return self.auth(request)
        try:
            self.pos = request.body.tell()
        except AttributeError:
            pass
        request.register_hook('response', self.handle_401)
        return request

class GuessProxyAuth(GuessAuth):
    """
    Guesses the auth type by WWW-Authentication and Proxy-Authentication
    headers
    """

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, 
                 proxy_username: Optional[str] = None, proxy_password: Optional[str] = None) -> None:
        super(GuessProxyAuth, self).__init__(username, password)
        self.proxy_username: Optional[str] = proxy_username
        self.proxy_password: Optional[str] = proxy_password
        self.proxy_auth: Optional[auth.AuthBase] = None

    def _handle_basic_auth_407(self, r: Response, kwargs: Dict[str, Any]) -> Response:
        if self.pos is not None:
            r.request.body.seek(self.pos)
        r.content
        r.raw.release_conn()
        prep: PreparedRequest = r.request.copy()
        if not hasattr(prep, '_cookies'):
            prep._cookies = cookies.RequestsCookieJar()
        cookies.extract_cookies_to_jar(prep._cookies, r.request, r.raw)
        prep.prepare_cookies(prep._cookies)
        self.proxy_auth = auth.HTTPProxyAuth(self.proxy_username, self.proxy_password)
        prep = self.proxy_auth(prep)
        _r: Response = r.connection.send(prep, **kwargs)
        _r.history.append(r)
        _r.request = prep
        return _r

    def _handle_digest_auth_407(self, r: Response, kwargs: Dict[str, Any]) -> Response:
        self.proxy_auth = http_proxy_digest.HTTPProxyDigestAuth(username=self.proxy_username, password=self.proxy_password)
        try:
            self.auth.init_per_thread_state()
        except AttributeError:
            pass
        return self.proxy_auth.handle_407(r, **kwargs)

    def handle_407(self, r: Response, **kwargs: Any) -> Response:
        proxy_authenticate: str = r.headers.get('Proxy-Authenticate', '').lower()
        if 'basic' in proxy_authenticate:
            return self._handle_basic_auth_407(r, kwargs)
        if 'digest' in proxy_authenticate:
            return self._handle_digest_auth_407(r, kwargs)
        return r

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        if self.proxy_auth is not None:
            request = self.proxy_auth(request)
        try:
            self.pos = request.body.tell()
        except AttributeError:
            pass
        request.register_hook('response', self.handle_407)
        return super(GuessProxyAuth, self).__call__(request)
