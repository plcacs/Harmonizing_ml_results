from requests import auth, cookies, Request, Response
from . import _digest_auth_compat as auth_compat, http_proxy_digest

class GuessAuth(auth.AuthBase):
    def __init__(self, username: str, password: str) -> None:
        self.username: str = username
        self.password: str = password
        self.auth: auth.AuthBase = None
        self.pos: int = None

    def _handle_basic_auth_401(self, r: Response, kwargs: dict) -> Response:
        ...

    def _handle_digest_auth_401(self, r: Response, kwargs: dict) -> Response:
        ...

    def handle_401(self, r: Response, **kwargs) -> Response:
        ...

    def __call__(self, request: Request) -> Request:
        ...

class GuessProxyAuth(GuessAuth):
    def __init__(self, username: str = None, password: str = None, proxy_username: str = None, proxy_password: str = None) -> None:
        ...

    def _handle_basic_auth_407(self, r: Response, kwargs: dict) -> Response:
        ...

    def _handle_digest_auth_407(self, r: Response, kwargs: dict) -> Response:
        ...

    def handle_407(self, r: Response, **kwargs) -> Response:
        ...

    def __call__(self, request: Request) -> Request:
        ...
