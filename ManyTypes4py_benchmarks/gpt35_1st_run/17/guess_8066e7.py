from requests.models import Response, PreparedRequest
from typing import Any, Dict, Optional

class GuessAuth(auth.AuthBase):
    def __init__(self, username: str, password: str) -> None:
        self.username: str = username
        self.password: str = password
        self.auth: Optional[auth.AuthBase] = None
        self.pos: Optional[int] = None

    def _handle_basic_auth_401(self, r: Response, kwargs: Dict[str, Any]) -> Response:
        ...

    def _handle_digest_auth_401(self, r: Response, kwargs: Dict[str, Any]) -> Response:
        ...

    def handle_401(self, r: Response, **kwargs: Any) -> Response:
        ...

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        ...

class GuessProxyAuth(GuessAuth):
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, proxy_username: Optional[str] = None, proxy_password: Optional[str] = None) -> None:
        ...

    def _handle_basic_auth_407(self, r: Response, kwargs: Dict[str, Any]) -> Response:
        ...

    def _handle_digest_auth_407(self, r: Response, kwargs: Dict[str, Any]) -> Response:
        ...

    def handle_407(self, r: Response, **kwargs: Any) -> Response:
        ...

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        ...
