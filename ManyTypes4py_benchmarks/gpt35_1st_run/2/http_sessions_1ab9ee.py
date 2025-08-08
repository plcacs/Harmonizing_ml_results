from typing import Any, Dict, List, Optional, Tuple, Union

class SessionRedirectMixin:
    def get_redirect_target(self, response: Any) -> Optional[str]:
    def resolve_redirects(self, response: Any, request: Any, stream: bool = False, timeout: Optional[float] = None, verify: bool = True, cert: Optional[Union[str, Tuple[str, str]]] = None, proxies: Optional[Dict[str, str]] = None, yield_requests: bool = False, **adapter_kwargs: Any) -> Any:
    def rebuild_auth(self, prepared_request: Any, response: Any) -> None:
    def rebuild_proxies(self, prepared_request: Any, proxies: Optional[Dict[str, str]]) -> Dict[str, str]:
    def rebuild_method(self, prepared_request: Any, response: Any) -> bool:

class HTTPSession(SessionRedirectMixin):
    def __init__(self) -> None:
    def prepare_request(self, request: Any) -> Any:
    def request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, cookies: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None, auth: Optional[Union[Tuple[str, str], Any]] = None, timeout: Optional[Union[float, Tuple[float, float]]] = None, allow_redirects: bool = True, proxies: Optional[Dict[str, str]] = None, hooks: Optional[Dict[str, List[Any]]] = None, stream: Optional[bool] = None, verify: Optional[Union[bool, str]] = None, cert: Optional[Union[str, Tuple[str, str]]] = None, json: Optional[Dict[str, Any]] = None) -> Any:
    def get(self, url: str, **kwargs: Any) -> Any:
    def head(self, url: str, **kwargs: Any) -> Any:
    def send(self, request: Any, **kwargs: Any) -> Any:
    def merge_environment_settings(self, url: str, proxies: Optional[Dict[str, str]], stream: Optional[bool], verify: Optional[Union[bool, str]], cert: Optional[Union[str, Tuple[str, str]]]) -> Dict[str, Any]:
    def get_adapter(self, url: str) -> Any:
    def close(self) -> None:
    def mount(self, prefix: str, adapter: Any) -> None:

class AsyncHTTPSession(HTTPSession):
    def __init__(self, backend: Any = None) -> None:
    async def get(self, url: str, **kwargs: Any) -> Any:
    async def head(self, url: str, **kwargs: Any) -> Any:
    async def request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, cookies: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None, auth: Optional[Union[Tuple[str, str], Any]] = None, timeout: Optional[Union[float, Tuple[float, float]]] = None, allow_redirects: bool = True, proxies: Optional[Dict[str, str]] = None, hooks: Optional[Dict[str, List[Any]]] = None, stream: Optional[bool] = None, verify: Optional[Union[bool, str]] = None, cert: Optional[Union[str, Tuple[str, str]]] = None, json: Optional[Dict[str, Any]] = None) -> Any:
    async def send(self, request: Any, **kwargs: Any) -> Any:
