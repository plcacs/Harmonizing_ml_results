from __future__ import annotations
import typing
from contextlib import contextmanager
from ._client import Client
from ._config import DEFAULT_TIMEOUT_CONFIG
from ._models import Response
from ._types import AuthTypes, CookieTypes, HeaderTypes, ProxyTypes, QueryParamTypes, RequestContent, RequestData, RequestFiles, TimeoutTypes
from ._urls import URL
if typing.TYPE_CHECKING:
    import ssl
__all__: typing.List[str] = ['delete', 'get', 'head', 'options', 'patch', 'post', 'put', 'request', 'stream']

def request(method: str, url: str, *, params: typing.Optional[QueryParamTypes] = None, content: typing.Optional[RequestContent] = None, data: typing.Optional[RequestData] = None, files: typing.Optional[RequestFiles] = None, json: typing.Optional[typing.Any] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, follow_redirects: bool = False, verify: bool = True, trust_env: bool = True) -> Response:
    ...

@contextmanager
def stream(method: str, url: str, *, params: typing.Optional[QueryParamTypes] = None, content: typing.Optional[RequestContent] = None, data: typing.Optional[RequestData] = None, files: typing.Optional[RequestFiles] = None, json: typing.Optional[typing.Any] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, follow_redirects: bool = False, verify: bool = True, trust_env: bool = True) -> typing.Iterator[Response]:
    ...

def get(url: str, *, params: typing.Optional[QueryParamTypes] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, follow_redirects: bool = False, verify: bool = True, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, trust_env: bool = True) -> Response:
    ...

def options(url: str, *, params: typing.Optional[QueryParamTypes] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, follow_redirects: bool = False, verify: bool = True, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, trust_env: bool = True) -> Response:
    ...

def head(url: str, *, params: typing.Optional[QueryParamTypes] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, follow_redirects: bool = False, verify: bool = True, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, trust_env: bool = True) -> Response:
    ...

def post(url: str, *, content: typing.Optional[RequestContent] = None, data: typing.Optional[RequestData] = None, files: typing.Optional[RequestFiles] = None, json: typing.Optional[typing.Any] = None, params: typing.Optional[QueryParamTypes] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, follow_redirects: bool = False, verify: bool = True, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, trust_env: bool = True) -> Response:
    ...

def put(url: str, *, content: typing.Optional[RequestContent] = None, data: typing.Optional[RequestData] = None, files: typing.Optional[RequestFiles] = None, json: typing.Optional[typing.Any] = None, params: typing.Optional[QueryParamTypes] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, follow_redirects: bool = False, verify: bool = True, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, trust_env: bool = True) -> Response:
    ...

def patch(url: str, *, content: typing.Optional[RequestContent] = None, data: typing.Optional[RequestData] = None, files: typing.Optional[RequestFiles] = None, json: typing.Optional[typing.Any] = None, params: typing.Optional[QueryParamTypes] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, follow_redirects: bool = False, verify: bool = True, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, trust_env: bool = True) -> Response:
    ...

def delete(url: str, *, params: typing.Optional[QueryParamTypes] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, follow_redirects: bool = False, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, verify: bool = True, trust_env: bool = True) -> Response:
    ...
