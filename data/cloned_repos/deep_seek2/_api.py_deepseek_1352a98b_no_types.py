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
__all__ = ['delete', 'get', 'head', 'options', 'patch', 'post', 'put', 'request', 'stream']

def request(method, url, *, params: QueryParamTypes | None=None, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: typing.Any | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, follow_redirects: bool=False, verify: ssl.SSLContext | str | bool=True, trust_env: bool=True):
    with Client(cookies=cookies, proxy=proxy, verify=verify, timeout=timeout, trust_env=trust_env) as client:
        return client.request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, auth=auth, follow_redirects=follow_redirects)

@contextmanager
def stream(method, url, *, params: QueryParamTypes | None=None, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: typing.Any | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, follow_redirects: bool=False, verify: ssl.SSLContext | str | bool=True, trust_env: bool=True):
    with Client(cookies=cookies, proxy=proxy, verify=verify, timeout=timeout, trust_env=trust_env) as client:
        with client.stream(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, auth=auth, follow_redirects=follow_redirects) as response:
            yield response

def get(url, *, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, follow_redirects: bool=False, verify: ssl.SSLContext | str | bool=True, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    return request('GET', url, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def options(url, *, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, follow_redirects: bool=False, verify: ssl.SSLContext | str | bool=True, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    return request('OPTIONS', url, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def head(url, *, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, follow_redirects: bool=False, verify: ssl.SSLContext | str | bool=True, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    return request('HEAD', url, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def post(url, *, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: typing.Any | None=None, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, follow_redirects: bool=False, verify: ssl.SSLContext | str | bool=True, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    return request('POST', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def put(url, *, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: typing.Any | None=None, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, follow_redirects: bool=False, verify: ssl.SSLContext | str | bool=True, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    return request('PUT', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def patch(url, *, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: typing.Any | None=None, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, follow_redirects: bool=False, verify: ssl.SSLContext | str | bool=True, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    return request('PATCH', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def delete(url, *, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | None=None, proxy: ProxyTypes | None=None, follow_redirects: bool=False, timeout: TimeoutTypes=DEFAULT_TIMEOUT_CONFIG, verify: ssl.SSLContext | str | bool=True, trust_env: bool=True):
    return request('DELETE', url, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)