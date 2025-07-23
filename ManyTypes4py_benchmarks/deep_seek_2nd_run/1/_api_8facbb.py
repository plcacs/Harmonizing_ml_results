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

def request(
    method: str,
    url: typing.Union[str, URL],
    *,
    params: typing.Optional[QueryParamTypes] = None,
    content: typing.Optional[RequestContent] = None,
    data: typing.Optional[RequestData] = None,
    files: typing.Optional[RequestFiles] = None,
    json: typing.Optional[typing.Any] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    follow_redirects: bool = False,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    trust_env: bool = True
) -> Response:
    with Client(cookies=cookies, proxy=proxy, verify=verify, timeout=timeout, trust_env=trust_env) as client:
        return client.request(
            method=method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            auth=auth,
            follow_redirects=follow_redirects
        )

@contextmanager
def stream(
    method: str,
    url: typing.Union[str, URL],
    *,
    params: typing.Optional[QueryParamTypes] = None,
    content: typing.Optional[RequestContent] = None,
    data: typing.Optional[RequestData] = None,
    files: typing.Optional[RequestFiles] = None,
    json: typing.Optional[typing.Any] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    follow_redirects: bool = False,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    trust_env: bool = True
) -> typing.Iterator[Response]:
    with Client(cookies=cookies, proxy=proxy, verify=verify, timeout=timeout, trust_env=trust_env) as client:
        with client.stream(
            method=method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            auth=auth,
            follow_redirects=follow_redirects
        ) as response:
            yield response

def get(
    url: typing.Union[str, URL],
    *,
    params: typing.Optional[QueryParamTypes] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    follow_redirects: bool = False,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True
) -> Response:
    return request(
        'GET',
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env
    )

def options(
    url: typing.Union[str, URL],
    *,
    params: typing.Optional[QueryParamTypes] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    follow_redirects: bool = False,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True
) -> Response:
    return request(
        'OPTIONS',
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env
    )

def head(
    url: typing.Union[str, URL],
    *,
    params: typing.Optional[QueryParamTypes] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    follow_redirects: bool = False,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True
) -> Response:
    return request(
        'HEAD',
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env
    )

def post(
    url: typing.Union[str, URL],
    *,
    content: typing.Optional[RequestContent] = None,
    data: typing.Optional[RequestData] = None,
    files: typing.Optional[RequestFiles] = None,
    json: typing.Optional[typing.Any] = None,
    params: typing.Optional[QueryParamTypes] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    follow_redirects: bool = False,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True
) -> Response:
    return request(
        'POST',
        url,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env
    )

def put(
    url: typing.Union[str, URL],
    *,
    content: typing.Optional[RequestContent] = None,
    data: typing.Optional[RequestData] = None,
    files: typing.Optional[RequestFiles] = None,
    json: typing.Optional[typing.Any] = None,
    params: typing.Optional[QueryParamTypes] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    follow_redirects: bool = False,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True
) -> Response:
    return request(
        'PUT',
        url,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env
    )

def patch(
    url: typing.Union[str, URL],
    *,
    content: typing.Optional[RequestContent] = None,
    data: typing.Optional[RequestData] = None,
    files: typing.Optional[RequestFiles] = None,
    json: typing.Optional[typing.Any] = None,
    params: typing.Optional[QueryParamTypes] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    follow_redirects: bool = False,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True
) -> Response:
    return request(
        'PATCH',
        url,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env
    )

def delete(
    url: typing.Union[str, URL],
    *,
    params: typing.Optional[QueryParamTypes] = None,
    headers: typing.Optional[HeaderTypes] = None,
    cookies: typing.Optional[CookieTypes] = None,
    auth: typing.Optional[AuthTypes] = None,
    proxy: typing.Optional[ProxyTypes] = None,
    follow_redirects: bool = False,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    verify: typing.Union[bool, ssl.SSLContext] = True,
    trust_env: bool = True
) -> Response:
    return request(
        'DELETE',
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env
    )
