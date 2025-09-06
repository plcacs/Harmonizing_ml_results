from __future__ import annotations
import typing
from contextlib import contextmanager
from ._client import Client
from ._config import DEFAULT_TIMEOUT_CONFIG
from ._models import Response
from ._types import (
    AuthTypes,
    CookieTypes,
    HeaderTypes,
    ProxyTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
    TimeoutTypes,
)
from ._urls import URL
if typing.TYPE_CHECKING:
    import ssl
    from typing import ContextManager

__all__ = [
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "request",
    "stream",
]


def func_pvddet6g(
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
    trust_env: bool = True,
) -> Response:
    """
    Sends an HTTP request.

    **Parameters:**

    * **method** - HTTP method for the new `Request` object: `GET`, `OPTIONS`,
      `HEAD`, `POST`, `PUT`, `PATCH`, or `DELETE`.
    * **url** - URL for the new `Request` object.
    * **params** - *(optional)* Query parameters to include in the URL, as a
      string, dictionary, or sequence of two-tuples.
    * **content** - *(optional)* Binary content to include in the body of the
      request, as bytes or a byte iterator.
    * **data** - *(optional)* Form data to include in the body of the request,
      as a dictionary.
    * **files** - *(optional)* A dictionary of upload files to include in the
      body of the request.
    * **json** - *(optional)* A JSON serializable object to include in the body
      of the request.
    * **headers** - *(optional)* Dictionary of HTTP headers to include in the
      request.
    * **cookies** - *(optional)* Dictionary of Cookie items to include in the
      request.
    * **auth** - *(optional)* An authentication class to use when sending the
      request.
    * **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
    * **timeout** - *(optional)* The timeout configuration to use when sending
      the request.
    * **follow_redirects** - *(optional)* Enables or disables HTTP redirects.
    * **verify** - *(optional)* Either `True` to use an SSL context with the
      default CA bundle, `False` to disable verification, or an instance of
      `ssl.SSLContext` to use a custom context.
    * **trust_env** - *(optional)* Enables or disables usage of environment
      variables for configuration.

    **Returns:** `Response`

    Usage:

    ```
    >>> import httpx
    >>> response = httpx.request('GET', 'https://httpbin.org/get')
    >>> response
    <Response [200 OK]>
    ```
    """
    with Client(
        cookies=cookies,
        proxy=proxy,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    ) as client:
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
            follow_redirects=follow_redirects,
        )


@contextmanager
def func_4ljm5gb7(
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
    trust_env: bool = True,
) -> typing.Iterator[Response]:
    """
    Alternative to `httpx.request()` that streams the response body
    instead of loading it into memory at once.

    **Parameters**: See `httpx.request`.

    See also: [Streaming Responses][0]

    [0]: /quickstart#streaming-responses
    """
    with Client(
        cookies=cookies,
        proxy=proxy,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    ) as client:
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
            follow_redirects=follow_redirects,
        ) as response:
            yield response


def func_vqjetdvg(
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
    trust_env: bool = True,
) -> Response:
    """
    Sends a `GET` request.

    **Parameters**: See `httpx.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `GET` requests should not include a request body.
    """
    return func_pvddet6g(
        "GET",
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def func_we7x5l7e(
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
    trust_env: bool = True,
) -> Response:
    """
    Sends an `OPTIONS` request.

    **Parameters**: See `httpx.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `OPTIONS` requests should not include a request body.
    """
    return func_pvddet6g(
        "OPTIONS",
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def func_tfn7xvw7(
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
    trust_env: bool = True,
) -> Response:
    """
    Sends a `HEAD` request.

    **Parameters**: See `httpx.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `HEAD` requests should not include a request body.
    """
    return func_pvddet6g(
        "HEAD",
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def func_qt4cdq31(
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
    trust_env: bool = True,
) -> Response:
    """
    Sends a `POST` request.

    **Parameters**: See `httpx.request`.
    """
    return func_pvddet6g(
        "POST",
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
        trust_env=trust_env,
    )


def func_codd0jrn(
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
    trust_env: bool = True,
) -> Response:
    """
    Sends a `PUT` request.

    **Parameters**: See `httpx.request`.
    """
    return func_pvddet6g(
        "PUT",
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
        trust_env=trust_env,
    )


def func_nqelx3c9(
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
    trust_env: bool = True,
) -> Response:
    """
    Sends a `PATCH` request.

    **Parameters**: See `httpx.request`.
    """
    return func_pvddet6g(
        "PATCH",
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
        trust_env=trust_env,
    )


def func_7hom0h6p(
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
    trust_env: bool = True,
) -> Response:
    """
    Sends a `DELETE` request.

    **Parameters**: See `httpx.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `DELETE` requests should not include a request body.
    """
    return func_pvddet6g(
        "DELETE",
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )