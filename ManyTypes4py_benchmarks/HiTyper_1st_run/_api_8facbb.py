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

def request(method: Union[_types.RequestContent, _types.AuthTypes, _types.URLTypes], url: Union[_types.RequestContent, _types.AuthTypes, _types.URLTypes], *, params: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, content: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, data: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, files: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, json: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, headers: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, cookies: Union[None, bool, _types.TimeoutTypes, _config.UnsetType, _types.AuthTypes]=None, auth: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, proxy: Union[None, bool, _types.TimeoutTypes, _config.UnsetType, _types.AuthTypes]=None, timeout: Any=DEFAULT_TIMEOUT_CONFIG, follow_redirects: bool=False, verify: bool=True, trust_env: bool=True) -> flask.Response:
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
    with Client(cookies=cookies, proxy=proxy, verify=verify, timeout=timeout, trust_env=trust_env) as client:
        return client.request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, auth=auth, follow_redirects=follow_redirects)

@contextmanager
def stream(method: Union[_types.RequestContent, _types.AuthTypes, _types.URLTypes], url: Union[_types.RequestContent, _types.AuthTypes, _types.URLTypes], *, params: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, content: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, data: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, files: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, json: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, headers: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, cookies: Union[None, bool, _types.TimeoutTypes, _config.UnsetType, _types.AuthTypes]=None, auth: Union[None, _types.RequestContent, _types.AuthTypes, _types.URLTypes]=None, proxy: Union[None, bool, _types.TimeoutTypes, _config.UnsetType, _types.AuthTypes]=None, timeout: Any=DEFAULT_TIMEOUT_CONFIG, follow_redirects: bool=False, verify: bool=True, trust_env: bool=True) -> typing.Generator:
    """
    Alternative to `httpx.request()` that streams the response body
    instead of loading it into memory at once.

    **Parameters**: See `httpx.request`.

    See also: [Streaming Responses][0]

    [0]: /quickstart#streaming-responses
    """
    with Client(cookies=cookies, proxy=proxy, verify=verify, timeout=timeout, trust_env=trust_env) as client:
        with client.stream(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, auth=auth, follow_redirects=follow_redirects) as response:
            yield response

def get(url: Union[_types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes], *, params: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, headers: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, cookies: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, auth: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, proxy: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, follow_redirects: bool=False, verify: bool=True, timeout: Any=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    """
    Sends a `GET` request.

    **Parameters**: See `httpx.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `GET` requests should not include a request body.
    """
    return request('GET', url, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def options(url: Union[_types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes], *, params: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, headers: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, cookies: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, auth: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, proxy: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, follow_redirects: bool=False, verify: bool=True, timeout: Any=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    """
    Sends an `OPTIONS` request.

    **Parameters**: See `httpx.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `OPTIONS` requests should not include a request body.
    """
    return request('OPTIONS', url, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def head(url: Union[_types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes], *, params: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, headers: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, cookies: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, auth: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, proxy: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, follow_redirects: bool=False, verify: bool=True, timeout: Any=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    """
    Sends a `HEAD` request.

    **Parameters**: See `httpx.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `HEAD` requests should not include a request body.
    """
    return request('HEAD', url, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def post(url: Union[bool, _types.URLTypes, _types.TimeoutTypes], *, content: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, data: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, files: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, json: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, params: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, headers: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, cookies: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, auth: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, proxy: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, follow_redirects: bool=False, verify: bool=True, timeout: Any=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    """
    Sends a `POST` request.

    **Parameters**: See `httpx.request`.
    """
    return request('POST', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def put(url: Union[bool, _types.URLTypes, _types.TimeoutTypes], *, content: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, data: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, files: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, json: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, params: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, headers: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, cookies: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, auth: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, proxy: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, follow_redirects: bool=False, verify: bool=True, timeout: Any=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    """
    Sends a `PUT` request.

    **Parameters**: See `httpx.request`.
    """
    return request('PUT', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def patch(url: Union[bool, _types.URLTypes, _types.TimeoutTypes], *, content: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, data: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, files: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, json: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, params: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, headers: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, cookies: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, auth: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, proxy: Union[None, bool, _types.URLTypes, _types.TimeoutTypes]=None, follow_redirects: bool=False, verify: bool=True, timeout: Any=DEFAULT_TIMEOUT_CONFIG, trust_env: bool=True):
    """
    Sends a `PATCH` request.

    **Parameters**: See `httpx.request`.
    """
    return request('PATCH', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)

def delete(url: Union[_types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes], *, params: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, headers: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, cookies: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, auth: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, proxy: Union[None, _types.TimeoutTypes, _config.UnsetType, bool, _types.CookieTypes]=None, follow_redirects: bool=False, timeout: Any=DEFAULT_TIMEOUT_CONFIG, verify: bool=True, trust_env: bool=True):
    """
    Sends a `DELETE` request.

    **Parameters**: See `httpx.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `DELETE` requests should not include a request body.
    """
    return request('DELETE', url, params=params, headers=headers, cookies=cookies, auth=auth, proxy=proxy, follow_redirects=follow_redirects, verify=verify, timeout=timeout, trust_env=trust_env)