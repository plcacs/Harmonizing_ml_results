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
__all__: typing.List[str] = ['delete', 'get', 'head', 'options', 'patch', 'post', 'put',
    'request', 'stream']


def func_pvddet6g(method: str, url: URL, *, params: typing.Optional[QueryParamTypes] = None, content: typing.Optional[RequestContent] = None, data: typing.Optional[RequestData] = None,
    files: typing.Optional[RequestFiles] = None, json: typing.Optional[typing.Any] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, auth: typing.Optional[AuthTypes] = None, proxy: typing.Optional[ProxyTypes] = None, timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG, follow_redirects: bool = False, verify: bool = True, trust_env: bool = True) -> Response:
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

    