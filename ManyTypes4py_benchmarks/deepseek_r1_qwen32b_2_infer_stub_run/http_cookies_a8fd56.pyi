"""
requests.cookies
~~~~~~~~~~~~~~~~

Compatibility code to be able to use `cookielib.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

import copy
import time
import calendar
from collections.abc import MutableMapping
from threading import RLock
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from ._internal_utils import to_native_string
from ._basics import (
    cookielib,
    urlparse,
    urlunparse,
    Morsel,
    HTTPMessage,
    HTTPResponse,
    Request,
)

class MockRequest:
    """Wraps a `requests.Request` to mimic a `urllib2.Request`."""
    def __init__(self, request: Request) -> None:
        ...

    def get_type(self) -> str:
        ...

    def get_host(self) -> str:
        ...

    def get_origin_req_host(self) -> str:
        ...

    def get_full_url(self) -> str:
        ...

    def is_unverifiable(self) -> bool:
        ...

    def has_header(self, name: str) -> bool:
        ...

    def get_header(self, name: str, default: Optional[Any] = None) -> Optional[str]:
        ...

    def add_unredirected_header(self, name: str, value: str) -> None:
        ...

    def get_new_headers(self) -> Dict[str, str]:
        ...

    @property
    def unverifiable(self) -> bool:
        ...

    @property
    def origin_req_host(self) -> str:
        ...

    @property
    def host(self) -> str:
        ...

class Headers:
    def __init__(self, headers: HTTPMessage) -> None:
        ...

    def get_all(self, key: str, default: Optional[Any] = None) -> List[str]:
        ...

class MockResponse:
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`."""
    def __init__(self, headers: HTTPMessage) -> None:
        ...

    def get_all(self, name: str, default: Optional[Any] = None) -> List[str]:
        ...

    def info(self) -> Headers:
        ...

    def getheaders(self, name: str) -> List[str]:
        ...

    @property
    def headers(self) -> Headers:
        ...

def extract_cookies_to_jar(
    jar: cookielib.CookieJar,
    request: Request,
    response: HTTPResponse,
) -> None:
    ...

def get_cookie_header(jar: cookielib.CookieJar, request: Request) -> Optional[str]:
    ...

def remove_cookie_by_name(
    cookiejar: cookielib.CookieJar,
    name: str,
    domain: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    ...

class CookieConflictError(RuntimeError):
    ...

class RequestsCookieJar(cookielib.CookieJar, MutableMapping):
    """Compatibility class; is a cookielib.CookieJar, but exposes a dict interface."""
    def get(
        self,
        name: str,
        default: Optional[Any] = None,
        domain: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Optional[str]:
        ...

    def set(
        self,
        name: str,
        value: Union[str, Morsel],
        **kwargs: Any,
    ) -> cookielib.Cookie:
        ...

    def iterkeys(self) -> Iterator[str]:
        ...

    def keys(self) -> List[str]:
        ...

    def itervalues(self) -> Iterator[str]:
        ...

    def values(self) -> List[str]:
        ...

    def iteritems(self) -> Iterator[Tuple[str, str]]:
        ...

    def items(self) -> List[Tuple[str, str]]:
        ...

    def list_domains(self) -> List[str]:
        ...

    def list_paths(self) -> List[str]:
        ...

    def multiple_domains(self) -> bool:
        ...

    def get_dict(
        self,
        domain: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Dict[str, str]:
        ...

    def __contains__(self, name: str) -> bool:
        ...

    def __getitem__(self, name: str) -> str:
        ...

    def __setitem__(self, name: str, value: str) -> None:
        ...

    def __delitem__(self, name: str) -> None:
        ...

    def set_cookie(self, cookie: cookielib.Cookie, *args: Any, **kwargs: Any) -> None:
        ...

    def update(self, other: Union[cookielib.CookieJar, MutableMapping]) -> None:
        ...

    def _find(
        self,
        name: str,
        domain: Optional[str] = None,
        path: Optional[str] = None,
    ) -> str:
        ...

    def _find_no_duplicates(
        self,
        name: str,
        domain: Optional[str] = None,
        path: Optional[str] = None,
    ) -> str:
        ...

    def __getstate__(self) -> Dict[str, Any]:
        ...

    def __setstate__(self, state: Dict[str, Any]) -> None:
        ...

    def copy(self) -> "RequestsCookieJar":
        ...

    def get_policy(self) -> cookielib.CookiePolicy:
        ...

def _copy_cookie_jar(jar: Optional[cookielib.CookieJar]) -> Optional[cookielib.CookieJar]:
    ...

def create_cookie(
    name: str,
    value: str,
    **kwargs: Any,
) -> cookielib.Cookie:
    ...

def morsel_to_cookie(morsel: Morsel) -> cookielib.Cookie:
    ...

def cookiejar_from_dict(
    cookie_dict: Optional[Dict[str, str]],
    cookiejar: Optional[cookielib.CookieJar] = None,
    overwrite: bool = True,
) -> cookielib.CookieJar:
    ...

def merge_cookies(
    cookiejar: cookielib.CookieJar,
    cookies: Union[Dict[str, str], cookielib.CookieJar],
) -> cookielib.CookieJar:
    ...