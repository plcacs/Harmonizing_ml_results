import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from typing import Tuple, Iterable, List, Mapping, Iterator, Dict, Union, Optional, Awaitable, Generator, AnyStr

class HTTPHeaders(StrMutableMapping):
    # ...

    @lru_cache(1000)
    def _normalize_header(name: str) -> str:
        """Map a header name to Http-Header-Case."""
        return '-'.join([w.capitalize() for w in name.split('-')])

class HTTPServerRequest:
    # ...

    def __init__(self, method: str, uri: str, version: str = 'HTTP/1.0', headers: HTTPHeaders = None, body: bytes = None, host: str = None, files: Dict[str, List[HTTPFile]] = None, connection: 'HTTPServerConnection' = None, start_line: RequestStartLine = None, server_connection: 'HTTPServerConnection' = None):
        # ...

    @property
    def cookies(self) -> http.cookies.SimpleCookie:
        # ...

    def full_url(self) -> str:
        # ...

    def request_time(self) -> float:
        # ...

    def get_ssl_certificate(self, binary_form: bool = False) -> Union[bytes, Dict]:
        # ...

class HTTPMessageDelegate:
    # ...

    def headers_received(self, start_line: RequestStartLine, headers: HTTPHeaders):
        # ...

    def data_received(self, chunk: bytes):
        # ...

    def finish(self):
        # ...

    def on_connection_close(self):
        # ...

class HTTPFile(ObjectDict):
    # ...

def url_concat(url: str, args: Union[Dict[str, str], List[Tuple[str, str]]]) -> str:
    # ...

def _parse_request_range(range_header: str) -> Tuple[Optional[int], Optional[int]]:
    # ...

def _get_content_range(start: Optional[int], end: Optional[int], total: int) -> str:
    # ...

def _int_or_none(val: str) -> Optional[int]:
    # ...

def parse_body_arguments(content_type: str, body: bytes, arguments: Dict[str, List[str]], files: Dict[str, List[HTTPFile]], headers: HTTPHeaders = None) -> None:
    # ...

def parse_multipart_form_data(boundary: bytes, data: bytes, arguments: Dict[str, List[str]], files: Dict[str, List[HTTPFile]]) -> None:
    # ...

def format_timestamp(ts: Union[int, Tuple[int, int, int], datetime.datetime]) -> str:
    # ...

class RequestStartLine(NamedTuple):
    pass

def parse_request_start_line(line: str) -> RequestStartLine:
    # ...

class ResponseStartLine(NamedTuple):
    pass

def parse_response_start_line(line: str) -> ResponseStartLine:
    # ...

def _parseparam(s: str) -> Iterator[str]:
    # ...

def _parse_header(line: str) -> Tuple[str, Dict[str, str]]:
    # ...

def _encode_header(key: str, pdict: Dict[str, str]) -> str:
    # ...

def encode_username_password(username: str, password: str) -> bytes:
    # ...

def split_host_and_port(netloc: str) -> Tuple[str, Optional[int]]:
    # ...

def qs_to_qsl(qs: Dict[str, List[str]]) -> Iterator[Tuple[str, str]]:
    # ...

def _unquote_cookie(s: str) -> str:
    # ...

def parse_cookie(cookie: str) -> Dict[str, str]:
    # ...
