#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""HTTP utility code shared by clients and servers.

This module also defines the `HTTPServerRequest` class which is exposed
via `tornado.web.RequestHandler.request`.
"""

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

from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type


# responses is unused in this file, but we re-export it to other files.
# Reference it so pyflakes doesn't complain.
responses

import typing
from typing import (
    Tuple,
    Iterable,
    List,
    Mapping,
    Iterator,
    Dict,
    Union,
    Optional,
    Awaitable,
    Generator,
    AnyStr,
    Any,
    Match,
    Pattern,
    NamedTuple,
    TypeVar,
    Callable,
    Type,
    cast,
)

if typing.TYPE_CHECKING:
    from typing import Deque  # noqa: F401
    from asyncio import Future  # noqa: F401
    import unittest  # noqa: F401

    # This can be done unconditionally in the base class of HTTPHeaders
    # after we drop support for Python 3.8.
    StrMutableMapping = collections.abc.MutableMapping[str, str]
else:
    StrMutableMapping = collections.abc.MutableMapping

# To be used with str.strip() and related methods.
HTTP_WHITESPACE: str = " \t"

_T = TypeVar('_T')

@lru_cache(1000)
def _normalize_header(name: str) -> str:
    """Map a header name to Http-Header-Case.

    >>> _normalize_header("coNtent-TYPE")
    'Content-Type'
    """
    return "-".join([w.capitalize() for w in name.split("-")])


class HTTPHeaders(StrMutableMapping):
    """A dictionary that maintains ``Http-Header-Case`` for all keys."""

    def __init__(self, *args: Any, **kwargs: str) -> None:
        self._dict: Dict[str, str] = {}
        self._as_list: Dict[str, List[str]] = {}
        self._last_key: Optional[str] = None
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], HTTPHeaders):
            for k, v in args[0].get_all():
                self.add(k, v)
        else:
            self.update(*args, **kwargs)

    def add(self, name: str, value: str) -> None:
        norm_name = _normalize_header(name)
        self._last_key = norm_name
        if norm_name in self:
            self._dict[norm_name] = native_str(self[norm_name]) + "," + native_str(value)
            self._as_list[norm_name].append(value)
        else:
            self[norm_name] = value

    def get_list(self, name: str) -> List[str]:
        norm_name = _normalize_header(name)
        return self._as_list.get(norm_name, [])

    def get_all(self) -> Iterable[Tuple[str, str]]:
        for name, values in self._as_list.items():
            for value in values:
                yield (name, value)

    def parse_line(self, line: str) -> None:
        if line[0].isspace():
            if self._last_key is None:
                raise HTTPInputError("first header line cannot start with whitespace")
            new_part = " " + line.lstrip(HTTP_WHITESPACE)
            self._as_list[self._last_key][-1] += new_part
            self._dict[self._last_key] += new_part
        else:
            try:
                name, value = line.split(":", 1)
            except ValueError:
                raise HTTPInputError("no colon in header line")
            self.add(name, value.strip(HTTP_WHITESPACE))

    @classmethod
    def parse(cls, headers: str) -> "HTTPHeaders":
        h = cls()
        for line in headers.split("\n"):
            if line.endswith("\r"):
                line = line[:-1]
            if line:
                h.parse_line(line)
        return h

    def __setitem__(self, name: str, value: str) -> None:
        norm_name = _normalize_header(name)
        self._dict[norm_name] = value
        self._as_list[norm_name] = [value]

    def __getitem__(self, name: str) -> str:
        return self._dict[_normalize_header(name)]

    def __delitem__(self, name: str) -> None:
        norm_name = _normalize_header(name)
        del self._dict[norm_name]
        del self._as_list[norm_name]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def copy(self) -> "HTTPHeaders":
        return HTTPHeaders(self)

    __copy__ = copy

    def __str__(self) -> str:
        lines = []
        for name, value in self.get_all():
            lines.append(f"{name}: {value}\n")
        return "".join(lines)

    __unicode__ = __str__


class HTTPServerRequest:
    path: Optional[str] = None
    query: Optional[str] = None
    _body_future: Optional["Future[None]"] = None

    def __init__(
        self,
        method: Optional[str] = None,
        uri: Optional[str] = None,
        version: str = "HTTP/1.0",
        headers: Optional[HTTPHeaders] = None,
        body: Optional[bytes] = None,
        host: Optional[str] = None,
        files: Optional[Dict[str, List["HTTPFile"]]] = None,
        connection: Optional["HTTPConnection"] = None,
        start_line: Optional["RequestStartLine"] = None,
        server_connection: Optional[object] = None,
    ) -> None:
        if start_line is not None:
            method, uri, version = start_line
        self.method = method
        self.uri = uri
        self.version = version
        self.headers = headers or HTTPHeaders()
        self.body = body or b""
        context = getattr(connection, "context", None)
        self.remote_ip = getattr(context, "remote_ip", None)
        self.protocol = getattr(context, "protocol", "http")
        self.host = host or self.headers.get("Host") or "127.0.0.1"
        self.host_name = split_host_and_port(self.host.lower())[0]
        self.files = files or {}
        self.connection = connection
        self.server_connection = server_connection
        self._start_time = time.time()
        self._finish_time = None
        if uri is not None:
            self.path, _, self.query = uri.partition("?")
        self.arguments = parse_qs_bytes(self.query, keep_blank_values=True) if self.query else {}
        self.query_arguments = copy.deepcopy(self.arguments)
        self.body_arguments: Dict[str, List[bytes]] = {}

    @property
    def cookies(self) -> Dict[str, http.cookies.Morsel]:
        if not hasattr(self, "_cookies"):
            self._cookies: http.cookies.SimpleCookie = http.cookies.SimpleCookie()
            if "Cookie" in self.headers:
                try:
                    parsed = parse_cookie(self.headers["Cookie"])
                except Exception:
                    pass
                else:
                    for k, v in parsed.items():
                        try:
                            self._cookies[k] = v
                        except Exception:
                            pass
        return self._cookies

    def full_url(self) -> str:
        return f"{self.protocol}://{self.host}{self.uri}"

    def request_time(self) -> float:
        if self._finish_time is None:
            return time.time() - self._start_time
        else:
            return self._finish_time - self._start_time

    def get_ssl_certificate(self, binary_form: bool = False) -> Union[None, Dict[Any, Any], bytes]:
        try:
            if self.connection is None:
                return None
            return self.connection.stream.socket.getpeercert(binary_form=binary_form)
        except SSLError:
            return None

    def _parse_body(self) -> None:
        parse_body_arguments(
            self.headers.get("Content-Type", ""),
            self.body,
            self.body_arguments,
            self.files,
            self.headers,
        )
        for k, v in self.body_arguments.items():
            self.arguments.setdefault(k, []).extend(v)

    def __repr__(self) -> str:
        attrs = ("protocol", "host", "method", "uri", "version", "remote_ip")
        args = ", ".join([f"{n}={getattr(self, n)!r}" for n in attrs])
        return f"{self.__class__.__name__}({args})"


class HTTPInputError(Exception):
    pass


class HTTPOutputError(Exception):
    pass


class HTTPServerConnectionDelegate:
    def start_request(
        self, server_conn: object, request_conn: "HTTPConnection"
    ) -> "HTTPMessageDelegate":
        raise NotImplementedError()

    def on_close(self, server_conn: object) -> None:
        pass


class HTTPMessageDelegate:
    def headers_received(
        self,
        start_line: Union["RequestStartLine", "ResponseStartLine"],
        headers: HTTPHeaders,
    ) -> Optional[Awaitable[None]]:
        pass

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def finish(self) -> None:
        pass

    def on_connection_close(self) -> None:
        pass


class HTTPConnection:
    def write_headers(
        self,
        start_line: Union["RequestStartLine", "ResponseStartLine"],
        headers: HTTPHeaders,
        chunk: Optional[bytes] = None,
    ) -> "Future[None]":
        raise NotImplementedError()

    def write(self, chunk: bytes) -> "Future[None]":
        raise NotImplementedError()

    def finish(self) -> None:
        raise NotImplementedError()


def url_concat(
    url: str,
    args: Union[None, Dict[str, str], List[Tuple[str, str]], Tuple[Tuple[str, str], ...]],
) -> str:
    if args is None:
        return url
    parsed_url = urlparse(url)
    if isinstance(args, dict):
        parsed_query = parse_qsl(parsed_url.query, keep_blank_values=True)
        parsed_query.extend(args.items())
    elif isinstance(args, (list, tuple)):
        parsed_query = parse_qsl(parsed_url.query, keep_blank_values=True)
        parsed_query.extend(args)
    else:
        raise TypeError(f"'args' parameter should be dict, list or tuple. Not {type(args)}")
    final_query = urlencode(parsed_query)
    return urlunparse((
        parsed_url[0],
        parsed_url[1],
        parsed_url[2],
        parsed_url[3],
        final_query,
        parsed_url[5],
    ))


class HTTPFile(ObjectDict):
    filename: str
    body: bytes
    content_type: str


def _parse_request_range(range_header: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    unit, _, value = range_header.partition("=")
    unit, value = unit.strip(), value.strip()
    if unit != "bytes":
        return None
    start_b, _, end_b = value.partition("-")
    try:
        start = _int_or_none(start_b)
        end = _int_or_none(end_b)
    except ValueError:
        return None
    if end is not None:
        if start is None:
            if end != 0:
                start = -end
                end = None
        else:
            end += 1
    return (start, end)


def _get_content_range(start: Optional[int], end: Optional[int], total: int) -> str:
    start = start or 0
    end = (end or total) - 1
    return f"bytes {start}-{end}/{total}"


def _int_or_none(val: str) -> Optional[int]:
    val = val.strip()
    if val == "":
        return None
    return int(val)


def parse_body_arguments(
    content_type: str,
    body: bytes,
    arguments: Dict[str, List[bytes]],
    files: Dict[str, List[HTTPFile]],
    headers: Optional[HTTPHeaders] = None,
) -> None:
    if content_type.startswith("application/x-www-form-urlencoded"):
        if headers and "Content-Encoding" in headers:
            gen_log.warning("Unsupported Content-Encoding: %s", headers["Content-Encoding"])
            return
        try:
            uri_arguments = parse_qs_bytes(body, keep_blank_values=True)
        except Exception as e:
            gen_log.warning("Invalid x-www-form-urlencoded body: %s", e)
            uri_arguments = {}
        for name, values in uri_arguments.items():
            if values:
                arguments.setdefault(name, []).extend(values)
    elif content_type.startswith("multipart/form-data"):
        if headers and "Content-Encoding" in headers:
            gen_log.warning("Unsupported Content-Encoding: %s", headers["Content-Encoding"])
            return
        try:
            fields = content_type.split(";")
            for field in fields:
                k, sep, v = field.strip().partition("=")
                if k == "boundary" and v:
                    parse_multipart_form_data(utf8(v), body, arguments, files)
                    break
            else:
                raise ValueError("multipart boundary not found")
        except Exception as e:
            gen_log.warning("Invalid multipart/form-data: %s", e)


def parse_multipart_form_data(
    boundary: bytes,
    data: bytes,
    arguments: Dict[str, List[bytes]],
    files: Dict[str, List[HTTPFile]],
) -> None:
    if boundary.startswith(b'"') and boundary.endswith(b'"'):
        boundary = boundary[1:-1]
    final_boundary_index = data.rfind(b"--" + boundary + b"--")
    if final_boundary_index == -1:
        gen_log.warning("Invalid multipart/form-data: no final boundary")
        return
    parts = data[:final_boundary_index].split(b"--" + boundary + b"\r\n")
    for part in parts:
        if not part:
            continue
        eoh = part.find(b"\r\n\r\n")
        if eoh == -1:
            gen_log.warning("multipart/form-data missing headers")
            continue
        headers = HTTPHeaders.parse(part[:eoh].decode("utf-8"))
        disp_header = headers.get("Content-Disposition", "")
        disposition, disp_params = _parse_header(disp_header)
        if disposition != "form-data" or not part.endswith(b"\r\n"):
            gen_log.warning("Invalid multipart/form-data")
            continue
        value = part[eoh + 4 : -2]
        if not disp_params.get("name"):
            gen_log.warning("multipart/form-data value missing name")
            continue
        name = disp_params["name"]
        if disp_params.get("filename"):
            ctype = headers.get("Content-Type", "application/unknown")
            files.setdefault(name, []).append(
                HTTPFile(filename=disp_params["filename"], body=value, content_type=ctype)
            )
        else:
            arguments.setdefault(name, []).append(value)


def format_timestamp(
    ts: Union[int, float, Tuple[Any, ...], time.struct_time, datetime.datetime]
) -> str:
    if isinstance(ts, (int, float)):
        time_num = ts
    elif isinstance(ts, (tuple, time.struct_time)):
        time_num = calendar.timegm(ts)
    elif isinstance(ts, datetime.datetime):
        time_num = calendar.timegm(ts.utctimetuple())
    else:
        raise TypeError(f"unknown timestamp type: {ts!r}")
    return email.utils.formatdate(time_num, usegmt=True)


class RequestStartLine(NamedTuple):
    method: str
    path: str
    version: str


_http_version_re: Pattern[str] = re.compile(r"^HTTP/1\.[0-9]$")


def parse_request_start_line(line: str) -> RequestStartLine:
    try:
        method, path, version = line.split(" ")
    except ValueError:
        raise HTTPInputError("Malformed HTTP request line")
    if not _http_version_re.match(version):
        raise HTTPInputError(f"Malformed HTTP version in HTTP Request-Line: {version!r}")
    return RequestStartLine(method, path, version)


class ResponseStartLine(NamedTuple):
    version: str
    code: int
    reason: str


_http_response_line_re: Pattern[str] = re.compile(r"(HTTP/1.[0-9]) ([0-9]+) ([^\r]*)")


def parse_response_start_line(line: str) -> ResponseStartLine:
    line = native_str(line)
    match = _http_response_line_re.match(line)
    if not match:
        raise HTTPInputError("Error parsing response start line")
    return ResponseStartLine(match.group(1), int(match.group(2)), match.group(3))


def _parseparam(s: str) -> Generator[str, None, None]:
    while s[:1] == ";":
        s = s[1:]
        end = s.find(";")
        while end > 0 and (s.count('"', 0, end) - s.count('\\"