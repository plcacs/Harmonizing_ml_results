#!/usr/bin/env python3
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
import typing
from typing import Tuple, Iterable, List, Mapping, Iterator, Dict, Union, Optional, Awaitable, Generator, AnyStr, Any, NamedTuple, Pattern

if typing.TYPE_CHECKING:
    from typing import Deque
    from asyncio import Future
    import unittest
    StrMutableMapping = collections.abc.MutableMapping[str, str]
else:
    StrMutableMapping = collections.abc.MutableMapping

HTTP_WHITESPACE: str = ' \t'

@lru_cache(1000)
def _normalize_header(name: str) -> str:
    """Map a header name to Http-Header-Case.

    >>> _normalize_header("coNtent-TYPE")
    'Content-Type'
    """
    return '-'.join([w.capitalize() for w in name.split('-')])


class HTTPHeaders(StrMutableMapping):
    """A dictionary that maintains ``Http-Header-Case`` for all keys.

    Supports multiple values per key via a pair of new methods,
    `add()` and `get_list()`.  The regular dictionary interface
    returns a single value per key, with multiple values joined by a
    comma.
    """

    @typing.overload
    def __init__(self, __arg: Any) -> None:
        ...

    @typing.overload
    def __init__(self, __arg: Any) -> None:
        ...

    @typing.overload
    def __init__(self, *args: Any) -> None:
        ...

    @typing.overload
    def __init__(self, **kwargs: Any) -> None:
        ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._dict: Dict[str, str] = {}
        self._as_list: Dict[str, List[str]] = {}
        self._last_key: Optional[str] = None
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], HTTPHeaders):
            for k, v in args[0].get_all():
                self.add(k, v)
        else:
            self.update(*args, **kwargs)

    def add(self, name: str, value: str) -> None:
        """Adds a new value for the given key."""
        norm_name: str = _normalize_header(name)
        self._last_key = norm_name
        if norm_name in self:
            self._dict[norm_name] = native_str(self[norm_name]) + ',' + native_str(value)
            self._as_list[norm_name].append(value)
        else:
            self[norm_name] = value

    def get_list(self, name: str) -> List[str]:
        """Returns all values for the given header as a list."""
        norm_name: str = _normalize_header(name)
        return self._as_list.get(norm_name, [])

    def get_all(self) -> Iterator[Tuple[str, str]]:
        """Returns an iterable of all (name, value) pairs.

        If a header has multiple values, multiple pairs will be
        returned with the same name.
        """
        for name, values in self._as_list.items():
            for value in values:
                yield (name, value)

    def parse_line(self, line: str) -> None:
        """Updates the dictionary with a single header line.

        >>> h = HTTPHeaders()
        >>> h.parse_line("Content-Type: text/html")
        >>> h.get('content-type')
        'text/html'
        """
        if line[0].isspace():
            if self._last_key is None:
                raise HTTPInputError('first header line cannot start with whitespace')
            new_part: str = ' ' + line.lstrip(HTTP_WHITESPACE)
            self._as_list[self._last_key][-1] += new_part
            self._dict[self._last_key] += new_part
        else:
            try:
                name, value = line.split(':', 1)
            except ValueError:
                raise HTTPInputError('no colon in header line')
            self.add(name, value.strip(HTTP_WHITESPACE))

    @classmethod
    def parse(cls, headers: str) -> "HTTPHeaders":
        """Returns a dictionary from HTTP header text.

        >>> h = HTTPHeaders.parse("Content-Type: text/html\\r\\nContent-Length: 42\\r\\n")
        >>> sorted(h.items())
        [('Content-Length', '42'), ('Content-Type', 'text/html')]

        .. versionchanged:: 5.1

           Raises `HTTPInputError` on malformed headers instead of a
           mix of `KeyError`, and `ValueError`.

        """
        h: HTTPHeaders = cls()
        for line in headers.split('\n'):
            if line.endswith('\r'):
                line = line[:-1]
            if line:
                h.parse_line(line)
        return h

    def __setitem__(self, name: str, value: str) -> None:
        norm_name: str = _normalize_header(name)
        self._dict[norm_name] = value
        self._as_list[norm_name] = [value]

    def __getitem__(self, name: str) -> str:
        return self._dict[_normalize_header(name)]

    def __delitem__(self, name: str) -> None:
        norm_name: str = _normalize_header(name)
        del self._dict[norm_name]
        del self._as_list[norm_name]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def copy(self) -> "HTTPHeaders":
        return HTTPHeaders(self)
    __copy__ = copy.copy

    def __str__(self) -> str:
        lines: List[str] = []
        for name, value in self.get_all():
            lines.append(f'{name}: {value}\n')
        return ''.join(lines)
    __unicode__ = __str__


class HTTPServerRequest:
    """A single HTTP request.

    All attributes are type `str` unless otherwise noted.
    """

    path: Optional[str] = None
    query: Optional[str] = None
    _body_future: Any = None

    def __init__(self, method: Optional[str] = None, uri: Optional[str] = None, version: str = 'HTTP/1.0', headers: Optional[HTTPHeaders] = None, body: Optional[bytes] = None, host: Optional[str] = None, files: Optional[Dict[str, List["HTTPFile"]]] = None, connection: Optional[Any] = None, start_line: Optional[Tuple[str, str, str]] = None, server_connection: Optional[Any] = None) -> None:
        if start_line is not None:
            method, uri, version = start_line
        self.method: Optional[str] = method
        self.uri: Optional[str] = uri
        self.version: str = version
        self.headers: HTTPHeaders = headers or HTTPHeaders()
        self.body: bytes = body or b''
        context: Any = getattr(connection, 'context', None)
        self.remote_ip: Optional[str] = getattr(context, 'remote_ip', None)
        self.protocol: str = getattr(context, 'protocol', 'http')
        self.host: str = host or self.headers.get('Host') or '127.0.0.1'
        self.host_name: str = split_host_and_port(self.host.lower())[0]
        self.files: Dict[str, List[HTTPFile]] = files or {}
        self.connection: Any = connection
        self.server_connection: Any = server_connection
        self._start_time: float = time.time()
        self._finish_time: Optional[float] = None
        if uri is not None:
            self.path, sep, self.query = uri.partition('?')
        else:
            self.path, self.query = None, None
        self.arguments: Dict[Any, List[bytes]] = parse_qs_bytes(self.query, keep_blank_values=True)  # type: ignore
        self.query_arguments: Dict[Any, List[bytes]] = copy.deepcopy(self.arguments)
        self.body_arguments: Dict[Any, List[bytes]] = {}

    @property
    def cookies(self) -> http.cookies.SimpleCookie:
        """A dictionary of ``http.cookies.Morsel`` objects."""
        if not hasattr(self, '_cookies'):
            self._cookies: http.cookies.SimpleCookie = http.cookies.SimpleCookie()
            if 'Cookie' in self.headers:
                try:
                    parsed: Dict[str, str] = parse_cookie(self.headers['Cookie'])
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
        """Reconstructs the full URL for this request."""
        return self.protocol + '://' + self.host + (self.uri or '')

    def request_time(self) -> float:
        """Returns the amount of time it took for this request to execute."""
        if self._finish_time is None:
            return time.time() - self._start_time
        else:
            return self._finish_time - self._start_time

    def get_ssl_certificate(self, binary_form: bool = False) -> Optional[Union[Dict[Any, Any], bytes]]:
        """Returns the client's SSL certificate, if any."""
        try:
            if self.connection is None:
                return None
            return self.connection.stream.socket.getpeercert(binary_form=binary_form)
        except SSLError:
            return None

    def _parse_body(self) -> None:
        parse_body_arguments(self.headers.get('Content-Type', ''), self.body, self.body_arguments, self.files, self.headers)
        for k, v in self.body_arguments.items():
            self.arguments.setdefault(k, []).extend(v)

    def __repr__(self) -> str:
        attrs: Tuple[str, ...] = ('protocol', 'host', 'method', 'uri', 'version', 'remote_ip')
        args: str = ', '.join([f'{n}={getattr(self, n)!r}' for n in attrs])
        return f'{self.__class__.__name__}({args})'


class HTTPInputError(Exception):
    """Exception class for malformed HTTP requests or responses
    from remote sources.
    """
    pass


class HTTPOutputError(Exception):
    """Exception class for errors in HTTP output.
    """
    pass


class HTTPServerConnectionDelegate:
    """Implement this interface to handle requests from `.HTTPServer`.
    """

    def start_request(self, server_conn: Any, request_conn: Any) -> "HTTPMessageDelegate":
        """Called by the server when a new request has started.

        :arg server_conn: an opaque object representing the long-lived connection.
        :arg request_conn: an `.HTTPConnection` object for a single request/response exchange.
        """
        raise NotImplementedError()

    def on_close(self, server_conn: Any) -> None:
        """Called when a connection has been closed.

        :arg server_conn: a server connection that has previously been passed to ``start_request``.
        """
        pass


class HTTPMessageDelegate:
    """Implement this interface to handle an HTTP request or response.
    """

    def headers_received(self, start_line: Any, headers: HTTPHeaders) -> Optional[Awaitable[Any]]:
        """Called when the HTTP headers have been received and parsed.

        :arg start_line: a `.RequestStartLine` or `.ResponseStartLine`.
        :arg headers: a `.HTTPHeaders` instance.
        """
        pass

    def data_received(self, chunk: bytes) -> Optional[Awaitable[Any]]:
        """Called when a chunk of data has been received."""
        pass

    def finish(self) -> None:
        """Called after the last chunk of data has been received."""
        pass

    def on_connection_close(self) -> None:
        """Called if the connection is closed without finishing the request."""
        pass


class HTTPConnection:
    """Applications use this interface to write their responses.
    """

    def write_headers(self, start_line: Any, headers: HTTPHeaders, chunk: Optional[bytes] = None) -> Awaitable[Any]:
        """Write an HTTP header block.

        :arg start_line: a `.RequestStartLine` or `.ResponseStartLine`.
        :arg headers: a `.HTTPHeaders` instance.
        :arg chunk: the first (optional) chunk of data.
        """
        raise NotImplementedError()

    def write(self, chunk: bytes) -> Awaitable[Any]:
        """Writes a chunk of body data.
        """
        raise NotImplementedError()

    def finish(self) -> Awaitable[Any]:
        """Indicates that the last body data has been written."""
        raise NotImplementedError()


def url_concat(url: str, args: Union[Dict[str, str], List[Tuple[str, str]], Tuple[Tuple[str, str], ...]]) -> str:
    """Concatenate url and arguments regardless of whether
    url has existing query parameters.
    """
    if args is None:
        return url
    parsed_url = urlparse(url)
    if isinstance(args, dict):
        parsed_query: List[Tuple[str, str]] = parse_qsl(parsed_url.query, keep_blank_values=True)
        parsed_query.extend(list(args.items()))
    elif isinstance(args, list) or isinstance(args, tuple):
        parsed_query = parse_qsl(parsed_url.query, keep_blank_values=True)
        parsed_query.extend(args)  # type: ignore
    else:
        err = "'args' parameter should be dict, list or tuple. Not {0}".format(type(args))
        raise TypeError(err)
    final_query: str = urlencode(parsed_query)
    url = urlunparse((parsed_url[0], parsed_url[1], parsed_url[2], parsed_url[3], final_query, parsed_url[5]))
    return url


class HTTPFile(ObjectDict):
    """Represents a file uploaded via a form.

    For backwards compatibility, its instance attributes are also
    accessible as dictionary keys.

    * ``filename``
    * ``body``
    * ``content_type``
    """
    # The actual attributes can be dynamically assigned.
    pass


def _parse_request_range(range_header: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """Parses a Range header.
    """
    unit, _, value = range_header.partition('=')
    unit, value = (unit.strip(), value.strip())
    if unit != 'bytes':
        return None
    start_b, _, end_b = value.partition('-')
    try:
        start: Optional[int] = _int_or_none(start_b)
        end: Optional[int] = _int_or_none(end_b)
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
    """Returns a suitable Content-Range header:
    """
    start_val: int = start or 0
    end_val: int = (end or total) - 1
    return f'bytes {start_val}-{end_val}/{total}'


def _int_or_none(val: str) -> Optional[int]:
    val = val.strip()
    if val == '':
        return None
    return int(val)


def parse_body_arguments(content_type: str, body: bytes, arguments: Dict[Any, List[bytes]], files: Dict[str, List[HTTPFile]], headers: Optional[Dict[str, str]] = None) -> None:
    """Parses a form request body.
    """
    if content_type.startswith('application/x-www-form-urlencoded'):
        if headers and 'Content-Encoding' in headers:
            gen_log.warning('Unsupported Content-Encoding: %s', headers['Content-Encoding'])
            return
        try:
            uri_arguments: Dict[Any, List[bytes]] = parse_qs_bytes(body, keep_blank_values=True)  # type: ignore
        except Exception as e:
            gen_log.warning('Invalid x-www-form-urlencoded body: %s', e)
            uri_arguments = {}
        for name, values in uri_arguments.items():
            if values:
                arguments.setdefault(name, []).extend(values)
    elif content_type.startswith('multipart/form-data'):
        if headers and 'Content-Encoding' in headers:
            gen_log.warning('Unsupported Content-Encoding: %s', headers['Content-Encoding'])
            return
        try:
            fields: List[str] = content_type.split(';')
            for field in fields:
                k, sep, v = field.strip().partition('=')
                if k == 'boundary' and v:
                    parse_multipart_form_data(utf8(v), body, arguments, files)
                    break
            else:
                raise ValueError('multipart boundary not found')
        except Exception as e:
            gen_log.warning('Invalid multipart/form-data: %s', e)


def parse_multipart_form_data(boundary: bytes, data: bytes, arguments: Dict[str, List[bytes]], files: Dict[str, List[HTTPFile]]) -> None:
    """Parses a ``multipart/form-data`` body.
    """
    if boundary.startswith(b'"') and boundary.endswith(b'"'):
        boundary = boundary[1:-1]
    final_boundary_index: int = data.rfind(b'--' + boundary + b'--')
    if final_boundary_index == -1:
        gen_log.warning('Invalid multipart/form-data: no final boundary')
        return
    parts: List[bytes] = data[:final_boundary_index].split(b'--' + boundary + b'\r\n')
    for part in parts:
        if not part:
            continue
        eoh: int = part.find(b'\r\n\r\n')
        if eoh == -1:
            gen_log.warning('multipart/form-data missing headers')
            continue
        headers: HTTPHeaders = HTTPHeaders.parse(part[:eoh].decode('utf-8'))
        disp_header: str = headers.get('Content-Disposition', '')
        disposition, disp_params = _parse_header(disp_header)
        if disposition != 'form-data' or not part.endswith(b'\r\n'):
            gen_log.warning('Invalid multipart/form-data')
            continue
        value: bytes = part[eoh + 4:-2]
        if not disp_params.get('name'):
            gen_log.warning('multipart/form-data value missing name')
            continue
        name: str = disp_params['name']
        if disp_params.get('filename'):
            ctype: str = headers.get('Content-Type', 'application/unknown')
            files.setdefault(name, []).append(HTTPFile(filename=disp_params['filename'], body=value, content_type=ctype))
        else:
            arguments.setdefault(name, []).append(value)


def format_timestamp(ts: Union[int, float, Tuple, time.struct_time, datetime.datetime]) -> str:
    """Formats a timestamp in the format used by HTTP.
    """
    if isinstance(ts, (int, float)):
        time_num: float = ts  # type: ignore
    elif isinstance(ts, (tuple, time.struct_time)):
        time_num = calendar.timegm(ts)
    elif isinstance(ts, datetime.datetime):
        time_num = calendar.timegm(ts.utctimetuple())
    else:
        raise TypeError('unknown timestamp type: %r' % ts)
    return email.utils.formatdate(time_num, usegmt=True)


class RequestStartLine(NamedTuple):
    method: str
    path: str
    version: str


_http_version_re: Pattern[str] = re.compile('^HTTP/1\\.[0-9]$')


def parse_request_start_line(line: str) -> RequestStartLine:
    """Returns a (method, path, version) tuple for an HTTP 1.x request line.
    """
    try:
        method, path, version = line.split(' ')
    except ValueError:
        raise HTTPInputError('Malformed HTTP request line')
    if not _http_version_re.match(version):
        raise HTTPInputError('Malformed HTTP version in HTTP Request-Line: %r' % version)
    return RequestStartLine(method, path, version)


class ResponseStartLine(NamedTuple):
    version: str
    code: int
    reason: str


_http_response_line_re: Pattern[str] = re.compile('(HTTP/1.[0-9]) ([0-9]+) ([^\\r]*)')


def parse_response_start_line(line: Union[str, bytes]) -> ResponseStartLine:
    """Returns a (version, code, reason) tuple for an HTTP 1.x response line.
    """
    line_str: str = native_str(line)
    match = _http_response_line_re.match(line_str)
    if not match:
        raise HTTPInputError('Error parsing response start line')
    return ResponseStartLine(match.group(1), int(match.group(2)), match.group(3))


def _parseparam(s: str) -> Iterator[str]:
    while s[:1] == ';':
        s = s[1:]
        end: int = s.find(';')
        while end > 0 and (s.count('"', 0, end) - s.count('\\"', 0, end)) % 2:
            end = s.find(';', end + 1)
        if end < 0:
            end = len(s)
        f: str = s[:end]
        yield f.strip()
        s = s[end:]


def _parse_header(line: str) -> Tuple[str, Dict[str, str]]:
    """Parse a Content-type like header.
    """
    parts: Iterator[str] = _parseparam(';' + line)
    key: str = next(parts)
    params: List[Tuple[str, str]] = [('Dummy', 'value')]
    for p in parts:
        i: int = p.find('=')
        if i >= 0:
            name: str = p[:i].strip().lower()
            value: str = p[i + 1:].strip()
            params.append((name, native_str(value)))
    decoded_params: List[Tuple[str, str]] = email.utils.decode_params(params)
    decoded_params.pop(0)
    pdict: Dict[str, str] = {}
    for name, decoded_value in decoded_params:
        value: str = email.utils.collapse_rfc2231_value(decoded_value)
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        pdict[name] = value
    return (key, pdict)


def _encode_header(key: str, pdict: Dict[str, Optional[Union[int, str]]]) -> str:
    """Inverse of _parse_header.
    """
    if not pdict:
        return key
    out: List[str] = [key]
    for k, v in sorted(pdict.items()):
        if v is None:
            out.append(k)
        else:
            out.append(f'{k}={v}')
    return '; '.join(out)


def encode_username_password(username: Union[str, unicode_type], password: Union[str, unicode_type]) -> bytes:
    """Encodes a username/password pair in the format used by HTTP auth.
    """
    if isinstance(username, unicode_type):
        username = unicodedata.normalize('NFC', username)
    if isinstance(password, unicode_type):
        password = unicodedata.normalize('NFC', password)
    return utf8(username) + b':' + utf8(password)


def doctests() -> Any:
    import doctest
    return doctest.DocTestSuite()


_netloc_re: Pattern[str] = re.compile('^(.+):(\\d+)$')


def split_host_and_port(netloc: str) -> Tuple[str, Optional[int]]:
    """Returns ``(host, port)`` tuple from ``netloc``.
    """
    match = _netloc_re.match(netloc)
    if match:
        host: str = match.group(1)
        port: int = int(match.group(2))
    else:
        host = netloc
        port = None
    return (host, port)


def qs_to_qsl(qs: Dict[str, List[str]]) -> Iterator[Tuple[str, str]]:
    """Generator converting a result of ``parse_qs`` back to name-value pairs.
    """
    for k, vs in qs.items():
        for v in vs:
            yield (k, v)


_unquote_sub: Any = re.compile(r'\\(?:([0-3][0-7][0-7])|(.))').sub

def _unquote_replace(m: re.Match) -> str:
    if m.group(1):
        return chr(int(m.group(1), 8))
    else:
        return m.group(2)  # type: ignore


def _unquote_cookie(s: Optional[str]) -> Optional[str]:
    """Handle double quotes and escaping in cookie values.
    """
    if s is None or len(s) < 2:
        return s
    if s[0] != '"' or s[-1] != '"':
        return s
    s_inner: str = s[1:-1]
    return _unquote_sub(_unquote_replace, s_inner)


def parse_cookie(cookie: str) -> Dict[str, str]:
    """Parse a ``Cookie`` HTTP header into a dict of name/value pairs.
    """
    cookiedict: Dict[str, str] = {}
    for chunk in cookie.split(';'):
        if '=' in chunk:
            key, val = chunk.split('=', 1)
        else:
            key, val = ('', chunk)
        key = key.strip()
        val = val.strip()
        if key or val:
            cookiedict[key] = _unquote_cookie(val) or val
    return cookiedict
