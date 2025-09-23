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
from typing import Tuple, Iterable, List, Mapping, Iterator, Dict, Union, Optional, Awaitable, Generator, AnyStr, Any, cast
if typing.TYPE_CHECKING:
    from typing import Deque
    from asyncio import Future
    import unittest
    StrMutableMapping = collections.abc.MutableMapping[str, str]
else:
    StrMutableMapping = collections.abc.MutableMapping
HTTP_WHITESPACE = ' \t'

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

    >>> h = HTTPHeaders({"content-type": "text/html"})
    >>> list(h.keys())
    ['Content-Type']
    >>> h["Content-Type"]
    'text/html'

    >>> h.add("Set-Cookie", "A=B")
    >>> h.add("Set-Cookie", "C=D")
    >>> h["set-cookie"]
    'A=B,C=D'
    >>> h.get_list("set-cookie")
    ['A=B', 'C=D']

    >>> for (k,v) in sorted(h.get_all()):
    ...    print('%s: %s' % (k,v))
    ...
    Content-Type: text/html
    Set-Cookie: A=B
    Set-Cookie: C=D
    """

    @typing.overload
    def __init__(self, __arg: Mapping[str, List[str]]) -> None:
        pass

    @typing.overload
    def __init__(self, __arg: Mapping[str, str]) -> None:
        pass

    @typing.overload
    def __init__(self, *args: Tuple[str, str]) -> None:
        pass

    @typing.overload
    def __init__(self, **kwargs: str) -> None:
        pass

    def __init__(self, *args: typing.Any, **kwargs: str) -> None:
        self._dict: Dict[str, str] = {}
        self._as_list: Dict[str, List[str]] = {}
        self._last_key: Optional[str] = None
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], HTTPHeaders):
            for (k, v) in args[0].get_all():
                self.add(k, v)
        else:
            self.update(*args, **kwargs)

    def add(self, name: str, value: str) -> None:
        """Adds a new value for the given key."""
        norm_name = _normalize_header(name)
        self._last_key = norm_name
        if norm_name in self:
            self._dict[norm_name] = native_str(self[norm_name]) + ',' + native_str(value)
            self._as_list[norm_name].append(value)
        else:
            self[norm_name] = value

    def get_list(self, name: str) -> List[str]:
        """Returns all values for the given header as a list."""
        norm_name = _normalize_header(name)
        return self._as_list.get(norm_name, [])

    def get_all(self) -> Iterable[Tuple[str, str]]:
        """Returns an iterable of all (name, value) pairs.

        If a header has multiple values, multiple pairs will be
        returned with the same name.
        """
        for (name, values) in self._as_list.items():
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
            new_part = ' ' + line.lstrip(HTTP_WHITESPACE)
            self._as_list[self._last_key][-1] += new_part
            self._dict[self._last_key] += new_part
        else:
            try:
                (name, value) = line.split(':', 1)
            except ValueError:
                raise HTTPInputError('no colon in header line')
            self.add(name, value.strip(HTTP_WHITESPACE))

    @classmethod
    def parse(cls, headers: str) -> 'HTTPHeaders':
        """Returns a dictionary from HTTP header text.

        >>> h = HTTPHeaders.parse("Content-Type: text/html\\r\\nContent-Length: 42\\r\\n")
        >>> sorted(h.items())
        [('Content-Length', '42'), ('Content-Type', 'text/html')]

        .. versionchanged:: 5.1

           Raises `HTTPInputError` on malformed headers instead of a
           mix of `KeyError`, and `ValueError`.

        """
        h = cls()
        for line in headers.split('\n'):
            if line.endswith('\r'):
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

    def copy(self) -> 'HTTPHeaders':
        return HTTPHeaders(self)
    __copy__ = copy

    def __str__(self) -> str:
        lines = []
        for (name, value) in self.get_all():
            lines.append(f'{name}: {value}\n')
        return ''.join(lines)
    __unicode__ = __str__

class HTTPServerRequest:
    """A single HTTP request.

    All attributes are type `str` unless otherwise noted.

    .. attribute:: method

       HTTP request method, e.g. "GET" or "POST"

    .. attribute:: uri

       The requested uri.

    .. attribute:: path

       The path portion of `uri`

    .. attribute:: query

       The query portion of `uri`

    .. attribute:: version

       HTTP version specified in request, e.g. "HTTP/1.1"

    .. attribute:: headers

       `.HTTPHeaders` dictionary-like object for request headers.  Acts like
       a case-insensitive dictionary with additional methods for repeated
       headers.

    .. attribute:: body

       Request body, if present, as a byte string.

    .. attribute:: remote_ip

       Client's IP address as a string.  If ``HTTPServer.xheaders`` is set,
       will pass along the real IP address provided by a load balancer
       in the ``X-Real-Ip`` or ``X-Forwarded-For`` header.

    .. versionchanged:: 3.1
       The list format of ``X-Forwarded-For`` is now supported.

    .. attribute:: protocol

       The protocol used, either "http" or "https".  If ``HTTPServer.xheaders``
       is set, will pass along the protocol used by a load balancer if
       reported via an ``X-Scheme`` header.

    .. attribute:: host

       The requested hostname, usually taken from the ``Host`` header.

    .. attribute:: arguments

       GET/POST arguments are available in the arguments property, which
       maps arguments names to lists of values (to support multiple values
       for individual names). Names are of type `str`, while arguments
       are byte strings.  Note that this is different from
       `.RequestHandler.get_argument`, which returns argument values as
       unicode strings.

    .. attribute:: query_arguments

       Same format as ``arguments``, but contains only arguments extracted
       from the query string.

       .. versionadded:: 3.2

    .. attribute:: body_arguments

       Same format as ``arguments``, but contains only arguments extracted
       from the request body.

       .. versionadded:: 3.2

    .. attribute:: files

       File uploads are available in the files property, which maps file
       names to lists of `.HTTPFile`.

    .. attribute:: connection

       An HTTP request is attached to a single HTTP connection, which can
       be accessed through the "connection" attribute. Since connections
       are typically kept open in HTTP/1.1, multiple requests can be handled
       sequentially on a single connection.

    .. versionchanged:: 4.0
       Moved from ``tornado.httpserver.HTTPRequest``.
    """
    path: Optional[str] = None
    query: Optional[str] = None
    _body_future: Optional['Future[bytes]'] = None

    def __init__(self, method: Optional[str]=None, uri: Optional[str]=None, version: str='HTTP/1.0', headers: Optional[HTTPHeaders]=None, body: Optional[bytes]=None, host: Optional[str]=None, files: Optional[Dict[str, List['HTTPFile']]]=None, connection: Optional['HTTPConnection']=None, start_line: Optional['RequestStartLine']=None, server_connection: Optional[object]=None) -> None:
        if start_line is not None:
            (method, uri, version) = start_line
        self.method = method
        self.uri = uri
        self.version = version
        self.headers = headers or HTTPHeaders()
        self.body = body or b''
        context = getattr(connection, 'context', None)
        self.remote_ip = getattr(context, 'remote_ip', None)
        self.protocol = getattr(context, 'protocol', 'http')
        self.host = host or self.headers.get('Host') or '127.0.0.1'
        self.host_name = split_host_and_port(self.host.lower())[0]
        self.files = files or {}
        self.connection = connection
        self.server_connection = server_connection
        self._start_time = time.time()
        self._finish_time: Optional[float] = None
        if uri is not None:
            (self.path, sep, self.query) = uri.partition('?')
        self.arguments: Dict[str, List[bytes]] = parse_qs_bytes(self.query, keep_blank_values=True)
        self.query_arguments: Dict[str, List[bytes]] = copy.deepcopy(self.arguments)
        self.body_arguments: Dict[str, List[bytes]] = {}

    @property
    def cookies(self) -> Dict[str, http.cookies.Morsel]:
        """A dictionary of ``http.cookies.Morsel`` objects."""
        if not hasattr(self, '_cookies'):
            self._cookies = http.cookies.SimpleCookie()
            if 'Cookie' in self.headers:
                try:
                    parsed = parse_cookie(self.headers['Cookie'])
                except Exception:
                    pass
                else:
                    for (k, v) in parsed.items():
                        try:
                            self._cookies[k] = v
                        except Exception:
                            pass
        return self._cookies

    def full_url(self) -> str:
        """Reconstructs the full URL for this request."""
        return self.protocol + '://' + self.host + self.uri

    def request_time(self) -> float:
        """Returns the amount of time it took for this request to execute."""
        if self._finish_time is None:
            return time.time() - self._start_time
        else:
            return self._finish_time - self._start_time

    def get_ssl_certificate(self, binary_form: bool=False) -> Union[None, Dict, bytes]:
        """Returns the client's SSL certificate, if any.

        To use client certificates, the HTTPServer's
        `ssl.SSLContext.verify_mode` field must be set, e.g.::

            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain("foo.crt", "foo.key")
            ssl_ctx.load_verify_locations("cacerts.pem")
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            server = HTTPServer(app, ssl_options=ssl_ctx)

        By default, the return value is a dictionary (or None, if no
        client certificate is present).  If ``binary_form`` is true, a
        DER-encoded form of the certificate is returned instead.  See
        SSLSocket.getpeercert() in the standard library for more
        details.
        http://docs.python.org/library/ssl.html#sslsocket-objects
        """
        try:
            if self.connection is None:
                return None
            return self.connection.stream.socket.getpeercert(binary_form=binary_form)
        except SSLError:
            return None

    def _parse_body(self) -> None:
        parse_body_arguments(self.headers.get('Content-Type', ''), self.body, self.body_arguments, self.files, self.headers)
        for (k, v) in self.body_arguments.items():
            self.arguments.setdefault(k, []).extend(v)

    def __repr__(self) -> str:
        attrs = ('protocol', 'host', 'method', 'uri', 'version', 'remote_ip')
        args = ', '.join([f'{n}={getattr(self, n)!r}' for n in attrs])
        return f'{self.__class__.__name__}({args})'

class HTTPInputError(Exception):
    """Exception class for malformed HTTP requests or responses
    from remote sources.

    .. versionadded:: 4.0
    """
    pass

class HTTPOutputError(Exception):
    """Exception class for errors in HTTP output.

    .. versionadded:: 4.0
    """
    pass

class HTTPServerConnectionDelegate:
    """Implement this interface to handle requests from `.HTTPServer`.

    .. versionadded:: 4.0
    """

    def start_request(self, server_conn: object, request_conn: 'HTTPConnection') -> 'HTTPMessageDelegate':
        """This method is called by the server when a new request has started.

        :arg server_conn: is an opaque object representing the long-lived
            (e.g. tcp-level) connection.
        :arg request_conn: is a `.HTTPConnection` object for a single
            request/response exchange.

        This method should return a `.HTTPMessageDelegate`.
        """
        raise NotImplementedError()

    def on_close(self, server_conn: object) -> None:
        """This method is called when a connection has been closed.

        :arg server_conn: is a server connection that has previously been
            passed to ``start_request``.
        """
        pass

class HTTPMessageDelegate:
    """Implement this interface to handle an HTTP request or response.

    .. versionadded:: 4.0
    """

    def headers_received(self, start_line: Union['RequestStartLine', 'ResponseStartLine'], headers: HTTPHeaders) -> Optional[Awaitable[None]]:
        """Called when the HTTP headers have been received and parsed.

        :arg start_line: a `.RequestStartLine` or `.ResponseStartLine`
            depending on whether this is a client or server message.
        :arg headers: a `.HTTPHeaders` instance.

        Some `.HTTPConnection` methods can only be called during
        ``headers_received``.

        May return a `.Future`; if it does the body will not be read
        until it is done.
        """
        pass

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        """Called when a chunk of data has been received.

        May return a `.Future` for flow control.
        """
        pass

    def finish(self) -> None:
        """Called after the last chunk of data has been received."""
        pass

    def on_connection_close(self) -> None:
        """Called if the connection is closed without finishing the request.

        If ``headers_received`` is called, either ``finish`` or
        ``on_connection_close`` will be called, but not both.
        """
        pass

class HTTPConnection:
    """Applications use this interface to write their responses.

    .. versionadded:: 4.0
    """

    def write_headers(self, start_line: Union['RequestStartLine', 'ResponseStartLine'], headers: HTTPHeaders, chunk: Optional[bytes]=None) -> 'Future[None]':
        """Write an HTTP header block.

        :arg start_line: a `.RequestStartLine` or `.ResponseStartLine`.
        :arg headers: a `.HTTPHeaders` instance.
        :arg chunk: the first (optional) chunk of data.  This is an optimization
            so that small responses can be written in the same call as their
            headers.

        The ``version`` field of ``start_line`` is ignored.

        Returns a future for flow control.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed.
        """
        raise NotImplementedError()

    def write(self, chunk: bytes) -> 'Future[None]':
        """Writes a chunk of body data.

        Returns a future for flow control.

        .. versionchanged:: 6.0

           The ``