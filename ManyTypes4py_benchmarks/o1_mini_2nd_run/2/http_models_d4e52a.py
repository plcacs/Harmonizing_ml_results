"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""
import datetime
import codecs
from typing import (
    Mapping,
    Callable,
    Union,
    List,
    Tuple,
    Any,
    Optional,
    Dict,
    Iterator,
    IO,
)
import rfc3986
import encodings.idna
from .core._http.fields import RequestField
from .core._http.filepost import encode_multipart_formdata
from .core._http.exceptions import (
    DecodeError,
    ReadTimeoutError,
    ProtocolError,
    LocationParseError,
)
from io import UnsupportedOperation
from ._hooks import default_hooks
from ._structures import CaseInsensitiveDict
import requests3 as requests
from .http_auth import HTTPBasicAuth
from .http_cookies import (
    cookiejar_from_dict,
    get_cookie_header,
    _copy_cookie_jar,
)
from .exceptions import (
    HTTPError,
    MissingScheme,
    InvalidURL,
    ChunkedEncodingError,
    ContentDecodingError,
    ConnectionError,
    StreamConsumedError,
    InvalidHeader,
    InvalidBodyError,
    ReadTimeout,
)
from ._internal_utils import (
    to_native_string,
    unicode_is_ascii,
)
from .http_utils import (
    guess_filename,
    get_auth_from_url,
    requote_uri,
    stream_decode_response_unicode,
    to_key_val_list,
    parse_header_links,
    iter_slices,
    guess_json_utf,
    super_len,
    check_header_validity,
    is_stream,
)
from ._basics import (
    cookielib,
    urlunparse,
    urlsplit,
    urlencode,
    str,
    bytes,
    chardet,
    builtin_str,
    basestring,
)
import json as complexjson
from .http_stati import codes
from .exceptions import (
    HTTPError,
    MissingScheme,
    InvalidURL,
    ChunkedEncodingError,
    ContentDecodingError,
    ConnectionError,
    StreamConsumedError,
    InvalidHeader,
    InvalidBodyError,
    ReadTimeout,
)

REDIRECT_STATI: Tuple[int, ...] = (
    codes['moved'],
    codes['found'],
    codes['other'],
    codes['temporary_redirect'],
    codes['permanent_redirect'],
)
DEFAULT_REDIRECT_LIMIT: int = 30
CONTENT_CHUNK_SIZE: int = 10 * 1024
ITER_CHUNK_SIZE: int = 512


class RequestEncodingMixin:
    @property
    def path_url(self) -> str:
        """Build the path URL to use."""
        url: List[str] = []
        p = urlsplit(self.url)
        path = p.path
        if not path:
            path = '/'
        url.append(path)
        query = p.query
        if query:
            url.append('?')
            url.append(query)
        return ''.join(url)

    @staticmethod
    def _encode_params(data: Any) -> Union[str, bytes, IO]:
        """Encode parameters in a piece of data.

        Will successfully encode parameters when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.
        """
        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, 'read'):
            return data  # type: ignore
        elif hasattr(data, '__iter__'):
            result: List[Tuple[Union[str, bytes], Union[str, bytes]]] = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, '__iter__'):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        key = k.encode('utf-8') if isinstance(k, str) else k
                        value = v.encode('utf-8') if isinstance(v, str) else v
                        result.append((key, value))
            return urlencode(result, doseq=True)
        else:
            return data

    @staticmethod
    def _encode_files(
        files: Any,
        data: Any,
    ) -> Tuple[bytes, str]:
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        tuples. Order is retained if data is a list of tuples but arbitrary
        if parameters are supplied as a dict.
        The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype)
        or 4-tuples (filename, fileobj, contentype, custom_headers).
        """
        if not files:
            raise ValueError('Files must be provided.')
        elif isinstance(data, basestring):
            raise ValueError('Data must not be a string.')
        new_fields: List[Union[Tuple[Union[str, bytes], Union[str, bytes]], RequestField]] = []
        fields = to_key_val_list(data or {})
        files_list = to_key_val_list(files or {})
        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, '__iter__'):
                val = [val]
            for v in val:
                if v is not None:
                    if not isinstance(v, bytes):
                        v = str(v)
                    key = (
                        field.decode('utf-8') if isinstance(field, bytes) else field
                    )
                    value = (
                        v.encode('utf-8') if isinstance(v, str) else v
                    )
                    new_fields.append((key, value))
        for k, v in files_list:
            ft: Optional[str] = None
            fh: Optional[Dict[str, str]] = None
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    fn, fp = v
                elif len(v) == 3:
                    fn, fp, ft = v
                else:
                    fn, fp, ft, fh = v
            else:
                fn = guess_filename(v) or k
                fp = v
            if isinstance(fp, (str, bytes, bytearray)):
                fdata = fp
            elif hasattr(fp, 'read'):
                fdata = fp.read()  # type: ignore
            elif fp is None:
                continue
            else:
                fdata = fp
            rf = RequestField(
                name=k,
                data=fdata,
                filename=fn,
                headers=fh,
            )
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)
        body, content_type = encode_multipart_formdata(new_fields)
        return body, content_type


class RequestHooksMixin:
    hooks: Dict[str, List[Callable[..., Any]]]

    def register_hook(
        self,
        event: str,
        hook: Union[Callable[..., Any], List[Callable[..., Any]]],
    ) -> None:
        """Properly register a hook."""
        if event not in self.hooks:
            raise ValueError(
                f'Unsupported event specified, with event name "{event}"'
            )
        if isinstance(hook, Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, '__iter__'):
            self.hooks[event].extend(
                h for h in hook if isinstance(h, Callable)
            )

    def deregister_hook(
        self,
        event: str,
        hook: Callable[..., Any],
    ) -> bool:
        """Deregister a previously registered hook.
        Returns True if the hook existed, False if not.
        """
        try:
            self.hooks[event].remove(hook)
            return True
        except ValueError:
            return False


class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object.

    Used to prepare a :class:`PreparedRequest <PreparedRequest>`, which is sent to the server.

    :param method: HTTP method to use.
    :param url: URL to send.
    :param headers: dictionary of headers to send.
    :param files: dictionary of {filename: fileobject} files to multipart upload.
    :param data: the body to attach to the request. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param json: json for the body to attach to the request (if files or data is not specified).
    :param params: URL parameters to append to the URL. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param auth: Auth handler or (user, pass) tuple.
    :param cookies: dictionary or CookieJar of cookies to attach to this request.
    :param hooks: dictionary of callback hooks, for internal usage.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> req.prepare()
      <PreparedRequest [GET]>
    """
    __slots__ = (
        'method',
        'url',
        'headers',
        'files',
        'data',
        'params',
        'auth',
        'cookies',
        'hooks',
        'json',
    )

    method: Optional[str]
    url: Optional[Union[str, bytes]]
    headers: Dict[str, str]
    files: Any
    data: Any
    params: Any
    auth: Any
    cookies: Any
    json: Any

    def __init__(
        self,
        method: Optional[str] = None,
        url: Optional[Union[str, bytes]] = None,
        headers: Optional[Mapping[str, str]] = None,
        files: Optional[Any] = None,
        data: Optional[Any] = None,
        params: Optional[Any] = None,
        auth: Optional[Any] = None,
        cookies: Optional[Any] = None,
        hooks: Optional[Mapping[str, Union[Callable[..., Any], List[Callable[..., Any]]]]] = None,
        json: Optional[Any] = None,
    ) -> None:
        data = [] if data is None else data
        files = [] if files is None else files
        headers = {} if headers is None else headers
        params = {} if params is None else params
        hooks = {} if hooks is None else hooks
        self.hooks = default_hooks()
        for k, v in list(hooks.items()):
            self.register_hook(event=k, hook=v)
        self.method = method
        self.url = url
        self.headers = headers
        self.files = files
        self.data = data
        self.json = json
        self.params = params
        self.auth = auth
        self.cookies = cookies

    def __repr__(self) -> str:
        return f'<Request [{self.method}]>'

    def prepare(self) -> 'PreparedRequest':
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for transmission and returns it."""
        p = PreparedRequest()
        p.prepare(
            method=self.method,
            url=self.url,
            headers=self.headers,
            files=self.files,
            data=self.data,
            json=self.json,
            params=self.params,
            auth=self.auth,
            cookies=self.cookies,
            hooks=self.hooks,
        )
        return p


class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):
    """The fully mutable :class:`PreparedRequest <PreparedRequest>` object,
    containing the exact bytes that will be sent to the server.

    Generated from either a :class:`Request <Request>` object or manually.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> r = req.prepare()
      <PreparedRequest [GET]>

      >>> s = requests.Session()
      >>> s.send(r)
      <Response [200]>
    """
    __slots__ = (
        'method',
        'url',
        'headers',
        '_cookies',
        'body',
        'hooks',
        '_body_position',
    )

    method: Optional[str]
    url: Optional[str]
    headers: Optional[CaseInsensitiveDict]
    _cookies: Optional[cookielib.CookieJar]
    body: Optional[Union[str, bytes, IO]]
    hooks: Dict[str, List[Callable[..., Any]]]
    _body_position: Optional[Any]

    def __init__(self) -> None:
        self.method = None
        self.url = None
        self.headers = None
        self._cookies = None
        self.body = None
        self.hooks = default_hooks()
        self._body_position = None

    def prepare(
        self,
        method: Optional[str] = None,
        url: Optional[Union[str, bytes]] = None,
        headers: Optional[Mapping[str, str]] = None,
        files: Optional[Any] = None,
        data: Optional[Any] = None,
        params: Optional[Any] = None,
        auth: Optional[Any] = None,
        cookies: Optional[Any] = None,
        hooks: Optional[Mapping[str, Union[Callable[..., Any], List[Callable[..., Any]]]]] = None,
        json: Optional[Any] = None,
    ) -> None:
        """Prepares the entire request with the given parameters."""
        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files, json)
        self.prepare_auth(auth, url)
        self.prepare_hooks(hooks)

    def __repr__(self) -> str:
        return f'<PreparedRequest [{self.method}]>'

    def copy(self) -> 'PreparedRequest':
        p = PreparedRequest()
        p.method = self.method
        p.url = self.url
        p.headers = self.headers.copy() if self.headers is not None else None
        p._cookies = _copy_cookie_jar(self._cookies) if self._cookies else None
        p.body = self.body
        p.hooks = self.hooks.copy()
        p._body_position = self._body_position
        return p

    def prepare_method(self, method: Optional[str]) -> None:
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is None:
            raise ValueError('Request method cannot be "None"')
        self.method = to_native_string(self.method.upper())

    @staticmethod
    def _get_idna_encoded_host(host: str) -> str:
        import idna
        try:
            host = idna.encode(host, uts46=True).decode('utf-8')
        except idna.IDNAError:
            raise UnicodeError
        return host

    def prepare_url(
        self,
        url: Optional[Union[str, bytes]],
        params: Optional[Any],
        validate: bool = False,
    ) -> None:
        """Prepares the given HTTP URL."""
        if isinstance(url, bytes):
            url = url.decode('utf8')
        else:
            url = str(url)
        url = url.strip()
        if ':' in url and (not url.lower().startswith('http')):
            self.url = url
            return
        try:
            uri = rfc3986.urlparse(url)
            if validate:
                rfc3986.normalize_uri(url)
        except rfc3986.exceptions.RFC3986Exception:
            raise InvalidURL(f'Invalid URL {url!r}: URL is imporoper.')
        if not uri.scheme:
            error = (
                'Invalid URL {0!r}: No scheme supplied. Perhaps you meant http://{0}?'
            )
            error = error.format(to_native_string(url, 'utf8'))
            raise MissingScheme(error)
        if not uri.host:
            raise InvalidURL(f'Invalid URL {url!r}: No host supplied')
        if not unicode_is_ascii(uri.host):
            try:
                uri = uri.copy_with(host=self._get_idna_encoded_host(uri.host))
            except UnicodeError:
                raise InvalidURL('URL has an invalid label.')
        elif uri.host.startswith('*'):
            raise InvalidURL('URL has an invalid label.')
        if not uri.path:
            uri = uri.copy_with(path='/')
        if isinstance(params, (str, bytes)):
            params = to_native_string(params)
        enc_params = self._encode_params(params)
        if enc_params:
            if uri.query:
                uri = uri.copy_with(query=f'{uri.query}&{enc_params}')
            else:
                uri = uri.copy_with(query=enc_params)
        self.url = rfc3986.normalize_uri(uri.unsplit())

    def prepare_headers(self, headers: Optional[Mapping[str, str]]) -> None:
        """Prepares the given HTTP headers."""
        self.headers = CaseInsensitiveDict()
        if headers:
            for header in headers.items():
                check_header_validity(header)
                name, value = header
                self.headers[to_native_string(name)] = value

    def prepare_body(
        self,
        data: Optional[Any],
        files: Optional[Any],
        json: Optional[Any] = None,
    ) -> None:
        """Prepares the given HTTP body data."""
        body: Optional[Union[str, bytes, IO]] = None
        content_type: Optional[str] = None
        if not data and json is not None:
            content_type = 'application/json'
            body = complexjson.dumps(json)
            if not isinstance(body, bytes):
                body = body.encode('utf-8')
        is_stream = all(
            [
                hasattr(data, '__iter__'),
                not isinstance(data, (basestring, list, tuple, Mapping)),
            ]
        )
        try:
            length = super_len(data)
        except (TypeError, AttributeError, UnsupportedOperation):
            length = None
        if is_stream:
            body = data  # type: ignore
            if getattr(body, 'tell', None) is not None:
                try:
                    self._body_position = body.tell()
                except (IOError, OSError):
                    self._body_position = object()
            if files:
                raise NotImplementedError(
                    'Streamed bodies and files are mutually exclusive.'
                )
        else:
            if files:
                body, content_type = self._encode_files(files, data)
            elif data:
                body = self._encode_params(data)
                if isinstance(data, basestring) or hasattr(data, 'read'):
                    content_type = None
                else:
                    content_type = 'application/x-www-form-urlencoded'
            if content_type and self.headers and 'content-type' not in self.headers:
                self.headers['Content-Type'] = content_type
        self.prepare_content_length(body)
        self.body = body

    def prepare_content_length(
        self,
        body: Optional[Union[str, bytes, IO]],
    ) -> None:
        """Prepares Content-Length header.

        If the length of the body of the request can be computed, Content-Length
        is set using ``super_len``. If user has manually set either a
        Transfer-Encoding or Content-Length header when it should not be set
        (they should be mutually exclusive) an InvalidHeader
        error will be raised.
        """
        if body is not None:
            length = super_len(body)
            if length:
                assert self.headers is not None
                self.headers['Content-Length'] = builtin_str(length)
            elif is_stream(body):
                assert self.headers is not None
                self.headers['Transfer-Encoding'] = 'chunked'
            else:
                raise InvalidBodyError(
                    'Non-null body must have length or be streamable.'
                )
        elif self.method not in ('GET', 'HEAD') and self.headers and 'Content-Length' not in self.headers:
            assert self.headers is not None
            self.headers['Content-Length'] = '0'
        if self.headers and 'Transfer-Encoding' in self.headers and 'Content-Length' in self.headers:
            raise InvalidHeader(
                'Conflicting Headers: Both Transfer-Encoding and Content-Length are set.'
            )

    def prepare_auth(
        self,
        auth: Optional[Any],
        url: Union[str, bytes] = '',
    ) -> None:
        """Prepares the given HTTP auth data."""
        if auth is None:
            url_auth = get_auth_from_url(self.url)
            auth = url_auth if any(url_auth) else None
        if auth:
            if isinstance(auth, tuple) and len(auth) == 2:
                auth = HTTPBasicAuth(*auth)
            r = auth(self)
            self.__dict__.update(r.__dict__)
            self.prepare_content_length(self.body)

    def prepare_cookies(self, cookies: Optional[Any]) -> None:
        """Prepares the given HTTP cookie data.

        This function eventually generates a ``Cookie`` header from the
        given cookies using cookielib. Due to cookielib's design, the header
        will not be regenerated if it already exists, meaning this function
        can only be called once for the life of the
        :class:`PreparedRequest <PreparedRequest>` object. Any subsequent calls
        to ``prepare_cookies`` will have no actual effect, unless the "Cookie"
        header is removed beforehand.
        """
        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)
        cookie_header = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            assert self.headers is not None
            self.headers['Cookie'] = cookie_header

    def prepare_hooks(
        self,
        hooks: Optional[Mapping[str, Union[Callable[..., Any], List[Callable[..., Any]]]]],
    ) -> None:
        """Prepares the given hooks."""
        hooks = hooks or {}
        for event, hook in hooks.items():
            self.register_hook(event, hook)

    def send(
        self,
        session: Optional['requests.Session'] = None,
        **send_kwargs: Any,
    ) -> 'Response':
        """Sends the PreparedRequest to the given Session.
        If none is provided, one is created for you."""
        session = requests.Session() if session is None else session
        with session:
            return session.send(self, **send_kwargs)


class Response:
    """The :class:`Response <Response>` object, which contains a
    server's response to an HTTP request.
    """
    __attrs__ = [
        '_content',
        'status_code',
        'headers',
        'url',
        'history',
        'encoding',
        'reason',
        'cookies',
        'elapsed',
        'request',
        'protocol',
    ]
    __slots__ = __attrs__ + ['_content_consumed', 'raw', '_next', 'connection']

    _content: Optional[Union[bytes, bool]]
    status_code: Optional[int]
    headers: CaseInsensitiveDict
    url: Optional[str]
    history: List['Response']
    encoding: Optional[str]
    reason: Optional[Union[str, bytes]]
    cookies: cookielib.CookieJar
    elapsed: datetime.timedelta
    request: Optional[PreparedRequest]
    protocol: Optional[str]
    _content_consumed: bool
    raw: Optional[Any]
    _next: Optional[PreparedRequest]
    connection: Optional[Any]

    def __init__(self) -> None:
        self._content = False
        self._content_consumed = False
        self._next = None
        self.status_code = None
        self.headers = CaseInsensitiveDict()
        self.raw = None
        self.url = None
        self.encoding = None
        self.history = []
        self.reason = None
        self.cookies = cookiejar_from_dict({})
        self.elapsed = datetime.timedelta(0)
        self.request = None
        self.protocol = None

    def __enter__(self) -> 'Response':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __getstate__(self) -> Dict[str, Any]:
        if not self._content_consumed:
            _ = self.content
        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        setattr(self, '_content_consumed', True)
        setattr(self, 'raw', None)

    def __repr__(self) -> str:
        authority = self.uri.authority if self.url else None
        elapsed_ms = self.elapsed.microseconds / 1000
        return (
            f'<Response status={self.status_code} authority={authority!r} '
            f'protocol={self.protocol!r} elapsed={elapsed_ms:.3f}ms>'
        )

    def __iter__(self) -> Iterator[bytes]:
        """Allows you to use a response as an iterator."""
        return self.iter_content(128)

    @property
    def uri(self) -> rfc3986.uri.URIReference:
        return rfc3986.urlparse(self.url)

    @property
    def ok(self) -> bool:
        """Returns True if :attr:`status_code` is less than 400, False if not.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        try:
            self.raise_for_status()
        except HTTPError:
            return False
        return True

    @property
    def is_redirect(self) -> bool:
        """True if this Response is a well-formed HTTP redirect that could have
        been processed automatically (by :meth:`Session.resolve_redirects`).
        """
        return 'location' in self.headers and self.status_code in REDIRECT_STATI

    @property
    def is_permanent_redirect(self) -> bool:
        """True if this Response one of the permanent versions of redirect."""
        return (
            'location' in self.headers
            and self.status_code in (codes.moved_permanently, codes.permanent_redirect)
        )

    @property
    def next(self) -> Optional[PreparedRequest]:
        """Returns a PreparedRequest for the next request in a redirect chain, if there is one."""
        return self._next

    @property
    def apparent_encoding(self) -> Optional[str]:
        """The apparent encoding, provided by the chardet library."""
        detect_result = chardet.detect(self.content) if self.content else None
        return detect_result['encoding'] if detect_result else None

    def iter_content(
        self,
        chunk_size: int = CONTENT_CHUNK_SIZE,
        decode_unicode: bool = False,
    ) -> Iterator[Union[bytes, str]]:
        """Iterates over the response data.  When stream=True is set on the
        request, this avoids reading the content at once into memory for
        large responses.  The chunk size is the number of bytes it should
        read into memory.  This is not necessarily the length of each item
        returned as decoding can take place.

        chunk_size must be of type int or None. A value of None will
        function differently depending on the value of `stream`.
        stream=True will read data as it arrives in whatever size the
        chunks are received. If stream=False, data is returned as
        a single chunk.

        If using decode_unicode, the encoding must be set to a valid encoding
        enumeration before invoking iter_content.
        """
        DEFAULT_CHUNK_SIZE = 1

        def generate() -> Iterator[bytes]:
            if hasattr(self.raw, 'stream'):
                try:
                    for chunk in self.raw.stream():
                        yield chunk
                except ProtocolError as e:
                    if self.headers.get('Transfer-Encoding') == 'chunked':
                        raise ChunkedEncodingError(e)
                    else:
                        raise ConnectionError(e)
                except DecodeError as e:
                    raise ContentDecodingError(e)
                except ReadTimeoutError as e:
                    raise ReadTimeout(e)
            else:
                while True:
                    chunk = self.raw.read(DEFAULT_CHUNK_SIZE)  # type: ignore
                    if not chunk:
                        break
                    yield chunk
            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        reused_chunks: Iterator[bytes] = iter_slices(
            self._content, DEFAULT_CHUNK_SIZE
        ) if isinstance(self._content, bytes) else iter([])
        stream_chunks = generate()
        chunks: Iterator[bytes] = reused_chunks if self._content_consumed else stream_chunks
        if decode_unicode:
            if self.encoding is None:
                raise TypeError(
                    'encoding must be set before consuming streaming responses'
                )
            codecs.lookup(self.encoding)
            chunks = stream_decode_response_unicode(chunks, self)
        return chunks

    def iter_lines(
        self,
        chunk_size: int = ITER_CHUNK_SIZE,
        decode_unicode: bool = False,
        delimiter: Optional[Union[str, bytes]] = None,
    ) -> Iterator[Union[str, bytes]]:
        """Iterates over the response data, one line at a time.  When
        stream=True is set on the request, this avoids reading the
        content at once into memory for large responses.

        .. note:: This method is not reentrant safe.
        """
        carriage_return: Union[str, bytes] = '\r' if decode_unicode else b'\r'
        line_feed: Union[str, bytes] = '\n' if decode_unicode else b'\n'
        pending: Optional[Union[str, bytes]] = None
        last_chunk_ends_with_cr: bool = False
        for chunk in self.iter_content(chunk_size=chunk_size, decode_unicode=decode_unicode):
            if not chunk:
                continue
            if pending is not None:
                chunk = pending + chunk  # type: ignore
                pending = None
            if delimiter:
                lines = chunk.split(delimiter)
            else:
                skip_first_char = last_chunk_ends_with_cr and chunk.startswith(line_feed)
                last_chunk_ends_with_cr = chunk.endswith(carriage_return)
                if skip_first_char:
                    chunk = chunk[1:]
                    if not chunk:
                        continue
                lines = chunk.splitlines()
            incomplete_line = bool(lines) and (lines[-1] == chunk[-1:])
            if delimiter or incomplete_line:
                pending = lines.pop() if lines else None
            for line in lines:
                yield line
        if pending is not None:
            yield pending

    @property
    def content(self) -> Optional[bytes]:
        """Content of the response, in bytes."""
        if self._content is False:
            if self._content_consumed:
                raise RuntimeError('The content for this response was already consumed')
            if self.status_code == 0 or self.raw is None:
                self._content = None
            else:
                self._content = bytes().join(self.iter_content()) or bytes()
        self._content_consumed = True
        return self._content

    @property
    def text(self) -> str:
        """Content of the response, in unicode.

        If Response.encoding is None, encoding will be guessed using
        ``chardet``.

        The encoding of the response content is determined based solely on HTTP
        headers, following RFC 2616 to the letter. If you can take advantage of
        non-HTTP knowledge to make a better guess at the encoding, you should
        set ``r.encoding`` appropriately before accessing this property.
        """
        content: Optional[Union[str, bytes]] = None
        encoding: Optional[str] = self.encoding
        if not self.content:
            return str('')
        if self.encoding is None:
            encoding = self.apparent_encoding
        try:
            content = str(self.content, encoding, errors='replace') if encoding else str(self.content, errors='replace')  # type: ignore
        except (LookupError, TypeError):
            content = str(self.content, errors='replace')
        return content  # type: ignore

    def json(self, **kwargs: Any) -> Any:
        """Returns the json-encoded content of a response, if any.

        :param \\*\\*kwargs: Optional arguments that ``json.loads`` takes.
        :raises ValueError: If the response body does not contain valid json.
        """
        if not self.encoding and self.content and (len(self.content) > 3):
            encoding = guess_json_utf(self.content)
            if encoding is not None:
                try:
                    content = self.content
                    return complexjson.loads(content.decode(encoding), **kwargs)
                except UnicodeDecodeError:
                    pass
        return complexjson.loads(self.text, **kwargs)

    @property
    def links(self) -> Dict[str, Dict[str, Any]]:
        """Returns the parsed header links of the response, if any."""
        header = self.headers.get('link')
        l: Dict[str, Dict[str, Any]] = {}
        if header:
            links = parse_header_links(header)
            for link in links:
                key = link.get('rel') or link.get('url')
                if key:
                    l[key] = link
        return l

    def raise_for_status(self) -> 'Response':
        """Raises stored :class:`HTTPError`, if one occurred.
        Otherwise, returns the response object (self)."""
        http_error_msg = ''
        if isinstance(self.reason, bytes):
            try:
                reason = self.reason.decode('utf-8')
            except UnicodeDecodeError:
                reason = self.reason.decode('iso-8859-1')
        else:
            reason = self.reason
        if self.status_code is not None:
            if 400 <= self.status_code < 500:
                http_error_msg = f'{self.status_code} Client Error: {reason} for url: {self.url}'
            elif 500 <= self.status_code < 600:
                http_error_msg = f'{self.status_code} Server Error: {reason} for url: {self.url}'
        if http_error_msg:
            raise HTTPError(http_error_msg, response=self)
        return self

    def close(self) -> None:
        """Releases the connection back to the pool. Once this method has been
        called the underlying ``raw`` object must not be accessed again.

        *Note: Should not normally need to be called explicitly.*
        """
        if not self._content_consumed and self.raw is not None:
            self.raw.close()
        release_conn = getattr(self.raw, 'release_conn', None)
        if release_conn is not None:
            release_conn()


class AsyncResponse(Response):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def json(self, **kwargs: Any) -> Any:
        """Returns the json-encoded content of a response, if any.

        :param \\*\\*kwargs: Optional arguments that ``json.loads`` takes.
        :raises ValueError: If the response body does not contain valid json.
        """
        if not self.encoding and self.content and (len(await self.content) > 3):
            encoding = guess_json_utf(await self.content)
            if encoding is not None:
                try:
                    content = await self.content
                    return complexjson.loads(content.decode(encoding), **kwargs)
                except UnicodeDecodeError:
                    pass
        return complexjson.loads(await self.text, **kwargs)

    @property
    async def text(self) -> str:
        """Content of the response, in unicode.

        If Response.encoding is None, encoding will be guessed using
        ``chardet``.

        The encoding of the response content is determined based solely on HTTP
        headers, following RFC 2616 to the letter. If you can take advantage of
        non-HTTP knowledge to make a better guess at the encoding, you should
        set ``r.encoding`` appropriately before accessing this property.
        """
        content: Optional[str] = None
        encoding: Optional[str] = self.encoding
        if not await self.content:
            return str('')
        if self.encoding is None:
            encoding = self.apparent_encoding
        try:
            content = str(await self.content, encoding, errors='replace') if encoding else str(await self.content, errors='replace')  # type: ignore
        except (LookupError, TypeError):
            content = str(await self.content, errors='replace')
        return content  # type: ignore

    @property
    async def content(self) -> Optional[bytes]:
        """Content of the response, in bytes."""
        if self._content is False:
            if self._content_consumed:
                raise RuntimeError('The content for this response was already consumed')
            if self.status_code == 0 or self.raw is None:
                self._content = None
            else:
                chunks = []
                async for chunk in self.iter_content():
                    chunks.append(chunk)
                self._content = b''.join(chunks) or b''
        self._content_consumed = True
        return self._content

    @property
    async def apparent_encoding(self) -> Optional[str]:
        """The apparent encoding, provided by the chardet library."""
        detected = chardet.detect(await self.content) if self.content else None
        return detected['encoding'] if detected else None

    async def iter_content(
        self,
        chunk_size: int = CONTENT_CHUNK_SIZE,
        decode_unicode: bool = False,
    ) -> Iterator[Union[bytes, str]]:
        """Iterates over the response data.  When stream=True is set on the
        request, this avoids reading the content at once into memory for
        large responses.  The chunk size is the number of bytes it should
        read into memory.  This is not necessarily the length of each item
        returned as decoding can take place.

        chunk_size must be of type int or None. A value of None will
        function differently depending on the value of `stream`.
        stream=True will read data as it arrives in whatever size the
        chunks are received. If stream=False, data is returned as
        a single chunk.

        If using decode_unicode, the encoding must be set to a valid encoding
        enumeration before invoking iter_content.
        """
        DEFAULT_CHUNK_SIZE = 1

        async def generate() -> Iterator[bytes]:
            if hasattr(self.raw, 'stream'):
                try:
                    async for chunk in self.raw.stream():
                        yield chunk
                except ProtocolError as e:
                    if self.headers.get('Transfer-Encoding') == 'chunked':
                        raise ChunkedEncodingError(e)
                    else:
                        raise ConnectionError(e)
                except DecodeError as e:
                    raise ContentDecodingError(e)
                except ReadTimeoutError as e:
                    raise ReadTimeout(e)
            else:
                while True:
                    chunk = await self.raw.read(DEFAULT_CHUNK_SIZE)  # type: ignore
                    if not chunk:
                        break
                    yield chunk
            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        reused_chunks: Iterator[bytes] = iter_slices(
            self._content, DEFAULT_CHUNK_SIZE
        ) if isinstance(self._content, bytes) else iter([])
        try:
            stream_chunks = generate().__anext__()
        except StopAsyncIteration:
            stream_chunks = None
        chunks: Iterator[bytes] = reused_chunks if self._content_consumed else iter([])
        if decode_unicode:
            if self.encoding is None:
                raise TypeError(
                    'encoding must be set before consuming streaming responses'
                )
            codecs.lookup(self.encoding)
            chunks = stream_decode_response_unicode(chunks, self)
        return chunks
