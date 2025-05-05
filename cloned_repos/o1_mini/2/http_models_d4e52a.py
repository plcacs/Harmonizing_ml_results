import datetime
import codecs
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Iterable, Iterator
import rfc3986
import encodings.idna
from .core._http.fields import RequestField
from .core._http.filepost import encode_multipart_formdata
from .core._http.exceptions import DecodeError, ReadTimeoutError, ProtocolError, LocationParseError
from io import UnsupportedOperation
from ._hooks import default_hooks
from ._structures import CaseInsensitiveDict
import requests3 as requests
from .http_auth import HTTPBasicAuth
from .http_cookies import cookiejar_from_dict, get_cookie_header, _copy_cookie_jar
from .exceptions import HTTPError, MissingScheme, InvalidURL, ChunkedEncodingError, ContentDecodingError, ConnectionError, StreamConsumedError, InvalidHeader, InvalidBodyError, ReadTimeout
from ._internal_utils import to_native_string, unicode_is_ascii
from .http_utils import guess_filename, get_auth_from_url, requote_uri, stream_decode_response_unicode, to_key_val_list, parse_header_links, iter_slices, guess_json_utf, super_len, check_header_validity, is_stream
from ._basics import cookielib, urlunparse, urlsplit, urlencode, str, bytes, chardet, builtin_str, basestring
import json as complexjson
from .http_stati import codes

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
    def _encode_params(data: Union[Dict[str, Any], List[Tuple[str, Any]], str, bytes, Any]) -> Union[str, bytes]:
        """Encode parameters in a piece of data."""
        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, 'read'):
            return data
        elif hasattr(data, '__iter__'):
            result: List[Tuple[Union[bytes, str], Union[bytes, str]]] = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, '__iter__'):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        k_encoded = k.encode('utf-8') if isinstance(k, str) else k
                        v_encoded = v.encode('utf-8') if isinstance(v, str) else v
                        result.append((k_encoded, v_encoded))
            return urlencode(result, doseq=True)
        else:
            return data

    @staticmethod
    def _encode_files(files: Union[Dict[str, Any], List[Tuple[str, Any]]], data: Optional[Union[Dict[str, Any], List[Tuple[str, Any]]]]) -> Tuple[bytes, str]:
        """Build the body for a multipart/form-data request."""
        if not files:
            raise ValueError('Files must be provided.')
        elif isinstance(data, basestring):
            raise ValueError('Data must not be a string.')
        new_fields: List[Any] = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})
        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, '__iter__'):
                val = [val]
            for v in val:
                if v is not None:
                    if not isinstance(v, bytes):
                        v = str(v)
                    field_final = field.decode('utf-8') if isinstance(field, bytes) else field
                    v_final = v.encode('utf-8') if isinstance(v, str) else v
                    new_fields.append((field_final, v_final))
        for k, v in files:
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
                fdata = fp.read()
            elif fp is None:
                continue
            else:
                fdata = fp
            rf = RequestField(name=k, data=fdata, filename=fn, headers=fh)
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)
        body, content_type = encode_multipart_formdata(new_fields)
        return (body, content_type)


class RequestHooksMixin:
    def register_hook(self, event: str, hook: Union[Callable[..., Any], List[Callable[..., Any]]]) -> None:
        """Properly register a hook."""
        if event not in self.hooks:
            raise ValueError(f'Unsupported event specified, with event name "{event}"')
        if isinstance(hook, Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, '__iter__'):
            self.hooks[event].extend([h for h in hook if isinstance(h, Callable)])

    def deregister_hook(self, event: str, hook: Callable[..., Any]) -> bool:
        """Deregister a previously registered hook. Returns True if the hook existed, False if not."""
        try:
            self.hooks[event].remove(hook)
            return True
        except ValueError:
            return False


class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object."""
    __slots__ = ('method', 'url', 'headers', 'files', 'data', 'params', 'auth', 'cookies', 'hooks', 'json')

    method: Optional[str]
    url: Optional[Union[str, bytes]]
    headers: Dict[str, Any]
    files: Any
    data: Any
    params: Any
    auth: Any
    cookies: Any
    hooks: Dict[str, List[Callable[..., Any]]]
    json: Any

    def __init__(
        self,
        method: Optional[str] = None,
        url: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, Any]] = None,
        files: Optional[Any] = None,
        data: Optional[Any] = None,
        params: Optional[Any] = None,
        auth: Optional[Any] = None,
        cookies: Optional[Any] = None,
        hooks: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None
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
            hooks=self.hooks
        )
        return p


class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):
    """The fully mutable :class:`PreparedRequest <PreparedRequest>` object."""
    __slots__ = ('method', 'url', 'headers', '_cookies', 'body', 'hooks', '_body_position')

    method: Optional[str]
    url: Optional[str]
    headers: CaseInsensitiveDict
    _cookies: Optional[cookielib.CookieJar]
    body: Optional[Union[str, bytes, Any]]
    hooks: Dict[str, List[Callable[..., Any]]]
    _body_position: Optional[Any]

    def __init__(self) -> None:
        self.method = None
        self.url = None
        self.headers = CaseInsensitiveDict()
        self._cookies = None
        self.body = None
        self.hooks = default_hooks()
        self._body_position = None

    def prepare(
        self,
        method: Optional[str] = None,
        url: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, Any]] = None,
        files: Optional[Any] = None,
        data: Optional[Any] = None,
        params: Optional[Any] = None,
        auth: Optional[Any] = None,
        cookies: Optional[Any] = None,
        hooks: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None
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
        p._cookies = _copy_cookie_jar(self._cookies)
        p.body = self.body
        p.hooks = self.hooks
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

    def prepare_url(self, url: Optional[Union[str, bytes]], params: Optional[Any], validate: bool = False) -> None:
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
            error = f'Invalid URL {to_native_string(url, "utf8")!r}: No scheme supplied. Perhaps you meant http://{to_native_string(url, "utf8")}?'
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

    def prepare_headers(self, headers: Optional[Dict[str, Any]]) -> None:
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
        json: Optional[Any] = None
    ) -> None:
        """Prepares the given HTTP body data."""
        body: Optional[Union[str, bytes, Any]] = None
        content_type: Optional[str] = None
        if not data and json is not None:
            content_type = 'application/json'
            body = complexjson.dumps(json)
            if not isinstance(body, bytes):
                body = body.encode('utf-8')
        is_stream = all([hasattr(data, '__iter__'), not isinstance(data, (basestring, list, tuple, Mapping))])
        try:
            length = super_len(data)
        except (TypeError, AttributeError, UnsupportedOperation):
            length = None
        if is_stream:
            body = data
            if getattr(body, 'tell', None) is not None:
                try:
                    self._body_position = body.tell()
                except (IOError, OSError):
                    self._body_position = object()
            if files:
                raise NotImplementedError('Streamed bodies and files are mutually exclusive.')
        else:
            if files:
                body, content_type = self._encode_files(files, data)
            elif data:
                body = self._encode_params(data)
                if isinstance(data, basestring) or hasattr(data, 'read'):
                    content_type = None
                else:
                    content_type = 'application/x-www-form-urlencoded'
            if content_type and 'content-type' not in self.headers:
                self.headers['Content-Type'] = content_type
        self.prepare_content_length(body)
        self.body = body

    def prepare_content_length(self, body: Optional[Union[str, bytes, Any]]) -> None:
        """Prepares Content-Length header."""
        if body is not None:
            length = super_len(body)
            if length:
                self.headers['Content-Length'] = builtin_str(length)
            elif is_stream(body):
                self.headers['Transfer-Encoding'] = 'chunked'
            else:
                raise InvalidBodyError('Non-null body must have length or be streamable.')
        elif self.method not in ('GET', 'HEAD') and self.headers.get('Content-Length') is None:
            self.headers['Content-Length'] = '0'
        if 'Transfer-Encoding' in self.headers and 'Content-Length' in self.headers:
            raise InvalidHeader('Conflicting Headers: Both Transfer-Encoding and Content-Length are set.')

    def prepare_auth(self, auth: Optional[Any], url: Optional[Union[str, bytes]] = None) -> None:
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
        """Prepares the given HTTP cookie data."""
        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)
        cookie_header = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            self.headers['Cookie'] = cookie_header

    def prepare_hooks(self, hooks: Optional[Dict[str, Any]]) -> None:
        """Prepares the given hooks."""
        hooks = hooks or {}
        for event in hooks:
            self.register_hook(event, hooks[event])

    def send(self, session: Optional['requests.Session'] = None, **send_kwargs: Any) -> 'requests.Response':
        """Sends the PreparedRequest to the given Session. If none is provided, one is created for you."""
        session = requests.Session() if session is None else session
        with session:
            return session.send(self, **send_kwargs)


class Response:
    """The :class:`Response <Response>` object, which contains a server's response to an HTTP request."""
    __attrs__ = ['_content', 'status_code', 'headers', 'url', 'history', 'encoding', 'reason', 'cookies', 'elapsed', 'request', 'protocol']
    __slots__ = __attrs__ + ['_content_consumed', 'raw', '_next', 'connection']

    _content: Union[bytes, bool]
    _content_consumed: bool
    _next: Optional['PreparedRequest']
    status_code: Optional[int]
    headers: CaseInsensitiveDict
    url: Optional[str]
    history: List['Response']
    encoding: Optional[str]
    reason: Optional[Union[str, bytes]]
    cookies: cookielib.CookieJar
    elapsed: datetime.timedelta
    request: Optional['PreparedRequest']
    protocol: Optional[str]
    raw: Optional[Any]
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
        return f'<Response status={self.status_code} authority={self.uri.authority!r} protocol={self.protocol!r} elapsed={self.elapsed.microseconds:_}ms>'

    def __iter__(self) -> Iterator[bytes]:
        """Allows you to use a response as an iterator."""
        return self.iter_content(128)

    @property
    def uri(self) -> rfc3986.URIReference:
        return rfc3986.urlparse(self.url)

    @property
    def ok(self) -> bool:
        """Returns True if :attr:`status_code` is less than 400, False if not."""
        try:
            self.raise_for_status()
        except HTTPError:
            return False
        return True

    @property
    def is_redirect(self) -> bool:
        """True if this Response is a well-formed HTTP redirect."""
        return 'location' in self.headers and self.status_code in REDIRECT_STATI

    @property
    def is_permanent_redirect(self) -> bool:
        """True if this Response one of the permanent versions of redirect."""
        return 'location' in self.headers and self.status_code in (codes.moved_permanently, codes.permanent_redirect)

    @property
    def next(self) -> Optional['PreparedRequest']:
        """Returns a PreparedRequest for the next request in a redirect chain, if there is one."""
        return self._next

    @property
    def apparent_encoding(self) -> Optional[str]:
        """The apparent encoding, provided by the chardet library."""
        return chardet.detect(self.content)['encoding']

    def iter_content(self, decode_unicode: bool = False) -> Iterator[bytes]:
        """Iterates over the response data."""
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
                    chunk = self.raw.read(DEFAULT_CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        reused_chunks = iter_slices(self._content, DEFAULT_CHUNK_SIZE)
        stream_chunks = generate()
        chunks = reused_chunks if self._content_consumed else stream_chunks
        if decode_unicode:
            if self.encoding is None:
                raise TypeError('encoding must be set before consuming streaming responses')
            codecs.lookup(self.encoding)  # type: ignore
            chunks = stream_decode_response_unicode(chunks, self)
        return chunks

    def iter_lines(
        self,
        chunk_size: int = ITER_CHUNK_SIZE,
        decode_unicode: bool = False,
        delimiter: Optional[Union[str, bytes]] = None
    ) -> Iterator[Union[str, bytes]]:
        """Iterates over the response data, one line at a time."""
        carriage_return = '\r' if decode_unicode else b'\r'
        line_feed = '\n' if decode_unicode else b'\n'
        pending: Optional[Union[str, bytes]] = None
        last_chunk_ends_with_cr = False
        for chunk in self.iter_content(chunk_size=chunk_size, decode_unicode=decode_unicode):
            if not chunk:
                continue
            if pending is not None:
                chunk = pending + chunk  # type: ignore
                pending = None
            if delimiter:
                lines = chunk.split(delimiter)
            else:
                skip_first_char = last_chunk_ends_with_cr and chunk.startswith(line_feed)  # type: ignore
                last_chunk_ends_with_cr = chunk.endswith(carriage_return)  # type: ignore
                if skip_first_char:
                    chunk = chunk[1:]
                    if not chunk:
                        continue
                lines = chunk.splitlines()
            incomplete_line = bool(lines) and bool(lines[-1] and lines[-1][-1] == chunk[-1])
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
        """Content of the response, in unicode."""
        content: Optional[Union[str, bytes]] = None
        encoding = self.encoding
        if not self.content:
            return ''
        if self.encoding is None:
            encoding = self.apparent_encoding
        try:
            content = str(self.content, encoding, errors='replace')  # type: ignore
        except (LookupError, TypeError):
            content = str(self.content, errors='replace')  # type: ignore
        return content

    def json(self, **kwargs: Any) -> Any:
        """Returns the json-encoded content of a response, if any."""
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
                l[key] = link
        return l

    def raise_for_status(self) -> 'Response':
        """Raises stored :class:`HTTPError`, if one occurred. Otherwise, returns the response object (self)."""
        http_error_msg = ''
        if isinstance(self.reason, bytes):
            try:
                reason = self.reason.decode('utf-8')
            except UnicodeDecodeError:
                reason = self.reason.decode('iso-8859-1')
        else:
            reason = self.reason
        if 400 <= self.status_code < 500:
            http_error_msg = f'{self.status_code} Client Error: {reason} for url: {self.url}'
        elif 500 <= self.status_code < 600:
            http_error_msg = f'{self.status_code} Server Error: {reason} for url: {self.url}'
        if http_error_msg:
            raise HTTPError(http_error_msg, response=self)
        return self

    def close(self) -> None:
        """Releases the connection back to the pool."""
        if not self._content_consumed and self.raw is not None:
            self.raw.close()
        release_conn = getattr(self.raw, 'release_conn', None)
        if release_conn is not None:
            release_conn()


class AsyncResponse(Response):
    """Asynchronous version of the :class:`Response <Response>` object."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def json(self, **kwargs: Any) -> Any:
        """Returns the json-encoded content of a response, if any."""
        if not self.encoding and await self.content and (len(await self.content) > 3):
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
        """Content of the response, in unicode."""
        content: Optional[Union[str, bytes]] = None
        encoding = self.encoding
        if not await self.content:
            return ''
        if self.encoding is None:
            encoding = self.apparent_encoding
        try:
            content = str(await self.content, encoding, errors='replace')  # type: ignore
        except (LookupError, TypeError):
            content = str(await self.content, errors='replace')  # type: ignore
        return content

    @property
    async def content(self) -> Optional[bytes]:
        """Content of the response, in bytes."""
        if self._content is False:
            if self._content_consumed:
                raise RuntimeError('The content for this response was already consumed')
            if self.status_code == 0 or self.raw is None:
                self._content = None
            else:
                self._content = bytes().join([await self.iter_content()]) or bytes()
        self._content_consumed = True
        return self._content

    @property
    async def apparent_encoding(self) -> Optional[str]:
        """The apparent encoding, provided by the chardet library."""
        return chardet.detect(await self.content)['encoding']

    async def iter_content(self, decode_unicode: bool = False) -> Iterator[bytes]:
        """Asynchronously iterates over the response data."""
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
                    chunk = await self.raw.read(DEFAULT_CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        reused_chunks = iter_slices(self._content, DEFAULT_CHUNK_SIZE)
        try:
            stream_chunks = await generate().__anext__()
        except StopAsyncIteration:
            stream_chunks = None
        chunks = reused_chunks if self._content_consumed else stream_chunks
        if decode_unicode:
            if self.encoding is None:
                raise TypeError('encoding must be set before consuming streaming responses')
            codecs.lookup(self.encoding)  # type: ignore
            chunks = stream_decode_response_unicode(chunks, self)
        return chunks
