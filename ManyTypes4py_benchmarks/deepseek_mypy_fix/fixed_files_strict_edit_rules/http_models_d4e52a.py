"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""
import datetime
import codecs
from typing import (
    Mapping, Callable, Optional, Union, List, Tuple, Dict, Any, Iterable, 
    Iterator, AsyncIterator, cast, TypeVar, Generic, IO, BinaryIO, TextIO
)
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

T = TypeVar('T')
REDIRECT_STATI: Tuple[int, ...] = (codes['moved'], codes['found'], codes['other'], codes['temporary_redirect'], codes['permanent_redirect'])
DEFAULT_REDIRECT_LIMIT: int = 30
CONTENT_CHUNK_SIZE: int = 10 * 1024
ITER_CHUNK_SIZE: int = 512

class RequestEncodingMixin(object):

    @property
    def path_url(self) -> str:
        """Build the path URL to use."""
        url = []
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
    def _encode_params(data: Union[str, bytes, Iterable, Mapping]) -> Union[str, bytes]:
        """Encode parameters in a piece of data."""
        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, 'read'):
            return cast(Union[str, bytes], data)
        elif hasattr(data, '__iter__'):
            result = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, '__iter__'):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        result.append((k.encode('utf-8') if isinstance(k, str) else k, v.encode('utf-8') if isinstance(v, str) else v))
            return urlencode(result, doseq=True)
        else:
            return cast(Union[str, bytes], data)

    @staticmethod
    def _encode_files(files: Dict[str, Any], data: Optional[Union[Dict, List[Tuple[str, str]]]]) -> Tuple[bytes, str]:
        """Build the body for a multipart/form-data request."""
        if not files:
            raise ValueError('Files must be provided.')
        elif isinstance(data, basestring):
            raise ValueError('Data must not be a string.')
        new_fields = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})
        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, '__iter__'):
                val = [val]
            for v in val:
                if v is not None:
                    if not isinstance(v, bytes):
                        v = str(v)
                    new_fields.append((field.decode('utf-8') if isinstance(field, bytes) else field, v.encode('utf-8') if isinstance(v, str) else v))
        for k, v in files:
            ft = None
            fh = None
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

class RequestHooksMixin(object):

    def register_hook(self, event: str, hook: Union[Callable, Iterable[Callable]]) -> None:
        """Properly register a hook."""
        if event not in self.hooks:
            raise ValueError('Unsupported event specified, with event name "%s"' % event)
        if isinstance(hook, Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, '__iter__'):
            self.hooks[event].extend((h for h in hook if isinstance(h, Callable)))

    def deregister_hook(self, event: str, hook: Callable) -> bool:
        """Deregister a previously registered hook."""
        try:
            self.hooks[event].remove(hook)
            return True
        except ValueError:
            return False

class Request(RequestHooksMixin):
    __slots__ = ('method', 'url', 'headers', 'files', 'data', 'params', 'auth', 'cookies', 'hooks', 'json')

    def __init__(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict, List[Tuple[str, str]], str, bytes]] = None,
        params: Optional[Union[Dict, List[Tuple[str, str]]]] = None,
        auth: Optional[Union[Tuple[str, str], HTTPBasicAuth]] = None,
        cookies: Optional[Union[Dict[str, str], cookielib.CookieJar]] = None,
        hooks: Optional[Dict[str, List[Callable]]] = None,
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
        return '<Request [%s]>' % self.method

    def prepare(self) -> 'PreparedRequest':
        """Constructs a PreparedRequest for transmission and returns it."""
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
    __slots__ = ('method', 'url', 'headers', '_cookies', 'body', 'hooks', '_body_position')

    def __init__(self) -> None:
        self.method: Optional[str] = None
        self.url: Optional[str] = None
        self.headers: Optional[CaseInsensitiveDict] = None
        self._cookies: Optional[cookielib.CookieJar] = None
        self.body: Optional[Union[bytes, str, Iterable[bytes]]] = None
        self.hooks = default_hooks()
        self._body_position: Optional[int] = None

    def prepare(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict, List[Tuple[str, str]], str, bytes]] = None,
        params: Optional[Union[Dict, List[Tuple[str, str]]]] = None,
        auth: Optional[Union[Tuple[str, str], HTTPBasicAuth]] = None,
        cookies: Optional[Union[Dict[str, str], cookielib.CookieJar]] = None,
        hooks: Optional[Dict[str, List[Callable]]] = None,
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

    def prepare_url(self, url: Optional[str], params: Optional[Union[Dict, List[Tuple[str, str]]]], validate: bool = False) -> None:
        """Prepares the given HTTP URL."""
        if url is None:
            raise ValueError('URL cannot be None')
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
            error = 'Invalid URL {0!r}: No scheme supplied. Perhaps you meant http://{0}?'
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

    def prepare_headers(self, headers: Optional[Dict[str, str]]) -> None:
        """Prepares the given HTTP headers."""
        self.headers = CaseInsensitiveDict()
        if headers:
            for header in headers.items():
                check_header_validity(header)
                name, value = header
                self.headers[to_native_string(name)] = value

    def prepare_body(self, data: Optional[Union[Dict, List[Tuple[str, str]], str, bytes]], files: Optional[Dict[str, Any]], json: Optional[Any] = None) -> None:
        """Prepares the given HTTP body data."""
        body = None
        content_type = None
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

    def prepare_content_length(self, body: Optional[Union[bytes, str, Iterable[bytes]]]) -> None:
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

    def prepare_auth(self, auth: Optional[Union[Tuple[str, str], HTTPBasicAuth]], url: str = '') -> None:
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

    def prepare_cookies(self, cookies: Optional[Union[Dict[str, str], cookielib.CookieJar]]) -> None:
        """Prepares the given HTTP cookie data."""
        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)
        cookie_header = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            self.headers['Cookie'] = cookie_header

    def prepare_hooks(self, hooks: Optional[Dict[str, List[Callable]]]) -> None:
        """Prepares the given hooks."""
        hooks = hooks or []
        for event in hooks:
            self.register_hook(event, hooks[event])

    def send(self, session: Optional[requests.Session] = None, **send_kwargs: Any) -> 'Response':
        """Sends the PreparedRequest to the given Session."""
        session = requests.Session() if session is None else session
        with session:
            return session.send(self, **send_kwargs)

class Response(object):
    __attrs__ = ['_content', 'status_code', 'headers', 'url', 'history', 'encoding', 'reason', 'cookies', 'elapsed', 'request', 'protocol']
    __slots__ = __attrs__ + ['_content_consumed', 'raw', '_next', 'connection']

    def __init__(self) -> None:
        self._content: Union[bool, bytes, None] = False
        self._content_consumed: bool = False
        self._next: Optional[PreparedRequest] = None
        self.status_code: Optional[int] = None
        self.headers: CaseInsensitiveDict = CaseInsensitiveDict()
        self.raw: Optional[Any] = None
        self.url: Optional[str] = None
        self.encoding: Optional[str] = None
        self.h