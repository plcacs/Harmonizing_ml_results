from __future__ import annotations
import codecs
import datetime
import email.message
import json as jsonlib
import re
import typing
import urllib.request
from collections.abc import Mapping
from http.cookiejar import Cookie, CookieJar
from ._content import ByteStream, UnattachedStream, encode_request, encode_response
from ._decoders import SUPPORTED_DECODERS, ByteChunker, ContentDecoder, IdentityDecoder, LineDecoder, MultiDecoder, TextChunker, TextDecoder
from ._exceptions import CookieConflict, HTTPStatusError, RequestNotRead, ResponseNotRead, StreamClosed, StreamConsumed, request_context
from ._multipart import get_multipart_boundary_from_content_type
from ._status_codes import codes
from ._types import AsyncByteStream, CookieTypes, HeaderTypes, QueryParamTypes, RequestContent, RequestData, RequestExtensions, RequestFiles, ResponseContent, ResponseExtensions, SyncByteStream
from ._urls import URL
from ._utils import to_bytes_or_str, to_str

__all__ = ['Cookies', 'Headers', 'Request', 'Response']
SENSITIVE_HEADERS = {'authorization', 'proxy-authorization'}

def _is_known_encoding(encoding: str) -> bool:
    try:
        codecs.lookup(encoding)
    except LookupError:
        return False
    return True

def _normalize_header_key(key: str | bytes, encoding: str | None = None) -> bytes:
    return key if isinstance(key, bytes) else key.encode(encoding or 'ascii')

def _normalize_header_value(value: str | bytes, encoding: str | None = None) -> bytes:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise TypeError(f'Header value must be str or bytes, not {type(value)}')
    return value.encode(encoding or 'ascii')

def _parse_content_type_charset(content_type: str) -> str | None:
    msg = email.message.Message()
    msg['content-type'] = content_type
    return msg.get_content_charset(failobj=None)

def _parse_header_links(value: str) -> list[dict[str, str]]:
    links = []
    replace_chars = ' \'"'
    value = value.strip(replace_chars)
    if not value:
        return links
    for val in re.split(', *<', value):
        try:
            url, params = val.split(';', 1)
        except ValueError:
            url, params = (val, '')
        link = {'url': url.strip('<> \'"')}
        for param in params.split(';'):
            try:
                key, value = param.split('=')
            except ValueError:
                break
            link[key.strip(replace_chars)] = value.strip(replace_chars)
        links.append(link)
    return links

def _obfuscate_sensitive_headers(items: typing.Iterable[tuple[bytes, bytes]]) -> typing.Generator[tuple[bytes, bytes], None, None]:
    for k, v in items:
        if to_str(k.lower()) in SENSITIVE_HEADERS:
            v = to_bytes_or_str('[secure]', match_type_of=v)
        yield (k, v)

class Headers(typing.MutableMapping[str, str]):
    def __init__(self, headers: HeaderTypes | None = None, encoding: str | None = None) -> None:
        self._list: list[tuple[bytes, bytes, bytes]] = []
        if isinstance(headers, Headers):
            self._list = list(headers._list)
        elif isinstance(headers, Mapping):
            for k, v in headers.items():
                bytes_key = _normalize_header_key(k, encoding)
                bytes_value = _normalize_header_value(v, encoding)
                self._list.append((bytes_key, bytes_key.lower(), bytes_value))
        elif headers is not None:
            for k, v in headers:
                bytes_key = _normalize_header_key(k, encoding)
                bytes_value = _normalize_header_value(v, encoding)
                self._list.append((bytes_key, bytes_key.lower(), bytes_value))
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        if self._encoding is None:
            for encoding in ['ascii', 'utf-8']:
                for key, value in self.raw:
                    try:
                        key.decode(encoding)
                        value.decode(encoding)
                    except UnicodeDecodeError:
                        break
                else:
                    self._encoding = encoding
                    break
            else:
                self._encoding = 'iso-8859-1'
        return self._encoding

    @encoding.setter
    def encoding(self, value: str) -> None:
        self._encoding = value

    @property
    def raw(self) -> list[tuple[bytes, bytes]]:
        return [(raw_key, value) for raw_key, _, value in self._list]

    def keys(self) -> typing.KeysView[str]:
        return {key.decode(self.encoding): None for _, key, value in self._list}.keys()

    def values(self) -> typing.ValuesView[str]:
        values_dict: dict[str, str] = {}
        for _, key, value in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            if str_key in values_dict:
                values_dict[str_key] += f', {str_value}'
            else:
                values_dict[str_key] = str_value
        return values_dict.values()

    def items(self) -> typing.ItemsView[str, str]:
        values_dict: dict[str, str] = {}
        for _, key, value in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            if str_key in values_dict:
                values_dict[str_key] += f', {str_value}'
            else:
                values_dict[str_key] = str_value
        return values_dict.items()

    def multi_items(self) -> list[tuple[str, str]]:
        return [(key.decode(self.encoding), value.decode(self.encoding)) for _, key, value in self._list]

    def get(self, key: str, default: str | None = None) -> str | None:
        try:
            return self[key]
        except KeyError:
            return default

    def get_list(self, key: str, split_commas: bool = False) -> list[str]:
        get_header_key = key.lower().encode(self.encoding)
        values = [item_value.decode(self.encoding) for _, item_key, item_value in self._list if item_key.lower() == get_header_key]
        if not split_commas:
            return values
        split_values = []
        for value in values:
            split_values.extend([item.strip() for item in value.split(',')])
        return split_values

    def update(self, headers: HeaderTypes | None = None) -> None:
        headers = Headers(headers)
        for key in headers.keys():
            if key in self:
                self.pop(key)
        self._list.extend(headers._list)

    def copy(self) -> Headers:
        return Headers(self, encoding=self.encoding)

    def __getitem__(self, key: str) -> str:
        normalized_key = key.lower().encode(self.encoding)
        items = [header_value.decode(self.encoding) for _, header_key, header_value in self._list if header_key == normalized_key]
        if items:
            return ', '.join(items)
        raise KeyError(key)

    def __setitem__(self, key: str, value: str) -> None:
        set_key = key.encode(self._encoding or 'utf-8')
        set_value = value.encode(self._encoding or 'utf-8')
        lookup_key = set_key.lower()
        found_indexes = [idx for idx, (_, item_key, _) in enumerate(self._list) if item_key == lookup_key]
        for idx in reversed(found_indexes[1:]):
            del self._list[idx]
        if found_indexes:
            idx = found_indexes[0]
            self._list[idx] = (set_key, lookup_key, set_value)
        else:
            self._list.append((set_key, lookup_key, set_value))

    def __delitem__(self, key: str) -> None:
        del_key = key.lower().encode(self.encoding)
        pop_indexes = [idx for idx, (_, item_key, _) in enumerate(self._list) if item_key.lower() == del_key]
        if not pop_indexes:
            raise KeyError(key)
        for idx in reversed(pop_indexes):
            del self._list[idx]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        header_key = key.lower().encode(self.encoding)
        return header_key in [key for _, key, _ in self._list]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Headers):
            try:
                other = Headers(other)
            except ValueError:
                return False
        self_list = [(key, value) for _, key, value in self._list]
        other_list = [(key, value) for _, key, value in other._list]
        return sorted(self_list) == sorted(other_list)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        encoding_str = ''
        if self.encoding != 'ascii':
            encoding_str = f', encoding={self.encoding!r}'
        as_list = list(_obfuscate_sensitive_headers(self.multi_items()))
        as_dict = dict(as_list)
        no_duplicate_keys = len(as_dict) == len(as_list)
        if no_duplicate_keys:
            return f'{class_name}({as_dict!r}{encoding_str})'
        return f'{class_name}({as_list!r}{encoding_str})'

class Request:
    def __init__(self, method: str, url: str, *, params: QueryParamTypes | None = None, headers: HeaderTypes | None = None, cookies: CookieTypes | None = None, content: RequestContent | None = None, data: RequestData | None = None, files: RequestFiles | None = None, json: typing.Any = None, stream: SyncByteStream | AsyncByteStream | None = None, extensions: RequestExtensions | None = None) -> None:
        self.method = method.upper()
        self.url = URL(url) if params is None else URL(url, params=params)
        self.headers = Headers(headers)
        self.extensions = {} if extensions is None else dict(extensions)
        if cookies:
            Cookies(cookies).set_cookie_header(self)
        if stream is None:
            content_type = self.headers.get('content-type')
            headers, stream = encode_request(content=content, data=data, files=files, json=json, boundary=get_multipart_boundary_from_content_type(content_type=content_type.encode(self.headers.encoding) if content_type else None))
            self._prepare(headers)
            self.stream = stream
            if isinstance(stream, ByteStream):
                self.read()
        else:
            self.stream = stream

    def _prepare(self, default_headers: HeaderTypes) -> None:
        for key, value in default_headers.items():
            if key.lower() == 'transfer-encoding' and 'Content-Length' in self.headers:
                continue
            self.headers.setdefault(key, value)
        auto_headers: list[tuple[bytes, str]] = []
        has_host = 'Host' in self.headers
        has_content_length = 'Content-Length' in self.headers or 'Transfer-Encoding' in self.headers
        if not has_host and self.url.host:
            auto_headers.append((b'Host', self.url.netloc))
        if not has_content_length and self.method in ('POST', 'PUT', 'PATCH'):
            auto_headers.append((b'Content-Length', b'0'))
        self.headers = Headers(auto_headers + self.headers.raw)

    @property
    def content(self) -> bytes:
        if not hasattr(self, '_content'):
            raise RequestNotRead()
        return self._content

    def read(self) -> bytes:
        if not hasattr(self, '_content'):
            assert isinstance(self.stream, typing.Iterable)
            self._content = b''.join(self.stream)
            if not isinstance(self.stream, ByteStream):
                self.stream = ByteStream(self._content)
        return self._content

    async def aread(self) -> bytes:
        if not hasattr(self, '_content'):
            assert isinstance(self.stream, typing.AsyncIterable)
            self._content = b''.join([part async for part in self.stream])
            if not isinstance(self.stream, ByteStream):
                self.stream = ByteStream(self._content)
        return self._content

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        url = str(self.url)
        return f'<{class_name}({self.method!r}, {url!r})>'

    def __getstate__(self) -> dict[str, typing.Any]:
        return {name: value for name, value in self.__dict__.items() if name not in ['extensions', 'stream']}

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.extensions = {}
        self.stream = UnattachedStream()

class Response:
    def __init__(self, status_code: int, *, headers: HeaderTypes | None = None, content: ResponseContent | None = None, text: str | None = None, html: str | None = None, json: typing.Any = None, stream: SyncByteStream | AsyncByteStream | None = None, request: Request | None = None, extensions: ResponseExtensions | None = None, history: list[Response] | None = None, default_encoding: str = 'utf-8') -> None:
        self.status_code = status_code
        self.headers = Headers(headers)
        self._request = request
        self.next_request: Request | None = None
        self.extensions = {} if extensions is None else dict(extensions)
        self.history = [] if history is None else list(history)
        self.is_closed = False
        self.is_stream_consumed = False
        self.default_encoding = default_encoding
        if stream is None:
            headers, stream = encode_response(content, text, html, json)
            self._prepare(headers)
            self.stream = stream
            if isinstance(stream, ByteStream):
                self.read()
        else:
            self.stream = stream
        self._num_bytes_downloaded = 0

    def _prepare(self, default_headers: HeaderTypes) -> None:
        for key, value in default_headers.items():
            if key.lower() == 'transfer-encoding' and 'content-length' in self.headers:
                continue
            self.headers.setdefault(key, value)

    @property
    def elapsed(self) -> datetime.timedelta:
        if not hasattr(self, '_elapsed'):
            raise RuntimeError("'.elapsed' may only be accessed after the response has been read or closed.")
        return self._elapsed

    @elapsed.setter
    def elapsed(self, elapsed: datetime.timedelta) -> None:
        self._elapsed = elapsed

    @property
    def request(self) -> Request:
        if self._request is None:
            raise RuntimeError('The request instance has not been set on this response.')
        return self._request

    @request.setter
    def request(self, value: Request) -> None:
        self._request = value

    @property
    def http_version(self) -> str:
        try:
            http_version = self.extensions['http_version']
        except KeyError:
            return 'HTTP/1.1'
        else:
            return http_version.decode('ascii', errors='ignore')

    @property
    def reason_phrase(self) -> str:
        try:
            reason_phrase = self.extensions['reason_phrase']
        except KeyError:
            return codes.get_reason_phrase(self.status_code)
        else:
            return reason_phrase.decode('ascii', errors='ignore')

    @property
    def url(self) -> URL:
        return self.request.url

    @property
    def content(self) -> bytes:
        if not hasattr(self, '_content'):
            raise ResponseNotRead()
        return self._content

    @property
    def text(self) -> str:
        if not hasattr(self, '_text'):
            content = self.content
            if not content:
                self._text = ''
            else:
                decoder = TextDecoder(encoding=self.encoding or 'utf-8')
                self._text = ''.join([decoder.decode(self.content), decoder.flush()])
        return self._text

    @property
    def encoding(self) -> str:
        if not hasattr(self, '_encoding'):
            encoding = self.charset_encoding
            if encoding is None or not _is_known_encoding(encoding):
                if isinstance(self.default_encoding, str):
                    encoding = self.default_encoding
                elif hasattr(self, '_content'):
                    encoding = self.default_encoding(self._content)
            self._encoding = encoding or 'utf-8'
        return self._encoding

    @encoding.setter
    def encoding(self, value: str) -> None:
        if hasattr(self, '_text'):
            raise ValueError('Setting encoding after `text` has been accessed is not allowed.')
        self._encoding = value

    @property
    def charset_encoding(self) -> str | None:
        content_type = self.headers.get('Content-Type')
        if content_type is None:
            return None
        return _parse_content_type_charset(content_type)

    def _get_content_decoder(self) -> ContentDecoder:
        if not hasattr(self, '_decoder'):
            decoders: list[ContentDecoder] = []
            values = self.headers.get_list('content-encoding', split_commas=True)
            for value in values:
                value = value.strip().lower()
                try:
                    decoder_cls = SUPPORTED_DECODERS[value]
                    decoders.append(decoder_cls())
                except KeyError:
                    continue
            if len(decoders) == 1:
                self._decoder = decoders[0]
            elif len(decoders) > 1:
                self._decoder = MultiDecoder(children=decoders)
            else:
                self._decoder = IdentityDecoder()
        return self._decoder

    @property
    def is_informational(self) -> bool:
        return codes.is_informational(self.status_code)

    @property
    def is_success(self) -> bool:
        return codes.is_success(self.status_code)

    @property
    def is_redirect(self) -> bool:
        return codes.is_redirect(self.status_code)

    @property
    def is_client_error(self) -> bool:
        return codes.is_client_error(self.status_code)

    @property
    def is_server_error(self) -> bool:
        return codes.is_server_error(self.status_code)

    @property
    def is_error(self) -> bool:
        return codes.is_error(self.status_code)

    @property
    def has_redirect_location(self) -> bool:
        return self.status_code in (codes.MOVED_PERMANENTLY, codes.FOUND, codes.SEE_OTHER, codes.TEMPORARY_REDIRECT, codes.PERMANENT_REDIRECT) and 'Location' in self.headers

    def raise_for_status(self) -> Response:
        request = self._request
        if request is None:
            raise RuntimeError('Cannot call `raise_for_status` as the request instance has not been set on this response.')
        if self.is_success:
            return self
        if self.has_redirect_location:
            message = "{error_type} '{0.status_code} {0.reason_phrase}' for url '{0.url}'\nRedirect location: '{0.headers[location]}'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{0.status_code}"
        else:
            message = "{error_type} '{0.status_code} {0.reason_phrase}' for url '{0.url}'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{0.status_code}"
        status_class = self.status_code // 100
        error_types = {1: 'Informational response', 3: 'Redirect response', 4: 'Client error', 5: 'Server error'}
        error_type = error_types.get(status_class, 'Invalid status code')
        message = message.format(self, error_type=error_type)
        raise HTTPStatusError(message, request=request, response=self)

    def json(self, **kwargs: typing.Any) -> typing.Any:
        return jsonlib.loads(self.content, **kwargs)

    @property
    def cookies(self) -> Cookies:
        if not hasattr(self, '_cookies'):
            self._cookies = Cookies()
            self._cookies.extract_cookies(self)
        return self._cookies

    @property
    def links(self) -> dict[str, dict[str, str]]:
        header = self.headers.get('link')
        if header is None:
            return {}
        return {link.get('rel') or link.get('url'): link for link in _parse_header_links(header)}

    @property
    def num_bytes_downloaded(self) -> int:
        return self._num_bytes_downloaded

    def __repr__(self) -> str:
        return f'<Response [{self.status_code} {self.reason_phrase}]>'

    def __getstate__(self) -> dict[str, typing.Any]:
        return {name: value for name, value in self.__dict__.items() if name not in ['extensions', 'stream', 'is_closed', '_decoder']}

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.is_closed = True
        self.extensions = {}
        self.stream = UnattachedStream()

    def read(self) -> bytes:
        if not hasattr(self, '_content'):
            self._content = b''.join(self.iter_bytes())
        return self._content

    def iter_bytes(self, chunk_size: int | None = None) -> typing.Iterator[bytes]:
        if hasattr(self, '_content'):
            chunk_size = len(self._content) if chunk_size is None else chunk_size
            for i in range(0, len(self._content), max(chunk_size, 1)):
                yield self._content[i:i + chunk_size]
        else:
            decoder = self._get_content_decoder()
            chunker = ByteChunker(chunk_size=chunk_size)
            with request_context(request=self._request):
                for raw_bytes in self.iter_raw():
                    decoded = decoder.decode(raw_bytes)
                    for chunk in chunker.decode(decoded):
                        yield chunk
                decoded = decoder.flush()
                for chunk in chunker.decode(decoded):
                    yield chunk
                for chunk in chunker.flush():
                    yield chunk

    def iter_text(self, chunk_size: int | None = None) -> typing.Iterator[str]:
        decoder = TextDecoder(encoding=self.encoding or 'utf-8')
        chunker = TextChunker(chunk_size=chunk_size)
        with request_context(request=self._request):
            for byte_content in self.iter_bytes():
                text_content = decoder.decode(byte_content)
                for chunk in chunker.decode(text_content):
                    yield chunk
            text_content = decoder.flush()
            for chunk in chunker.decode(text_content):
                yield chunk
            for chunk in chunker.flush():
                yield chunk

    def iter_lines(self) -> typing.Iterator[str]:
        decoder = LineDecoder()
        with request_context(request=self._request):
            for text in self.iter_text():
                for line in decoder.decode(text):
                    yield line
            for line in decoder.flush():
                yield line

    def iter_raw(self, chunk_size: int | None = None) -> typing.Iterator[bytes]:
        if self.is_stream_consumed:
            raise StreamConsumed()
        if self.is_closed:
            raise StreamClosed()
        if not isinstance(self.stream, SyncByteStream):
            raise RuntimeError('Attempted to call a sync iterator on an async stream.')
        self.is_stream_consumed = True
        self._num_bytes_downloaded = 0
        chunker = ByteChunker(chunk_size=chunk_size)
        with request_context(request=self._request):
            for raw_stream_bytes in self.stream:
                self._num_bytes_downloaded += len(raw_stream_bytes)
                for chunk in chunker.decode(raw_stream_bytes):
                    yield chunk
        for chunk in chunker.flush():
            yield chunk
        self.close()

    def close(self) -> None:
        if not isinstance(self.stream, SyncByteStream):
            raise RuntimeError('Attempted to call an sync close on an async stream.')
        if not self.is_closed:
            self.is_closed = True
            with request_context(request=self._request):
                self.stream.close()

    async def aread(self) -> bytes:
        if not hasattr(self, '_content'):
            self._content = b''.join([part async for part in self.aiter_bytes()])
        return self._content

    async def aiter_bytes(self, chunk_size: int | None = None) -> typing.AsyncIterator[bytes]:
        if hasattr(self, '_content'):
            chunk_size = len(self._content) if chunk_size is None else chunk_size
            for i in range(0, len(self._content), max(chunk_size, 1)):
                yield self._content[i:i + chunk_size]
        else:
            decoder = self._get_content_decoder()
            chunker = ByteChunker(chunk_size=chunk_size)
            with request_context(request=self._request):
                async for raw_bytes in self.aiter_raw():
                    decoded = decoder.decode(raw_bytes)
                    for chunk in chunker.decode(decoded):
                        yield chunk
                decoded = decoder.flush()
                for chunk in chunker.decode(decoded):
                    yield chunk
                for chunk in chunker.flush():
                    yield chunk

    async def aiter_text(self, chunk_size: int | None = None) -> typing.AsyncIterator[str]:
        decoder = TextDecoder(encoding=self.encoding or 'utf-8')
        chunker = TextChunker(chunk_size=chunk_size)
        with request_context(request=self._request):
            async for byte_content in self.aiter_bytes():
                text_content = decoder.decode(byte_content)
                for chunk in chunker.decode(text_content):
                    yield chunk
            text_content = decoder.flush()
            for chunk in chunker.decode(text_content):
                yield chunk
            for chunk in chunker.flush():
                yield chunk

    async def aiter_lines(self) -> typing.AsyncIterator[str]:
        decoder = LineDecoder()
        with request_context(request=self._request):
            async for text in self.aiter_text():
                for line in decoder.decode(text):
                    yield line
            for line in decoder.flush():
                yield line

    async def aiter_raw(self, chunk_size: int | None = None) -> typing.AsyncIterator[bytes]:
        if self.is_stream_consumed:
            raise StreamConsumed()
        if self.is_closed:
            raise StreamClosed()
        if not isinstance(self.stream, AsyncByteStream):
            raise RuntimeError('Attempted to call an async iterator on an sync stream.')
        self.is_stream_consumed = True
        self._num_bytes_downloaded = 0
        chunker = ByteChunker(chunk_size=chunk_size)
        with request_context(request=self._request):
            async for raw_stream_bytes in self.stream:
                self._num_bytes_downloaded += len(raw_stream_bytes)
                for chunk in chunker.decode(raw_stream_bytes):
                    yield chunk
        for chunk in chunker.flush():
            yield chunk
        await self.aclose()

    async def aclose(self) -> None:
        if not isinstance(self.stream, AsyncByteStream):
            raise RuntimeError('Attempted to call an async close on an sync stream.')
        if not self.is_closed:
            self.is_closed = True
            with request_context(request=self._request):
                await self.stream.aclose()

class Cookies(typing.MutableMapping[str, str]):
    def __init__(self, cookies: CookieTypes | None = None) -> None:
        if cookies is None or isinstance(cookies, dict):
            self.jar = CookieJar()
            if isinstance(cookies, dict):
                for key, value in cookies.items():
                    self.set(key, value)
        elif isinstance(cookies, list):
            self.jar = CookieJar()
            for key, value in cookies:
                self.set(key, value)
        elif isinstance(cookies, Cookies):
            self.jar = CookieJar()
            for cookie in cookies.jar:
                self.jar.set_cookie(cookie)
        else:
            self.jar = cookies

    def extract_cookies(self, response: Response) -> None:
        urllib_response = self._CookieCompatResponse(response)
        urllib_request = self._CookieCompatRequest(response.request)
        self.jar.extract_cookies(urllib_response, urllib_request)

    def set_cookie_header(self, request: Request) -> None:
        urllib_request = self._CookieCompatRequest(request)
        self.jar.add_cookie_header(urllib_request)

    def set(self, name: str, value: str, domain: str = '', path: str = '/') -> None:
        kwargs = {'version': 0, 'name': name, 'value': value, 'port': None, 'port_specified': False, 'domain': domain, 'domain_specified': bool(domain), 'domain_initial_dot': domain.startswith('.'), 'path': path, 'path_specified': bool(path), 'secure': False, 'expires': None, 'discard': True, 'comment': None, 'comment_url': None, 'rest': {'HttpOnly': None}, 'rfc2109': False}
        cookie = Cookie(**kwargs)
        self.jar.set_cookie(cookie)

    def get(self, name: str, default: str | None = None, domain: str | None = None, path: str | None = None) -> str | None:
        value = None
        for cookie in self.jar:
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if value is not None:
                            message = f'Multiple cookies exist with name={name}'
                            raise CookieConflict(message)
                        value = cookie.value
        if value is None:
            return default
        return value

    def delete(self, name: str, domain: str | None = None, path: str | None = None) -> None:
        if domain is not None and path is not None:
            return self.jar.clear(domain, path, name)
        remove = [cookie for cookie in self.jar if cookie.name == name and (domain is None or cookie.domain == domain) and (path is None or cookie.path == path)]
        for cookie in remove:
            self.jar.clear(cookie.domain, cookie.path, cookie.name)

    def clear(self, domain: str | None = None, path: str | None = None) -> None:
        args = []
        if domain is not None:
            args.append(domain)
        if path is not None:
            assert domain is not None
            args.append(path)
        self.jar.clear(*args)

    def update(self, cookies: CookieTypes | None = None) -> None:
        cookies = Cookies(cookies)
        for cookie in cookies.jar:
            self.jar.set_cookie(cookie)

    def __setitem__(self, name: str, value: str) -> None:
        return self.set(name, value)

    def __getitem__(self, name: str) -> str:
        value = self.get(name)
        if value is None:
            raise KeyError(name)
        return value

    def __delitem__(self, name: str) -> None:
        return self.delete(name)

    def __len__(self) -> int:
        return len(self.jar)

    def __iter__(self) -> typing.Iterator[str]:
        return (cookie.name for cookie in self.jar)

    def __bool__(self) -> bool:
        for _ in self.jar:
            return True
        return False

    def __repr__(self) -> str:
        cookies_repr = ', '.join([f'<Cookie {cookie.name}={cookie.value} for {cookie.domain} />' for cookie in self.jar])
        return f'<Cookies[{cookies_repr}]>'

    class _CookieCompatRequest(urllib.request.Request):
        def __init__(self, request: Request) -> None:
            super().__init__(url=str(request.url), headers=dict(request.headers), method=request.method)
            self.request = request

        def add_unredirected_header(self, key: str, value: str) -> None:
            super().add_unredirected_header(key, value)
            self.request.headers[key] = value

    class _CookieCompatResponse:
        def __init__(self, response: Response) -> None:
            self.response = response

        def info(self) -> email.message.Message:
            info = email.message.Message()
            for key, value in self.response.headers.multi_items():
                info[key] = value
            return info
