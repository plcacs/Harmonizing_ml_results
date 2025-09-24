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
    """
    Return `True` if `encoding` is a known codec.
    """
    try:
        codecs.lookup(encoding)
    except LookupError:
        return False
    return True

def _normalize_header_key(key: typing.Union[str, bytes], encoding: typing.Optional[str] = None) -> bytes:
    """
    Coerce str/bytes into a strictly byte-wise HTTP header key.
    """
    return key if isinstance(key, bytes) else key.encode(encoding or 'ascii')

def _normalize_header_value(value: typing.Union[str, bytes], encoding: typing.Optional[str] = None) -> bytes:
    """
    Coerce str/bytes into a strictly byte-wise HTTP header value.
    """
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise TypeError(f'Header value must be str or bytes, not {type(value)}')
    return value.encode(encoding or 'ascii')

def _parse_content_type_charset(content_type: str) -> typing.Optional[str]:
    msg = email.message.Message()
    msg['content-type'] = content_type
    return msg.get_content_charset(failobj=None)

def _parse_header_links(value: str) -> typing.List[typing.Dict[str, str]]:
    """
    Returns a list of parsed link headers, for more info see:
    https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Link
    The generic syntax of those is:
    Link: < uri-reference >; param1=value1; param2="value2"
    So for instance:
    Link; '<http:/.../front.jpeg>; type="image/jpeg",<http://.../back.jpeg>;'
    would return
        [
            {"url": "http:/.../front.jpeg", "type": "image/jpeg"},
            {"url": "http://.../back.jpeg"},
        ]
    :param value: HTTP Link entity-header field
    :return: list of parsed link headers
    """
    links: typing.List[typing.Dict[str, str]] = []
    replace_chars = ' \'"'
    value = value.strip(replace_chars)
    if not value:
        return links
    for val in re.split(', *<', value):
        try:
            (url, params) = val.split(';', 1)
        except ValueError:
            (url, params) = (val, '')
        link = {'url': url.strip('<> \'"')}
        for param in params.split(';'):
            try:
                (key, value) = param.split('=')
            except ValueError:
                break
            link[key.strip(replace_chars)] = value.strip(replace_chars)
        links.append(link)
    return links

def _obfuscate_sensitive_headers(items: typing.Iterable[typing.Tuple[typing.Union[str, bytes], typing.Union[str, bytes]]]) -> typing.Iterator[typing.Tuple[typing.Union[str, bytes], typing.Union[str, bytes]]]:
    for (k, v) in items:
        if to_str(k.lower()) in SENSITIVE_HEADERS:
            v = to_bytes_or_str('[secure]', match_type_of=v)
        yield (k, v)

class Headers(typing.MutableMapping[str, str]):
    """
    HTTP headers, as a case-insensitive multi-dict.
    """

    def __init__(self, headers: typing.Optional[HeaderTypes] = None, encoding: typing.Optional[str] = None) -> None:
        self._list: typing.List[typing.Tuple[bytes, bytes, bytes]] = []
        if isinstance(headers, Headers):
            self._list = list(headers._list)
        elif isinstance(headers, Mapping):
            for (k, v) in headers.items():
                bytes_key = _normalize_header_key(k, encoding)
                bytes_value = _normalize_header_value(v, encoding)
                self._list.append((bytes_key, bytes_key.lower(), bytes_value))
        elif headers is not None:
            for (k, v) in headers:
                bytes_key = _normalize_header_key(k, encoding)
                bytes_value = _normalize_header_value(v, encoding)
                self._list.append((bytes_key, bytes_key.lower(), bytes_value))
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        """
        Header encoding is mandated as ascii, but we allow fallbacks to utf-8
        or iso-8859-1.
        """
        if self._encoding is None:
            for encoding in ['ascii', 'utf-8']:
                for (key, value) in self.raw:
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
    def raw(self) -> typing.List[typing.Tuple[bytes, bytes]]:
        """
        Returns a list of the raw header items, as byte pairs.
        """
        return [(raw_key, value) for (raw_key, _, value) in self._list]

    def keys(self) -> typing.KeysView[str]:
        return {key.decode(self.encoding): None for (_, key, value) in self._list}.keys()

    def values(self) -> typing.ValuesView[str]:
        values_dict: typing.Dict[str, str] = {}
        for (_, key, value) in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            if str_key in values_dict:
                values_dict[str_key] += f', {str_value}'
            else:
                values_dict[str_key] = str_value
        return values_dict.values()

    def items(self) -> typing.ItemsView[str, str]:
        """
        Return `(key, value)` items of headers. Concatenate headers
        into a single comma separated value when a key occurs multiple times.
        """
        values_dict: typing.Dict[str, str] = {}
        for (_, key, value) in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            if str_key in values_dict:
                values_dict[str_key] += f', {str_value}'
            else:
                values_dict[str_key] = str_value
        return values_dict.items()

    def multi_items(self) -> typing.List[typing.Tuple[str, str]]:
        """
        Return a list of `(key, value)` pairs of headers. Allow multiple
        occurrences of the same key without concatenating into a single
        comma separated value.
        """
        return [(key.decode(self.encoding), value.decode(self.encoding)) for (_, key, value) in self._list]

    def get(self, key: str, default: typing.Optional[str] = None) -> typing.Optional[str]:
        """
        Return a header value. If multiple occurrences of the header occur
        then concatenate them together with commas.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def get_list(self, key: str, split_commas: bool = False) -> typing.List[str]:
        """
        Return a list of all header values for a given key.
        If `split_commas=True` is passed, then any comma separated header
        values are split into multiple return strings.
        """
        get_header_key = key.lower().encode(self.encoding)
        values = [item_value.decode(self.encoding) for (_, item_key, item_value) in self._list if item_key.lower() == get_header_key]
        if not split_commas:
            return values
        split_values = []
        for value in values:
            split_values.extend([item.strip() for item in value.split(',')])
        return split_values

    def update(self, headers: typing.Optional[HeaderTypes] = None) -> None:
        headers = Headers(headers)
        for key in headers.keys():
            if key in self:
                self.pop(key)
        self._list.extend(headers._list)

    def copy(self) -> Headers:
        return Headers(self, encoding=self.encoding)

    def __getitem__(self, key: str) -> str:
        """
        Return a single header value.

        If there are multiple headers with the same key, then we concatenate
        them with commas. See: https://tools.ietf.org/html/rfc7230#section-3.2.2
        """
        normalized_key = key.lower().encode(self.encoding)
        items = [header_value.decode(self.encoding) for (_, header_key, header_value) in self._list if header_key == normalized_key]
        if items:
            return ', '.join(items)
        raise KeyError(key)

    def __setitem__(self, key: str, value: str) -> None:
        """
        Set the header `key` to `value`, removing any duplicate entries.
        Retains insertion order.
        """
        set_key = key.encode(self._encoding or 'utf-8')
        set_value = value.encode(self._encoding or 'utf-8')
        lookup_key = set_key.lower()
        found_indexes = [idx for (idx, (_, item_key, _)) in enumerate(self._list) if item_key == lookup_key]
        for idx in reversed(found_indexes[1:]):
            del self._list[idx]
        if found_indexes:
            idx = found_indexes[0]
            self._list[idx] = (set_key, lookup_key, set_value)
        else:
            self._list.append((set_key, lookup_key, set_value))

    def __delitem__(self, key: str) -> None:
        """
        Remove the header `key`.
        """
        del_key = key.lower().encode(self.encoding)
        pop_indexes = [idx for (idx, (_, item_key, _)) in enumerate(self._list) if item_key.lower() == del_key]
        if not pop_indexes:
            raise KeyError(key)
        for idx in reversed(pop_indexes):
            del self._list[idx]

    def __contains__(self, key: typing.Any) -> bool:
        header_key = key.lower().encode(self.encoding)
        return header_key in [key for (_, key, _) in self._list]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._list)

    def __eq__(self, other: typing.Any) -> bool:
        try:
            other_headers = Headers(other)
        except ValueError:
            return False
        self_list = [(key, value) for (_, key, value) in self._list]
        other_list = [(key, value) for (_, key, value) in other_headers._list]
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

    def __init__(self, method: str, url: typing.Union[URL, str], *, params: typing.Optional[QueryParamTypes] = None, headers: typing.Optional[HeaderTypes] = None, cookies: typing.Optional[CookieTypes] = None, content: typing.Optional[RequestContent] = None, data: typing.Optional[RequestData] = None, files: typing.Optional[RequestFiles] = None, json: typing.Optional[typing.Any] = None, stream: typing.Optional[typing.Union[SyncByteStream, AsyncByteStream]] = None, extensions: typing.Optional[RequestExtensions] = None) -> None:
        self.method = method.upper()
        self.url = URL(url) if params is None else URL(url, params=params)
        self.headers = Headers(headers)
        self.extensions: typing.Dict[str, typing.Any] = {} if extensions is None else dict(extensions)
        if cookies:
            Cookies(cookies).set_cookie_header(self)
        if stream is None:
            content_type: typing.Optional[str] = self.headers.get('content-type')
            (headers, stream) = encode_request(content=content, data=data, files=files, json=json, boundary=get_multipart_boundary_from_content_type(content_type=content_type.encode(self.headers.encoding) if content_type else None))
            self._prepare(headers)
            self.stream = stream
            if isinstance(stream, ByteStream):
                self.read()
        else:
            self.stream = stream

    def _prepare(self, default_headers: typing.Dict[str, str]) -> None:
        for (key, value) in default_headers.items():
            if key.lower() == 'transfer-encoding' and 'Content-Length' in self.headers:
                continue
            self.headers.setdefault(key, value)
        auto_headers: typing.List[typing.Tuple[bytes, bytes]] = []
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
        """
        Read and return the request content.
        """
        if not hasattr(self, '_content'):
            assert isinstance(self.stream, typing.Iterable)
            self._content = b''.join(self.stream)
            if not isinstance(self.stream, ByteStream):
                self.stream = ByteStream(self._content)
        return self._content

    async def aread(self) -> bytes:
        """
        Read and return the request content.
        """
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

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        return {name: value for (name, value) in self.__dict__.items() if name not in ['extensions', 'stream']}

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        for (name, value) in state.items():
            setattr(self, name, value)
        self.extensions = {}
        self.stream = UnattachedStream()

class Response:

    def __init__(self, status_code: int, *, headers: typing.Optional[HeaderTypes] = None, content: typing.Optional[ResponseContent] = None, text: typing.Optional[str] = None, html: typing.Optional[str] = None, json: typing.Optional[typing.Any] = None, stream: typing.Optional[typing.Union[SyncByteStream, AsyncByteStream]] = None, request: typing.Optional[Request] = None, extensions: typing.Optional[ResponseExtensions] = None, history: typing.Optional[typing.List[Response]] = None, default_encoding: typing.Union[str, typing.Callable[[bytes], str]] = 'utf-8') -> None:
        self.status_code = status_code
        self.headers = Headers(headers)
        self._request: typing.Optional[Request] = request
        self.next_request: typing.Optional[Request] = None
        self.extensions: typing.Dict[str, typing.Any] = {} if extensions is None else dict(extensions)
        self.history: typing.List[Response] = [] if history is None else list(history)
        self.is_closed = False
        self.is_stream_consumed = False
        self.default_encoding: typing.Union[str, typing.Callable[[bytes], str]] = default_encoding
        if stream is None:
            (headers, stream) = encode_response(content, text, html, json)
            self._prepare(headers)
            self.stream = stream
            if isinstance(stream, ByteStream):
                self.read()
        else:
            self.stream = stream
        self._num_bytes_downloaded = 0

    def _prepare(self, default_headers: typing.Dict[str, str]) ->