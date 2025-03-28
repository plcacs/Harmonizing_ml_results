```python
from __future__ import annotations

import codecs
import datetime
import email.message
import json as jsonlib
import re
import typing
import urllib.request
from collections.abc import Mapping, Iterable, Iterator, AsyncIterator, KeysView, ValuesView, ItemsView, MutableMapping
from http.cookiejar import Cookie, CookieJar

from ._content import ByteStream, UnattachedStream, encode_request, encode_response
from ._decoders import (
    SUPPORTED_DECODERS,
    ByteChunker,
    ContentDecoder,
    IdentityDecoder,
    LineDecoder,
    MultiDecoder,
    TextChunker,
    TextDecoder,
)
from ._exceptions import (
    CookieConflict,
    HTTPStatusError,
    RequestNotRead,
    ResponseNotRead,
    StreamClosed,
    StreamConsumed,
    request_context,
)
from ._multipart import get_multipart_boundary_from_content_type
from ._status_codes import codes
from ._types import (
    AsyncByteStream,
    CookieTypes,
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestExtensions,
    RequestFiles,
    ResponseContent,
    ResponseExtensions,
    SyncByteStream,
)
from ._urls import URL
from ._utils import to_bytes_or_str, to_str

__all__ = ["Cookies", "Headers", "Request", "Response"]

SENSITIVE_HEADERS = {"authorization", "proxy-authorization"}


def _is_known_encoding(encoding: str) -> bool:
    try:
        codecs.lookup(encoding)
    except LookupError:
        return False
    return True


def _normalize_header_key(key: str | bytes, encoding: str | None = None) -> bytes:
    return key if isinstance(key, bytes) else key.encode(encoding or "ascii")


def _normalize_header_value(value: str | bytes, encoding: str | None = None) -> bytes:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Header value must be str or bytes, not {type(value)}")
    return value.encode(encoding or "ascii")


def _parse_content_type_charset(content_type: str) -> str | None:
    msg = email.message.Message()
    msg["content-type"] = content_type
    return msg.get_content_charset(failobj=None)


def _parse_header_links(value: str) -> list[dict[str, str]]:
    replace_chars = " '\""
    value = value.strip(replace_chars)
    if not value:
        return []
    links: list[dict[str, str]] = []
    for val in re.split(", *<", value):
        try:
            url, params = val.split(";", 1)
        except ValueError:
            url, params = val, ""
        link = {"url": url.strip("<> '\"")}
        for param in params.split(";"):
            try:
                key, value = param.split("=")
            except ValueError:
                break
            link[key.strip(replace_chars)] = value.strip(replace_chars)
        links.append(link)
    return links


def _obfuscate_sensitive_headers(
    items: Iterable[tuple[typing.AnyStr, typing.AnyStr]],
) -> Iterator[tuple[typing.AnyStr, typing.AnyStr]]:
    for k, v in items:
        if to_str(k.lower()) in SENSITIVE_HEADERS:
            v = to_bytes_or_str("[secure]", match_type_of=v)
        yield k, v


class Headers(MutableMapping[str, str]):
    def __init__(
        self,
        headers: HeaderTypes | None = None,
        encoding: str | None = None,
    ) -> None:
        self._list: list[tuple[bytes, bytes, bytes]] = []
        self._encoding: str | None = encoding

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

    @property
    def encoding(self) -> str:
        if self._encoding is None:
            for encoding in ["ascii", "utf-8"]:
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
                self._encoding = "iso-8859-1"
        return self._encoding

    @encoding.setter
    def encoding(self, value: str) -> None:
        self._encoding = value

    @property
    def raw(self) -> list[tuple[bytes, bytes]]:
        return [(raw_key, value) for raw_key, _, value in self._list]

    def keys(self) -> KeysView[str]:
        return {key.decode(self.encoding): None for _, key, _ in self._list}.keys()

    def values(self) -> ValuesView[str]:
        values_dict: dict[str, str] = {}
        for _, key, value in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            values_dict[str_key] = values_dict.get(str_key, "") + f", {str_value}" if str_key in values_dict else str_value
        return values_dict.values()

    def items(self) -> ItemsView[str, str]:
        values_dict: dict[str, str] = {}
        for _, key, value in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            values_dict[str_key] = values_dict.get(str_key, "") + f", {str_value}" if str_key in values_dict else str_value
        return values_dict.items()

    def multi_items(self) -> list[tuple[str, str]]:
        return [
            (key.decode(self.encoding), value.decode(self.encoding))
            for _, key, value in self._list
        ]

    def get(self, key: str, default: typing.Any = None) -> typing.Any:
        try:
            return self[key]
        except KeyError:
            return default

    def get_list(self, key: str, split_commas: bool = False) -> list[str]:
        get_header_key = key.lower().encode(self.encoding)
        values = [
            item_value.decode(self.encoding)
            for _, item_key, item_value in self._list
            if item_key.lower() == get_header_key
        ]
        if not split_commas:
            return values
        split_values: list[str] = []
        for value in values:
            split_values.extend([item.strip() for item in value.split(",")])
        return split_values

    def update(self, headers: HeaderTypes | None = None) -> None:  # type: ignore
        headers = Headers(headers)
        for key in headers.keys():
            if key in self:
                del self[key]
        self._list.extend(headers._list)

    def copy(self) -> Headers:
        return Headers(self, encoding=self.encoding)

    def __getitem__(self, key: str) -> str:
        normalized_key = key.lower().encode(self.encoding)
        items = [
            header_value.decode(self.encoding)
            for _, header_key, header_value in self._list
            if header_key == normalized_key
        ]
        if items:
            return ", ".join(items)
        raise KeyError(key)

    def __setitem__(self, key: str, value: str) -> None:
        set_key = key.encode(self._encoding or "utf-8")
        set_value = value.encode(self._encoding or "utf-8")
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

    def __contains__(self, key: typing.Any) -> bool:
        return key.lower().encode(self.encoding) in [k for _, k, _ in self._list]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._list)

    def __eq__(self, other: typing.Any) -> bool:
        try:
            other_headers = Headers(other)
        except ValueError:
            return False
        return sorted((k, v) for _, k, v in self._list) == sorted((k, v) for _, k, v in other_headers._list)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        encoding_str = f", encoding={self.encoding!r}" if self.encoding != "ascii" else ""
        as_list = list(_obfuscate_sensitive_headers(self.multi_items()))
        return f"{class_name}({dict(as_list)!r}{encoding_str})" if len(as_list) == len(dict(as_list)) else f"{class_name}({as_list!r}{encoding_str})"


class Request:
    def __init__(
        self,
        method: str,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        stream: SyncByteStream | AsyncByteStream | None = None,
        extensions: RequestExtensions | None = None,
    ) -> None:
        self.method = method.upper()
        self.url = URL(url) if params is None else URL(url, params=params)
        self.headers = Headers(headers)
        self.extensions: RequestExtensions = {} if extensions is None else dict(extensions)

        if cookies:
            Cookies(cookies).set_cookie_header(self)

        if stream is None:
            content_type: str | None = self.headers.get("content-type")
            boundary = get_multipart_boundary_from_content_type(
                content_type.encode(self.headers.encoding) if content_type else None
            )
            headers, stream = encode_request(content, data, files, json, boundary)
            self._prepare(headers)
            self.stream = stream
            if isinstance(stream, ByteStream):
                self.read()
        else:
            self.stream = stream

    def _prepare(self, default_headers: dict[str, str]) -> None:
        for key, value in default_headers.items():
            if key.lower() == "transfer-encoding" and "Content-Length" in self.headers:
                continue
            self.headers.setdefault(key, value)
        auto_headers: list[tuple[bytes, bytes]] = []
        if "Host" not in self.headers and self.url.host:
            auto_headers.append((b"Host", self.url.netloc))
        if "Content-Length" not in self.headers and self.method in ("POST", "PUT", "PATCH"):
            auto_headers.append((b"Content-Length", b"0"))
        self.headers = Headers(auto_headers + self.headers.raw)

    @property
    def content(self) -> bytes:
        if not hasattr(self, "_content"):
            raise RequestNotRead()
        return self._content

    def read(self) -> bytes:
        if not hasattr(self, "_content"):
            assert isinstance(self.stream, typing.Iterable)
            self._content = b"".join(self.stream)
            if not isinstance(self.stream, ByteStream):
                self.stream = ByteStream(self._content)
        return self._content

    async def aread(self) -> bytes:
        if not hasattr(self, "_content"):
            assert isinstance(self.stream, typing.AsyncIterable)
            self._content = b"".join([part async for part in self.stream])
            if not isinstance(self.stream, ByteStream):
                self.stream = ByteStream(self._content)
        return self._content

    def __repr__(self) -> str:
        return f"<Request({self.method!r}, {str(self.url)!r})>"

    def __getstate__(self) -> dict[str, typing.Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ["extensions", "stream"]}

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.extensions = {}
        self.stream = UnattachedStream()


class Response:
    def __init__(
        self,
        status_code: int,
        *,
        headers: HeaderTypes | None = None,
        content: ResponseContent | None = None,
        text: str | None = None,
        html: str | None = None,
        json: typing.Any = None,
        stream: SyncByteStream | AsyncByteStream | None = None,
        request: Request | None = None,
        extensions: ResponseExtensions | None = None,
        history: list[Response] | None = None,
        default_encoding: str | typing.Callable[[bytes], str] = "utf-8",
    ) -> None:
        self.status_code = status_code
        self.headers = Headers(headers)
        self._request: Request | None = request
        self.next_request: Request | None = None
        self.extensions: ResponseExtensions = {} if extensions is None else dict(extensions)
        self.history: list[Response] = [] if history is None else list(history)
        self.is_closed = False
        self.is_stream_consumed = False
        self.default_encoding = default_encoding
        self._num_bytes_downloaded = 0

        if stream is None:
            headers, stream = encode_response(content, text, html, json)
            self._prepare(headers)
            self.stream = stream
            if isinstance(stream, ByteStream):
                self.read()
        else:
            self.stream = stream

    def _prepare(self, default_headers: dict[str, str]) -> None:
        for key, value in default_headers.items():
            if key.lower() == "transfer-encoding" and "content-length" in self.headers:
                continue
            self.headers.setdefault(key, value)

    @property
    def elapsed(self) -> datetime.timedelta:
        if not hasattr(self, "_elapsed"):
            raise RuntimeError("'.elapsed' accessed before response read/closed")
        return self._elapsed

    @elapsed.setter
    def elapsed(self, elapsed: datetime.timedelta) -> None:
        self._elapsed = elapsed

    @property
    def request(self) -> Request:
        if self._request is None:
            raise RuntimeError("Request instance not set")
        return self._request

    @request.setter
    def request(self, value: Request) -> None:
        self._request = value

    @property
    def http_version(self) -> str:
        return self.extensions.get("http_version", b"HTTP/1.1").decode("ascii", errors="ignore")

    @property
    def reason_phrase(self) -> str:
        return self.extensions.get("reason_phrase", codes.get_reason_phrase(self.status_code).encode()).decode("ascii", errors="ignore")

    @property
    def url(self) -> URL:
        return self.request.url

    @property
    def content(self) -> bytes:
        if not hasattr(self, "_content"):
            raise ResponseNotRead()
        return self._content

    @property
    def text(self) -> str:
        if not hasattr(self, "_text"):
            content = self.content
            decoder = TextDecoder(self.encoding or "utf-8")
            self._text = decoder.decode(content) + decoder.flush()
        return self._text

    @property
    def encoding(self) -> str | None:
        if not hasattr(self, "_encoding"):
            encoding = self.charset_encoding
            if not encoding or not _is_known_encoding(encoding):
                if isinstance(self.default_encoding, str):
                    encoding = self.default_encoding
                elif hasattr(self, "_content"):
                    encoding = self.default_encoding(self._content)
            self._encoding = encoding or "utf-8"
        return self._encoding

    @encoding.setter
    def encoding(self, value: str) -> None:
        if hasattr(self, "_text"):
            raise ValueError("Cannot set encoding after text accessed")
        self._encoding = value

    @property
    def charset_encoding(self) -> str | None:
        return _parse_content_type_charset(self.headers.get("Content-Type", ""))

    def _get_content_decoder(self) -> ContentDecoder:
        if not hasattr(self, "_decoder"):
            decoders: list[ContentDecoder] = []
            for value in self.headers.get_list("content-encoding", True):
                value = value.strip().lower()
                decoder_cls = SUPPORTED_DECODERS.get(value)
                if decoder_cls:
                    decoders.append(decoder_cls())
            self._decoder = MultiDecoder(decoders) if len(decoders) > 1 else decoders[0] if decoders else IdentityDecoder()
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

   