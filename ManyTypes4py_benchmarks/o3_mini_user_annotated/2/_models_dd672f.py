#!/usr/bin/env python3
from __future__ import annotations

import codecs
import datetime
import email.message
import json as jsonlib
import re
import typing
import urllib.request
from collections.abc import Mapping, Iterator, KeysView, ValuesView, ItemsView
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
    """
    Return `True` if `encoding` is a known codec.
    """
    try:
        codecs.lookup(encoding)
    except LookupError:
        return False
    return True


def _normalize_header_key(key: str | bytes, encoding: str | None = None) -> bytes:
    """
    Coerce str/bytes into a strictly byte-wise HTTP header key.
    """
    return key if isinstance(key, bytes) else key.encode(encoding or "ascii")


def _normalize_header_value(value: str | bytes, encoding: str | None = None) -> bytes:
    """
    Coerce str/bytes into a strictly byte-wise HTTP header value.
    """
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Header value must be str or bytes, not {type(value)}")
    return value.encode(encoding or "ascii")


def _parse_content_type_charset(content_type: str) -> str | None:
    # We used to use `cgi.parse_header()` here, but `cgi` became a dead battery.
    # See: https://peps.python.org/pep-0594/#cgi
    msg = email.message.Message()
    msg["content-type"] = content_type
    return msg.get_content_charset(failobj=None)


def _parse_header_links(value: str) -> list[dict[str, str]]:
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
    links: list[dict[str, str]] = []
    replace_chars = " '\""
    value = value.strip(replace_chars)
    if not value:
        return links
    for val in re.split(", *<", value):
        try:
            url, params = val.split(";", 1)
        except ValueError:
            url, params = val, ""
        link = {"url": url.strip("<> '\"")}
        for param in params.split(";"):
            try:
                key, value_param = param.split("=")
            except ValueError:
                break
            link[key.strip(replace_chars)] = value_param.strip(replace_chars)
        links.append(link)
    return links


def _obfuscate_sensitive_headers(
    items: typing.Iterable[tuple[typing.AnyStr, typing.AnyStr]],
) -> typing.Iterator[tuple[typing.AnyStr, typing.AnyStr]]:
    for k, v in items:
        if to_str(k.lower()) in SENSITIVE_HEADERS:
            v = to_bytes_or_str("[secure]", match_type_of=v)
        yield k, v


class Headers(typing.MutableMapping[str, str]):
    """
    HTTP headers, as a case-insensitive multi-dict.
    """

    def __init__(
        self,
        headers: HeaderTypes | None = None,
        encoding: str | None = None,
    ) -> None:
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
        """
        Header encoding is mandated as ascii, but we allow fallbacks to utf-8
        or iso-8859-1.
        """
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
        """
        Returns a list of the raw header items, as byte pairs.
        """
        return [(raw_key, value) for raw_key, _, value in self._list]

    def keys(self) -> KeysView[str]:
        return {key.decode(self.encoding): None for _, key, _ in self._list}.keys()

    def values(self) -> ValuesView[str]:
        values_dict: dict[str, str] = {}
        for _, key, value in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            if str_key in values_dict:
                values_dict[str_key] += f", {str_value}"
            else:
                values_dict[str_key] = str_value
        return values_dict.values()

    def items(self) -> ItemsView[str, str]:
        values_dict: dict[str, str] = {}
        for _, key, value in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            if str_key in values_dict:
                values_dict[str_key] += f", {str_value}"
            else:
                values_dict[str_key] = str_value
        return values_dict.items()

    def multi_items(self) -> list[tuple[str, str]]:
        """
        Return a list of `(key, value)` pairs of headers. Allow multiple
        occurrences of the same key without concatenating into a single
        comma separated value.
        """
        return [
            (key.decode(self.encoding), value.decode(self.encoding))
            for _, key, value in self._list
        ]

    def get(self, key: str, default: typing.Any = None) -> typing.Any:
        """
        Return a header value. If multiple occurrences of the header occur
        then concatenate them together with commas.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def get_list(self, key: str, split_commas: bool = False) -> list[str]:
        """
        Return a list of all header values for a given key.
        If `split_commas=True` is passed, then any comma separated header
        values are split into multiple return strings.
        """
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

        found_indexes = [
            idx
            for idx, (_, item_key, _) in enumerate(self._list)
            if item_key == lookup_key
        ]

        for idx in reversed(found_indexes[1:]):
            del self._list[idx]

        if found_indexes:
            idx = found_indexes[0]
            self._list[idx] = (set_key, lookup_key, set_value)
        else:
            self._list.append((set_key, lookup_key, set_value))

    def __delitem__(self, key: str) -> None:
        del_key = key.lower().encode(self.encoding)

        pop_indexes = [
            idx
            for idx, (_, item_key, _) in enumerate(self._list)
            if item_key.lower() == del_key
        ]

        if not pop_indexes:
            raise KeyError(key)

        for idx in reversed(pop_indexes):
            del self._list[idx]

    def __contains__(self, key: typing.Any) -> bool:
        header_key = key.lower().encode(self.encoding)
        return header_key in [item_key for _, item_key, _ in self._list]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._list)

    def __eq__(self, other: typing.Any) -> bool:
        try:
            other_headers = Headers(other)
        except ValueError:
            return False

        self_list = [(key, value) for _, key, value in self._list]
        other_list = [(key, value) for _, key, value in other_headers._list]
        return sorted(self_list) == sorted(other_list)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__

        encoding_str = ""
        if self.encoding != "ascii":
            encoding_str = f", encoding={self.encoding!r}"

        as_list = list(_obfuscate_sensitive_headers(self.multi_items()))
        as_dict = dict(as_list)

        no_duplicate_keys = len(as_dict) == len(as_list)
        if no_duplicate_keys:
            return f"{class_name}({as_dict!r}{encoding_str})"
        return f"{class_name}({as_list!r}{encoding_str})"


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
        self.method: str = method.upper()
        self.url: URL = URL(url) if params is None else URL(url, params=params)
        self.headers: Headers = Headers(headers)
        self.extensions: dict[str, typing.Any] = {} if extensions is None else dict(extensions)

        if cookies:
            Cookies(cookies).set_cookie_header(self)

        if stream is None:
            content_type: str | None = self.headers.get("content-type")
            default_boundary: bytes | None = (
                get_multipart_boundary_from_content_type(
                    content_type=content_type.encode(self.headers.encoding)
                    if content_type
                    else None
                )
            )
            default_headers, stream_obj = encode_request(
                content=content,
                data=data,
                files=files,
                json=json,
                boundary=default_boundary,
            )
            self._prepare(default_headers)
            self.stream: SyncByteStream | AsyncByteStream = stream_obj
            if isinstance(stream_obj, ByteStream):
                self.read()
        else:
            self.stream = stream

    def _prepare(self, default_headers: dict[str, str]) -> None:
        for key, value in default_headers.items():
            if key.lower() == "transfer-encoding" and "Content-Length" in self.headers:
                continue
            self.headers.setdefault(key, value)

        auto_headers: list[tuple[bytes, bytes]] = []

        has_host: bool = "Host" in self.headers
        has_content_length: bool = (
            "Content-Length" in self.headers or "Transfer-Encoding" in self.headers
        )

        if not has_host and self.url.host:
            auto_headers.append((b"Host", self.url.netloc))
        if not has_content_length and self.method in ("POST", "PUT", "PATCH"):
            auto_headers.append((b"Content-Length", b"0"))

        self.headers = Headers(auto_headers + self.headers.raw)

    @property
    def content(self) -> bytes:
        if not hasattr(self, "_content"):
            raise RequestNotRead()
        return self._content  # type: ignore

    def read(self) -> bytes:
        """
        Read and return the request content.
        """
        if not hasattr(self, "_content"):
            assert isinstance(self.stream, typing.Iterable)
            self._content = b"".join(self.stream)  # type: ignore
            if not isinstance(self.stream, ByteStream):
                self.stream = ByteStream(self._content)
        return self._content  # type: ignore

    async def aread(self) -> bytes:
        """
        Read and return the request content asynchronously.
        """
        if not hasattr(self, "_content"):
            assert isinstance(self.stream, typing.AsyncIterable)
            self._content = b"".join([part async for part in self.stream])  # type: ignore
            if not isinstance(self.stream, ByteStream):
                self.stream = ByteStream(self._content)
        return self._content  # type: ignore

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        url_str: str = str(self.url)
        return f"<{class_name}({self.method!r}, {url_str!r})>"

    def __getstate__(self) -> dict[str, typing.Any]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ["extensions", "stream"]
        }

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
        self.status_code: int = status_code
        self.headers: Headers = Headers(headers)
        self._request: Request | None = request
        self.next_request: Request | None = None
        self.extensions: dict[str, typing.Any] = {} if extensions is None else dict(extensions)
        self.history: list[Response] = [] if history is None else list(history)
        self.is_closed: bool = False
        self.is_stream_consumed: bool = False
        self.default_encoding: str | typing.Callable[[bytes], str] = default_encoding

        if stream is None:
            default_headers, stream_obj = encode_response(content, text, html, json)
            self._prepare(default_headers)
            self.stream: SyncByteStream | AsyncByteStream = stream_obj
            if isinstance(stream_obj, ByteStream):
                self.read()
        else:
            self.stream = stream

        self._num_bytes_downloaded: int = 0

    def _prepare(self, default_headers: dict[str, str]) -> None:
        for key, value in default_headers.items():
            if key.lower() == "transfer-encoding" and "content-length" in self.headers:
                continue
            self.headers.setdefault(key, value)

    @property
    def elapsed(self) -> datetime.timedelta:
        if not hasattr(self, "_elapsed"):
            raise RuntimeError(
                "'.elapsed' may only be accessed after the response "
                "has been read or closed."
            )
        return self._elapsed  # type: ignore

    @elapsed.setter
    def elapsed(self, elapsed: datetime.timedelta) -> None:
        self._elapsed = elapsed

    @property
    def request(self) -> Request:
        if self._request is None:
            raise RuntimeError(
                "The request instance has not been set on this response."
            )
        return self._request

    @request.setter
    def request(self, value: Request) -> None:
        self._request = value

    @property
    def http_version(self) -> str:
        try:
            http_version: bytes = self.extensions["http_version"]
        except KeyError:
            return "HTTP/1.1"
        else:
            return http_version.decode("ascii", errors="ignore")

    @property
    def reason_phrase(self) -> str:
        try:
            reason_phrase: bytes = self.extensions["reason_phrase"]
        except KeyError:
            return codes.get_reason_phrase(self.status_code)
        else:
            return reason_phrase.decode("ascii", errors="ignore")

    @property
    def url(self) -> URL:
        return self.request.url

    @property
    def content(self) -> bytes:
        if not hasattr(self, "_content"):
            raise ResponseNotRead()
        return self._content  # type: ignore

    @property
    def text(self) -> str:
        if not hasattr(self, "_text"):
            content_bytes: bytes = self.content
            if not content_bytes:
                self._text = ""
            else:
                decoder = TextDecoder(encoding=self.encoding or "utf-8")
                self._text = "".join([decoder.decode(content_bytes), decoder.flush()])
        return self._text  # type: ignore

    @property
    def encoding(self) -> str | None:
        if not hasattr(self, "_encoding"):
            encoding: str | None = self.charset_encoding
            if encoding is None or not _is_known_encoding(encoding):
                if isinstance(self.default_encoding, str):
                    encoding = self.default_encoding
                elif hasattr(self, "_content"):
                    encoding = self.default_encoding(self._content)  # type: ignore
            self._encoding = encoding or "utf-8"
        return self._encoding  # type: ignore

    @encoding.setter
    def encoding(self, value: str) -> None:
        if hasattr(self, "_text"):
            raise ValueError(
                "Setting encoding after `text` has been accessed is not allowed."
            )
        self._encoding = value

    @property
    def charset_encoding(self) -> str | None:
        content_type: str | None = self.headers.get("Content-Type")
        if content_type is None:
            return None
        return _parse_content_type_charset(content_type)

    def _get_content_decoder(self) -> ContentDecoder:
        if not hasattr(self, "_decoder"):
            decoders: list[ContentDecoder] = []
            values: list[str] = self.headers.get_list("content-encoding", split_commas=True)
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
        return self._decoder  # type: ignore

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
        return (
            self.status_code
            in (
                codes.MOVED_PERMANENTLY,
                codes.FOUND,
                codes.SEE_OTHER,
                codes.TEMPORARY_REDIRECT,
                codes.PERMANENT_REDIRECT,
            )
            and "Location" in self.headers
        )

    def raise_for_status(self) -> Response:
        request_inst = self._request
        if request_inst is None:
            raise RuntimeError(
                "Cannot call `raise_for_status` as the request "
                "instance has not been set on this response."
            )
        if self.is_success:
            return self

        if self.has_redirect_location:
            message = (
                "{error_type} '{0.status_code} {0.reason_phrase}' for url '{0.url}'\n"
                "Redirect location: '{0.headers[location]}'\n"
                "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{0.status_code}"
            )
        else:
            message = (
                "{error_type} '{0.status_code} {0.reason_phrase}' for url '{0.url}'\n"
                "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{0.status_code}"
            )

        status_class: int = self.status_code // 100
        error_types: dict[int, str] = {
            1: "Informational response",
            3: "Redirect response",
            4: "Client error",
            5: "Server error",
        }
        error_type: str = error_types.get(status_class, "Invalid status code")
        message = message.format(self, error_type=error_type)
        raise HTTPStatusError(message, request=request_inst, response=self)

    def json(self, **kwargs: typing.Any) -> typing.Any:
        return jsonlib.loads(self.content, **kwargs)

    @property
    def cookies(self) -> Cookies:
        if not hasattr(self, "_cookies"):
            self._cookies = Cookies()
            self._cookies.extract_cookies(self)
        return self._cookies

    @property
    def links(self) -> dict[str | None, dict[str, str]]:
        header: str | None = self.headers.get("link")
        if header is None:
            return {}
        return {
            (link.get("rel") or link.get("url")): link
            for link in _parse_header_links(header)
        }

    @property
    def num_bytes_downloaded(self) -> int:
        return self._num_bytes_downloaded

    def __repr__(self) -> str:
        return f"<Response [{self.status_code} {self.reason_phrase}]>"

    def __getstate__(self) -> dict[str, typing.Any]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ["extensions", "stream", "is_closed", "_decoder"]
        }

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.is_closed = True
        self.extensions = {}
        self.stream = UnattachedStream()

    def read(self) -> bytes:
        """
        Read and return the response content.
        """
        if not hasattr(self, "_content"):
            self._content = b"".join(self.iter_bytes())
        return self._content  # type: ignore

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        if hasattr(self, "_content"):
            actual_chunk_size: int = len(self._content) if chunk_size is None else chunk_size  # type: ignore
            for i in range(0, len(self._content), max(actual_chunk_size, 1)):  # type: ignore
                yield self._content[i : i + actual_chunk_size]  # type: ignore
        else:
            decoder: ContentDecoder = self._get_content_decoder()
            chunker: ByteChunker = ByteChunker(chunk_size=chunk_size)
            with request_context(request=self._request):
                for raw_bytes in self.iter_raw():
                    decoded: bytes = decoder.decode(raw_bytes)
                    for chunk in chunker.decode(decoded):
                        yield chunk
                decoded = decoder.flush()
                for chunk in chunker.decode(decoded):
                    yield chunk
                for chunk in chunker.flush():
                    yield chunk

    def iter_text(self, chunk_size: int | None = None) -> Iterator[str]:
        decoder: TextDecoder = TextDecoder(encoding=self.encoding or "utf-8")
        chunker = TextChunker(chunk_size=chunk_size)
        with request_context(request=self._request):
            for byte_content in self.iter_bytes():
                text_content: str = decoder.decode(byte_content)
                for chunk in chunker.decode(text_content):
                    yield chunk
            text_content = decoder.flush()
            for chunk in chunker.decode(text_content):
                yield chunk
            for chunk in chunker.flush():
                yield chunk

    def iter_lines(self) -> Iterator[str]:
        decoder: LineDecoder = LineDecoder()
        with request_context(request=self._request):
            for text in self.iter_text():
                for line in decoder.decode(text):
                    yield line
            for line in decoder.flush():
                yield line

    def iter_raw(self, chunk_size: int | None = None) -> Iterator[bytes]:
        if self.is_stream_consumed:
            raise StreamConsumed()
        if self.is_closed:
            raise StreamClosed()
        if not isinstance(self.stream, ByteStream):
            raise RuntimeError("Attempted to call a sync iterator on an async stream.")

        self.is_stream_consumed = True
        self._num_bytes_downloaded = 0
        chunker: ByteChunker = ByteChunker(chunk_size=chunk_size)

        with request_context(request=self._request):
            for raw_stream_bytes in self.stream:  # type: ignore
                self._num_bytes_downloaded += len(raw_stream_bytes)
                for chunk in chunker.decode(raw_stream_bytes):
                    yield chunk

        for chunk in chunker.flush():
            yield chunk

        self.close()
        return  # type: ignore

    def close(self) -> None:
        if not isinstance(self.stream, ByteStream):
            raise RuntimeError("Attempted to call an sync close on an async stream.")
        if not self.is_closed:
            self.is_closed = True
            with request_context(request=self._request):
                self.stream.close()  # type: ignore

    async def aread(self) -> bytes:
        if not hasattr(self, "_content"):
            self._content = b"".join([part async for part in self.aiter_bytes()])  # type: ignore
        return self._content  # type: ignore

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> typing.AsyncIterator[bytes]:
        if hasattr(self, "_content"):
            actual_chunk_size: int = len(self._content) if chunk_size is None else chunk_size  # type: ignore
            for i in range(0, len(self._content), max(actual_chunk_size, 1)):  # type: ignore
                yield self._content[i : i + actual_chunk_size]  # type: ignore
        else:
            decoder: ContentDecoder = self._get_content_decoder()
            chunker: ByteChunker = ByteChunker(chunk_size=chunk_size)
            with request_context(request=self._request):
                async for raw_bytes in self.aiter_raw():
                    decoded: bytes = decoder.decode(raw_bytes)
                    for chunk in chunker.decode(decoded):
                        yield chunk
                decoded = decoder.flush()
                for chunk in chunker.decode(decoded):
                    yield chunk
                for chunk in chunker.flush():
                    yield chunk

    async def aiter_text(
        self, chunk_size: int | None = None
    ) -> typing.AsyncIterator[str]:
        decoder: TextDecoder = TextDecoder(encoding=self.encoding or "utf-8")
        chunker: TextChunker = TextChunker(chunk_size=chunk_size)
        with request_context(request=self._request):
            async for byte_content in self.aiter_bytes():
                text_content: str = decoder.decode(byte_content)
                for chunk in chunker.decode(text_content):
                    yield chunk
            text_content: str = decoder.flush()
            for chunk in chunker.decode(text_content):
                yield chunk
            for chunk in chunker.flush():
                yield chunk

    async def aiter_lines(self) -> typing.AsyncIterator[str]:
        decoder: LineDecoder = LineDecoder()
        with request_context(request=self._request):
            async for text in self.aiter_text():
                for line in decoder.decode(text):
                    yield line
            for line in decoder.flush():
                yield line

    async def aiter_raw(
        self, chunk_size: int | None = None
    ) -> typing.AsyncIterator[bytes]:
        if self.is_stream_consumed:
            raise StreamConsumed()
        if self.is_closed:
            raise StreamClosed()
        if not isinstance(self.stream, AsyncByteStream):
            raise RuntimeError("Attempted to call an async iterator on a sync stream.")

        self.is_stream_consumed = True
        self._num_bytes_downloaded = 0
        chunker: ByteChunker = ByteChunker(chunk_size=chunk_size)

        with request_context(request=self._request):
            async for raw_stream_bytes in self.stream:  # type: ignore
                self._num_bytes_downloaded += len(raw_stream_bytes)
                for chunk in chunker.decode(raw_stream_bytes):
                    yield chunk

        for chunk in chunker.flush():
            yield chunk

        await self.aclose()

    async def aclose(self) -> None:
        if not isinstance(self.stream, AsyncByteStream):
            raise RuntimeError("Attempted to call an async close on a sync stream.")
        if not self.is_closed:
            self.is_closed = True
            with request_context(request=self._request):
                await self.stream.aclose()  # type: ignore


class Cookies(typing.MutableMapping[str, str]):
    """
    HTTP Cookies, as a mutable mapping.
    """

    def __init__(self, cookies: CookieTypes | None = None) -> None:
        if cookies is None or isinstance(cookies, dict):
            self.jar: CookieJar = CookieJar()
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
        self.jar.extract_cookies(urllib_response, urllib_request)  # type: ignore

    def set_cookie_header(self, request: Request) -> None:
        urllib_request = self._CookieCompatRequest(request)
        self.jar.add_cookie_header(urllib_request)

    def set(self, name: str, value: str, domain: str = "", path: str = "/") -> None:
        kwargs: dict[str, typing.Any] = {
            "version": 0,
            "name": name,
            "value": value,
            "port": None,
            "port_specified": False,
            "domain": domain,
            "domain_specified": bool(domain),
            "domain_initial_dot": domain.startswith("."),
            "path": path,
            "path_specified": bool(path),
            "secure": False,
            "expires": None,
            "discard": True,
            "comment": None,
            "comment_url": None,
            "rest": {"HttpOnly": None},
            "rfc2109": False,
        }
        cookie: Cookie = Cookie(**kwargs)  # type: ignore
        self.jar.set_cookie(cookie)

    def get(
        self,
        name: str,
        default: str | None = None,
        domain: str | None = None,
        path: str | None = None,
    ) -> str | None:
        value: str | None = None
        for cookie in self.jar:
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if value is not None:
                            message = f"Multiple cookies exist with name={name}"
                            raise CookieConflict(message)
                        value = cookie.value
        if value is None:
            return default
        return value

    def delete(
        self,
        name: str,
        domain: str | None = None,
        path: str | None = None,
    ) -> None:
        if domain is not None and path is not None:
            return self.jar.clear(domain, path, name)
        remove = [
            cookie
            for cookie in self.jar
            if cookie.name == name
            and (domain is None or cookie.domain == domain)
            and (path is None or cookie.path == path)
        ]
        for cookie in remove:
            self.jar.clear(cookie.domain, cookie.path, cookie.name)

    def clear(self, domain: str | None = None, path: str | None = None) -> None:
        args: list[str] = []
        if domain is not None:
            args.append(domain)
        if path is not None:
            assert domain is not None
            args.append(path)
        self.jar.clear(*args)

    def update(self, cookies: CookieTypes | None = None) -> None:
        cookies_obj = Cookies(cookies)
        for cookie in cookies_obj.jar:
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

    def __iter__(self) -> Iterator[str]:
        return (cookie.name for cookie in self.jar)

    def __bool__(self) -> bool:
        for _ in self.jar:
            return True
        return False

    def __repr__(self) -> str:
        cookies_repr = ", ".join(
            [
                f"<Cookie {cookie.name}={cookie.value} for {cookie.domain} />"
                for cookie in self.jar
            ]
        )
        return f"<Cookies[{cookies_repr}]>"

    class _CookieCompatRequest(urllib.request.Request):
        def __init__(self, request: Request) -> None:
            super().__init__(
                url=str(request.url),
                headers=dict(request.headers),
                method=request.method,
            )
            self.request: Request = request

        def add_unredirected_header(self, key: str, value: str) -> None:
            super().add_unredirected_header(key, value)
            self.request.headers[key] = value

    class _CookieCompatResponse:
        def __init__(self, response: Response) -> None:
            self.response: Response = response

        def info(self) -> email.message.Message:
            info = email.message.Message()
            for key, value in self.response.headers.multi_items():
                info[key] = value
            return info
