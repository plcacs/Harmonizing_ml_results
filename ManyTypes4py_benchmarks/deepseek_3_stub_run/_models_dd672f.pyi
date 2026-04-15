from __future__ import annotations
import datetime
import email.message
import json as jsonlib
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from http.cookiejar import Cookie, CookieJar
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

from ._content import ByteStream, UnattachedStream
from ._decoders import (
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
)
from ._multipart import get_multipart_boundary_from_content_type
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

__all__: Tuple[str, ...] = ("Cookies", "Headers", "Request", "Response")

SENSITIVE_HEADERS: set[str] = {"authorization", "proxy-authorization"}


def _is_known_encoding(encoding: str) -> bool: ...


def _normalize_header_key(
    key: Union[str, bytes], encoding: Optional[str] = None
) -> bytes: ...


def _normalize_header_value(
    value: Union[str, bytes], encoding: Optional[str] = None
) -> bytes: ...


def _parse_content_type_charset(content_type: str) -> Optional[str]: ...


def _parse_header_links(value: str) -> List[Dict[str, str]]: ...


def _obfuscate_sensitive_headers(
    items: Iterable[Tuple[Union[str, bytes], Union[str, bytes]]]
) -> Iterator[Tuple[Union[str, bytes], Union[str, bytes]]]: ...


class Headers(MutableMapping[str, str]):
    _list: List[Tuple[bytes, bytes, bytes]]
    _encoding: Optional[str]

    def __init__(
        self,
        headers: Optional[
            Union[
                Headers,
                Mapping[str, str],
                Mapping[bytes, bytes],
                Iterable[Tuple[str, str]],
                Iterable[Tuple[bytes, bytes]],
            ]
        ] = None,
        encoding: Optional[str] = None,
    ) -> None: ...

    @property
    def encoding(self) -> str: ...

    @encoding.setter
    def encoding(self, value: str) -> None: ...

    @property
    def raw(self) -> List[Tuple[bytes, bytes]]: ...

    def keys(self) -> Iterator[str]: ...

    def values(self) -> Iterator[str]: ...

    def items(self) -> Iterator[Tuple[str, str]]: ...

    def multi_items(self) -> List[Tuple[str, str]]: ...

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]: ...

    def get_list(
        self, key: str, split_commas: bool = False
    ) -> List[str]: ...

    def update(
        self,
        headers: Optional[
            Union[
                Headers,
                Mapping[str, str],
                Mapping[bytes, bytes],
                Iterable[Tuple[str, str]],
                Iterable[Tuple[bytes, bytes]],
            ]
        ] = None,
    ) -> None: ...

    def copy(self) -> Headers: ...

    def __getitem__(self, key: str) -> str: ...

    def __setitem__(self, key: str, value: str) -> None: ...

    def __delitem__(self, key: str) -> None: ...

    def __contains__(self, key: object) -> bool: ...

    def __iter__(self) -> Iterator[str]: ...

    def __len__(self) -> int: ...

    def __eq__(self, other: object) -> bool: ...

    def __repr__(self) -> str: ...


class Request:
    method: str
    url: URL
    headers: Headers
    extensions: RequestExtensions
    stream: Union[ByteStream, SyncByteStream, AsyncByteStream]

    def __init__(
        self,
        method: str,
        url: Union[str, URL],
        *,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        stream: Optional[Union[SyncByteStream, AsyncByteStream]] = None,
        extensions: Optional[RequestExtensions] = None,
    ) -> None: ...

    def _prepare(self, default_headers: Mapping[bytes, bytes]) -> None: ...

    @property
    def content(self) -> bytes: ...

    def read(self) -> bytes: ...

    async def aread(self) -> bytes: ...

    def __repr__(self) -> str: ...

    def __getstate__(self) -> Dict[str, Any]: ...

    def __setstate__(self, state: Dict[str, Any]) -> None: ...


class Response:
    status_code: int
    headers: Headers
    _request: Optional[Request]
    next_request: Optional[Request]
    extensions: ResponseExtensions
    history: List[Response]
    is_closed: bool
    is_stream_consumed: bool
    default_encoding: Union[str, Callable[[bytes], str]]
    stream: Union[ByteStream, SyncByteStream, AsyncByteStream]
    _num_bytes_downloaded: int

    def __init__(
        self,
        status_code: int,
        *,
        headers: Optional[HeaderTypes] = None,
        content: Optional[ResponseContent] = None,
        text: Optional[str] = None,
        html: Optional[str] = None,
        json: Optional[Any] = None,
        stream: Optional[Union[SyncByteStream, AsyncByteStream]] = None,
        request: Optional[Request] = None,
        extensions: Optional[ResponseExtensions] = None,
        history: Optional[List[Response]] = None,
        default_encoding: Union[str, Callable[[bytes], str]] = "utf-8",
    ) -> None: ...

    def _prepare(self, default_headers: Mapping[bytes, bytes]) -> None: ...

    @property
    def elapsed(self) -> datetime.timedelta: ...

    @elapsed.setter
    def elapsed(self, elapsed: datetime.timedelta) -> None: ...

    @property
    def request(self) -> Request: ...

    @request.setter
    def request(self, value: Request) -> None: ...

    @property
    def http_version(self) -> str: ...

    @property
    def reason_phrase(self) -> str: ...

    @property
    def url(self) -> URL: ...

    @property
    def content(self) -> bytes: ...

    @property
    def text(self) -> str: ...

    @property
    def encoding(self) -> str: ...

    @encoding.setter
    def encoding(self, value: str) -> None: ...

    @property
    def charset_encoding(self) -> Optional[str]: ...

    def _get_content_decoder(self) -> ContentDecoder: ...

    @property
    def is_informational(self) -> bool: ...

    @property
    def is_success(self) -> bool: ...

    @property
    def is_redirect(self) -> bool: ...

    @property
    def is_client_error(self) -> bool: ...

    @property
    def is_server_error(self) -> bool: ...

    @property
    def is_error(self) -> bool: ...

    @property
    def has_redirect_location(self) -> bool: ...

    def raise_for_status(self) -> Response: ...

    def json(self, **kwargs: Any) -> Any: ...

    @property
    def cookies(self) -> Cookies: ...

    @property
    def links(self) -> Dict[str, Dict[str, str]]: ...

    @property
    def num_bytes_downloaded(self) -> int: ...

    def __repr__(self) -> str: ...

    def __getstate__(self) -> Dict[str, Any]: ...

    def __setstate__(self, state: Dict[str, Any]) -> None: ...

    def read(self) -> bytes: ...

    def iter_bytes(
        self, chunk_size: Optional[int] = None
    ) -> Iterator[bytes]: ...

    def iter_text(
        self, chunk_size: Optional[int] = None
    ) -> Iterator[str]: ...

    def iter_lines(self) -> Iterator[str]: ...

    def iter_raw(self, chunk_size: Optional[int] = None) -> Iterator[bytes]: ...

    def close(self) -> None: ...

    async def aread(self) -> bytes: ...

    async def aiter_bytes(
        self, chunk_size: Optional[int] = None
    ) -> AsyncIterator[bytes]: ...

    async def aiter_text(
        self, chunk_size: Optional[int] = None
    ) -> AsyncIterator[str]: ...

    async def aiter_lines(self) -> AsyncIterator[str]: ...

    async def aiter_raw(
        self, chunk_size: Optional[int] = None
    ) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...


class Cookies(MutableMapping[str, str]):
    jar: CookieJar

    def __init__(
        self,
        cookies: Optional[
            Union[
                Cookies,
                CookieJar,
                Dict[str, str],
                List[Tuple[str, str]],
            ]
        ] = None,
    ) -> None: ...

    def extract_cookies(self, response: Response) -> None: ...

    def set_cookie_header(self, request: Request) -> None: ...

    def set(
        self,
        name: str,
        value: str,
        domain: str = "",
        path: str = "/",
    ) -> None: ...

    def get(
        self,
        name: str,
        default: Optional[str] = None,
        domain: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Optional[str]: ...

    def delete(
        self,
        name: str,
        domain: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None: ...

    def clear(
        self, domain: Optional[str] = None, path: Optional[str] = None
    ) -> None: ...

    def update(
        self,
        cookies: Optional[
            Union[
                Cookies,
                CookieJar,
                Dict[str, str],
                List[Tuple[str, str]],
            ]
        ] = None,
    ) -> None: ...

    def __setitem__(self, name: str, value: str) -> None: ...

    def __getitem__(self, name: str) -> str: ...

    def __delitem__(self, name: str) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[str]: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

    class _CookieCompatRequest(urllib.request.Request):
        request: Request

        def __init__(self, request: Request) -> None: ...

        def add_unredirected_header(self, key: str, value: str) -> None: ...

    class _CookieCompatResponse:
        response: Response

        def __init__(self, response: Response) -> None: ...

        def info(self) -> email.message.Message: ...