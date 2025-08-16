from typing import Any, Dict, Iterator, MutableMapping, Optional, Union, cast

class StreamResponse(MutableMapping[str, Any]):
    _length_check: bool = True
    _body: Optional[bytes] = None
    _keep_alive: Optional[bool] = None
    _chunked: bool = False
    _compression: bool = False
    _compression_strategy: int = zlib.Z_DEFAULT_STRATEGY
    _compression_force: Optional[ContentCoding] = None
    _req: Optional[Any] = None
    _payload_writer: Optional[AbstractStreamWriter] = None
    _eof_sent: bool = False
    _must_be_empty_body: Optional[bool] = None
    _body_length: int = 0

    def __init__(self, *, status: int = 200, reason: Optional[str] = None, headers: Optional[Dict[str, str]] = None, _real_headers: Optional[CIMultiDict] = None) -> None:
        ...

    @property
    def prepared(self) -> bool:
        ...

    @property
    def task(self) -> Optional[Any]:
        ...

    @property
    def status(self) -> int:
        ...

    @property
    def chunked(self) -> bool:
        ...

    @property
    def compression(self) -> bool:
        ...

    @property
    def reason(self) -> str:
        ...

    def set_status(self, status: int, reason: Optional[str] = None) -> None:
        ...

    def _set_status(self, status: int, reason: Optional[str]) -> None:
        ...

    @property
    def keep_alive(self) -> Optional[bool]:
        ...

    def force_close(self) -> None:
        ...

    @property
    def body_length(self) -> int:
        ...

    def enable_chunked_encoding(self) -> None:
        ...

    def enable_compression(self, force: Optional[ContentCoding] = None, strategy: int = zlib.Z_DEFAULT_STRATEGY) -> None:
        ...

    @property
    def headers(self) -> CIMultiDict:
        ...

    @property
    def content_length(self) -> Optional[int]:
        ...

    @content_length.setter
    def content_length(self, value: Optional[int]) -> None:
        ...

    @property
    def content_type(self) -> str:
        ...

    @content_type.setter
    def content_type(self, value: str) -> None:
        ...

    @property
    def charset(self) -> Optional[str]:
        ...

    @charset.setter
    def charset(self, value: Optional[str]) -> None:
        ...

    @property
    def last_modified(self) -> Optional[datetime.datetime]:
        ...

    @last_modified.setter
    def last_modified(self, value: Optional[Union[int, float, datetime.datetime, str]]) -> None:
        ...

    @property
    def etag(self) -> Optional[ETag]:
        ...

    @etag.setter
    def etag(self, value: Optional[Union[str, ETag]]) -> None:
        ...

    def _generate_content_type_header(self, CONTENT_TYPE: str = hdrs.CONTENT_TYPE) -> None:
        ...

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        ...

    async def _start_compression(self, request: 'BaseRequest') -> None:
        ...

    async def prepare(self, request: 'BaseRequest') -> Optional[AbstractStreamWriter]:
        ...

    async def _start(self, request: 'BaseRequest') -> AbstractStreamWriter:
        ...

    async def _prepare_headers(self) -> None:
        ...

    async def _write_headers(self) -> None:
        ...

    async def write(self, data: Union[bytes, bytearray, memoryview]) -> None:
        ...

    async def drain(self) -> None:
        ...

    async def write_eof(self, data: Union[bytes, bytearray, memoryview] = b'') -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        ...

    def __delitem__(self, key: str) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __bool__(self) -> bool:
        ...

class Response(StreamResponse):
    _compressed_body: Optional[bytes] = None

    def __init__(self, *, body: Optional[bytes] = None, status: int = 200, reason: Optional[str] = None, text: Optional[str] = None, headers: Optional[Dict[str, str]] = None, content_type: Optional[str] = None, charset: Optional[str] = None, zlib_executor_size: Optional[int] = None, zlib_executor: Optional[Executor] = None) -> None:
        ...

    @property
    def body(self) -> Optional[bytes]:
        ...

    @body.setter
    def body(self, body: Optional[bytes]) -> None:
        ...

    @property
    def text(self) -> Optional[str]:
        ...

    @text.setter
    def text(self, text: str) -> None:
        ...

    @property
    def content_length(self) -> Optional[int]:
        ...

    @content_length.setter
    def content_length(self, value: Optional[int]) -> None:
        ...

    async def write_eof(self, data: Union[bytes, bytearray, memoryview] = b'') -> None:
        ...

    async def _start(self, request: 'BaseRequest') -> AbstractStreamWriter:
        ...

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        ...

def json_response(data: Any = sentinel, *, text: Optional[str] = None, body: Optional[bytes] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Dict[str, str]] = None, content_type: str = 'application/json', dumps: JSONEncoder = json.dumps) -> Response:
    ...
