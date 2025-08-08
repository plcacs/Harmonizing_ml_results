from typing import Any, Dict, Iterator, MutableMapping, Optional, Union

class StreamResponse(MutableMapping[str, Any]):
    _length_check: bool = True
    _body: Optional[Any] = None
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

    def __init__(self, *, status: int = 200, reason: Optional[str] = None, headers: Optional[Dict[str, str]] = None, _real_headers: Optional[Dict[str, str]] = None) -> None:
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

    def __init__(self, *, body: Optional[Union[bytes, bytearray]] = None, status: int = 200, reason: Optional[str] = None, text: Optional[str] = None, headers: Optional[Dict[str, str]] = None, content_type: Optional[str] = None, charset: Optional[str] = None, zlib_executor_size: Optional[int] = None, zlib_executor: Optional[Executor] = None) -> None:
        ...

    @property
    def body(self) -> Optional[Union[bytes, bytearray]]:
        ...

    @property
    def text(self) -> Optional[str]:
        ...

    @property
    def content_length(self) -> Optional[int]:
        ...

    async def write_eof(self, data: Union[bytes, bytearray] = b'') -> None:
        ...

    async def _start(self, request: Any) -> Any:
        ...

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        ...

def json_response(data: Any = sentinel, *, text: Optional[str] = None, body: Optional[Union[bytes, bytearray]] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Dict[str, str]] = None, content_type: str = 'application/json', dumps: JSONEncoder = json.dumps) -> Response:
    ...
