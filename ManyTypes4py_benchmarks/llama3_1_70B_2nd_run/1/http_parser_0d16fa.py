class RawRequestMessage(NamedTuple):
    method: str
    path: str
    version: HttpVersion
    headers: CIMultiDictProxy[str]
    raw_headers: tuple[tuple[bytes, bytes], ...]
    close: bool
    compression: str
    upgrade: bool
    chunked: bool
    url: URL

class RawResponseMessage(NamedTuple):
    version: HttpVersion
    status: int
    reason: str
    headers: CIMultiDictProxy[str]
    raw_headers: tuple[tuple[bytes, bytes], ...]
    close: bool
    compression: str
    upgrade: bool
    chunked: bool

class HeadersParser:
    def __init__(self, max_line_size: int = 8190, max_field_size: int = 8190, lax: bool = False) -> None:
        self.max_line_size = max_line_size
        self.max_field_size = max_field_size
        self._lax = lax

    def parse_headers(self, lines: list[bytes]) -> tuple[CIMultiDictProxy[str], tuple[tuple[bytes, bytes], ...]]:
        ...

class HttpParser(abc.ABC, Generic[_MsgT]):
    lax: bool = False

    def __init__(self, 
                 protocol: BaseProtocol, 
                 loop: asyncio.AbstractEventLoop, 
                 limit: int, 
                 max_line_size: int = 8190, 
                 max_field_size: int = 8190, 
                 timer: Optional[BaseTimerContext] = None, 
                 code: Optional[int] = None, 
                 method: Optional[str] = None, 
                 payload_exception: Optional[Type[Exception]] = None, 
                 response_with_body: bool = True, 
                 read_until_eof: bool = False, 
                 auto_decompress: bool = True) -> None:
        ...

    @abc.abstractmethod
    def parse_message(self, lines: list[bytes]) -> _MsgT:
        ...

    @abc.abstractmethod
    def _is_chunked_te(self, te: str) -> bool:
        ...

    def feed_eof(self) -> Optional[_MsgT]:
        ...

    def feed_data(self, 
                  data: bytes, 
                  SEP: bytes = b'\r\n', 
                  EMPTY: bytes = b'', 
                  CONTENT_LENGTH: str = hdrs.CONTENT_LENGTH, 
                  METH_CONNECT: str = hdrs.METH_CONNECT, 
                  SEC_WEBSOCKET_KEY1: str = hdrs.SEC_WEBSOCKET_KEY1) -> tuple[list[tuple[_MsgT, StreamReader]], bool, bytes]:
        ...

    def parse_headers(self, lines: list[bytes]) -> tuple[CIMultiDictProxy[str], tuple[tuple[bytes, bytes], ...], Optional[bool], Optional[str], bool, bool]:
        ...

    def set_upgraded(self, val: bool) -> None:
        ...

class HttpRequestParser(HttpParser[RawRequestMessage]):
    ...

    def parse_message(self, lines: list[bytes]) -> RawRequestMessage:
        ...

    def _is_chunked_te(self, te: str) -> bool:
        ...

class HttpResponseParser(HttpParser[RawResponseMessage]):
    lax: bool = not DEBUG

    def feed_data(self, 
                  data: bytes, 
                  SEP: Optional[bytes] = None, 
                  *args, 
                  **kwargs) -> tuple[list[tuple[RawResponseMessage, StreamReader]], bool, bytes]:
        ...

    def parse_message(self, lines: list[bytes]) -> RawResponseMessage:
        ...

    def _is_chunked_te(self, te: str) -> bool:
        ...

class HttpPayloadParser:
    def __init__(self, 
                 payload: StreamReader, 
                 length: Optional[int] = None, 
                 chunked: bool = False, 
                 compression: Optional[str] = None, 
                 code: Optional[int] = None, 
                 method: Optional[str] = None, 
                 response_with_body: bool = True, 
                 auto_decompress: bool = True, 
                 lax: bool = False) -> None:
        ...

    def feed_eof(self) -> None:
        ...

    def feed_data(self, chunk: bytes, SEP: bytes = b'\r\n', CHUNK_EXT: bytes = b';') -> tuple[bool, bytes]:
        ...

class DeflateBuffer:
    def __init__(self, out: StreamReader, encoding: str) -> None:
        ...

    def set_exception(self, exc: Exception, exc_cause: Optional[Exception] = None) -> None:
        ...

    def feed_data(self, chunk: bytes) -> None:
        ...

    def feed_eof(self) -> None:
        ...

    def begin_http_chunk_receiving(self) -> None:
        ...

    def end_http_chunk_receiving(self) -> None:
        ...
