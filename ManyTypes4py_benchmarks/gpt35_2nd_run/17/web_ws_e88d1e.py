from typing import Any, Final, Iterable, Optional, Tuple, Union
from multidict import CIMultiDict
from . import hdrs
from ._websocket.reader import WebSocketDataQueue
from ._websocket.writer import DEFAULT_LIMIT
from .abc import AbstractStreamWriter
from .client_exceptions import WSMessageTypeError
from .helpers import calculate_timeout_when, frozen_dataclass_decorator, set_exception, set_result
from .http import WS_CLOSED_MESSAGE, WS_CLOSING_MESSAGE, WS_KEY, WebSocketError, WebSocketReader, WebSocketWriter, WSCloseCode, WSMessage, WSMsgType, ws_ext_gen, ws_ext_parse
from .http_websocket import _INTERNAL_RECEIVE_TYPES, WSMessageError
from .log import ws_logger
from .streams import EofStream
from .typedefs import JSONDecoder, JSONEncoder
from .web_exceptions import HTTPBadRequest, HTTPException
from .web_request import BaseRequest
from .web_response import StreamResponse
if sys.version_info >= (3, 11):
    import asyncio as async_timeout
else:
    import async_timeout

__all__: Tuple[str, ...] = ('WebSocketResponse', 'WebSocketReady', 'WSMsgType')

THRESHOLD_CONNLOST_ACCESS: Final[int] = 5

@frozen_dataclass_decorator
class WebSocketReady:
    ok: bool

class WebSocketResponse(StreamResponse):
    _length_check: bool = False
    _ws_protocol: Optional[str] = None
    _writer: Optional[WebSocketWriter] = None
    _reader: Optional[WebSocketDataQueue] = None
    _closed: bool = False
    _closing: bool = False
    _conn_lost: int = 0
    _close_code: Optional[WSCloseCode] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _waiting: bool = False
    _close_wait: Optional[asyncio.Future] = None
    _exception: Optional[Exception] = None
    _heartbeat_when: float = 0.0
    _heartbeat_cb: Optional[asyncio.TimerHandle] = None
    _pong_response_cb: Optional[asyncio.TimerHandle] = None
    _ping_task: Optional[asyncio.Task] = None

    def __init__(self, *, timeout: float = 10.0, receive_timeout: Optional[float] = None, autoclose: bool = True, autoping: bool = True, heartbeat: Optional[float] = None, protocols: Iterable[str] = (), compress: bool = True, max_msg_size: int = 4 * 1024 * 1024, writer_limit: int = DEFAULT_LIMIT) -> None:
        super().__init__(status=101)
        self._protocols: Iterable[str] = protocols
        self._timeout: float = timeout
        self._receive_timeout: Optional[float] = receive_timeout
        self._autoclose: bool = autoclose
        self._autoping: bool = autoping
        self._heartbeat: Optional[float] = heartbeat
        if heartbeat is not None:
            self._pong_heartbeat: float = heartbeat / 2.0
        self._compress: bool = compress
        self._max_msg_size: int = max_msg_size
        self._writer_limit: int = writer_limit

    def _cancel_heartbeat(self) -> None:
        ...

    def _cancel_pong_response_cb(self) -> None:
        ...

    def _reset_heartbeat(self) -> None:
        ...

    def _send_heartbeat(self) -> None:
        ...

    def _ping_task_done(self, task: asyncio.Task) -> None:
        ...

    def _pong_not_received(self) -> None:
        ...

    def _handle_ping_pong_exception(self, exc: Exception) -> None:
        ...

    def _set_closed(self) -> None:
        ...

    async def prepare(self, request: BaseRequest) -> AbstractStreamWriter:
        ...

    def _handshake(self, request: BaseRequest) -> Tuple[CIMultiDict, Optional[str], int, bool]:
        ...

    def _pre_start(self, request: BaseRequest) -> Tuple[Optional[str], WebSocketWriter]:
        ...

    def _post_start(self, request: BaseRequest, protocol: Optional[str], writer: WebSocketWriter) -> None:
        ...

    def can_prepare(self, request: BaseRequest) -> WebSocketReady:
        ...

    @property
    def closed(self) -> bool:
        ...

    @property
    def close_code(self) -> Optional[WSCloseCode]:
        ...

    @property
    def ws_protocol(self) -> Optional[str]:
        ...

    @property
    def compress(self) -> bool:
        ...

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        ...

    def exception(self) -> Optional[Exception]:
        ...

    async def ping(self, message: bytes = b'') -> None:
        ...

    async def pong(self, message: bytes = b'') -> None:
        ...

    async def send_frame(self, message: bytes, opcode: WSMsgType, compress: Optional[int] = None) -> None:
        ...

    async def send_str(self, data: str, compress: Optional[int] = None) -> None:
        ...

    async def send_bytes(self, data: Union[bytes, bytearray, memoryview], compress: Optional[int] = None) -> None:
        ...

    async def send_json(self, data: Any, compress: Optional[int] = None, *, dumps: JSONEncoder = json.dumps) -> None:
        ...

    async def write_eof(self) -> None:
        ...

    async def close(self, *, code: WSCloseCode = WSCloseCode.OK, message: bytes = b'', drain: bool = True) -> bool:
        ...

    def _set_closing(self, code: WSCloseCode) -> None:
        ...

    def _set_code_close_transport(self, code: WSCloseCode) -> None:
        ...

    def _close_transport(self) -> None:
        ...

    async def receive(self, timeout: Optional[float] = None) -> Union[WSMessage, WSMessageError]:
        ...

    async def receive_str(self, *, timeout: Optional[float] = None) -> str:
        ...

    async def receive_bytes(self, *, timeout: Optional[float] = None) -> bytes:
        ...

    async def receive_json(self, *, loads: JSONDecoder = json.loads, timeout: Optional[float] = None) -> Any:
        ...

    async def write(self, data: Any) -> None:
        ...

    def __aiter__(self) -> 'WebSocketResponse':
        ...

    async def __anext__(self) -> WSMessage:
        ...

    def _cancel(self, exc: Exception) -> None:
        ...
