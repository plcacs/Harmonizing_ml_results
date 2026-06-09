from typing import Any

# === Internal dependency: aiohttp._websocket.helpers ===
def _websocket_mask_python(mask: bytes, data: bytearray) -> None: ...
PACK_LEN1: Any
PACK_LEN2: Any
PACK_CLOSE_CODE: Any

# === Internal dependency: aiohttp._websocket.models ===
WS_DEFLATE_TRAILING: Final[bytes]

# === Internal dependency: aiohttp._websocket.reader ===
WebSocketDataQueue: Any

# === Internal dependency: aiohttp.base_protocol ===
class BaseProtocol(asyncio.Protocol):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None: ...
    def connection_made(self, transport: asyncio.BaseTransport) -> None: ...

# === Internal dependency: aiohttp.http ===
# re-export: from .http_websocket import WebSocketError
# re-export: from .http_websocket import WSCloseCode
# re-export: from .http_websocket import WSMsgType

# === Internal dependency: aiohttp.http_websocket ===
# re-export: from ._websocket.models import WSMessageBinary
# re-export: from ._websocket.models import WSMessageClose
# re-export: from ._websocket.models import WSMessagePing
# re-export: from ._websocket.models import WSMessagePong
# re-export: from ._websocket.models import WSMessageText
# re-export: from ._websocket.reader import WebSocketReader

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises