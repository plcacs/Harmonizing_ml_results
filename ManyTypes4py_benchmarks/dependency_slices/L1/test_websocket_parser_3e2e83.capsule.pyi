from typing import Any

# === Internal dependency: aiohttp._websocket.helpers ===
def _websocket_mask_python(mask, data): ...
PACK_LEN1 = ...
PACK_LEN2 = ...
PACK_CLOSE_CODE = ...

# === Internal dependency: aiohttp._websocket.models ===
WS_DEFLATE_TRAILING = bytes(...)

# === Internal dependency: aiohttp._websocket.reader ===
WebSocketDataQueue: Any

# === Internal dependency: aiohttp.base_protocol ===
class BaseProtocol(asyncio.Protocol):
    def __init__(self, loop): ...
    def connection_made(self, transport): ...

# === Internal dependency: aiohttp.http ===
from .http_websocket import WebSocketError
from .http_websocket import WSCloseCode
from .http_websocket import WSMsgType

# === Internal dependency: aiohttp.http_websocket ===
from ._websocket.models import WSMessageBinary
from ._websocket.models import WSMessageClose
from ._websocket.models import WSMessagePing
from ._websocket.models import WSMessagePong
from ._websocket.models import WSMessageText
from ._websocket.reader import WebSocketReader

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises