from typing import Any

# === Third-party dependency: aiohttp ===
# Used symbols: ClientSession

# === Third-party dependency: aiohttp.client_exceptions ===
class ClientError(Exception): ...
class ClientResponseError(ClientError): ...
class ClientConnectionError(ClientError): ...

# === Third-party dependency: aiohttp.streams ===
class StreamReader(AsyncStreamReaderMixin):
    def __init__(self, protocol: BaseProtocol, limit: int, *, timer: Optional[BaseTimerContext] = ..., loop: Optional[asyncio.AbstractEventLoop] = ...) -> None: ...
    def feed_eof(self) -> None: ...
    def feed_data(self, data: bytes, size: int = ...) -> None: ...

# === Internal dependency: homeassistant.const ===
EVENT_HOMEASSISTANT_CLOSE: EventType[NoEventData]

# === Internal dependency: homeassistant.helpers.json ===
def json_dumps(data: Any) -> str: ...

# === Internal dependency: homeassistant.util.json ===
def json_loads(__obj: bytes | bytearray | memoryview | str) -> JsonValueType: ...

# === Third-party dependency: multidict ===
# Used symbols: CIMultiDict

# === Third-party dependency: yarl ===
# Used symbols: URL