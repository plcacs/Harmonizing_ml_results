from typing import Any

# === Third-party dependency: anyio ===
# Used symbols: CancelScope, Event, create_memory_object_stream, sleep_forever

# === Third-party dependency: anyio.abc ===
# Used symbols: BlockingPortal

# === Third-party dependency: anyio.from_thread ===
def start_blocking_portal(backend: str = ..., backend_options: dict[str, Any] | None = ..., *, name: str | None = ...) -> Generator[BlockingPortal, Any, None]: ...

# === Third-party dependency: anyio.streams.stapled ===
class StapledObjectStream(Generic[T_Item], ObjectStream[T_Item]):
    ...

# === Third-party dependency: httpx ===
# Used symbols: BaseTransport, ByteStream, Client, Response, _client

# === Internal dependency: starlette._utils ===
def is_async_callable(obj: AwaitableCallable[T]) -> TypeGuard[AwaitableCallable[T]]: ...
def is_async_callable(obj: typing.Any) -> TypeGuard[AwaitableCallable[typing.Any]]: ...
def is_async_callable(obj: typing.Any) -> typing.Any: ...

# === Internal dependency: starlette.types ===
Scope: Any
Message: Any
Receive: Any
Send: Any

# === Internal dependency: starlette.websockets ===
class WebSocketDisconnect(Exception):
    ...