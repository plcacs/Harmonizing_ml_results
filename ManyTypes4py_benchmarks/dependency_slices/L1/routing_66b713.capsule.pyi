from typing import Any

# === Internal dependency: starlette._exception_handler ===
def wrap_app_handling_exceptions(app, conn): ...

# === Internal dependency: starlette._utils ===
def is_async_callable(obj): ...
def get_route_path(scope): ...

# === Internal dependency: starlette.concurrency ===
async def run_in_threadpool(func, *args, **kwargs): ...

# === Internal dependency: starlette.convertors ===
class Convertor(typing.Generic[T]): ...
class StringConvertor(Convertor[str]):
    ...
class PathConvertor(Convertor[str]):
class IntegerConvertor(Convertor[int]):
class FloatConvertor(Convertor[float]):
class UUIDConvertor(Convertor[uuid.UUID]):
CONVERTOR_TYPES = {'str': StringConvertor(...), 'path': PathConvertor(...), 'int': IntegerConvertor(...), 'float': FloatConvertor(...), 'uuid': UUIDConvertor(...)}

# === Internal dependency: starlette.datastructures ===
class URL:
    def __init__(self, url=..., scope=..., **components): ...
class URLPath(str): ...
class Headers(typing.Mapping[str, str]):
    def __init__(self, headers=..., raw=..., scope=...): ...

# === Internal dependency: starlette.exceptions ===
class HTTPException(Exception): ...

# === Internal dependency: starlette.requests ===
class Request(HTTPConnection):
    def __init__(self, scope, receive=..., send=...): ...

# === Internal dependency: starlette.responses ===
class Response: ...
class PlainTextResponse(Response):
    ...
class RedirectResponse(Response):
    def __init__(self, url, status_code=..., headers=..., background=...): ...

# === Internal dependency: starlette.types ===
AppType = typing.TypeVar(...)
Scope = typing.MutableMapping[str, typing.Any]
Message = typing.MutableMapping[str, typing.Any]
Receive = typing.Callable[[], typing.Awaitable[Message]]
Send = typing.Callable[[Message], typing.Awaitable[None]]
ASGIApp = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]
StatelessLifespan = typing.Callable[[AppType], typing.AsyncContextManager[None]]
StatefulLifespan = typing.Callable[[AppType], typing.AsyncContextManager[typing.Mapping[str, typing.Any]]]
Lifespan = typing.Union[StatelessLifespan[AppType], StatefulLifespan[AppType]]

# === Internal dependency: starlette.websockets ===
class WebSocket(HTTPConnection):
    def __init__(self, scope, receive, send): ...
class WebSocketClose:
    def __init__(self, code=..., reason=...): ...