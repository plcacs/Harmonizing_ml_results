from typing import Any

# === Internal dependency: starlette.datastructures ===
class URLPath(str): ...
class State:
    def __init__(self, state=...): ...

# === Internal dependency: starlette.middleware ===
class Middleware: ...

# === Internal dependency: starlette.middleware.base ===
class BaseHTTPMiddleware: ...

# === Internal dependency: starlette.middleware.errors ===
class ServerErrorMiddleware: ...

# === Internal dependency: starlette.middleware.exceptions ===
class ExceptionMiddleware: ...

# === Internal dependency: starlette.requests ===
class Request(HTTPConnection): ...

# === Internal dependency: starlette.responses ===
class Response: ...

# === Internal dependency: starlette.routing ===
class BaseRoute: ...
class Router:
    def __init__(self, routes=..., redirect_slashes=..., default=..., on_startup=..., on_shutdown=..., lifespan=..., *, middleware=...): ...

# === Internal dependency: starlette.types ===
Scope = typing.MutableMapping[str, typing.Any]
Message = typing.MutableMapping[str, typing.Any]
Receive = typing.Callable[[], typing.Awaitable[Message]]
Send = typing.Callable[[Message], typing.Awaitable[None]]
ASGIApp = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]