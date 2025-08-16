from typing import Any, Awaitable, Callable, MutableMapping

class ASGIApp(Protocol):

    async def __call__(self, scope, receive, send):
        ...

async def app_lifespan_context(app: ASGIApp) -> AsyncGenerator[None, None]:
    ...

class PrefectResponse(httpx.Response):
    ...

class PrefectHttpxAsyncClient(httpx.AsyncClient):
    ...

class PrefectHttpxSyncClient(httpx.Client):
    ...

def determine_server_type() -> ServerType:
    ...
