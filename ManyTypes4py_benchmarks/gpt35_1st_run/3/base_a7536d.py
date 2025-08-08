from typing import Any, Awaitable, Callable, MutableMapping

class ASGIApp(Protocol):

    async def __call__(self, scope, receive, send):
        ...

async def app_lifespan_context(app: ASGIApp) -> AsyncGenerator[None, None]:
    ...

class PrefectResponse(httpx.Response):

    def raise_for_status(self) -> None:
        ...

    @classmethod
    def from_httpx_response(cls, response: httpx.Response) -> 'PrefectResponse':
        ...

class PrefectHttpxAsyncClient(httpx.AsyncClient):

    def __init__(self, *args, enable_csrf_support: bool = False, raise_on_all_errors: bool = True, **kwargs) -> None:
        ...

    async def _send_with_retry(self, request: Request, send: Callable, send_args: tuple, send_kwargs: dict, retry_codes: set, retry_exceptions: tuple) -> Response:
        ...

    async def send(self, request: Request, *args, **kwargs) -> PrefectResponse:
        ...

    async def _add_csrf_headers(self, request: Request) -> None:
        ...

class PrefectHttpxSyncClient(httpx.Client):

    def __init__(self, *args, enable_csrf_support: bool = False, raise_on_all_errors: bool = True, **kwargs) -> None:
        ...

    def _send_with_retry(self, request: Request, send: Callable, send_args: tuple, send_kwargs: dict, retry_codes: set, retry_exceptions: tuple) -> Response:
        ...

    def send(self, request: Request, *args, **kwargs) -> PrefectResponse:
        ...

    def _add_csrf_headers(self, request: Request) -> None:
        ...

def determine_server_type() -> ServerType:
    ...
