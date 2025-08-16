from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Union
import aiohttp_cors
from aiohttp import __version__ as aiohttp_version
from aiohttp.web import AppRunner, Application, BaseSite, Request, Response, TCPSite, UnixSite
from aiohttp.payload import Payload
from aiohttp_cors import CorsConfig, ResourceOptions
from faust.types import AppT
from faust.utils import json as _json
from faust.web import base
from faust.types.web import ResourceOptions as _ResourceOptions
from mode import Service
from mode.threads import ServiceThread

_bytes: Any = bytes
NON_OPTIONS_METHODS: Any = frozenset({'GET', 'PUT', 'POST', 'DELETE'})

def _prepare_cors_options(opts: Mapping[str, Any]) -> Mapping[str, Any]:
    return {k: _faust_to_aiohttp_options(v) for k, v in opts.items()}

def _faust_to_aiohttp_options(opts: _ResourceOptions) -> ResourceOptions:
    return ResourceOptions(**opts._asdict())

class ServerThread(ServiceThread):
    def __init__(self, web: Any, **kwargs: Any) -> None:
        self.web = web
        super().__init__(**kwargs)

    async def on_start(self) -> None:
        await self.web.start_server()

    async def on_thread_stop(self) -> None:
        await self.web.stop_server()

class Server(Service):
    def __init__(self, web: Any, **kwargs: Any) -> None:
        self.web = web
        super().__init__(**kwargs)

    async def on_start(self) -> None:
        await self.web.start_server()

    async def on_stop(self) -> None:
        await self.web.stop_server()

class Web(base.Web):
    driver_version: str = f'aiohttp={aiohttp_version}'
    handler_shutdown_timeout: float = 60.0
    _thread: Optional[Any] = None
    _cors: Optional[Any] = None

    def __init__(self, app: AppT, **kwargs: Any) -> None:
        super().__init__(app, **kwargs)
        self.web_app: Application = Application()
        self.cors_options: Mapping[str, Any] = _prepare_cors_options(app.conf.web_cors_options or {})
        self._runner: AppRunner = AppRunner(self.web_app, access_log=None)
        self._transport_handlers: Mapping[str, Callable[[], BaseSite]] = {'tcp': self._new_transport_tcp, 'unix': self._new_transport_unix}

    @property
    def cors(self) -> CorsConfig:
        if self._cors is None:
            self._cors = aiohttp_cors.setup(self.web_app, defaults=self.cors_options)
        return self._cors

    async def on_start(self) -> None:
        cors = self.cors
        assert cors
        self.init_server()
        server_cls = ServerThread if self.app.conf.web_in_thread else Server
        self._thread = server_cls(self, loop=self.loop, beacon=self.beacon)
        self.add_dependency(self._thread)

    async def wsgi(self) -> Application:
        self.init_server()
        return self.web_app

    def text(self, value: str, *, content_type: Optional[str] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Mapping[str, str]] = None) -> base.Response:
        response = Response(text=value, content_type=content_type, status=status, reason=reason, headers=headers)
        return cast(base.Response, response)

    def html(self, value: str, *, content_type: Optional[str] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Mapping[str, str]] = None) -> base.Response:
        return self.text(value, status=status, content_type=content_type or 'text/html', reason=reason, headers=headers)

    def json(self, value: Any, *, content_type: Optional[str] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Mapping[str, str]] = None) -> base.Response:
        ctype = content_type or 'application/json'
        payload = _json.dumps(value)
        if isinstance(payload, _bytes):
            return self.bytes(payload, content_type=ctype, status=status, reason=reason, headers=headers)
        else:
            return self.text(payload, content_type=ctype, status=status, reason=reason, headers=headers)

    def bytes(self, value: _bytes, *, content_type: Optional[str] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Mapping[str, str]] = None) -> base.Response:
        response = Response(body=value, content_type=content_type, status=status, reason=reason, headers=headers)
        return cast(base.Response, response)

    async def read_request_content(self, request: Request) -> _bytes:
        return await cast(Request, request).content.read()

    def route(self, pattern: str, handler: Callable, cors_options: Optional[Mapping[str, Any]] = None) -> None:
        if cors_options or self.cors_options:
            async_handler = self._wrap_into_asyncdef(handler)
            for method in NON_OPTIONS_METHODS:
                r = self.web_app.router.add_route(method, pattern, async_handler)
                self.cors.add(r, _prepare_cors_options(cors_options or {}))
        else:
            self.web_app.router.add_route('*', pattern, self._wrap_into_asyncdef(handler))

    def _wrap_into_asyncdef(self, handler: Callable) -> Callable:
        async def _dispatch(request: Request) -> Any:
            return await handler(request)
        return _dispatch

    def add_static(self, prefix: str, path: Path, **kwargs: Any) -> None:
        self.web_app.router.add_static(prefix, str(path), **kwargs)

    def bytes_to_response(self, s: _bytes) -> base.Response:
        status, headers, body = self._bytes_to_response(s)
        response = Response(body=body, status=status, headers=headers)
        return cast(base.Response, response)

    def response_to_bytes(self, response: base.Response) -> _bytes:
        resp = cast(Response, response)
        if resp.body is None:
            body = b''
        elif isinstance(resp.body, Payload):
            raise NotImplementedError('Does not support Payload')
        else:
            body = resp.body
        return self._response_to_bytes(resp.status, resp.headers, body)

    def _create_site(self) -> BaseSite:
        return self._new_transport(self.app.conf.web_transport.scheme)

    def _new_transport(self, type_: str) -> BaseSite:
        return self._transport_handlers[type_]()

    def _new_transport_tcp(self) -> TCPSite:
        return TCPSite(self._runner, self.app.conf.web_bind, self.app.conf.web_port)

    def _new_transport_unix(self) -> UnixSite:
        return UnixSite(self._runner, self.app.conf.web_transport.path)

    async def start_server(self) -> None:
        await self._runner.setup()
        site = self._create_site()
        await site.start()

    async def stop_server(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        await self._cleanup_app()

    async def _cleanup_app(self) -> None:
        if self.web_app is not None:
            self.log.info('Cleanup')
            await self.web_app.cleanup()

    @property
    def _app(self) -> Application:
        return self.web_app
