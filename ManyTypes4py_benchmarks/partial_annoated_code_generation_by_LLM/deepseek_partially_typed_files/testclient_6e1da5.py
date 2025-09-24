from __future__ import annotations
import contextlib
import inspect
import io
import json
import math
import sys
import typing
from concurrent.futures import Future
from types import GeneratorType
from urllib.parse import unquote, urljoin
import anyio
import anyio.abc
import anyio.from_thread
from anyio.streams.stapled import StapledObjectStream
from starlette._utils import is_async_callable
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect
if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard
try:
    import httpx
except ModuleNotFoundError:
    raise RuntimeError('The starlette.testclient module requires the httpx package to be installed.\nYou can install this with:\n    $ pip install httpx\n')
_PortalFactoryType = typing.Callable[[], typing.ContextManager[anyio.abc.BlockingPortal]]
ASGIInstance = typing.Callable[[Receive, Send], typing.Awaitable[None]]
ASGI2App = typing.Callable[[Scope], ASGIInstance]
ASGI3App = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]
_RequestData = typing.Mapping[str, typing.Union[str, typing.Iterable[str], bytes]]

def _is_asgi3(app: typing.Any) -> bool:
    if inspect.isclass(app):
        return hasattr(app, '__await__')
    return is_async_callable(app)

class _WrapASGI2:
    """
    Provide an ASGI3 interface onto an ASGI2 app.
    """

    def __init__(self, app: ASGI2App) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        instance = self.app(scope)
        await instance(receive, send)

class _AsyncBackend(typing.TypedDict):
    backend: str
    backend_options: dict[str, typing.Any]

class _Upgrade(Exception):

    def __init__(self, session: WebSocketTestSession) -> None:
        self.session = session

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect):
    """
    A special case of `WebSocketDisconnect`, raised in the `TestClient` if the
    `WebSocket` is closed before being accepted with a `send_denial_response()`.
    """

class WebSocketTestSession:

    def __init__(self, app: ASGI3App, scope: Scope, portal_factory: _PortalFactoryType) -> None:
        self.app = app
        self.scope = scope
        self.accepted_subprotocol: typing.Optional[str] = None
        self.portal_factory = portal_factory
        self.extra_headers: typing.Optional[typing.List[typing.Tuple[bytes, bytes]]] = None
        self.exit_stack: typing.Optional[contextlib.ExitStack] = None
        self.portal: typing.Optional[anyio.abc.BlockingPortal] = None
        self._receive_tx: typing.Optional[anyio.abc.ObjectSendStream[Message]] = None
        self._send_rx: typing.Optional[anyio.abc.ObjectReceiveStream[Message]] = None

    def __enter__(self) -> WebSocketTestSession:
        with contextlib.ExitStack() as stack:
            self.portal = portal = stack.enter_context(self.portal_factory())
            (fut, cs) = portal.start_task(self._run)
            stack.callback(fut.result)
            stack.callback(portal.call, cs.cancel)
            self.send({'type': 'websocket.connect'})
            message = self.receive()
            self._raise_on_close(message)
            self.accepted_subprotocol = message.get('subprotocol', None)
            self.extra_headers = message.get('headers', None)
            stack.callback(self.close, 1000)
            self.exit_stack = stack.pop_all()
            return self

    def __exit__(self, *args: typing.Any) -> typing.Optional[bool]:
        if self.exit_stack:
            return self.exit_stack.__exit__(*args)
        return None

    async def _run(self, *, task_status: anyio.abc.TaskStatus[anyio.CancelScope]) -> None:
        """
        The sub-thread in which the websocket session runs.
        """
        send: typing.Tuple[anyio.abc.ObjectSendStream[Message], anyio.abc.ObjectReceiveStream[Message]] = anyio.create_memory_object_stream(math.inf)
        (send_tx, send_rx) = send
        receive: typing.Tuple[anyio.abc.ObjectSendStream[Message], anyio.abc.ObjectReceiveStream[Message]] = anyio.create_memory_object_stream(math.inf)
        (receive_tx, receive_rx) = receive
        with send_tx, send_rx, receive_tx, receive_rx, anyio.CancelScope() as cs:
            self._receive_tx = receive_tx
            self._send_rx = send_rx
            task_status.started(cs)
            await self.app(self.scope, receive_rx.receive, send_tx.send)
            await anyio.sleep_forever()

    def _raise_on_close(self, message: Message) -> None:
        if message['type'] == 'websocket.close':
            raise WebSocketDisconnect(code=message.get('code', 1000), reason=message.get('reason', ''))
        elif message['type'] == 'websocket.http.response.start':
            status_code: int = message['status']
            headers: list[tuple[bytes, bytes]] = message['headers']
            body: list[bytes] = []
            while True:
                message = self.receive()
                assert message['type'] == 'websocket.http.response.body'
                body.append(message['body'])
                if not message.get('more_body', False):
                    break
            raise WebSocketDenialResponse(status_code=status_code, headers=headers, content=b''.join(body))

    def send(self, message: Message) -> None:
        if self.portal and self._receive_tx:
            self.portal.call(self._receive_tx.send, message)

    def send_text(self, data: str) -> None:
        self.send({'type': 'websocket.receive', 'text': data})

    def send_bytes(self, data: bytes) -> None:
        self.send({'type': 'websocket.receive', 'bytes': data})

    def send_json(self, data: typing.Any, mode: typing.Literal['text', 'binary']='text') -> None:
        text = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        if mode == 'text':
            self.send({'type': 'websocket.receive', 'text': text})
        else:
            self.send({'type': 'websocket.receive', 'bytes': text.encode('utf-8')})

    def close(self, code: int=1000, reason: typing.Optional[str]=None) -> None:
        self.send({'type': 'websocket.disconnect', 'code': code, 'reason': reason})

    def receive(self) -> Message:
        if self.portal and self._send_rx:
            return self.portal.call(self._send_rx.receive)
        raise RuntimeError("WebSocket session not properly initialized")

    def receive_text(self) -> str:
        message = self.receive()
        self._raise_on_close(message)
        return typing.cast(str, message['text'])

    def receive_bytes(self) -> bytes:
        message = self.receive()
        self._raise_on_close(message)
        return typing.cast(bytes, message['bytes'])

    def receive_json(self, mode: typing.Literal['text', 'binary']='text') -> typing.Any:
        message = self.receive()
        self._raise_on_close(message)
        if mode == 'text':
            text = message['text']
        else:
            text = message['bytes'].decode('utf-8')
        return json.loads(text)

class _TestClientTransport(httpx.BaseTransport):

    def __init__(self, app: ASGI3App, portal_factory: _PortalFactoryType, raise_server_exceptions: bool=True, root_path: str='', *, client: tuple[str, int], app_state: dict[str, typing.Any]) -> None:
        self.app = app
        self.raise_server_exceptions = raise_server_exceptions
        self.root_path = root_path
        self.portal_factory = portal_factory
        self.app_state = app_state
        self.client = client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        scheme = request.url.scheme
        netloc = request.url.netloc.decode(encoding='ascii')
        path = request.url.path
        raw_path = request.url.raw_path
        query = request.url.query.decode(encoding='ascii')
        default_port = {'http': 80, 'ws': 80, 'https': 443, 'wss': 443}[scheme]
        if ':' in netloc:
            (host, port_string) = netloc.split(':', 1)
            port = int(port_string)
        else:
            host = netloc
            port = default_port
        if 'host' in request.headers:
            headers: list[tuple[bytes, bytes]] = []
        elif port == default_port:
            headers = [(b'host', host.encode())]
        else:
            headers = [(b'host', f'{host}:{port}'.encode())]
        headers += [(key.lower().encode(), value.encode()) for (key, value) in request.headers.multi_items()]
        scope: dict[str, typing.Any]
        if scheme in {'ws', 'wss'}:
            subprotocol = request.headers.get('sec-websocket-protocol', None)
            if subprotocol is None:
                subprotocols: typing.Sequence[str] = []
            else:
                subprotocols = [value.strip() for value in subprotocol.split(',')]
            scope = {'type': 'websocket', 'path': unquote(path), 'raw_path': raw_path.split(b'?', 1)[0], 'root_path': self.root_path, 'scheme': scheme, 'query_string': query.encode(), 'headers': headers, 'client': self.client, 'server': [host, port], 'subprotocols': subprotocols, 'state': self.app_state.copy(), 'extensions': {'websocket.http.response': {}}}
            session = WebSocketTestSession(self.app, scope, self.portal_factory)
            raise _Upgrade(session)
        scope = {'type': 'http', 'http_version': '1.1', 'method': request.method, 'path': unquote(path), 'raw_path': raw_path.split(b'?', 1)[0], 'root_path': self.root_path, 'scheme': scheme, 'query_string': query.encode(), 'headers': headers, 'client': self.client, 'server': [host, port], 'extensions': {'http.response.debug': {}}, 'state': self.app_state.copy()}
        request_complete = False
        response_started = False
        response_complete: typing.Optional[anyio.Event] = None
        raw_kwargs: dict[str, typing.Any] = {'stream': io.BytesIO()}
        template: typing.Optional[str] = None
        context: typing.Optional[dict[str, typing.Any]] = None

        async def receive() -> Message:
            nonlocal request_complete
            if request_complete:
                if response_complete and not response_complete.is_set():
                    await response_complete.wait()
                return {'type': 'http.disconnect'}
            body = request.read()
            if isinstance(body, str):
                body_bytes: bytes = body.encode('utf-8')
            elif body is None:
                body_bytes = b''
            elif isinstance(body, GeneratorType):
                try:
                    chunk = body.send(None)
                    if isinstance(chunk, str):
                        chunk = chunk.encode('utf-8')
                    return {'type': 'http.request', 'body': chunk, 'more_body': True}
                except StopIteration:
                    request_complete = True
                    return {'type': 'http.request', 'body': b''}
            else:
                body_bytes = body
            request_complete = True
            return {'type': 'http.request', 'body': body_bytes}

        async def send(message: Message) -> None:
            nonlocal raw_kwargs, response_started, template, context
            if message['type'] == 'http.response.start':
                assert not response_started, 'Received multiple "http.response.start" messages.'
                raw_kwargs['status_code'] = message['status']
                raw_kwargs['headers'] = [(key.decode(), value.decode()) for (key, value) in message.get('headers', [])]
                response_started = True
            elif message['type'] == 'http.response.body':
                assert response_started, 'Received "http.response.body" without "http.response.start".'
                assert response_complete is not None and not response_complete.is_set(), 'Received "http.response.body" after response completed.'
                body = message.get('body', b'')
                more_body = message.get('more_body', False)
                if request.method != 'HEAD':
                    raw_kwargs['stream'].write(body)
                if not more_body:
                    raw_kwargs['stream'].seek(0)
                    response_complete.set()
            elif message['type'] == 'http.response.debug':
                template = message['info']['template']
                context = message['info']['context']
        try:
            with self.portal_factory() as portal:
                response_complete = portal.call(anyio.Event)
                portal.call(self.app, scope, receive, send)
        except BaseException as exc:
            if self.raise_server_exceptions:
                raise exc
        if self.raise_server_exceptions:
            assert response_started, 'TestClient did not receive any response.'
        elif not response_started:
            raw_kwargs = {'status_code': 500, 'headers': [], 'stream': io.BytesIO()}
        raw_kwargs['stream'] = httpx.ByteStream(raw_kwargs['stream'].read())
        response = httpx.Response(**raw_kwargs, request=request)
        if template is not None:
            response.template = template  # type: ignore
            response.context = context  # type: ignore
        return response

class TestClient(httpx.Client):
    __test__ = False
    task: typing.Optional[Future[None]] = None
    portal: typing.Optional[anyio.abc.BlockingPortal] = None
    stream_send: typing.Optional[StapledObjectStream] = None
    stream_receive: typing.Optional[StapledObjectStream] = None
    exit_stack: typing.Optional[contextlib.ExitStack] = None

    def __init__(self, app: ASGIApp, base_url: str='http://testserver', raise_server_exceptions: bool=True, root_path: str='', backend: typing.Literal['asyncio', 'trio']='asyncio', backend_options: typing.Optional[dict[str, typing.Any]]=None, cookies: typing.Optional[httpx._types.CookieTypes]=None, headers: typing.Optional[dict[str, str]]=None, follow_redirects: bool=True, client: tuple[str, int]=('testclient', 50000)) -> None:
        self.async_backend = _AsyncBackend(backend=backend, backend_options=backend_options or {})
        if _is_asgi3(app):
            asgi_app = app
        else:
            app = typing.cast(ASGI2App, app)
            asgi_app = _WrapASGI2(app)
        self.app = asgi_app
        self.app_state: dict[str, typing.Any] = {}
        transport = _TestClientTransport(self.app, portal_factory=self._portal_factory, raise_server_exceptions=raise_server_exceptions, root_path=root_path, app_state=self.app_state, client=client)
        if headers is None:
            headers = {}
        headers.setdefault('user-agent', 'testclient')
        super().__init__(base_url=base_url, headers=headers, transport=transport, follow_redirects=follow_redirects, cookies=cookies)

    @contextlib.contextmanager
    def _portal_factory(self) -> typing.Generator[anyio.abc.BlockingPortal, None, None]:
        if self.portal is not None:
            yield self.portal
        else:
            with anyio.from_thread.start_blocking_portal(**self.async_backend) as portal:
                yield portal

    def request(self, method: str, url: httpx._types.URLTypes, *, content: typing.Optional[httpx._types.RequestContent]=None, data: typing.Optional[_RequestData]=None, files: typing.Optional[httpx._types.RequestFiles]=None, json: typing.Any=None, params: typing.Optional[httpx._types.QueryParamTypes]=None, headers: typing.Optional[httpx._types.HeaderTypes]=None, cookies: typing.Optional[httpx._types.CookieTypes]=None, auth: typing.Union[httpx._types.AuthTypes, httpx._client.UseClientDefault]=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: typing.Union[bool, httpx._client.UseClientDefault]=httpx._client.USE_CLIENT_DEFAULT, timeout: typing.Union[httpx._types.TimeoutTypes, httpx._client.UseClientDefault]=httpx._client.USE_CLIENT_DEFAULT, extensions: typing.Optional[dict[str, typing.Any]]=None) -> httpx.Response:
        url = self._merge_url(url)
        return super().request(method, url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def get(self, url: httpx._types.URLTypes, *, params: typing.Optional[httpx._types.QueryParamTypes]=None, headers: typing.Optional[httpx._types.HeaderTypes]=None, cookies: typing.Optional[httpx._types.CookieTypes]=None, auth: typing.Union[httpx._types.AuthTypes, httpx._client.UseClientDefault]=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: typing.Union[bool,