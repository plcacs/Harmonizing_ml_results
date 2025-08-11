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

def _is_asgi3(app: Any) -> bool:
    if inspect.isclass(app):
        return hasattr(app, '__await__')
    return is_async_callable(app)

class _WrapASGI2:
    """
    Provide an ASGI3 interface onto an ASGI2 app.
    """

    def __init__(self, app: Union[starlette.types.ASGIApp, typing.Any, None]) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        instance = self.app(scope)
        await instance(receive, send)

class _AsyncBackend(typing.TypedDict):
    pass

class _Upgrade(Exception):

    def __init__(self, session) -> None:
        self.session = session

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect):
    """
    A special case of `WebSocketDisconnect`, raised in the `TestClient` if the
    `WebSocket` is closed before being accepted with a `send_denial_response()`.
    """

class WebSocketTestSession:

    def __init__(self, app: Union[starlette.types.ASGIApp, typing.Any, None], scope, portal_factory) -> None:
        self.app = app
        self.scope = scope
        self.accepted_subprotocol = None
        self.portal_factory = portal_factory
        self.extra_headers = None

    def __enter__(self) -> WebSocketTestSession:
        with contextlib.ExitStack() as stack:
            self.portal = portal = stack.enter_context(self.portal_factory())
            fut, cs = portal.start_task(self._run)
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

    def __exit__(self, *args):
        return self.exit_stack.__exit__(*args)

    async def _run(self, *, task_status):
        """
        The sub-thread in which the websocket session runs.
        """
        send = anyio.create_memory_object_stream(math.inf)
        send_tx, send_rx = send
        receive = anyio.create_memory_object_stream(math.inf)
        receive_tx, receive_rx = receive
        with send_tx, send_rx, receive_tx, receive_rx, anyio.CancelScope() as cs:
            self._receive_tx = receive_tx
            self._send_rx = send_rx
            task_status.started(cs)
            await self.app(self.scope, receive_rx.receive, send_tx.send)
            await anyio.sleep_forever()

    def _raise_on_close(self, message: Union[message.Message, dict]) -> None:
        if message['type'] == 'websocket.close':
            raise WebSocketDisconnect(code=message.get('code', 1000), reason=message.get('reason', ''))
        elif message['type'] == 'websocket.http.response.start':
            status_code = message['status']
            headers = message['headers']
            body = []
            while True:
                message = self.receive()
                assert message['type'] == 'websocket.http.response.body'
                body.append(message['body'])
                if not message.get('more_body', False):
                    break
            raise WebSocketDenialResponse(status_code=status_code, headers=headers, content=b''.join(body))

    def send(self, message: Any) -> None:
        self.portal.call(self._receive_tx.send, message)

    def send_text(self, data: Union[bytes, golem_messages.message.Message, list[float]]) -> None:
        self.send({'type': 'websocket.receive', 'text': data})

    def send_bytes(self, data: bytes) -> None:
        self.send({'type': 'websocket.receive', 'bytes': data})

    def send_json(self, data: Union[dict, str, None], mode: typing.Text='text') -> None:
        text = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        if mode == 'text':
            self.send({'type': 'websocket.receive', 'text': text})
        else:
            self.send({'type': 'websocket.receive', 'bytes': text.encode('utf-8')})

    def close(self, code: int=1000, reason: Union[None, str, int, bytes]=None) -> None:
        self.send({'type': 'websocket.disconnect', 'code': code, 'reason': reason})

    def receive(self) -> Union[dict[typing.Text, typing.Text], dict[typing.Text, typing.Union[typing.Text,bytes,bool]], dict[typing.Text, typing.Union[typing.Text,bytes]]]:
        return self.portal.call(self._send_rx.receive)

    def receive_text(self):
        message = self.receive()
        self._raise_on_close(message)
        return typing.cast(str, message['text'])

    def receive_bytes(self):
        message = self.receive()
        self._raise_on_close(message)
        return typing.cast(bytes, message['bytes'])

    def receive_json(self, mode: typing.Text='text'):
        message = self.receive()
        self._raise_on_close(message)
        if mode == 'text':
            text = message['text']
        else:
            text = message['bytes'].decode('utf-8')
        return json.loads(text)

class _TestClientTransport(httpx.BaseTransport):

    def __init__(self, app: Union[starlette.types.ASGIApp, typing.Any, None], portal_factory, raise_server_exceptions=True, root_path='', *, client, app_state) -> None:
        self.app = app
        self.raise_server_exceptions = raise_server_exceptions
        self.root_path = root_path
        self.portal_factory = portal_factory
        self.app_state = app_state
        self.client = client

    def handle_request(self, request: httpx.Request):
        scheme = request.url.scheme
        netloc = request.url.netloc.decode(encoding='ascii')
        path = request.url.path
        raw_path = request.url.raw_path
        query = request.url.query.decode(encoding='ascii')
        default_port = {'http': 80, 'ws': 80, 'https': 443, 'wss': 443}[scheme]
        if ':' in netloc:
            host, port_string = netloc.split(':', 1)
            port = int(port_string)
        else:
            host = netloc
            port = default_port
        if 'host' in request.headers:
            headers = []
        elif port == default_port:
            headers = [(b'host', host.encode())]
        else:
            headers = [(b'host', f'{host}:{port}'.encode())]
        headers += [(key.lower().encode(), value.encode()) for key, value in request.headers.multi_items()]
        if scheme in {'ws', 'wss'}:
            subprotocol = request.headers.get('sec-websocket-protocol', None)
            if subprotocol is None:
                subprotocols = []
            else:
                subprotocols = [value.strip() for value in subprotocol.split(',')]
            scope = {'type': 'websocket', 'path': unquote(path), 'raw_path': raw_path.split(b'?', 1)[0], 'root_path': self.root_path, 'scheme': scheme, 'query_string': query.encode(), 'headers': headers, 'client': self.client, 'server': [host, port], 'subprotocols': subprotocols, 'state': self.app_state.copy(), 'extensions': {'websocket.http.response': {}}}
            session = WebSocketTestSession(self.app, scope, self.portal_factory)
            raise _Upgrade(session)
        scope = {'type': 'http', 'http_version': '1.1', 'method': request.method, 'path': unquote(path), 'raw_path': raw_path.split(b'?', 1)[0], 'root_path': self.root_path, 'scheme': scheme, 'query_string': query.encode(), 'headers': headers, 'client': self.client, 'server': [host, port], 'extensions': {'http.response.debug': {}}, 'state': self.app_state.copy()}
        request_complete = False
        response_started = False
        raw_kwargs = {'stream': io.BytesIO()}
        template = None
        context = None

        async def receive():
            nonlocal request_complete
            if request_complete:
                if not response_complete.is_set():
                    await response_complete.wait()
                return {'type': 'http.disconnect'}
            body = request.read()
            if isinstance(body, str):
                body_bytes = body.encode('utf-8')
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

        async def send(message):
            nonlocal raw_kwargs, response_started, template, context
            if message['type'] == 'http.response.start':
                assert not response_started, 'Received multiple "http.response.start" messages.'
                raw_kwargs['status_code'] = message['status']
                raw_kwargs['headers'] = [(key.decode(), value.decode()) for key, value in message.get('headers', [])]
                response_started = True
            elif message['type'] == 'http.response.body':
                assert response_started, 'Received "http.response.body" without "http.response.start".'
                assert not response_complete.is_set(), 'Received "http.response.body" after response completed.'
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
            response.template = template
            response.context = context
        return response

class TestClient(httpx.Client):
    __test__ = False
    portal = None

    def __init__(self, app: Union[starlette.types.ASGIApp, typing.Any, None], base_url='http://testserver', raise_server_exceptions=True, root_path='', backend='asyncio', backend_options=None, cookies=None, headers=None, follow_redirects=True, client=('testclient', 50000) -> None):
        self.async_backend = _AsyncBackend(backend=backend, backend_options=backend_options or {})
        if _is_asgi3(app):
            asgi_app = app
        else:
            app = typing.cast(ASGI2App, app)
            asgi_app = _WrapASGI2(app)
        self.app = asgi_app
        self.app_state = {}
        transport = _TestClientTransport(self.app, portal_factory=self._portal_factory, raise_server_exceptions=raise_server_exceptions, root_path=root_path, app_state=self.app_state, client=client)
        if headers is None:
            headers = {}
        headers.setdefault('user-agent', 'testclient')
        super().__init__(base_url=base_url, headers=headers, transport=transport, follow_redirects=follow_redirects, cookies=cookies)

    @contextlib.contextmanager
    def _portal_factory(self) -> typing.Generator:
        if self.portal is not None:
            yield self.portal
        else:
            with anyio.from_thread.start_blocking_portal(**self.async_backend) as portal:
                yield portal

    def request(self, method: Union[str, bool], url: str, *, content: Union[None, str, bool]=None, data: Union[None, str, bool]=None, files: Union[None, str, bool]=None, json: Union[None, str, bool]=None, params: Union[None, str, bool]=None, headers: Union[None, str, bool]=None, cookies: Union[None, str, bool]=None, auth: Any=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: Any=httpx._client.USE_CLIENT_DEFAULT, timeout: Any=httpx._client.USE_CLIENT_DEFAULT, extensions: Union[None, str, bool]=None):
        url = self._merge_url(url)
        return super().request(method, url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def get(self, url: bool, *, params: Union[None, bool]=None, headers: Union[None, bool]=None, cookies: Union[None, bool]=None, auth: Any=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: Any=httpx._client.USE_CLIENT_DEFAULT, timeout: Any=httpx._client.USE_CLIENT_DEFAULT, extensions: Union[None, bool]=None) -> Union[str, dict[str, typing.Any]]:
        return super().get(url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def options(self, url: bool, *, params: Union[None, bool]=None, headers: Union[None, bool]=None, cookies: Union[None, bool]=None, auth: Any=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: Any=httpx._client.USE_CLIENT_DEFAULT, timeout: Any=httpx._client.USE_CLIENT_DEFAULT, extensions: Union[None, bool]=None) -> Union[str, dict[str, typing.Any]]:
        return super().options(url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def head(self, url: bool, *, params: Union[None, bool]=None, headers: Union[None, bool]=None, cookies: Union[None, bool]=None, auth: Any=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: Any=httpx._client.USE_CLIENT_DEFAULT, timeout: Any=httpx._client.USE_CLIENT_DEFAULT, extensions: Union[None, bool]=None) -> Union[str, None, dict[str, str]]:
        return super().head(url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def post(self, url: Union[str, bool], *, content: Union[None, str, bool]=None, data: Union[None, str, bool]=None, files: Union[None, str, bool]=None, json: Union[None, str, bool]=None, params: Union[None, str, bool]=None, headers: Union[None, str, bool]=None, cookies: Union[None, str, bool]=None, auth: Any=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: Any=httpx._client.USE_CLIENT_DEFAULT, timeout: Any=httpx._client.USE_CLIENT_DEFAULT, extensions: Union[None, str, bool]=None) -> Union[starlette.requests.Request, str, dict[str, str]]:
        return super().post(url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def put(self, url: Union[str, bool], *, content: Union[None, str, bool]=None, data: Union[None, str, bool]=None, files: Union[None, str, bool]=None, json: Union[None, str, bool]=None, params: Union[None, str, bool]=None, headers: Union[None, str, bool]=None, cookies: Union[None, str, bool]=None, auth: Any=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: Any=httpx._client.USE_CLIENT_DEFAULT, timeout: Any=httpx._client.USE_CLIENT_DEFAULT, extensions: Union[None, str, bool]=None) -> str:
        return super().put(url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def patch(self, url: bool, *, content: Union[None, bool]=None, data: Union[None, bool]=None, files: Union[None, bool]=None, json: Union[None, bool]=None, params: Union[None, bool]=None, headers: Union[None, bool]=None, cookies: Union[None, bool]=None, auth: Any=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: Any=httpx._client.USE_CLIENT_DEFAULT, timeout: Any=httpx._client.USE_CLIENT_DEFAULT, extensions: Union[None, bool]=None) -> Union[str, bool, typing.IO]:
        return super().patch(url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def delete(self, url: bool, *, params: Union[None, bool]=None, headers: Union[None, bool]=None, cookies: Union[None, bool]=None, auth: Any=httpx._client.USE_CLIENT_DEFAULT, follow_redirects: Any=httpx._client.USE_CLIENT_DEFAULT, timeout: Any=httpx._client.USE_CLIENT_DEFAULT, extensions: Union[None, bool]=None) -> Union[str, bool, dict[str, bool]]:
        return super().delete(url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def websocket_connect(self, url: str, subprotocols: Union[str, list[str], None]=None, **kwargs):
        url = urljoin('ws://testserver', url)
        headers = kwargs.get('headers', {})
        headers.setdefault('connection', 'upgrade')
        headers.setdefault('sec-websocket-key', 'testserver==')
        headers.setdefault('sec-websocket-version', '13')
        if subprotocols is not None:
            headers.setdefault('sec-websocket-protocol', ', '.join(subprotocols))
        kwargs['headers'] = headers
        try:
            super().request('GET', url, **kwargs)
        except _Upgrade as exc:
            session = exc.session
        else:
            raise RuntimeError('Expected WebSocket upgrade')
        return session

    def __enter__(self) -> WebSocketTestSession:
        with contextlib.ExitStack() as stack:
            self.portal = portal = stack.enter_context(anyio.from_thread.start_blocking_portal(**self.async_backend))

            @stack.callback
            def reset_portal() -> None:
                self.portal = None
            send = anyio.create_memory_object_stream(math.inf)
            receive = anyio.create_memory_object_stream(math.inf)
            for channel in (*send, *receive):
                stack.callback(channel.close)
            self.stream_send = StapledObjectStream(*send)
            self.stream_receive = StapledObjectStream(*receive)
            self.task = portal.start_task_soon(self.lifespan)
            portal.call(self.wait_startup)

            @stack.callback
            def wait_shutdown() -> None:
                portal.call(self.wait_shutdown)
            self.exit_stack = stack.pop_all()
        return self

    def __exit__(self, *args):
        self.exit_stack.close()

    async def lifespan(self):
        scope = {'type': 'lifespan', 'state': self.app_state}
        try:
            await self.app(scope, self.stream_receive.receive, self.stream_send.send)
        finally:
            await self.stream_send.send(None)

    async def wait_startup(self):
        await self.stream_receive.send({'type': 'lifespan.startup'})

        async def receive():
            message = await self.stream_send.receive()
            if message is None:
                self.task.result()
            return message
        message = await receive()
        assert message['type'] in ('lifespan.startup.complete', 'lifespan.startup.failed')
        if message['type'] == 'lifespan.startup.failed':
            await receive()

    async def wait_shutdown(self):

        async def receive():
            message = await self.stream_send.receive()
            if message is None:
                self.task.result()
            return message
        await self.stream_receive.send({'type': 'lifespan.shutdown'})
        message = await receive()
        assert message['type'] in ('lifespan.shutdown.complete', 'lifespan.shutdown.failed')
        if message['type'] == 'lifespan.shutdown.failed':
            await receive()