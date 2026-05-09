import asyncio
import time
from typing import Optional, Protocol
from unittest import mock
import aiosignal
import pytest
from aiohttp import WSMessageTypeError, WSMsgType, web
from aiohttp.http import WS_CLOSED_MESSAGE, WS_CLOSING_MESSAGE
from aiohttp.http_websocket import WSMessageClose
from aiohttp.streams import EofStream
from aiohttp.test_utils import make_mocked_coro, make_mocked_request
from aiohttp.web_ws import WebSocketReady

class _RequestMaker(Protocol):
    def __call__(self, method: str, path: str, headers: Optional[CIMultiDict] = None, protocols: bool = False) -> web.Request: ...

@pytest.fixture
def app(loop: asyncio.AbstractEventLoop) -> web.Application:
    ret = mock.create_autospec(web.Application, spec_set=True)
    ret.on_response_prepare = aiosignal.Signal(ret)
    ret.on_response_prepare.freeze()
    return ret

@pytest.fixture
def protocol() -> mock.Mock:
    ret = mock.Mock()
    ret.set_parser.return_value = ret
    return ret

@pytest.fixture
def make_request(app: web.Application, protocol: mock.Mock) -> Callable[[str, str, Optional[CIMultiDict], bool], web.Request]:
    def maker(method: str, path: str, headers: Optional[CIMultiDict] = None, protocols: bool = False) -> web.Request:
        if headers is None:
            headers = CIMultiDict({'HOST': 'server.example.com', 'UPGRADE': 'websocket', 'CONNECTION': 'Upgrade', 'SEC-WEBSOCKET-KEY': 'dGhlIHNhbXBsZSBub25jZQ==', 'ORIGIN': 'http://example.com', 'SEC-WEBSOCKET-VERSION': '13'})
        if protocols:
            headers['SEC-WEBSOCKET-PROTOCOL'] = 'chat, superchat'
        return make_mocked_request(method, path, headers, app=app, protocol=protocol)
    return maker

async def test_nonstarted_ping() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.ping()

async def test_nonstarted_pong() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.pong()

async def test_nonstarted_send_frame() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.send_frame(b'string', WSMsgType.TEXT)

async def test_nonstarted_send_str() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.send_str('string')

async def test_nonstarted_send_bytes() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.send_bytes(b'bytes')

async def test_nonstarted_send_json() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.send_json({'type': 'json'})

async def test_nonstarted_close() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.close()

async def test_nonstarted_receive_str() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.receive_str()

async def test_nonstarted_receive_bytes() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.receive_bytes()

async def test_nonstarted_receive_json() -> None:
    ws = web.WebSocketResponse()
    with pytest.raises(RuntimeError):
        await ws.receive_json()

# ... rest of the code ...
