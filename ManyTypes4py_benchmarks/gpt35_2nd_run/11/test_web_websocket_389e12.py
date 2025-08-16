from typing import Optional, Protocol

class _RequestMaker(Protocol):

    def __call__(self, method: str, path: str, headers: Optional[dict] = None, protocols: bool = False) -> None:
        ...

def test_websocket_ready() -> None:
    websocket_ready: WebSocketReady = WebSocketReady(True, 'chat')
    assert websocket_ready.ok is True
    assert websocket_ready.protocol == 'chat'

def test_websocket_not_ready() -> None:
    websocket_ready: WebSocketReady = WebSocketReady(False, None)
    assert websocket_ready.ok is False
    assert websocket_ready.protocol is None

def test_websocket_ready_unknown_protocol() -> None:
    websocket_ready: WebSocketReady = WebSocketReady(True, None)
    assert websocket_ready.ok is True
    assert websocket_ready.protocol is None

def test_bool_websocket_ready() -> None:
    websocket_ready: WebSocketReady = WebSocketReady(True, None)
    assert bool(websocket_ready) is True

def test_bool_websocket_not_ready() -> None:
    websocket_ready: WebSocketReady = WebSocketReady(False, None)
    assert bool(websocket_ready) is False
