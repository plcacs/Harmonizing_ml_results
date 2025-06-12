"""Connection/Session management module."""
import asyncio
import json
import logging
from typing import Awaitable, Callable, Dict, List, Union, Any, Optional, TypeVar, cast
from pyee import EventEmitter
import websockets
from pyppeteer.errors import NetworkError
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)
logger_connection = logging.getLogger(__name__ + '.Connection')
logger_session = logging.getLogger(__name__ + '.CDPSession')

T = TypeVar('T')
CallbackType = Callable[[], None]
MessageType = Dict[str, Any]
FutureType = asyncio.Future[Any]

class Connection(EventEmitter):
    """Connection management class."""

    def __init__(self, url: str, loop: asyncio.AbstractEventLoop, delay: int = 0) -> None:
        """Make connection.

        :arg str url: WebSocket url to connect devtool.
        :arg int delay: delay to wait before processing received messages.
        """
        super().__init__()
        self._url = url
        self._lastId = 0
        self._callbacks: Dict[int, asyncio.Future[Any]] = {}
        self._delay = delay / 1000
        self._loop = loop
        self._sessions: Dict[str, 'CDPSession'] = {}
        self._connected = False
        self._ws = websockets.client.connect(self._url, max_size=None, loop=self._loop)
        self._recv_fut: asyncio.Task[None] = self._loop.create_task(self._recv_loop())
        self._closeCallback: Optional[CallbackType] = None
        self.connection: Optional[WebSocketClientProtocol] = None

    @property
    def url(self) -> str:
        """Get connected WebSocket url."""
        return self._url

    async def _recv_loop(self) -> None:
        async with self._ws as connection:
            self._connected = True
            self.connection = connection
            while self._connected:
                try:
                    resp = await self.connection.recv()
                    if resp:
                        await self._on_message(resp)
                except (websockets.ConnectionClosed, ConnectionResetError):
                    logger.info('connection closed')
                    break
                await asyncio.sleep(0)
        if self._connected:
            self._loop.create_task(self.dispose())

    async def _async_send(self, msg: str, callback_id: int) -> None:
        while not self._connected:
            await asyncio.sleep(self._delay)
        try:
            if self.connection:
                await self.connection.send(msg)
        except websockets.ConnectionClosed:
            logger.error('connection unexpectedly closed')
            callback = self._callbacks.get(callback_id, None)
            if callback and (not callback.done()):
                callback.set_result(None)
                await self.dispose()

    def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> asyncio.Future[Any]:
        """Send message via the connection."""
        if self._lastId and (not self._connected):
            raise ConnectionError('Connection is closed')
        if params is None:
            params = {}
        self._lastId += 1
        _id = self._lastId
        msg = json.dumps(dict(id=_id, method=method, params=params))
        logger_connection.debug(f'SEND: {msg}')
        self._loop.create_task(self._async_send(msg, _id))
        callback = self._loop.create_future()
        self._callbacks[_id] = callback
        callback.error = NetworkError()  # type: ignore
        callback.method = method  # type: ignore
        return callback

    def _on_response(self, msg: MessageType) -> None:
        callback = self._callbacks.pop(msg.get('id', -1))
        if msg.get('error'):
            callback.set_exception(_createProtocolError(callback.error, callback.method, msg))  # type: ignore
        else:
            callback.set_result(msg.get('result'))

    def _on_query(self, msg: MessageType) -> None:
        params = msg.get('params', {})
        method = msg.get('method', '')
        sessionId = params.get('sessionId')
        if method == 'Target.receivedMessageFromTarget':
            session = self._sessions.get(sessionId)
            if session:
                session._on_message(params.get('message'))
        elif method == 'Target.detachedFromTarget':
            session = self._sessions.get(sessionId)
            if session:
                session._on_closed()
                del self._sessions[sessionId]
        else:
            self.emit(method, params)

    def setClosedCallback(self, callback: Optional[CallbackType]) -> None:
        """Set closed callback."""
        self._closeCallback = callback

    async def _on_message(self, message: str) -> None:
        await asyncio.sleep(self._delay)
        logger_connection.debug(f'RECV: {message}')
        msg = json.loads(message)
        if msg.get('id') in self._callbacks:
            self._on_response(msg)
        else:
            self._on_query(msg)

    async def _on_close(self) -> None:
        if self._closeCallback:
            self._closeCallback()
            self._closeCallback = None
        for cb in self._callbacks.values():
            cb.set_exception(_rewriteError(cb.error, f'Protocol error {cb.method}: Target closed.'))  # type: ignore
        self._callbacks.clear()
        for session in self._sessions.values():
            session._on_closed()
        self._sessions.clear()
        if hasattr(self, 'connection') and self.connection:
            await self.connection.close()
        if not self._recv_fut.done():
            self._recv_fut.cancel()

    async def dispose(self) -> None:
        """Close all connection."""
        self._connected = False
        await self._on_close()

    async def createSession(self, targetInfo: Dict[str, Any]) -> 'CDPSession':
        """Create new session."""
        resp = await self.send('Target.attachToTarget', {'targetId': targetInfo['targetId']})
        sessionId = resp.get('sessionId')
        session = CDPSession(self, targetInfo['type'], sessionId, self._loop)
        self._sessions[sessionId] = session
        return session

class CDPSession(EventEmitter):
    """Chrome Devtools Protocol Session."""

    def __init__(self, connection: Connection, targetType: str, sessionId: str, loop: asyncio.AbstractEventLoop) -> None:
        """Make new session."""
        super().__init__()
        self._lastId = 0
        self._callbacks: Dict[int, asyncio.Future[Any]] = {}
        self._connection: Optional[Connection] = connection
        self._targetType = targetType
        self._sessionId = sessionId
        self._sessions: Dict[str, 'CDPSession'] = {}
        self._loop = loop

    def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> asyncio.Future[Any]:
        """Send message to the connected session."""
        if not self._connection:
            raise NetworkError(f'Protocol Error ({method}): Session closed. Most likely the {self._targetType} has been closed.')
        self._lastId += 1
        _id = self._lastId
        msg = json.dumps(dict(id=_id, method=method, params=params if params else {}))
        logger_session.debug(f'SEND: {msg}')
        callback = self._loop.create_future()
        self._callbacks[_id] = callback
        callback.error = NetworkError()  # type: ignore
        callback.method = method  # type: ignore
        try:
            if self._connection:
                self._connection.send('Target.sendMessageToTarget', {'sessionId': self._sessionId, 'message': msg})
        except Exception as e:
            if _id in self._callbacks:
                del self._callbacks[_id]
                _callback = self._callbacks[_id]
                _callback.set_exception(_rewriteError(_callback.error, e.args[0]))  # type: ignore
        return callback

    def _on_message(self, msg: str) -> None:
        logger_session.debug(f'RECV: {msg}')
        obj = json.loads(msg)
        _id = obj.get('id')
        if _id:
            callback = self._callbacks.get(_id)
            if callback:
                del self._callbacks[_id]
                if obj.get('error'):
                    callback.set_exception(_createProtocolError(callback.error, callback.method, obj))  # type: ignore
                else:
                    result = obj.get('result')
                    if callback and (not callback.done()):
                        callback.set_result(result)
        else:
            params = obj.get('params', {})
            if obj.get('method') == 'Target.receivedMessageFromTarget':
                session = self._sessions.get(params.get('sessionId'))
                if session:
                    session._on_message(params.get('message'))
            elif obj.get('method') == 'Target.detachFromTarget':
                sessionId = params.get('sessionId')
                session = self._sessions.get(sessionId)
                if session:
                    session._on_closed()
                    del self._sessions[sessionId]
            self.emit(obj.get('method'), obj.get('params'))

    async def detach(self) -> None:
        """Detach session from target."""
        if not self._connection:
            raise NetworkError('Connection already closed.')
        await self._connection.send('Target.detachFromTarget', {'sessionId': self._sessionId})

    def _on_closed(self) -> None:
        for cb in self._callbacks.values():
            cb.set_exception(_rewriteError(cb.error, f'Protocol error {cb.method}: Target closed.'))  # type: ignore
        self._callbacks.clear()
        self._connection = None

    def _createSession(self, targetType: str, sessionId: str) -> 'CDPSession':
        session = CDPSession(self, targetType, sessionId, self._loop)
        self._sessions[sessionId] = session
        return session

def _createProtocolError(error: NetworkError, method: str, obj: MessageType) -> NetworkError:
    message = f'Protocol error ({method}): {obj["error"]["message"]}'
    if 'data' in obj['error']:
        message += f' {obj["error"]["data"]}'
    return _rewriteError(error, message)

def _rewriteError(error: NetworkError, message: str) -> NetworkError:
    error.args = (message,)
    return error
