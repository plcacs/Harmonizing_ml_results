"""Connection/Session management module."""
import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING, cast
from pyee import EventEmitter
import websockets
from pyppeteer.errors import NetworkError
if TYPE_CHECKING:
    from typing import Optional
logger = logging.getLogger(__name__)
logger_connection = logging.getLogger(__name__ + '.Connection')
logger_session = logging.getLogger(__name__ + '.CDPSession')

class Connection(EventEmitter):
    """Connection management class."""

    def __init__(self, url: str, loop: asyncio.AbstractEventLoop, delay: float = 0) -> None:
        """Make connection.

        :arg str url: WebSocket url to connect devtool.
        :arg int delay: delay to wait before processing received messages.
        """
        super().__init__()
        self._url: str = url
        self._lastId: int = 0
        self._callbacks: Dict[int, asyncio.Future] = dict()
        self._delay: float = delay / 1000
        self._loop: asyncio.AbstractEventLoop = loop
        self._sessions: Dict[str, 'CDPSession'] = dict()
        self._connected: bool = False
        self._ws: websockets.client.Connect = websockets.client.connect(self._url, max_size=None, loop=self._loop)
        self._recv_fut: asyncio.Task = self._loop.create_task(self._recv_loop())
        self._closeCallback: Optional[Callable[[], None]] = None
        self.connection: Optional[websockets.WebSocketClientProtocol] = None

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
                    resp: str = await self.connection.recv()
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
            await self.connection.send(msg)  # type: ignore
        except websockets.ConnectionClosed:
            logger.error('connection unexpectedly closed')
            callback = self._callbacks.get(callback_id, None)
            if callback and (not callback.done()):
                callback.set_result(None)
                await self.dispose()

    def send(self, method: str, params: Optional[Dict] = None) -> asyncio.Future:
        """Send message via the connection."""
        if self._lastId and (not self._connected):
            raise ConnectionError('Connection is closed')
        if params is None:
            params = dict()
        self._lastId += 1
        _id: int = self._lastId
        msg: str = json.dumps(dict(id=_id, method=method, params=params))
        logger_connection.debug(f'SEND: {msg}')
        self._loop.create_task(self._async_send(msg, _id))
        callback: asyncio.Future = self._loop.create_future()
        self._callbacks[_id] = callback
        callback.error = NetworkError()  # type: ignore
        callback.method = method  # type: ignore
        return callback

    def _on_response(self, msg: Dict) -> None:
        callback: asyncio.Future = self._callbacks.pop(msg.get('id', -1))
        if msg.get('error'):
            callback.set_exception(_createProtocolError(callback.error, callback.method, msg))  # type: ignore
        else:
            callback.set_result(msg.get('result'))

    def _on_query(self, msg: Dict) -> None:
        params: Dict = msg.get('params', {})
        method: str = msg.get('method', '')
        sessionId: Optional[str] = params.get('sessionId')
        if method == 'Target.receivedMessageFromTarget':
            session: Optional[CDPSession] = self._sessions.get(sessionId)  # type: ignore
            if session:
                session._on_message(params.get('message'))
        elif method == 'Target.detachedFromTarget':
            session = self._sessions.get(sessionId)  # type: ignore
            if session:
                session._on_closed()
                del self._sessions[sessionId]  # type: ignore
        else:
            self.emit(method, params)

    def setClosedCallback(self, callback: Callable[[], None]) -> None:
        """Set closed callback."""
        self._closeCallback = callback

    async def _on_message(self, message: str) -> None:
        await asyncio.sleep(self._delay)
        logger_connection.debug(f'RECV: {message}')
        msg: Dict = json.loads(message)
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
        if hasattr(self, 'connection'):
            await self.connection.close()  # type: ignore
        if not self._recv_fut.done():
            self._recv_fut.cancel()

    async def dispose(self) -> None:
        """Close all connection."""
        self._connected = False
        await self._on_close()

    async def createSession(self, targetInfo: Dict[str, Any]) -> 'CDPSession':
        """Create new session."""
        resp: Dict = await self.send('Target.attachToTarget', {'targetId': targetInfo['targetId']})
        sessionId: str = resp.get('sessionId')
        session: CDPSession = CDPSession(self, targetInfo['type'], sessionId, self._loop)
        self._sessions[sessionId] = session
        return session

class CDPSession(EventEmitter):
    """Chrome Devtools Protocol Session.

    The :class:`CDPSession` instances are used to talk raw Chrome Devtools
    Protocol:

    * protocol methods can be called with :meth:`send` method.
    * protocol events can be subscribed to with :meth:`on` method.

    Documentation on DevTools Protocol can be found
    `here <https://chromedevtools.github.io/devtools-protocol/>`__.
    """

    def __init__(self, connection: Union[Connection, 'CDPSession'], targetType: str, sessionId: str, loop: asyncio.AbstractEventLoop) -> None:
        """Make new session."""
        super().__init__()
        self._lastId: int = 0
        self._callbacks: Dict[int, asyncio.Future] = {}
        self._connection: Optional[Union[Connection, 'CDPSession']] = connection
        self._targetType: str = targetType
        self._sessionId: str = sessionId
        self._sessions: Dict[str, 'CDPSession'] = dict()
        self._loop: asyncio.AbstractEventLoop = loop

    def send(self, method: str, params: Optional[Dict] = None) -> asyncio.Future:
        """Send message to the connected session.

        :arg str method: Protocol method name.
        :arg dict params: Optional method parameters.
        """
        if not self._connection:
            raise NetworkError(f'Protocol Error ({method}): Session closed. Most likely the {self._targetType} has been closed.')
        self._lastId += 1
        _id: int = self._lastId
        msg: str = json.dumps(dict(id=_id, method=method, params=params))
        logger_session.debug(f'SEND: {msg}')
        callback: asyncio.Future = self._loop.create_future()
        self._callbacks[_id] = callback
        callback.error = NetworkError()  # type: ignore
        callback.method = method  # type: ignore
        try:
            self._connection.send('Target.sendMessageToTarget', {'sessionId': self._sessionId, 'message': msg})
        except Exception as e:
            if _id in self._callbacks:
                del self._callbacks[_id]
                _callback: asyncio.Future = self._callbacks[_id]
                _callback.set_exception(_rewriteError(_callback.error, e.args[0]))  # type: ignore
        return callback

    def _on_message(self, msg: str) -> None:
        logger_session.debug(f'RECV: {msg}')
        obj: Dict = json.loads(msg)
        _id: Optional[int] = obj.get('id')
        if _id:
            callback: Optional[asyncio.Future] = self._callbacks.get(_id)
            if callback:
                del self._callbacks[_id]
                if obj.get('error'):
                    callback.set_exception(_createProtocolError(callback.error, callback.method, obj))  # type: ignore
                else:
                    result: Any = obj.get('result')
                    if callback and (not callback.done()):
                        callback.set_result(result)
        else:
            params: Dict = obj.get('params', {})
            if obj.get('method') == 'Target.receivedMessageFromTarget':
                session: Optional[CDPSession] = self._sessions.get(params.get('sessionId'))
                if session:
                    session._on_message(params.get('message'))
            elif obj.get('method') == 'Target.detachFromTarget':
                sessionId: str = params.get('sessionId')
                session = self._sessions.get(sessionId)
                if session:
                    session._on_closed()
                    del self._sessions[sessionId]
            self.emit(obj.get('method'), obj.get('params'))

    async def detach(self) -> None:
        """Detach session from target.

        Once detached, session won't emit any events and can't be used to send
        messages.
        """
        if not self._connection:
            raise NetworkError('Connection already closed.')
        await self._connection.send('Target.detachFromTarget', {'sessionId': self._sessionId})

    def _on_closed(self) -> None:
        for cb in self._callbacks.values():
            cb.set_exception(_rewriteError(cb.error, f'Protocol error {cb.method}: Target closed.'))  # type: ignore
        self._callbacks.clear()
        self._connection = None

    def _createSession(self, targetType: str, sessionId: str) -> 'CDPSession':
        session: CDPSession = CDPSession(self, targetType, sessionId, self._loop)
        self._sessions[sessionId] = session
        return session

def _createProtocolError(error: NetworkError, method: str, obj: Dict) -> NetworkError:
    message: str = f'Protocol error ({method}): {obj["error"]["message"]}'
    if 'data' in obj['error']:
        message += f' {obj["error"]["data"]}'
    return _rewriteError(error, message)

def _rewriteError(error: NetworkError, message: str) -> NetworkError:
    error.args = (message,)
    return error
