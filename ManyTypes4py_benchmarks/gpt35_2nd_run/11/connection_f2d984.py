from typing import Awaitable, Callable, Dict, Union, TYPE_CHECKING, Optional

class Connection(EventEmitter):
    def __init__(self, url: str, loop: asyncio.AbstractEventLoop, delay: int = 0):
    def send(self, method: str, params: Optional[Dict] = None) -> Awaitable:
    def setClosedCallback(self, callback: Callable):
    async def createSession(self, targetInfo: Dict) -> Awaitable

class CDPSession(EventEmitter):
    def __init__(self, connection: Connection, targetType: str, sessionId: str, loop: asyncio.AbstractEventLoop):
    def send(self, method: str, params: Optional[Dict] = None) -> Awaitable
    async def detach(self) -> Awaitable
