"""Execution Context Module."""
import logging
import math
import re
from typing import Any, Dict, Optional, TYPE_CHECKING, Union
from pyppeteer import helper
from pyppeteer.connection import CDPSession
from pyppeteer.errors import ElementHandleError, NetworkError
from pyppeteer.helper import debugError

class ExecutionContext(object):
    """Execution Context class."""

    def __init__(self, client: CDPSession, contextPayload: Dict[str, Any], objectHandleFactory: Any, frame: 'Frame' = None):
        self._client: CDPSession = client
        self._frame: 'Frame' = frame
        self._contextId: str = contextPayload.get('id')
        auxData: Dict[str, Any] = contextPayload.get('auxData', {'isDefault': False})
        self._isDefault: bool = bool(auxData.get('isDefault'))
        self._objectHandleFactory: Any = objectHandleFactory

    # ...

class JSHandle(object):
    """JSHandle class.

    JSHandle represents an in-page JavaScript object. JSHandle can be created
    with the :meth:`~pyppeteer.page.Page.evaluateHandle` method.
    """

    def __init__(self, context: ExecutionContext, client: CDPSession, remoteObject: Dict[str, Any]):
        self._context: ExecutionContext = context
        self._client: CDPSession = client
        self._remoteObject: Dict[str, Any] = remoteObject
        self._disposed: bool = False

    # ...

    async def getProperty(self, propertyName: str) -> Any:
        # ...

    async def getProperties(self) -> Dict[str, Any]:
        # ...

    async def jsonValue(self) -> Any:
        # ...

    def asElement(self) -> Optional['ElementHandle']:
        # ...

    async def dispose(self) -> None:
        # ...

    def toString(self) -> str:
        # ...

def _rewriteError(error: Exception) -> None:
    # ...
