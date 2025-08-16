import asyncio
import json
import logging
import math
from typing import Any, Awaitable, Callable, Dict, List, Union
from pyee import EventEmitter
import pyppeteer
from pyppeteer.connection import CDPSession
from pyppeteer.errors import ElementHandleError, TimeoutError

logger: logging.Logger = logging.getLogger(__name__)

def debugError(_logger: logging.Logger, msg: str) -> None:
    ...

def evaluationString(fun: str, *args: Any) -> str:
    ...

def getExceptionMessage(exceptionDetails: Dict[str, Any]) -> str:
    ...

def addEventListener(emitter: EventEmitter, eventName: str, handler: Callable) -> Dict[str, Union[EventEmitter, str, Callable]]:
    ...

def removeEventListeners(listeners: List[Dict[str, Union[EventEmitter, str, Callable]]]) -> None:
    ...

unserializableValueMap: Dict[str, Union[int, None, float]] = {'-0': -0, 'NaN': None, None: None, 'Infinity': math.inf, '-Infinity': -math.inf}

def valueFromRemoteObject(remoteObject: Dict[str, Any]) -> Any:
    ...

def releaseObject(client: Any, remoteObject: Dict[str, Any]) -> asyncio.Future:
    ...

def waitForEvent(emitter: EventEmitter, eventName: str, predicate: Callable, timeout: int, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
    ...

def get_positive_int(obj: Dict[str, Any], name: str) -> int:
    ...

def is_jsfunc(func: str) -> bool:
    ...
