"""Helper functions."""
import asyncio
import json
import logging
import math
from typing import Any, Awaitable, Callable, Dict, List, Union
from pyee import EventEmitter
import pyppeteer
from pyppeteer.connection import CDPSession
from pyppeteer.errors import ElementHandleError, TimeoutError

logger = logging.getLogger(__name__)

def debugError(_logger: logging.Logger, msg: str) -> None:
    """Log error messages."""
    if pyppeteer.DEBUG:
        _logger.error(msg)
    else:
        _logger.debug(msg)

def evaluationString(fun: str, *args: Any) -> str:
    """Convert function and arguments to str."""
    _args = ', '.join([json.dumps('undefined' if arg is None else arg) for arg in args])
    expr = f'({fun})({_args})'
    return expr

def getExceptionMessage(exceptionDetails: Dict[str, Any]) -> str:
    """Get exception message from `exceptionDetails` object."""
    exception = exceptionDetails.get('exception')
    if exception:
        return exception.get('description') or exception.get('value')
    message = exceptionDetails.get('text', '')
    stackTrace = exceptionDetails.get('stackTrace', dict())
    if stackTrace:
        for callframe in stackTrace.get('callFrames', []):
            location = str(callframe.get('url', '')) + ':' + str(callframe.get('lineNumber', '')) + ':' + str(callframe.get('columnNumber'))
            functionName = callframe.get('functionName', '<anonymous>')
            message = message + f'\n    at {functionName} ({location})'
    return message

def addEventListener(emitter: EventEmitter, eventName: str, handler: Callable) -> Dict[str, Any]:
    """Add handler to the emitter and return emitter/handler."""
    emitter.on(eventName, handler)
    return {'emitter': emitter, 'eventName': eventName, 'handler': handler}

def removeEventListeners(listeners: List[Dict[str, Any]]) -> None:
    """Remove listeners from emitter."""
    for listener in listeners:
        emitter = listener['emitter']
        eventName = listener['eventName']
        handler = listener['handler']
        emitter.remove_listener(eventName, handler)
    listeners.clear()

unserializableValueMap: Dict[Union[str, None], Union[float, None]] = {'-0': -0, 'NaN': None, None: None, 'Infinity': math.inf, '-Infinity': -math.inf}

def valueFromRemoteObject(remoteObject: Dict[str, Any]) -> Any:
    """Serialize value of remote object."""
    if remoteObject.get('objectId'):
        raise ElementHandleError('Cannot extract value when objectId is given')
    value = remoteObject.get('unserializableValue')
    if value:
        if value == '-0':
            return -0
        elif value == 'NaN':
            return None
        elif value == 'Infinity':
            return math.inf
        elif value == '-Infinity':
            return -math.inf
        else:
            raise ElementHandleError('Unsupported unserializable value: {}'.format(value))
    return remoteObject.get('value')

def releaseObject(client: CDPSession, remoteObject: Dict[str, Any]) -> Awaitable[None]:
    """Release remote object."""
    objectId = remoteObject.get('objectId')
    fut_none = client._loop.create_future()
    fut_none.set_result(None)
    if not objectId:
        return fut_none
    try:
        return client.send('Runtime.releaseObject', {'objectId': objectId})
    except Exception as e:
        debugError(logger, str(e))
    return fut_none

def waitForEvent(emitter: EventEmitter, eventName: str, predicate: Callable[[Any], bool], timeout: int, loop: asyncio.AbstractEventLoop) -> Awaitable[Any]:
    """Wait for an event emitted from the emitter."""
    promise = loop.create_future()

    def resolveCallback(target: Any) -> None:
        promise.set_result(target)

    def rejectCallback(exception: Exception) -> None:
        promise.set_exception(exception)

    async def timeoutTimer() -> None:
        await asyncio.sleep(timeout / 1000)
        rejectCallback(TimeoutError('Timeout exceeded while waiting for event'))

    def _listener(target: Any) -> None:
        if not predicate(target):
            return
        cleanup()
        resolveCallback(target)

    listener = addEventListener(emitter, eventName, _listener)
    if timeout:
        eventTimeout = loop.create_task(timeoutTimer())

    def cleanup() -> None:
        removeEventListeners([listener])
        if timeout:
            eventTimeout.cancel()

    return promise

def get_positive_int(obj: Dict[str, Any], name: str) -> int:
    """Get and check the value of name in obj is positive integer."""
    value = obj[name]
    if not isinstance(value, int):
        raise TypeError(f'{name} must be integer: {type(value)}')
    elif value < 0:
        raise ValueError(f'{name} must be positive integer: {value}')
    return value

def is_jsfunc(func: str) -> bool:
    """Heuristically check function or expression."""
    func = func.strip()
    if func.startswith('function') or func.startswith('async '):
        return True
    elif '=>' in func:
        return True
    return False
