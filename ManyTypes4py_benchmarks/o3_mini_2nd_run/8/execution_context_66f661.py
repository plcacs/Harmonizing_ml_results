"""Execution Context Module."""
import logging
import math
import re
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, Union

from pyppeteer import helper
from pyppeteer.connection import CDPSession
from pyppeteer.errors import ElementHandleError, NetworkError
from pyppeteer.helper import debugError

if TYPE_CHECKING:
    from pyppeteer.element_handle import ElementHandle
    from pyppeteer.frame_manager import Frame

logger = logging.getLogger(__name__)
EVALUATION_SCRIPT_URL = '__pyppeteer_evaluation_script__'
SOURCE_URL_REGEX = re.compile('^[\\040\\t]*//[@#] sourceURL=\\s*(\\S*?)\\s*$', re.MULTILINE)


class ExecutionContext(object):
    """Execution Context class."""

    def __init__(self, client: Any, contextPayload: Dict[str, Any],
                 objectHandleFactory: Callable[[Dict[str, Any]], "JSHandle"],
                 frame: Optional["Frame"] = None) -> None:
        self._client = client
        self._frame = frame
        self._contextId = contextPayload.get('id')
        auxData = contextPayload.get('auxData', {'isDefault': False})
        self._isDefault = bool(auxData.get('isDefault'))
        self._objectHandleFactory = objectHandleFactory

    @property
    def frame(self) -> Optional["Frame"]:
        """Return frame associated with this execution context."""
        return self._frame

    async def evaluate(self, pageFunction: Union[str, Callable[..., Any]], *args: Any,
                       force_expr: bool = False) -> Any:
        """Execute ``pageFunction`` on this context.

        Details see :meth:`pyppeteer.page.Page.evaluate`.
        """
        handle = await self.evaluateHandle(pageFunction, *args, force_expr=force_expr)
        try:
            result = await handle.jsonValue()
        except NetworkError as e:
            if 'Object reference chain is too long' in e.args[0]:
                return
            if "Object couldn't be returned by value" in e.args[0]:
                return
            raise
        await handle.dispose()
        return result

    async def evaluateHandle(self, pageFunction: Union[str, Callable[..., Any]], *args: Any,
                             force_expr: bool = False) -> "JSHandle":
        """Execute ``pageFunction`` on this context.

        Details see :meth:`pyppeteer.page.Page.evaluateHandle`.
        """
        suffix = f'//# sourceURL={EVALUATION_SCRIPT_URL}'
        if force_expr or (not args and (not helper.is_jsfunc(pageFunction))):
            try:
                if SOURCE_URL_REGEX.match(pageFunction):
                    expressionWithSourceUrl = pageFunction
                else:
                    expressionWithSourceUrl = f'{pageFunction}\n{suffix}'
                _obj: Dict[str, Any] = await self._client.send(
                    'Runtime.evaluate',
                    {
                        'expression': expressionWithSourceUrl,
                        'contextId': self._contextId,
                        'returnByValue': False,
                        'awaitPromise': True,
                        'userGesture': True
                    }
                )
            except Exception as e:
                _rewriteError(e)
            exceptionDetails = _obj.get('exceptionDetails')
            if exceptionDetails:
                raise ElementHandleError('Evaluation failed: {}'.format(helper.getExceptionMessage(exceptionDetails)))
            remoteObject = _obj.get('result')
            return self._objectHandleFactory(remoteObject)
        try:
            _obj = await self._client.send(
                'Runtime.callFunctionOn',
                {
                    'functionDeclaration': f'{pageFunction}\n{suffix}\n',
                    'executionContextId': self._contextId,
                    'arguments': [self._convertArgument(arg) for arg in args],
                    'returnByValue': False,
                    'awaitPromise': True,
                    'userGesture': True
                }
            )
        except Exception as e:
            _rewriteError(e)
        exceptionDetails = _obj.get('exceptionDetails')
        if exceptionDetails:
            raise ElementHandleError('Evaluation failed: {}'.format(helper.getExceptionMessage(exceptionDetails)))
        remoteObject = _obj.get('result')
        return self._objectHandleFactory(remoteObject)

    def _convertArgument(self, arg: Any) -> Dict[str, Any]:
        if arg == math.inf:
            return {'unserializableValue': 'Infinity'}
        if arg == -math.inf:
            return {'unserializableValue': '-Infinity'}
        objectHandle = arg if isinstance(arg, JSHandle) else None
        if objectHandle:
            if objectHandle._context != self:
                raise ElementHandleError('JSHandles can be evaluated only in the context they were created!')
            if objectHandle._disposed:
                raise ElementHandleError('JSHandle is disposed!')
            if objectHandle._remoteObject.get('unserializableValue'):
                return {'unserializableValue': objectHandle._remoteObject.get('unserializableValue')}
            if not objectHandle._remoteObject.get('objectId'):
                return {'value': objectHandle._remoteObject.get('value')}
            return {'objectId': objectHandle._remoteObject.get('objectId')}
        return {'value': arg}

    async def queryObjects(self, prototypeHandle: "JSHandle") -> "JSHandle":
        """Send query.

        Details see :meth:`pyppeteer.page.Page.queryObjects`.
        """
        if prototypeHandle._disposed:
            raise ElementHandleError('Prototype JSHandle is disposed!')
        if not prototypeHandle._remoteObject.get('objectId'):
            raise ElementHandleError('Prototype JSHandle must not be referencing primitive value')
        response: Dict[str, Any] = await self._client.send(
            'Runtime.queryObjects',
            {'prototypeObjectId': prototypeHandle._remoteObject['objectId']}
        )
        return self._objectHandleFactory(response.get('objects'))


class JSHandle(object):
    """JSHandle class.

    JSHandle represents an in-page JavaScript object. JSHandle can be created
    with the :meth:`~pyppeteer.page.Page.evaluateHandle` method.
    """

    def __init__(self, context: ExecutionContext, client: Any, remoteObject: Dict[str, Any]) -> None:
        self._context = context
        self._client = client
        self._remoteObject = remoteObject
        self._disposed = False

    @property
    def executionContext(self) -> ExecutionContext:
        """Get execution context of this handle."""
        return self._context

    async def getProperty(self, propertyName: str) -> "JSHandle":
        """Get property value of ``propertyName``."""
        objectHandle: JSHandle = await self._context.evaluateHandle(
            '(object, propertyName) => {\n                const result = {__proto__: null};\n                result[propertyName] = object[propertyName];\n                return result;\n            }', self, propertyName)
        properties: Dict[str, JSHandle] = await objectHandle.getProperties()
        result: JSHandle = properties[propertyName]
        await objectHandle.dispose()
        return result

    async def getProperties(self) -> Dict[str, "JSHandle"]:
        """Get all properties of this handle."""
        response: Dict[str, Any] = await self._client.send(
            'Runtime.getProperties',
            {
                'objectId': self._remoteObject.get('objectId', ''),
                'ownProperties': True
            }
        )
        result: Dict[str, JSHandle] = {}
        for prop in response['result']:
            if not prop.get('enumerable'):
                continue
            result[prop.get('name')] = self._context._objectHandleFactory(prop.get('value'))
        return result

    async def jsonValue(self) -> Any:
        """Get Jsonized value of this object."""
        objectId: Optional[Any] = self._remoteObject.get('objectId')
        if objectId:
            response: Dict[str, Any] = await self._client.send(
                'Runtime.callFunctionOn',
                {
                    'functionDeclaration': 'function() { return this; }',
                    'objectId': objectId,
                    'returnByValue': True,
                    'awaitPromise': True
                }
            )
            return helper.valueFromRemoteObject(response['result'])
        return helper.valueFromRemoteObject(self._remoteObject)

    def asElement(self) -> Optional["JSHandle"]:
        """Return either null or the object handle itself."""
        return None

    async def dispose(self) -> None:
        """Stop referencing the handle."""
        if self._disposed:
            return
        self._disposed = True
        try:
            await helper.releaseObject(self._client, self._remoteObject)
        except Exception as e:
            debugError(logger, e)

    def toString(self) -> str:
        """Get string representation."""
        if self._remoteObject.get('objectId'):
            _type: Optional[str] = self._remoteObject.get('subtype') or self._remoteObject.get('type')
            return f'JSHandle@{_type}'
        return 'JSHandle:{}'.format(helper.valueFromRemoteObject(self._remoteObject))


def _rewriteError(error: Exception) -> None:
    if error.args[0].endswith('Cannot find context with specified id'):
        msg = 'Execution context was destroyed, most likely because of a navigation.'
        raise type(error)(msg)
    raise error
