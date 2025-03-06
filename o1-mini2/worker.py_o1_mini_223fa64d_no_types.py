"""Worker module."""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from pyee import EventEmitter
from pyppeteer.execution_context import ExecutionContext, JSHandle
from pyppeteer.helper import debugError
if TYPE_CHECKING:
    from pyppeteer.connection import CDPSession
logger = logging.getLogger(__name__)


class Worker(EventEmitter):
    """The Worker class represents a WebWorker.

    The events `workercreated` and `workerdestroyed` are emitted on the page
    object to signal the worker lifecycle.

    .. code::

        page.on('workercreated', lambda worker: print('Worker created:', worker.url))
    """

    def __init__(self, client, url, consoleAPICalled, exceptionThrown):
        super().__init__()
        self._client: 'CDPSession' = client
        self._url: str = url
        self._loop: asyncio.AbstractEventLoop = client._loop
        self._executionContextPromise: asyncio.Future[ExecutionContext
            ] = self._loop.create_future()

        def jsHandleFactory(remoteObject):
            return None

        def onExecutionContentCreated(event):
            nonlocal jsHandleFactory

            def jsHandleFactory(remoteObject):
                return JSHandle(executionContext, client, remoteObject)
            executionContext: ExecutionContext = ExecutionContext(client,
                event['context'], jsHandleFactory)
            self._executionContextCallback(executionContext)
        self._client.on('Runtime.executionContextCreated',
            onExecutionContentCreated)
        try:
            self._client.send('Runtime.enable', {})
        except Exception as e:
            debugError(logger, e)

        def onConsoleAPICalled(event):
            args: List[JSHandle] = []
            for arg in event.get('args', []):
                handle: Optional[JSHandle] = jsHandleFactory(arg)
                if handle is not None:
                    args.append(handle)
            consoleAPICalled(event['type'], args)
        self._client.on('Runtime.consoleAPICalled', onConsoleAPICalled)
        self._client.on('Runtime.exceptionThrown', lambda exception:
            exceptionThrown(exception['exceptionDetails']))

    def _executionContextCallback(self, value):
        self._executionContextPromise.set_result(value)

    @property
    def url(self):
        """Return URL."""
        return self._url

    async def executionContext(self) ->ExecutionContext:
        """Return ExecutionContext."""
        return await self._executionContextPromise

    async def evaluate(self, pageFunction: str, *args: Any) ->Any:
        """Evaluate ``pageFunction`` with ``args``.

        Shortcut for ``(await worker.executionContext).evaluate(pageFunction, *args)``.
        """
        execution_context: ExecutionContext = (await self.
            _executionContextPromise)
        return await execution_context.evaluate(pageFunction, *args)

    async def evaluateHandle(self, pageFunction: str, *args: Any) ->JSHandle:
        """Evaluate ``pageFunction`` with ``args`` and return :class:`~pyppeteer.execution_context.JSHandle`.

        Shortcut for ``(await worker.executionContext).evaluateHandle(pageFunction, *args)``.
        """
        execution_context: ExecutionContext = (await self.
            _executionContextPromise)
        return await execution_context.evaluateHandle(pageFunction, *args)
