"""Worker module."""
import logging
from typing import Any, Callable, Dict, List, TYPE_CHECKING
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
        self._client = client
        self._url = url
        self._loop = client._loop
        self._executionContextPromise = self._loop.create_future()

        def jsHandleFactory(remoteObject):
            return None

        def onExecutionContentCreated(event):
            nonlocal jsHandleFactory

            def jsHandleFactory(remoteObject):
                return JSHandle(executionContext, client, remoteObject)
            executionContext = ExecutionContext(client, event['context'], jsHandleFactory)
            self._executionContextCallback(executionContext)
        self._client.on('Runtime.executionContextCreated', onExecutionContentCreated)
        try:
            self._client.send('Runtime.enable', {})
        except Exception as e:
            debugError(logger, e)

        def onConsoleAPICalled(event):
            args = []
            for arg in event.get('args', []):
                args.append(jsHandleFactory(arg))
            consoleAPICalled(event['type'], args)
        self._client.on('Runtime.consoleAPICalled', onConsoleAPICalled)
        self._client.on('Runtime.exceptionThrown', lambda exception: exceptionThrown(exception['exceptionDetails']))

    def _executionContextCallback(self, value):
        self._executionContextPromise.set_result(value)

    @property
    def url(self):
        """Return URL."""
        return self._url

    async def executionContext(self):
        """Return ExecutionContext."""
        return await self._executionContextPromise

    async def evaluate(self, pageFunction, *args):
        """Evaluate ``pageFunction`` with ``args``.

        Shortcut for ``(await worker.executionContext).evaluate(pageFunction, *args)``.
        """
        return await (await self._executionContextPromise).evaluate(pageFunction, *args)

    async def evaluateHandle(self, pageFunction, *args):
        """Evaluate ``pageFunction`` with ``args`` and return :class:`~pyppeteer.execution_context.JSHandle`.

        Shortcut for ``(await worker.executionContext).evaluateHandle(pageFunction, *args)``.
        """
        return await (await self._executionContextPromise).evaluateHandle(pageFunction, *args)