"""Worker module."""
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

    def __init__(
        self,
        client: 'CDPSession',
        url: str,
        consoleAPICalled: Callable[[str, List[JSHandle]], None],
        exceptionThrown: Callable[[Dict[str, Any]], None]
    ) -> None:
        super().__init__()
        self._client = client
        self._url = url
        self._loop = client._loop
        self._executionContextPromise = self._loop.create_future()

        # Define jsHandleFactory as a variable that can be reassigned
        jsHandleFactory: Callable[[Dict[str, Any]], Optional[JSHandle]] = lambda remoteObject: None

        def onExecutionContentCreated(event: Dict[str, Any]) -> None:
            nonlocal jsHandleFactory
            executionContext = ExecutionContext(client, event['context'], jsHandleFactory)
            
            # Define the actual factory function
            def actualFactory(remoteObject: Dict[str, Any]) -> JSHandle:
                return JSHandle(executionContext, client, remoteObject)
            
            # Reassign jsHandleFactory to the actual factory
            jsHandleFactory = actualFactory
            self._executionContextCallback(executionContext)
            
        self._client.on('Runtime.executionContextCreated', onExecutionContentCreated)
        try:
            self._client.send('Runtime.enable', {})
        except Exception as e:
            debugError(logger, e)

        def onConsoleAPICalled(event: Dict[str, Any]) -> None:
            args: List[JSHandle] = []
            for arg in event.get('args', []):
                factory_result = jsHandleFactory(arg)
                if factory_result is not None:
                    args.append(factory_result)
            consoleAPICalled(event['type'], args)
        self._client.on('Runtime.consoleAPICalled', onConsoleAPICalled)
        self._client.on('Runtime.exceptionThrown', lambda exception: exceptionThrown(exception['exceptionDetails']))

    def _executionContextCallback(self, value: ExecutionContext) -> None:
        self._executionContextPromise.set_result(value)

    @property
    def url(self) -> str:
        """Return URL."""
        return self._url

    async def executionContext(self) -> ExecutionContext:
        """Return ExecutionContext."""
        return await self._executionContextPromise

    async def evaluate(self, pageFunction: str, *args: Any) -> Any:
        """Evaluate ``pageFunction`` with ``args``.

        Shortcut for ``(await worker.executionContext).evaluate(pageFunction, *args)``.
        """
        return await (await self._executionContextPromise).evaluate(pageFunction, *args)

    async def evaluateHandle(self, pageFunction: str, *args: Any) -> JSHandle:
        """Evaluate ``pageFunction`` with ``args`` and return :class:`~pyppeteer.execution_context.JSHandle`.

        Shortcut for ``(await worker.executionContext).evaluateHandle(pageFunction, *args)``.
        """
        return await (await self._executionContextPromise).evaluateHandle(pageFunction, *args)