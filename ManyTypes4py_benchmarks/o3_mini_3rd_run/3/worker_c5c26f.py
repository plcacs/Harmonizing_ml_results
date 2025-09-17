import asyncio
import logging
from typing import Any, Callable, Dict, List, TYPE_CHECKING

from pyee import EventEmitter
from pyppeteer.execution_context import ExecutionContext, JSHandle
from pyppeteer.helper import debugError

if TYPE_CHECKING:
    from pyppeteer.connection import CDPSession

logger = logging.getLogger(__name__)


class Worker(EventEmitter):
    def __init__(
        self,
        client: Any,
        url: str,
        consoleAPICalled: Callable[[str, List[JSHandle]], None],
        exceptionThrown: Callable[[Dict[str, Any]], None],
    ) -> None:
        super().__init__()
        self._client: Any = client
        self._url: str = url
        self._loop: asyncio.AbstractEventLoop = client._loop
        self._executionContextPromise: "asyncio.Future[ExecutionContext]" = self._loop.create_future()

        def jsHandleFactory(remoteObject: Dict[str, Any]) -> Any:
            return None

        def onExecutionContentCreated(event: Dict[str, Any]) -> None:
            nonlocal jsHandleFactory

            def jsHandleFactory(remoteObject: Dict[str, Any]) -> JSHandle:
                return JSHandle(executionContext, client, remoteObject)

            executionContext: ExecutionContext = ExecutionContext(client, event["context"], jsHandleFactory)
            self._executionContextCallback(executionContext)

        self._client.on("Runtime.executionContextCreated", onExecutionContentCreated)
        try:
            self._client.send("Runtime.enable", {})
        except Exception as e:
            debugError(logger, e)

        def onConsoleAPICalled(event: Dict[str, Any]) -> None:
            args: List[JSHandle] = []
            for arg in event.get("args", []):
                args.append(jsHandleFactory(arg))
            consoleAPICalled(event["type"], args)

        self._client.on("Runtime.consoleAPICalled", onConsoleAPICalled)
        self._client.on("Runtime.exceptionThrown", lambda exception: exceptionThrown(exception["exceptionDetails"]))

    def _executionContextCallback(self, value: ExecutionContext) -> None:
        self._executionContextPromise.set_result(value)

    @property
    def url(self) -> str:
        return self._url

    async def executionContext(self) -> ExecutionContext:
        return await self._executionContextPromise

    async def evaluate(self, pageFunction: Any, *args: Any) -> Any:
        return await (await self._executionContextPromise).evaluate(pageFunction, *args)

    async def evaluateHandle(self, pageFunction: Any, *args: Any) -> JSHandle:
        return await (await self._executionContextPromise).evaluateHandle(pageFunction, *args)