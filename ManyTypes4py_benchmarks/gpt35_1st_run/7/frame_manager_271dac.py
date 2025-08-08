import asyncio
from collections import OrderedDict
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Dict, Generator, List, Optional, Set, Union
from pyee import EventEmitter
from pyppeteer import helper
from pyppeteer.connection import CDPSession
from pyppeteer.element_handle import ElementHandle
from pyppeteer.errors import NetworkError, ElementHandleError, PageError, TimeoutError
from pyppeteer.execution_context import ExecutionContext, JSHandle
from pyppeteer.util import merge_dict

logger: logging.Logger = logging.getLogger(__name__)

class FrameManager(EventEmitter):
    Events: SimpleNamespace = SimpleNamespace(FrameAttached='frameattached', FrameNavigated='framenavigated', FrameDetached='framedetached', LifecycleEvent='lifecycleevent', FrameNavigatedWithinDocument='framenavigatedwithindocument')

    def __init__(self, client: CDPSession, frameTree: Dict[str, Any], page: Any) -> None:
        super().__init__()
        self._client: CDPSession = client
        self._page: Any = page
        self._frames: OrderedDict[str, Frame] = OrderedDict()
        self._mainFrame: Optional[Frame] = None
        self._contextIdToContext: Dict[str, ExecutionContext] = dict()
        client.on('Page.frameAttached', lambda event: self._onFrameAttached(event.get('frameId', ''), event.get('parentFrameId', '')))
        client.on('Page.frameNavigated', lambda event: self._onFrameNavigated(event.get('frame')))
        client.on('Page.navigatedWithinDocument', lambda event: self._onFrameNavigatedWithinDocument(event.get('frameId'), event.get('url')))
        client.on('Page.frameDetached', lambda event: self._onFrameDetached(event.get('frameId')))
        client.on('Page.frameStoppedLoading', lambda event: self._onFrameStoppedLoading(event.get('frameId')))
        client.on('Runtime.executionContextCreated', lambda event: self._onExecutionContextCreated(event.get('context')))
        client.on('Runtime.executionContextDestroyed', lambda event: self._onExecutionContextDestroyed(event.get('executionContextId')))
        client.on('Runtime.executionContextsCleared', lambda event: self._onExecutionContextsCleared())
        client.on('Page.lifecycleEvent', lambda event: self._onLifecycleEvent(event))
        self._handleFrameTree(frameTree)

    def _onLifecycleEvent(self, event: Dict[str, Any]) -> None:
        frame: Optional[Frame] = self._frames.get(event['frameId'])
        if not frame:
            return
        frame._onLifecycleEvent(event['loaderId'], event['name'])
        self.emit(FrameManager.Events.LifecycleEvent, frame)

    def _onFrameStoppedLoading(self, frameId: str) -> None:
        frame: Optional[Frame] = self._frames.get(frameId)
        if not frame:
            return
        frame._onLoadingStopped()
        self.emit(FrameManager.Events.LifecycleEvent, frame)

    def _handleFrameTree(self, frameTree: Dict[str, Any]) -> None:
        frame: Dict[str, Any] = frameTree['frame']
        if 'parentId' in frame:
            self._onFrameAttached(frame['id'], frame['parentId'])
        self._onFrameNavigated(frame)
        if 'childFrames' not in frameTree:
            return
        for child in frameTree['childFrames']:
            self._handleFrameTree(child)

    @property
    def mainFrame(self) -> Optional[Frame]:
        return self._mainFrame

    def frames(self) -> List[Frame]:
        return list(self._frames.values())

    def frame(self, frameId: str) -> Optional[Frame]:
        return self._frames.get(frameId)

    def _onFrameAttached(self, frameId: str, parentFrameId: str) -> None:
        if frameId in self._frames:
            return
        parentFrame: Optional[Frame] = self._frames.get(parentFrameId)
        frame: Frame = Frame(self._client, parentFrame, frameId)
        self._frames[frameId] = frame
        self.emit(FrameManager.Events.FrameAttached, frame)

    def _onFrameNavigated(self, framePayload: Dict[str, Any]) -> None:
        isMainFrame: bool = not framePayload.get('parentId')
        if isMainFrame:
            frame: Frame = self._mainFrame
        else:
            frame: Optional[Frame] = self._frames.get(framePayload.get('id', ''))
        if not (isMainFrame or frame):
            raise PageError('We either navigate top level or have old version of the navigated frame')
        if frame:
            for child in frame.childFrames:
                self._removeFramesRecursively(child)
        _id: str = framePayload.get('id', '')
        if isMainFrame:
            if frame:
                self._frames.pop(frame._id, None)
                frame._id = _id
            else:
                frame = Frame(self._client, None, _id)
            self._frames[_id] = frame
            self._mainFrame = frame
        frame._navigated(framePayload)
        self.emit(FrameManager.Events.FrameNavigated, frame)

    def _onFrameNavigatedWithinDocument(self, frameId: str, url: str) -> None:
        frame: Optional[Frame] = self._frames.get(frameId)
        if not frame:
            return
        frame._navigatedWithinDocument(url)
        self.emit(FrameManager.Events.FrameNavigatedWithinDocument, frame)
        self.emit(FrameManager.Events.FrameNavigated, frame)

    def _onFrameDetached(self, frameId: str) -> None:
        frame: Optional[Frame] = self._frames.get(frameId)
        if frame:
            self._removeFramesRecursively(frame)

    def _onExecutionContextCreated(self, contextPayload: Dict[str, Any]) -> None:
        if contextPayload.get('auxData') and contextPayload['auxData'].get('frameId'):
            frameId: Optional[str] = contextPayload['auxData']['frameId']
        else:
            frameId = None
        frame: Optional[Frame] = self._frames.get(frameId)

        def _createJSHandle(obj: Any) -> JSHandle:
            context: ExecutionContext = self.executionContextById(contextPayload['id'])
            return self.createJSHandle(context, obj)
        context: ExecutionContext = ExecutionContext(self._client, contextPayload, _createJSHandle, frame)
        self._contextIdToContext[contextPayload['id']] = context
        if frame:
            frame._addExecutionContext(context)

    def _onExecutionContextDestroyed(self, executionContextId: str) -> None:
        context: Optional[ExecutionContext] = self._contextIdToContext.get(executionContextId)
        if not context:
            return
        del self._contextIdToContext[executionContextId]
        frame: Optional[Frame] = context.frame
        if frame:
            frame._removeExecutionContext(context)

    def _onExecutionContextsCleared(self) -> None:
        for context in self._contextIdToContext.values():
            frame: Optional[Frame] = context.frame
            if frame:
                frame._removeExecutionContext(context)
        self._contextIdToContext.clear()

    def executionContextById(self, contextId: str) -> ExecutionContext:
        context: Optional[ExecutionContext] = self._contextIdToContext.get(contextId)
        if not context:
            raise ElementHandleError(f'INTERNAL ERROR: missing context with id = {contextId}')
        return context

    def createJSHandle(self, context: ExecutionContext, remoteObject: Optional[Dict[str, Any]] = None) -> Union[ElementHandle, JSHandle]:
        if remoteObject is None:
            remoteObject = dict()
        if remoteObject.get('subtype') == 'node':
            return ElementHandle(context, self._client, remoteObject, self._page, self)
        return JSHandle(context, self._client, remoteObject)

    def _removeFramesRecursively(self, frame: Frame) -> None:
        for child in frame.childFrames:
            self._removeFramesRecursively(child)
        frame._detach()
        self._frames.pop(frame._id, None)
        self.emit(FrameManager.Events.FrameDetached, frame)

class Frame:
    def __init__(self, client: CDPSession, parentFrame: Optional['Frame'], frameId: str) -> None:
        self._client: CDPSession = client
        self._parentFrame: Optional['Frame'] = parentFrame
        self._url: str = ''
        self._detached: bool = False
        self._id: str = frameId
        self._documentPromise: Optional[Awaitable] = None
        self._contextResolveCallback: Callable[[Any], None] = lambda _: None
        self._setDefaultContext(None)
        self._waitTasks: Set[WaitTask] = set()
        self._loaderId: str = ''
        self._lifecycleEvents: Set[str] = set()
        self._childFrames: Set['Frame'] = set()
        if self._parentFrame:
            self._parentFrame._childFrames.add(self)

    def _addExecutionContext(self, context: ExecutionContext) -> None:
        if context._isDefault:
            self._setDefaultContext(context)

    def _removeExecutionContext(self, context: ExecutionContext) -> None:
        if context._isDefault:
            self._setDefaultContext(None)

    def _setDefaultContext(self, context: Optional[ExecutionContext]) -> None:
        if context is not None:
            self._contextResolveCallback(context)
            self._contextResolveCallback = lambda _: None
            for waitTask in self._waitTasks:
                self._client._loop.create_task(waitTask.rerun())
        else:
            self._documentPromise = None
            self._contextPromise = self._client._loop.create_future()
            self._contextResolveCallback = lambda _context: self._contextPromise.set_result(_context)

    async def executionContext(self) -> ExecutionContext:
        return await self._contextPromise

    async def evaluateHandle(self, pageFunction: str, *args: Any) -> JSHandle:
        context: ExecutionContext = await self.executionContext()
        if context is None:
            raise PageError('this frame has no context.')
        return await context.evaluateHandle(pageFunction, *args)

    async def evaluate(self, pageFunction: str, *args: Any, force_expr: bool = False) -> Any:
        context: ExecutionContext = await self.executionContext()
        if context is None:
            raise ElementHandleError('ExecutionContext is None.')
        return await context.evaluate(pageFunction, *args, force_expr=force_expr)

    async def querySelector(self, selector: str) -> ElementHandle:
        document: ElementHandle = await self._document()
        value: ElementHandle = await document.querySelector(selector)
        return value

    async def _document(self) -> ElementHandle:
        if self._documentPromise:
            return self._documentPromise
        context: ExecutionContext = await self.executionContext()
        if context is None:
            raise PageError('No context exists.')
        document: ElementHandle = (await context.evaluateHandle('document')).asElement()
        self._documentPromise = document
        if document is None:
            raise PageError('Could not find `document`.')
        return document

    async def xpath(self, expression: str) -> List[ElementHandle]:
        document: ElementHandle = await self._document()
        value: List[ElementHandle] = await document.xpath(expression)
        return value

    async def querySelectorEval(self, selector: str, pageFunction: str, *args: Any) -> Any:
        document: ElementHandle = await self._document()
        return await document.querySelectorEval(selector, pageFunction, *args)

    async def querySelectorAllEval(self, selector: str, pageFunction: str, *args: Any) -> Any:
        document: ElementHandle = await self._document()
        value: Any = await document.JJeval(selector, pageFunction, *args)
        return value

    async def querySelectorAll(self, selector: str) -> List[ElementHandle]:
        document: ElementHandle = await self._document()
        value: List[ElementHandle] = await document.querySelectorAll(selector)
        return value

    J: Any = querySelector
    Jx: Any = xpath
    Jeval: Any = querySelectorEval
    JJ: Any = querySelectorAll
    JJeval: Any = querySelectorAllEval

    async def content(self) -> str:
        return await self.evaluate("\n() => {\n  let retVal = '';\n  if (document.doctype)\n    retVal = new XMLSerializer().serializeToString(document.doctype);\n  if (document.documentElement)\n    retVal += document.documentElement.outerHTML;\n  return retVal;\n}\n        ".strip())

    async def setContent(self, html: str) -> None:
        func: str = '\nfunction(html) {\n  document.open();\n  document.write(html);\n  document.close();\n}\n'
        await self.evaluate(func, html)

    @property
    def name(self) -> str:
        return self.__dict__.get('_name', '')

    @property
    def url(self) -> str:
        return self._url

    @property
    def parentFrame(self) -> Optional['Frame']:
        return self._parentFrame

    @property
    def childFrames(self) -> List['Frame']:
        return list(self._childFrames)

    def isDetached(self) -> bool:
        return self._detached

    async def injectFile(self, filePath: str) -> Any:
        logger.warning('`injectFile` method is deprecated. Use `addScriptTag` method instead.')
        with open(filePath) as f:
            contents: str = f.read()
        contents += '/* # sourceURL= {} */'.format(filePath.replace('\n', ''))
        return await self.evaluate(contents)

    async def addScriptTag(self, options: Dict[str, Any]) -> ElementHandle:
        context: ExecutionContext = await self.executionContext()
        if context is None:
            raise ElementHandleError('ExecutionContext is None.')
        addScriptUrl: str = "\n        async function addScriptUrl(url, type) {\n            const script = document.createElement('script');\n            script.src = url;\n            if (type)\n                script.type = type;\n            const promise = new Promise((res, rej) => {\n                script.onload = res;\n                script.onerror = rej;\n            });\n            document.head.appendChild(script);\n            await promise;\n            return script;\n        }"
        addScriptContent: str = "\n        function addScriptContent(content, type = 'text/javascript') {\n            const script = document.createElement('script');\n            script.type = type;\n            script.text = content;\n            let error = null;\n            script.onerror = e => error = e;\n            document.head.appendChild(script);\n            if (error)\n                throw error;\n            return script;\n        }"
        if isinstance(options.get('url'), str):
            url: str = options['url']
            args: List[Any] = [addScriptUrl, url]
            if 'type' in options:
                args.append(options['type'])
            try:
                return (await context.evaluateHandle(*args)).asElement()
            except ElementHandleError as e:
                raise PageError(f'Loading script from {url} failed') from e
        if isinstance(options.get('path'), str):
            with open(options['path']) as f:
                contents: str = f.read()
            contents = contents + '//# sourceURL={}'.format(options['path'].replace('\n', ''))
            args: List[Any] = [addScriptContent, contents]
            if 'type' in options:
                args.append(options['type'])
            return (await context.evaluateHandle(*args)).asElement()
        if isinstance(options.get('content'), str):
            args: List[Any] = [addScriptContent, options['content']]
            if 'type' in options:
                args.append(options['type'])
            return (await context.evaluateHandle(*args)).asElement()
        raise ValueError('Provide an object with a `url`, `path` or `content` property')

    async def addStyleTag(self, options: Dict[str, Any]) -> ElementHandle:
        context: ExecutionContext = await self.executionContext()
        if context is None:
            raise ElementHandleError('ExecutionContext is None.')
        addStyleUrl: str = "\n        async function (url) {\n            const link = document.createElement('link');\n            link.rel = 'stylesheet';\n            link.href = url;\n            const promise = new Promise((res, rej) => {\n                link.onload = res;\n                link.onerror = rej;\n            });\n            document.head.appendChild(link);\n            await promise;\n            return link;\n        }"
        addStyleContent: str = "\n        async function (content) {\n            const style = document.createElement('style');\n            style.type = 'text/css';\n            style.appendChild(document.createTextNode(content));\n            const promise = new Promise((res, rej) => {\n                style.onload = res;\n                style.onerror = rej;\n            });\n            document.head.appendChild(style);\n            await promise;\n            return style;\n        }"
        if isinstance(options.get('url'), str):
            url: str = options['url']
            try:
                return (await context.evaluateHandle(addStyleUrl, url)).asElement()
            except ElementHandleError as e:
                raise PageError(f'Loading style from {url} failed') from e
        if isinstance(options.get('path'), str):
            with open(options['path']) as f:
                contents: str = f.read()
            contents = contents + '/*# sourceURL={}*/'.format(options['path'].replace('\n', ''))
            return (await context.evaluateHandle(addStyleContent, contents)).asElement()
        if isinstance(options.get('content'), str):
            return (await context.evaluateHandle(addStyleContent, options['content'])).asElement()
        raise ValueError('Provide an object with a `url`, `path` or `content` property')

    async def click(self, selector: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        options: Dict[str, Any] = merge_dict(options, kwargs)
        handle: ElementHandle = await self.J(selector)
        if not handle:
            raise PageError('No node found for selector: ' + selector)
        await handle.click(options)
        await handle.dispose()

    async def focus(self, selector: str) -> None:
        handle: ElementHandle = await self.J(selector)
        if not handle:
            raise PageError('No node found for selector: ' + selector)
        await self.evaluate('element => element.focus()', handle)
        await handle.dispose()

    async def hover(self, selector: str) -> None:
        handle: ElementHandle = await self.J(selector)
        if not handle:
            raise PageError('No node found for selector: ' + selector)
        await handle.hover()
        await handle.dispose()

    async def select(self, selector: str, *values: str) -> List[str]:
        for value in values:
           