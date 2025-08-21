from __future__ import annotations

"""Frame Manager module."""
import asyncio
from collections import OrderedDict
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Generator, List, Optional, Set, Tuple, Union, TYPE_CHECKING
from pyee import EventEmitter
from pyppeteer import helper
from pyppeteer.connection import CDPSession
from pyppeteer.element_handle import ElementHandle
from pyppeteer.errors import NetworkError
from pyppeteer.execution_context import ExecutionContext, JSHandle
from pyppeteer.errors import ElementHandleError, PageError, TimeoutError
from pyppeteer.util import merge_dict

if TYPE_CHECKING:
    from pyppeteer.page import Page

logger = logging.getLogger(__name__)


class FrameManager(EventEmitter):
    """FrameManager class."""
    Events = SimpleNamespace(
        FrameAttached='frameattached',
        FrameNavigated='framenavigated',
        FrameDetached='framedetached',
        LifecycleEvent='lifecycleevent',
        FrameNavigatedWithinDocument='framenavigatedwithindocument',
    )

    def __init__(self, client: CDPSession, frameTree: Dict[str, Any], page: "Page") -> None:
        """Make new frame manager."""
        super().__init__()
        self._client: CDPSession = client
        self._page: "Page" = page
        self._frames: "OrderedDict[str, Frame]" = OrderedDict()
        self._mainFrame: Optional[Frame] = None
        self._contextIdToContext: Dict[int, ExecutionContext] = dict()
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
        frame = self._frames.get(event['frameId'])
        if not frame:
            return
        frame._onLifecycleEvent(event['loaderId'], event['name'])
        self.emit(FrameManager.Events.LifecycleEvent, frame)

    def _onFrameStoppedLoading(self, frameId: str) -> None:
        frame = self._frames.get(frameId)
        if not frame:
            return
        frame._onLoadingStopped()
        self.emit(FrameManager.Events.LifecycleEvent, frame)

    def _handleFrameTree(self, frameTree: Dict[str, Any]) -> None:
        frame = frameTree['frame']
        if 'parentId' in frame:
            self._onFrameAttached(frame['id'], frame['parentId'])
        self._onFrameNavigated(frame)
        if 'childFrames' not in frameTree:
            return
        for child in frameTree['childFrames']:
            self._handleFrameTree(child)

    @property
    def mainFrame(self) -> Optional[Frame]:
        """Return main frame."""
        return self._mainFrame

    def frames(self) -> List[Frame]:
        """Return all frames."""
        return list(self._frames.values())

    def frame(self, frameId: str) -> Optional[Frame]:
        """Return :class:`Frame` of ``frameId``."""
        return self._frames.get(frameId)

    def _onFrameAttached(self, frameId: str, parentFrameId: str) -> None:
        if frameId in self._frames:
            return
        parentFrame = self._frames.get(parentFrameId)
        frame = Frame(self._client, parentFrame, frameId)
        self._frames[frameId] = frame
        self.emit(FrameManager.Events.FrameAttached, frame)

    def _onFrameNavigated(self, framePayload: Dict[str, Any]) -> None:
        isMainFrame = not framePayload.get('parentId')
        if isMainFrame:
            frame = self._mainFrame
        else:
            frame = self._frames.get(framePayload.get('id', ''))
        if not (isMainFrame or frame):
            raise PageError('We either navigate top level or have old version of the navigated frame')
        if frame:
            for child in frame.childFrames:
                self._removeFramesRecursively(child)
        _id = framePayload.get('id', '')
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
        frame = self._frames.get(frameId)
        if not frame:
            return
        frame._navigatedWithinDocument(url)
        self.emit(FrameManager.Events.FrameNavigatedWithinDocument, frame)
        self.emit(FrameManager.Events.FrameNavigated, frame)

    def _onFrameDetached(self, frameId: str) -> None:
        frame = self._frames.get(frameId)
        if frame:
            self._removeFramesRecursively(frame)

    def _onExecutionContextCreated(self, contextPayload: Dict[str, Any]) -> None:
        if contextPayload.get('auxData') and contextPayload['auxData'].get('frameId'):
            frameId = contextPayload['auxData']['frameId']
        else:
            frameId = None
        frame = self._frames.get(frameId)

        def _createJSHandle(obj: Dict[str, Any]) -> Union[ElementHandle, JSHandle]:
            context = self.executionContextById(contextPayload['id'])
            return self.createJSHandle(context, obj)

        context = ExecutionContext(self._client, contextPayload, _createJSHandle, frame)
        self._contextIdToContext[contextPayload['id']] = context
        if frame:
            frame._addExecutionContext(context)

    def _onExecutionContextDestroyed(self, executionContextId: int) -> None:
        context = self._contextIdToContext.get(executionContextId)
        if not context:
            return
        del self._contextIdToContext[executionContextId]
        frame = context.frame
        if frame:
            frame._removeExecutionContext(context)

    def _onExecutionContextsCleared(self) -> None:
        for context in self._contextIdToContext.values():
            frame = context.frame
            if frame:
                frame._removeExecutionContext(context)
        self._contextIdToContext.clear()

    def executionContextById(self, contextId: int) -> ExecutionContext:
        """Get stored ``ExecutionContext`` by ``id``."""
        context = self._contextIdToContext.get(contextId)
        if not context:
            raise ElementHandleError(f'INTERNAL ERROR: missing context with id = {contextId}')
        return context

    def createJSHandle(self, context: ExecutionContext, remoteObject: Optional[Dict[str, Any]] = None) -> Union[ElementHandle, JSHandle]:
        """Create JS handle associated to the context id and remote object."""
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


class Frame(object):
    """Frame class.

    Frame objects can be obtained via :attr:`pyppeteer.page.Page.mainFrame`.
    """

    def __init__(self, client: CDPSession, parentFrame: Optional[Frame], frameId: str) -> None:
        self._client: CDPSession = client
        self._parentFrame: Optional[Frame] = parentFrame
        self._url: str = ''
        self._detached: bool = False
        self._id: str = frameId
        self._documentPromise: Optional[ElementHandle] = None
        self._contextResolveCallback: Callable[[ExecutionContext], None] = lambda _: None
        self._setDefaultContext(None)
        self._waitTasks: Set[WaitTask] = set()
        self._loaderId: str = ''
        self._lifecycleEvents: Set[str] = set()
        self._childFrames: Set[Frame] = set()
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
            self._contextPromise: asyncio.Future[ExecutionContext] = self._client._loop.create_future()
            self._contextResolveCallback = lambda _context: self._contextPromise.set_result(_context)

    async def executionContext(self) -> ExecutionContext:
        """Return execution context of this frame.

        Return :class:`~pyppeteer.execution_context.ExecutionContext`
        associated to this frame.
        """
        return await self._contextPromise

    async def evaluateHandle(self, pageFunction: Any, *args: Any) -> JSHandle:
        """Execute function on this frame.

        Details see :meth:`pyppeteer.page.Page.evaluateHandle`.
        """
        context = await self.executionContext()
        if context is None:
            raise PageError('this frame has no context.')
        return await context.evaluateHandle(pageFunction, *args)

    async def evaluate(self, pageFunction: Any, *args: Any, force_expr: bool = False) -> Any:
        """Evaluate pageFunction on this frame.

        Details see :meth:`pyppeteer.page.Page.evaluate`.
        """
        context = await self.executionContext()
        if context is None:
            raise ElementHandleError('ExecutionContext is None.')
        return await context.evaluate(pageFunction, *args, force_expr=force_expr)

    async def querySelector(self, selector: str) -> Optional[ElementHandle]:
        """Get element which matches `selector` string.

        Details see :meth:`pyppeteer.page.Page.querySelector`.
        """
        document = await self._document()
        value = await document.querySelector(selector)
        return value

    async def _document(self) -> ElementHandle:
        if self._documentPromise:
            return self._documentPromise
        context = await self.executionContext()
        if context is None:
            raise PageError('No context exists.')
        document = (await context.evaluateHandle('document')).asElement()
        self._documentPromise = document
        if document is None:
            raise PageError('Could not find `document`.')
        return document

    async def xpath(self, expression: str) -> List[ElementHandle]:
        """Evaluate the XPath expression.

        If there are no such elements in this frame, return an empty list.

        :arg str expression: XPath string to be evaluated.
        """
        document = await self._document()
        value = await document.xpath(expression)
        return value

    async def querySelectorEval(self, selector: str, pageFunction: Any, *args: Any) -> Any:
        """Execute function on element which matches selector.

        Details see :meth:`pyppeteer.page.Page.querySelectorEval`.
        """
        document = await self._document()
        return await document.querySelectorEval(selector, pageFunction, *args)

    async def querySelectorAllEval(self, selector: str, pageFunction: Any, *args: Any) -> Any:
        """Execute function on all elements which matches selector.

        Details see :meth:`pyppeteer.page.Page.querySelectorAllEval`.
        """
        document = await self._document()
        value = await document.JJeval(selector, pageFunction, *args)
        return value

    async def querySelectorAll(self, selector: str) -> List[ElementHandle]:
        """Get all elements which matches `selector`.

        Details see :meth:`pyppeteer.page.Page.querySelectorAll`.
        """
        document = await self._document()
        value = await document.querySelectorAll(selector)
        return value

    J = querySelector
    Jx = xpath
    Jeval = querySelectorEval
    JJ = querySelectorAll
    JJeval = querySelectorAllEval

    async def content(self) -> str:
        """Get the whole HTML contents of the page."""
        return await self.evaluate(
            "() => {\n  let retVal = '';\n  if (document.doctype)\n    retVal = new XMLSerializer().serializeToString(document.doctype);\n  if (document.documentElement)\n    retVal += document.documentElement.outerHTML;\n  return retVal;\n}\n".strip()
        )

    async def setContent(self, html: str) -> None:
        """Set content to this page."""
        func = '\nfunction(html) {\n  document.open();\n  document.write(html);\n  document.close();\n}\n'
        await self.evaluate(func, html)

    @property
    def name(self) -> str:
        """Get frame name."""
        return self.__dict__.get('_name', '')

    @property
    def url(self) -> str:
        """Get url of the frame."""
        return self._url

    @property
    def parentFrame(self) -> Optional[Frame]:
        """Get parent frame.

        If this frame is main frame or detached frame, return ``None``.
        """
        return self._parentFrame

    @property
    def childFrames(self) -> List[Frame]:
        """Get child frames."""
        return list(self._childFrames)

    def isDetached(self) -> bool:
        """Return ``True`` if this frame is detached.

        Otherwise return ``False``.
        """
        return self._detached

    async def injectFile(self, filePath: str) -> Any:
        """[Deprecated] Inject file to the frame."""
        logger.warning('`injectFile` method is deprecated. Use `addScriptTag` method instead.')
        with open(filePath) as f:
            contents = f.read()
        contents += '/* # sourceURL= {} */'.format(filePath.replace('\n', ''))
        return await self.evaluate(contents)

    async def addScriptTag(self, options: Dict[str, Any]) -> ElementHandle:
        """Add script tag to this frame.

        Details see :meth:`pyppeteer.page.Page.addScriptTag`.
        """
        context = await self.executionContext()
        if context is None:
            raise ElementHandleError('ExecutionContext is None.')
        addScriptUrl = "\n        async function addScriptUrl(url, type) {\n            const script = document.createElement('script');\n            script.src = url;\n            if (type)\n                script.type = type;\n            const promise = new Promise((res, rej) => {\n                script.onload = res;\n                script.onerror = rej;\n            });\n            document.head.appendChild(script);\n            await promise;\n            return script;\n        }"
        addScriptContent = "\n        function addScriptContent(content, type = 'text/javascript') {\n            const script = document.createElement('script');\n            script.type = type;\n            script.text = content;\n            let error = null;\n            script.onerror = e => error = e;\n            document.head.appendChild(script);\n            if (error)\n                throw error;\n            return script;\n        }"
        if isinstance(options.get('url'), str):
            url = options['url']
            args: List[Any] = [addScriptUrl, url]
            if 'type' in options:
                args.append(options['type'])
            try:
                handle = await context.evaluateHandle(*args)
                return handle.asElement()  # type: ignore[return-value]
            except ElementHandleError as e:
                raise PageError(f'Loading script from {url} failed') from e
        if isinstance(options.get('path'), str):
            with open(options['path']) as f:
                contents = f.read()
            contents = contents + '//# sourceURL={}'.format(options['path'].replace('\n', ''))
            args2: List[Any] = [addScriptContent, contents]
            if 'type' in options:
                args2.append(options['type'])
            handle2 = await context.evaluateHandle(*args2)
            return handle2.asElement()  # type: ignore[return-value]
        if isinstance(options.get('content'), str):
            args3: List[Any] = [addScriptContent, options['content']]
            if 'type' in options:
                args3.append(options['type'])
            handle3 = await context.evaluateHandle(*args3)
            return handle3.asElement()  # type: ignore[return-value]
        raise ValueError('Provide an object with a `url`, `path` or `content` property')

    async def addStyleTag(self, options: Dict[str, Any]) -> ElementHandle:
        """Add style tag to this frame.

        Details see :meth:`pyppeteer.page.Page.addStyleTag`.
        """
        context = await self.executionContext()
        if context is None:
            raise ElementHandleError('ExecutionContext is None.')
        addStyleUrl = "\n        async function (url) {\n            const link = document.createElement('link');\n            link.rel = 'stylesheet';\n            link.href = url;\n            const promise = new Promise((res, rej) => {\n                link.onload = res;\n                link.onerror = rej;\n            });\n            document.head.appendChild(link);\n            await promise;\n            return link;\n        }"
        addStyleContent = "\n        async function (content) {\n            const style = document.createElement('style');\n            style.type = 'text/css';\n            style.appendChild(document.createTextNode(content));\n            const promise = new Promise((res, rej) => {\n                style.onload = res;\n                style.onerror = rej;\n            });\n            document.head.appendChild(style);\n            await promise;\n            return style;\n        }"
        if isinstance(options.get('url'), str):
            url = options['url']
            try:
                handle = await context.evaluateHandle(addStyleUrl, url)
                return handle.asElement()  # type: ignore[return-value]
            except ElementHandleError as e:
                raise PageError(f'Loading style from {url} failed') from e
        if isinstance(options.get('path'), str):
            with open(options['path']) as f:
                contents = f.read()
            contents = contents + '/*# sourceURL={}*/'.format(options['path'].replace('\n', ''))
            handle2 = await context.evaluateHandle(addStyleContent, contents)
            return handle2.asElement()  # type: ignore[return-value]
        if isinstance(options.get('content'), str):
            handle3 = await context.evaluateHandle(addStyleContent, options['content'])
            return handle3.asElement()  # type: ignore[return-value]
        raise ValueError('Provide an object with a `url`, `path` or `content` property')

    async def click(self, selector: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Click element which matches ``selector``.

        Details see :meth:`pyppeteer.page.Page.click`.
        """
        options = merge_dict(options, kwargs)
        handle = await self.J(selector)
        if not handle:
            raise PageError('No node found for selector: ' + selector)
        await handle.click(options)
        await handle.dispose()

    async def focus(self, selector: str) -> None:
        """Focus element which matches ``selector``.

        Details see :meth:`pyppeteer.page.Page.focus`.
        """
        handle = await self.J(selector)
        if not handle:
            raise PageError('No node found for selector: ' + selector)
        await self.evaluate('element => element.focus()', handle)
        await handle.dispose()

    async def hover(self, selector: str) -> None:
        """Mouse hover the element which matches ``selector``.

        Details see :meth:`pyppeteer.page.Page.hover`.
        """
        handle = await self.J(selector)
        if not handle:
            raise PageError('No node found for selector: ' + selector)
        await handle.hover()
        await handle.dispose()

    async def select(self, selector: str, *values: str) -> List[str]:
        """Select options and return selected values.

        Details see :meth:`pyppeteer.page.Page.select`.
        """
        for value in values:
            if not isinstance(value, str):
                raise TypeError(f'Values must be string. Found {value} of type {type(value)}')
        return await self.querySelectorEval(
            selector,
            "\n(element, values) => {\n    if (element.nodeName.toLowerCase() !== 'select')\n        throw new Error('Element is not a <select> element.');\n\n    const options = Array.from(element.options);\n    element.value = undefined;\n    for (const option of options) {\n        option.selected = values.includes(option.value);\n        if (option.selected && !element.multiple)\n            break;\n    }\n\n    element.dispatchEvent(new Event('input', { 'bubbles': true }));\n    element.dispatchEvent(new Event('change', { 'bubbles': true }));\n    return options.filter(option => option.selected).map(options => options.value)\n}\n        ",
            values,
        )

    async def tap(self, selector: str) -> None:
        """Tap the element which matches the ``selector``.

        Details see :meth:`pyppeteer.page.Page.tap`.
        """
        handle = await self.J(selector)
        if not handle:
            raise PageError('No node found for selector: ' + selector)
        await handle.tap()
        await handle.dispose()

    async def type(self, selector: str, text: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Type ``text`` on the element which matches ``selector``.

        Details see :meth:`pyppeteer.page.Page.type`.
        """
        options = merge_dict(options, kwargs)
        handle = await self.querySelector(selector)
        if handle is None:
            raise PageError('Cannot find {} on this page'.format(selector))
        await handle.type(text, options)
        await handle.dispose()

    def waitFor(self, selectorOrFunctionOrTimeout: Union[str, int, float], options: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any) -> Awaitable[Any]:
        """Wait until `selectorOrFunctionOrTimeout`.

        Details see :meth:`pyppeteer.page.Page.waitFor`.
        """
        options = merge_dict(options, kwargs)
        if isinstance(selectorOrFunctionOrTimeout, (int, float)):
            fut: asyncio.Task[Any] = self._client._loop.create_task(asyncio.sleep(selectorOrFunctionOrTimeout / 1000))
            return fut
        if not isinstance(selectorOrFunctionOrTimeout, str):
            fut2: asyncio.Future[Any] = self._client._loop.create_future()
            fut2.set_exception(TypeError('Unsupported target type: ' + str(type(selectorOrFunctionOrTimeout))))
            return fut2
        if args or helper.is_jsfunc(selectorOrFunctionOrTimeout):
            return self.waitForFunction(selectorOrFunctionOrTimeout, options, *args)
        if selectorOrFunctionOrTimeout.startswith('//'):
            return self.waitForXPath(selectorOrFunctionOrTimeout, options)
        return self.waitForSelector(selectorOrFunctionOrTimeout, options)

    def waitForSelector(self, selector: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> WaitTask:
        """Wait until element which matches ``selector`` appears on page.

        Details see :meth:`pyppeteer.page.Page.waitForSelector`.
        """
        options = merge_dict(options, kwargs)
        return self._waitForSelectorOrXPath(selector, False, options)

    def waitForXPath(self, xpath: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> WaitTask:
        """Wait until element which matches ``xpath`` appears on page.

        Details see :meth:`pyppeteer.page.Page.waitForXPath`.
        """
        options = merge_dict(options, kwargs)
        return self._waitForSelectorOrXPath(xpath, True, options)

    def waitForFunction(self, pageFunction: Any, options: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any) -> WaitTask:
        """Wait until the function completes.

        Details see :meth:`pyppeteer.page.Page.waitForFunction`.
        """
        options = merge_dict(options, kwargs)
        timeout = options.get('timeout', 30000)
        polling: Union[str, int, float] = options.get('polling', 'raf')
        return WaitTask(self, pageFunction, 'function', polling, timeout, self._client._loop, *args)

    def _waitForSelectorOrXPath(self, selectorOrXPath: str, isXPath: bool, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> WaitTask:
        options = merge_dict(options, kwargs)
        timeout: Union[int, float, None] = options.get('timeout', 30000)
        waitForVisible = bool(options.get('visible'))
        waitForHidden = bool(options.get('hidden'))
        polling: Union[str, int, float] = 'raf' if waitForHidden or waitForVisible else 'mutation'
        title = '{} "{}"{}'.format('XPath' if isXPath else 'selector', selectorOrXPath, ' to be hidden' if waitForHidden else '')
        predicate = "\n(selectorOrXPath, isXPath, waitForVisible, waitForHidden) => {\n    const node = isXPath\n        ? document.evaluate(selectorOrXPath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue\n        : document.querySelector(selectorOrXPath);\n    if (!node)\n        return waitForHidden;\n    if (!waitForVisible && !waitForHidden)\n        return node;\n    const element = /** @type {Element} */ (node.nodeType === Node.TEXT_NODE ? node.parentElement : node);\n\n    const style = window.getComputedStyle(element);\n    const isVisible = style && style.visibility !== 'hidden' && hasVisibleBoundingBox();\n    const success = (waitForVisible === isVisible || waitForHidden === !isVisible)\n    return success ? node : null\n\n    function hasVisibleBoundingBox() {\n        const rect = element.getBoundingClientRect();\n        return !!(rect.top || rect.bottom || rect.width || rect.height);\n    }\n}\n        "
        return WaitTask(self, predicate, title, polling, timeout, self._client._loop, selectorOrXPath, isXPath, waitForVisible, waitForHidden)

    async def title(self) -> str:
        """Get title of the frame."""
        return await self.evaluate('() => document.title')

    def _navigated(self, framePayload: Dict[str, Any]) -> None:
        self._name = framePayload.get('name', '')
        self._navigationURL = framePayload.get('url', '')
        self._url = framePayload.get('url', '')

    def _navigatedWithinDocument(self, url: str) -> None:
        self._url = url

    def _onLifecycleEvent(self, loaderId: str, name: str) -> None:
        if name == 'init':
            self._loaderId = loaderId
            self._lifecycleEvents.clear()
        else:
            self._lifecycleEvents.add(name)

    def _onLoadingStopped(self) -> None:
        self._lifecycleEvents.add('DOMContentLoaded')
        self._lifecycleEvents.add('load')

    def _detach(self) -> None:
        for waitTask in self._waitTasks:
            waitTask.terminate(PageError('waitForFunction failed: frame got detached.'))
        self._detached = True
        if self._parentFrame:
            self._parentFrame._childFrames.remove(self)
        self._parentFrame = None


class WaitTask(object):
    """WaitTask class.

    Instance of this class is awaitable.
    """

    def __init__(
        self,
        frame: Frame,
        predicateBody: str,
        title: str,
        polling: Union[str, int, float],
        timeout: Optional[Union[int, float]],
        loop: asyncio.AbstractEventLoop,
        *args: Any
    ) -> None:
        if isinstance(polling, str):
            if polling not in ['raf', 'mutation']:
                raise ValueError(f'Unknown polling: {polling}')
        elif isinstance(polling, (int, float)):
            if polling <= 0:
                raise ValueError(f'Cannot poll with non-positive interval: {polling}')
        else:
            raise ValueError(f'Unknown polling option: {polling}')
        self._frame: Frame = frame
        self._polling: Union[str, int, float] = polling
        self._timeout: Optional[Union[int, float]] = timeout
        self._loop: asyncio.AbstractEventLoop = loop
        if args or helper.is_jsfunc(predicateBody):
            self._predicateBody: str = f'return ({predicateBody})(...args)'
        else:
            self._predicateBody = f'return {predicateBody}'
        self._args: Tuple[Any, ...] = args
        self._runCount: int = 0
        self._terminated: bool = False
        self._timeoutError: bool = False
        frame._waitTasks.add(self)
        self.promise: asyncio.Future[Union[JSHandle, Exception]] = self._loop.create_future()

        async def timer(timeout_val: Union[int, float]) -> None:
            await asyncio.sleep(timeout_val / 1000)
            self._timeoutError = True
            self.terminate(TimeoutError(f'Waiting for {title} failed: timeout {timeout_val}ms exceeds.'))

        if timeout:
            self._timeoutTimer: asyncio.Task[Any] = self._loop.create_task(timer(self._timeout))  # type: ignore[arg-type]
        self._runningTask: asyncio.Task[Any] = self._loop.create_task(self.rerun())

    def __await__(self):
        """Make this class **awaitable**."""
        result = (yield from self.promise)
        if isinstance(result, Exception):
            raise result
        return result

    def terminate(self, error: Exception) -> None:
        """Terminate this task."""
        self._terminated = True
        if not self.promise.done():
            self.promise.set_result(error)
        self._cleanup()

    async def rerun(self) -> None:
        """Start polling."""
        runCount = self._runCount = self._runCount + 1
        success: Optional[JSHandle] = None
        error: Optional[Exception] = None
        try:
            context = await self._frame.executionContext()
            if context is None:
                raise PageError('No execution context.')
            success = await context.evaluateHandle(waitForPredicatePageFunction, self._predicateBody, self._polling, self._timeout, *self._args)
        except Exception as e:
            error = e
        if self.promise.done():
            return
        if self._terminated or runCount != self._runCount:
            if success:
                await success.dispose()
            return
        try:
            if not error and success and await self._frame.evaluate('s => !s', success):
                await success.dispose()
                return
        except NetworkError:
            if success is not None:
                await success.dispose()
            return
        if isinstance(error, NetworkError) and 'Execution context was destroyed' in error.args[0]:
            return
        if isinstance(error, NetworkError) and 'Cannot find context with specified id' in error.args[0]:
            return
        if error:
            self.promise.set_exception(error)
        else:
            self.promise.set_result(success)  # type: ignore[arg-type]
        self._cleanup()

    def _cleanup(self) -> None:
        if self._timeout and (not self._timeoutError):
            self._timeoutTimer.cancel()  # type: ignore[attr-defined]
        self._frame._waitTasks.remove(self)


waitForPredicatePageFunction: str = "\nasync function waitForPredicatePageFunction(predicateBody, polling, timeout, ...args) {\n  const predicate = new Function('...args', predicateBody);\n  let timedOut = false;\n  if (timeout)\n    setTimeout(() => timedOut = true, timeout);\n  if (polling === 'raf')\n    return await pollRaf();\n  if (polling === 'mutation')\n    return await pollMutation();\n  if (typeof polling === 'number')\n    return await pollInterval(polling);\n\n  /**\n   * @return {!Promise<*>}\n   */\n  function pollMutation() {\n    const success = predicate.apply(null, args);\n    if (success)\n      return Promise.resolve(success);\n\n    let fulfill;\n    const result = new Promise(x => fulfill = x);\n    const observer = new MutationObserver(mutations => {\n      if (timedOut) {\n        observer.disconnect();\n        fulfill();\n      }\n      const success = predicate.apply(null, args);\n      if (success) {\n        observer.disconnect();\n        fulfill(success);\n      }\n    });\n    observer.observe(document, {\n      childList: true,\n      subtree: true,\n      attributes: true\n    });\n    return result;\n  }\n\n  /**\n   * @return {!Promise<*>}\n   */\n  function pollRaf() {\n    let fulfill;\n    const result = new Promise(x => fulfill = x);\n    onRaf();\n    return result;\n\n    function onRaf() {\n      if (timedOut) {\n        fulfill();\n        return;\n      }\n      const success = predicate.apply(null, args);\n      if (success)\n        fulfill(success);\n      else\n        requestAnimationFrame(onRaf);\n    }\n  }\n\n  /**\n   * @param {number} pollInterval\n   * @return {!Promise<*>}\n   */\n  function pollInterval(pollInterval) {\n    let fulfill;\n    const result = new Promise(x => fulfill = x);\n    onTimeout();\n    return result;\n\n    function onTimeout() {\n      if (timedOut) {\n        fulfill();\n        return;\n      }\n      const success = predicate.apply(null, args);\n      if (success)\n        fulfill(success);\n      else\n        setTimeout(onTimeout, pollInterval);\n    }\n  }\n}\n"