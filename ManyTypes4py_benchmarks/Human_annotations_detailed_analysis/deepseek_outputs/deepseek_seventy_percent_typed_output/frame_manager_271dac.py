#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Frame Manager module."""

import asyncio
from collections import OrderedDict
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Dict, Generator, List, Optional, Set, Union, Callable

from pyee import EventEmitter

from pyppeteer import helper
from pyppeteer.connection import CDPSession
from pyppeteer.element_handle import ElementHandle
from pyppeteer.errors import NetworkError
from pyppeteer.execution_context import ExecutionContext, JSHandle
from pyppeteer.errors import ElementHandleError, PageError, TimeoutError
from pyppeteer.util import merge_dict

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

    def __init__(self, client: CDPSession, frameTree: Dict, page: Any) -> None:
        """Make new frame manager."""
        super().__init__()
        self._client = client
        self._page = page
        self._frames: OrderedDict[str, 'Frame'] = OrderedDict()
        self._mainFrame: Optional['Frame'] = None
        self._contextIdToContext: Dict[str, ExecutionContext] = dict()

        client.on('Page.frameAttached',
                  lambda event: self._onFrameAttached(
                      event.get('frameId', ''), event.get('parentFrameId', ''))
                  )
        client.on('Page.frameNavigated',
                  lambda event: self._onFrameNavigated(event.get('frame')))
        client.on('Page.navigatedWithinDocument',
                  lambda event: self._onFrameNavigatedWithinDocument(
                      event.get('frameId'), event.get('url')
                  ))
        client.on('Page.frameDetached',
                  lambda event: self._onFrameDetached(event.get('frameId')))
        client.on('Page.frameStoppedLoading',
                  lambda event: self._onFrameStoppedLoading(
                      event.get('frameId')
                  ))
        client.on('Runtime.executionContextCreated',
                  lambda event: self._onExecutionContextCreated(
                      event.get('context')))
        client.on('Runtime.executionContextDestroyed',
                  lambda event: self._onExecutionContextDestroyed(
                      event.get('executionContextId')))
        client.on('Runtime.executionContextsCleared',
                  lambda event: self._onExecutionContextsCleared())
        client.on('Page.lifecycleEvent',
                  lambda event: self._onLifecycleEvent(event))

        self._handleFrameTree(frameTree)

    def _onLifecycleEvent(self, event: Dict) -> None:
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

    def _handleFrameTree(self, frameTree: Dict) -> None:
        frame = frameTree['frame']
        if 'parentId' in frame:
            self._onFrameAttached(
                frame['id'],
                frame['parentId'],
            )
        self._onFrameNavigated(frame)
        if 'childFrames' not in frameTree:
            return
        for child in frameTree['childFrames']:
            self._handleFrameTree(child)

    @property
    def mainFrame(self) -> Optional['Frame']:
        """Return main frame."""
        return self._mainFrame

    def frames(self) -> List['Frame']:
        """Return all frames."""
        return list(self._frames.values())

    def frame(self, frameId: str) -> Optional['Frame']:
        """Return :class:`Frame` of ``frameId``."""
        return self._frames.get(frameId)

    def _onFrameAttached(self, frameId: str, parentFrameId: str) -> None:
        if frameId in self._frames:
            return
        parentFrame = self._frames.get(parentFrameId)
        frame = Frame(self._client, parentFrame, frameId)
        self._frames[frameId] = frame
        self.emit(FrameManager.Events.FrameAttached, frame)

    def _onFrameNavigated(self, framePayload: Dict) -> None:
        isMainFrame = not framePayload.get('parentId')
        if isMainFrame:
            frame = self._mainFrame
        else:
            frame = self._frames.get(framePayload.get('id', ''))
        if not (isMainFrame or frame):
            raise PageError('We either navigate top level or have old version '
                            'of the navigated frame')

        # Detach all child frames first.
        if frame:
            for child in frame.childFrames:
                self._removeFramesRecursively(child)

        # Update or create main frame.
        _id = framePayload.get('id', '')
        if isMainFrame:
            if frame:
                # Update frame id to retain frame identity on cross-process navigation.  # noqa: E501
                self._frames.pop(frame._id, None)
                frame._id = _id
            else:
                # Initial main frame navigation.
                frame = Frame(self._client, None, _id)
            self._frames[_id] = frame
            self._mainFrame = frame

        # Update frame payload.
        frame._navigated(framePayload)  # type: ignore
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

    def _onExecutionContextCreated(self, contextPayload: Dict) -> None:
        if (contextPayload.get('auxData') and
                contextPayload['auxData'].get('frameId')):
            frameId = contextPayload['auxData']['frameId']
        else:
            frameId = None

        frame = self._frames.get(frameId)

        def _createJSHandle(obj: Dict) -> JSHandle:
            context = self.executionContextById(contextPayload['id'])
            return self.createJSHandle(context, obj)

        context = ExecutionContext(
            self._client,
            contextPayload,
            _createJSHandle,
            frame,
        )
        self._contextIdToContext[contextPayload['id']] = context

        if frame:
            frame._addExecutionContext(context)

    def _onExecutionContextDestroyed(self, executionContextId: str) -> None:
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

    def executionContextById(self, contextId: str) -> ExecutionContext:
        """Get stored ``ExecutionContext`` by ``id``."""
        context = self._contextIdToContext.get(contextId)
        if not context:
            raise ElementHandleError(
                f'INTERNAL ERROR: missing context with id = {contextId}'
            )
        return context

    def createJSHandle(self, context: ExecutionContext,
                       remoteObject: Dict = None) -> JSHandle:
        """Create JS handle associated to the context id and remote object."""
        if remoteObject is None:
            remoteObject = dict()
        if remoteObject.get('subtype') == 'node':
            return ElementHandle(context, self._client, remoteObject,
                                 self._page, self)
        return JSHandle(context, self._client, remoteObject)

    def _removeFramesRecursively(self, frame: 'Frame') -> None:
        for child in frame.childFrames:
            self._removeFramesRecursively(child)
        frame._detach()
        self._frames.pop(frame._id, None)
        self.emit(FrameManager.Events.FrameDetached, frame)


class Frame(object):
    """Frame class.

    Frame objects can be obtained via :attr:`pyppeteer.page.Page.mainFrame`.
    """

    def __init__(self, client: CDPSession, parentFrame: Optional['Frame'],
                 frameId: str) -> None:
        self._client = client
        self._parentFrame = parentFrame
        self._url = ''
        self._detached = False
        self._id = frameId

        self._documentPromise: Optional[ElementHandle] = None
        self._contextResolveCallback: Callable[[Optional[ExecutionContext]], None] = lambda _: None
        self._setDefaultContext(None)

        self._waitTasks: Set['WaitTask'] = set()  # maybe list
        self._loaderId = ''
        self._lifecycleEvents: Set[str] = set()
        self._childFrames: Set['Frame'] = set()  # maybe list
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
            self._contextResolveCallback(context)  # type: ignore
            self._contextResolveCallback = lambda _: None
            for waitTask in self._waitTasks:
                self._client._loop.create_task(waitTask.rerun())
        else:
            self._documentPromise = None
            self._contextPromise = self._client._loop.create_future()
            self._contextResolveCallback = (
                lambda _context: self._contextPromise.set_result(_context)
            )

    async def executionContext(self) -> Optional[ExecutionContext]:
        """Return execution context of this frame.

        Return :class:`~pyppeteer.execution_context.ExecutionContext`
        associated to this frame.
        """
        return await self._contextPromise

    async def evaluateHandle(self, pageFunction: str, *args: Any) -> JSHandle:
        """Execute function on this frame.

        Details see :meth:`pyppeteer.page.Page.evaluateHandle`.
        """
        context = await self.executionContext()
        if context is None:
            raise PageError('this frame has no context.')
        return await context.evaluateHandle(pageFunction, *args)

    async def evaluate(self, pageFunction: str, *args: Any,
                       force_expr: bool = False) -> Any:
        """Evaluate pageFunction on this frame.

        Details see :meth:`pyppeteer.page.Page.evaluate`.
        """
        context = await self.executionContext()
        if context is None:
            raise ElementHandleError('ExecutionContext is None.')
        return await context.evaluate(
            pageFunction, *args, force_expr=force_expr)

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

    async def querySelectorEval(self, selector: str, pageFunction: str,
                                *args: Any) -> Any:
        """Execute function on element which matches selector.

        Details see :meth:`pyppeteer.page.Page.querySelectorEval`.
        """
        document = await self._document()
        return await document.querySelectorEval(selector, pageFunction, *args)

    async def querySelectorAllEval(self, selector: str, pageFunction: str,
                                   *args: Any) -> Optional[Dict]:
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

    #: Alias to :meth:`querySelector`
    J = querySelector
    #: Alias to :meth:`xpath`
    Jx = xpath
    #: Alias to :meth:`querySelectorEval`
    Jeval = querySelectorEval
    #: Alias to :meth:`querySelectorAll`
    JJ = querySelectorAll
    #: Alias to :meth:`querySelectorAllEval`
    JJeval = querySelectorAllEval

    async def content(self) -> str:
        """Get the whole HTML contents of the page."""
        return await self.evaluate('''
() => {
  let retVal = '';
  if (document.doctype)
    retVal = new XMLSerializer().serializeToString(document.doctype);
  if (document.documentElement)
    retVal += document.documentElement.outerHTML;
  return retVal;
}
        '''.strip())

    async def setContent(self, html: str) -> None:
        """Set content to this page."""
        func = '''
function(html) {
  document.open();
  document.write(html);
  document.close();
}
'''
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
    def parentFrame(self) -> Optional['Frame']:
        """Get parent frame.

        If this frame is main frame or detached frame, return ``None``.
        """
        return self._parentFrame

    @property
    def childFrames(self) -> List['Frame']:
        """Get child frames."""
        return list(self._childFrames)

    def isDetached(self) -> bool:
        """Return ``True`` if this frame is detached.

        Otherwise return ``False``.
        """
        return self._detached

    async def injectFile(self, filePath: str) -> str:
        """[Deprecated] Inject file to the frame."""
        logger.warning('`injectFile` method is deprecated.'
                       ' Use `addScriptTag` method instead.')
        with open(filePath) as f:
            contents = f.read()
        contents += '/* # sourceURL= {} */'.format(filePath.replace('\n', ''))
        return await self.evaluate(contents)

    async def addScriptTag(self, options: Dict) -> ElementHandle:  # noqa: C901
        """Add script tag to this frame.

        Details see :meth:`pyppeteer.page.Page.addScriptTag`.
        """
        context = await self.executionContext()
        if context is None:
            raise ElementHandleError('ExecutionContext is None.')

        addScriptUrl = '''
        async function addScriptUrl(url, type) {
            const script = document.createElement('script');
            script.src = url;
            if (type)
                script.type = type;
            const promise = new Promise((res, rej) => {
                script.onload = res;
                script.onerror = rej;
            });
            document.head.appendChild(script);
            await promise;
            return script;
        }'''

        addScriptContent = '''
        function addScriptContent(content, type = 'text/javascript') {
            const script = document.createElement('script');
            script.type = type;
            script.text = content;
            let error = null;
            script.onerror = e => error = e;
            document.head.appendChild(script);
            if (error)
                throw error;
            return script;
        }'''

        if isinstance(options.get('url'), str):
            url = options['url']
            args = [addScriptUrl, url]
            if 'type' in options:
                args.append(options['type'])
            try:
                return (await context.evaluateHandle(*args)  # type: ignore
                        ).asElement()
            except ElementHandleError as e:
                raise PageError(f'Loading script from {url} failed') from e

        if isinstance(options.get('path'), str):
            with open(options['path']) as f:
                contents = f.read()
            contents = contents + '//# sourceURL={}'.format(
                options['path'].replace('\n', ''))
            args = [addScriptContent, contents]
            if 'type' in options:
                args.append(options['type'])
            return (await context.evaluateHandle(*args)  # type: ignore
                    ).asElement()

        if isinstance(options.get('content'), str):
            args = [addScriptContent, options['content']]
            if 'type' in options:
                args.append(options['type'])
            return (await context.evaluate