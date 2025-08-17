#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Page module."""

import asyncio
import base64
import json
import logging
import math
import mimetypes
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, Tuple
from typing import TYPE_CHECKING

from pyee import EventEmitter

from pyppeteer import helper
from pyppeteer.connection import CDPSession
from pyppeteer.coverage import Coverage
from pyppeteer.dialog import Dialog
from pyppeteer.element_handle import ElementHandle
from pyppeteer.emulation_manager import EmulationManager
from pyppeteer.errors import PageError
from pyppeteer.execution_context import JSHandle  # noqa: F401
from pyppeteer.frame_manager import Frame  # noqa: F401
from pyppeteer.frame_manager import FrameManager
from pyppeteer.helper import debugError
from pyppeteer.input import Keyboard, Mouse, Touchscreen
from pyppeteer.navigator_watcher import NavigatorWatcher
from pyppeteer.network_manager import NetworkManager, Response, Request
from pyppeteer.tracing import Tracing
from pyppeteer.util import merge_dict
from pyppeteer.worker import Worker

if TYPE_CHECKING:
    from pyppeteer.browser import Browser, Target  # noqa: F401

logger = logging.getLogger(__name__)


class Page(EventEmitter):
    """Page class."""

    Events: SimpleNamespace = SimpleNamespace(
        Close='close',
        Console='console',
        Dialog='dialog',
        DOMContentLoaded='domcontentloaded',
        Error='error',
        PageError='pageerror',
        Request='request',
        Response='response',
        RequestFailed='requestfailed',
        RequestFinished='requestfinished',
        FrameAttached='frameattached',
        FrameDetached='framedetached',
        FrameNavigated='framenavigated',
        Load='load',
        Metrics='metrics',
        WorkerCreated='workercreated',
        WorkerDestroyed='workerdestroyed',
    )

    PaperFormats: Dict[str, Dict[str, float]] = {
        'letter': {'width': 8.5, 'height': 11},
        'legal': {'width': 8.5, 'height': 14},
        'tabloid': {'width': 11, 'height': 17},
        'ledger': {'width': 17, 'height': 11},
        'a0': {'width': 33.1, 'height': 46.8},
        'a1': {'width': 23.4, 'height': 33.1},
        'a2': {'width': 16.5, 'height': 23.4},
        'a3': {'width': 11.7, 'height': 16.5},
        'a4': {'width': 8.27, 'height': 11.7},
        'a5': {'width': 5.83, 'height': 8.27},
    }

    @staticmethod
    async def create(client: CDPSession, target: 'Target',
                     ignoreHTTPSErrors: bool, defaultViewport: Optional[Dict[str, Any]],
                     screenshotTaskQueue: Optional[List[Any]] = None) -> 'Page':
        """Async function which makes new page object."""
        await client.send('Page.enable')
        frameTree = (await client.send('Page.getFrameTree'))['frameTree']
        page = Page(client, target, frameTree, ignoreHTTPSErrors,
                    screenshotTaskQueue)

        await asyncio.gather(
            client.send('Target.setAutoAttach', {'autoAttach': True, 'waitForDebuggerOnStart': False}),
            client.send('Page.setLifecycleEventsEnabled', {'enabled': True}),
            client.send('Network.enable', {}),
            client.send('Runtime.enable', {}),
            client.send('Security.enable', {}),
            client.send('Performance.enable', {}),
            client.send('Log.enable', {}),
        )
        if ignoreHTTPSErrors:
            await client.send('Security.setOverrideCertificateErrors',
                             {'override': True})
        if defaultViewport:
            await page.setViewport(defaultViewport)
        return page

    def __init__(self, client: CDPSession, target: 'Target',
                 frameTree: Dict[str, Any], ignoreHTTPSErrors: bool,
                 screenshotTaskQueue: Optional[List[Any]] = None) -> None:
        super().__init__()
        self._closed: bool = False
        self._client: CDPSession = client
        self._target: 'Target' = target
        self._keyboard: Keyboard = Keyboard(client)
        self._mouse: Mouse = Mouse(client, self._keyboard)
        self._touchscreen: Touchscreen = Touchscreen(client, self._keyboard)
        self._frameManager: FrameManager = FrameManager(client, frameTree, self)
        self._networkManager: NetworkManager = NetworkManager(client, self._frameManager)
        self._emulationManager: EmulationManager = EmulationManager(client)
        self._tracing: Tracing = Tracing(client)
        self._pageBindings: Dict[str, Callable[..., Any]] = {}
        self._ignoreHTTPSErrors: bool = ignoreHTTPSErrors
        self._defaultNavigationTimeout: int = 30000
        self._javascriptEnabled: bool = True
        self._coverage: Coverage = Coverage(client)
        self._viewport: Optional[Dict[str, Any]] = None

        if screenshotTaskQueue is None:
            screenshotTaskQueue = []
        self._screenshotTaskQueue: List[Any] = screenshotTaskQueue

        self._workers: Dict[str, Worker] = {}

        def _onTargetAttached(event: Dict[str, Any]) -> None:
            targetInfo = event['targetInfo']
            if targetInfo['type'] != 'worker':
                try:
                    client.send('Target.detachFromTarget', {
                        'sessionId': event['sessionId'],
                    })
                except Exception as e:
                    debugError(logger, e)
                return
            sessionId = event['sessionId']
            session = client._createSession(targetInfo['type'], sessionId)
            worker = Worker(
                session,
                targetInfo['url'],
                self._addConsoleMessage,
                self._handleException,
            )
            self._workers[sessionId] = worker
            self.emit(Page.Events.WorkerCreated, worker)

        def _onTargetDetached(event: Dict[str, Any]) -> None:
            sessionId = event['sessionId']
            worker = self._workers.get(sessionId)
            if worker is None:
                return
            self.emit(Page.Events.WorkerDestroyed, worker)
            del self._workers[sessionId]

        client.on('Target.attachedToTarget', _onTargetAttached)
        client.on('Target.detachedFromTarget', _onTargetDetached)

        _fm = self._frameManager
        _fm.on(FrameManager.Events.FrameAttached,
               lambda event: self.emit(Page.Events.FrameAttached, event))
        _fm.on(FrameManager.Events.FrameDetached,
               lambda event: self.emit(Page.Events.FrameDetached, event))
        _fm.on(FrameManager.Events.FrameNavigated,
               lambda event: self.emit(Page.Events.FrameNavigated, event))

        _nm = self._networkManager
        _nm.on(NetworkManager.Events.Request,
               lambda event: self.emit(Page.Events.Request, event))
        _nm.on(NetworkManager.Events.Response,
               lambda event: self.emit(Page.Events.Response, event))
        _nm.on(NetworkManager.Events.RequestFailed,
               lambda event: self.emit(Page.Events.RequestFailed, event))
        _nm.on(NetworkManager.Events.RequestFinished,
               lambda event: self.emit(Page.Events.RequestFinished, event))

        client.on('Page.domContentEventFired',
                  lambda event: self.emit(Page.Events.DOMContentLoaded))
        client.on('Page.loadEventFired',
                  lambda event: self.emit(Page.Events.Load))
        client.on('Runtime.consoleAPICalled',
                  lambda event: self._onConsoleAPI(event))
        client.on('Runtime.bindingCalled',
                  lambda event: self._onBindingCalled(event))
        client.on('Page.javascriptDialogOpening',
                  lambda event: self._onDialog(event))
        client.on('Runtime.exceptionThrown',
                  lambda exception: self._handleException(
                      exception.get('exceptionDetails')))
        client.on('Security.certificateError',
                  lambda event: self._onCertificateError(event))
        client.on('Inspector.targetCrashed',
                  lambda event: self._onTargetCrashed())
        client.on('Performance.metrics',
                  lambda event: self._emitMetrics(event))
        client.on('Log.entryAdded',
                  lambda event: self._onLogEntryAdded(event))

        def closed(fut: asyncio.Future) -> None:
            self.emit(Page.Events.Close)
            self._closed = True

        self._target._isClosedPromise.add_done_callback(closed)

    @property
    def target(self) -> 'Target':
        """Return a target this page created from."""
        return self._target

    @property
    def browser(self) -> 'Browser':
        """Get the browser the page belongs to."""
        return self._target.browser

    def _onTargetCrashed(self, *args: Any, **kwargs: Any) -> None:
        self.emit('error', PageError('Page crashed!'))

    def _onLogEntryAdded(self, event: Dict[str, Any]) -> None:
        entry = event.get('entry', {})
        level = entry.get('level', '')
        text = entry.get('text', '')
        args = entry.get('args', [])
        source = entry.get('source', '')
        for arg in args:
            helper.releaseObject(self._client, arg)

        if source != 'worker':
            self.emit(Page.Events.Console, ConsoleMessage(level, text))

    @property
    def mainFrame(self) -> Optional['Frame']:
        """Get main frame of this page."""
        return self._frameManager._mainFrame

    @property
    def keyboard(self) -> Keyboard:
        """Get Keyboard object."""
        return self._keyboard

    @property
    def touchscreen(self) -> Touchscreen:
        """Get Touchscreen object."""
        return self._touchscreen

    @property
    def coverage(self) -> Coverage:
        """Return Coverage."""
        return self._coverage

    async def tap(self, selector: str) -> None:
        """Tap the element which matches the selector."""
        frame = self.mainFrame
        if frame is None:
            raise PageError('no main frame')
        await frame.tap(selector)

    @property
    def tracing(self) -> Tracing:
        """Get tracing object."""
        return self._tracing

    @property
    def frames(self) -> List['Frame']:
        """Get all frames of this page."""
        return list(self._frameManager.frames())

    @property
    def workers(self) -> List[Worker]:
        """Get all workers of this page."""
        return list(self._workers.values())

    async def setRequestInterception(self, value: bool) -> None:
        """Enable/disable request interception."""
        return await self._networkManager.setRequestInterception(value)

    async def setOfflineMode(self, enabled: bool) -> None:
        """Set offline mode enable/disable."""
        await self._networkManager.setOfflineMode(enabled)

    def setDefaultNavigationTimeout(self, timeout: int) -> None:
        """Change the default maximum navigation timeout."""
        self._defaultNavigationTimeout = timeout

    async def _send(self, method: str, msg: Dict[str, Any]) -> None:
        try:
            await self._client.send(method, msg)
        except Exception as e:
            debugError(logger, e)

    def _onCertificateError(self, event: Any) -> None:
        if not self._ignoreHTTPSErrors:
            return
        self._client._loop.create_task(
            self._send('Security.handleCertificateError', {
                'eventId': event.get('eventId'),
                'action': 'continue'
            })
        )

    async def querySelector(self, selector: str) -> Optional[ElementHandle]:
        """Get an Element which matches selector."""
        frame = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelector(selector)

    async def evaluateHandle(self, pageFunction: str, *args: Any
                            ) -> JSHandle:
        """Execute function on this page."""
        if not self.mainFrame:
            raise PageError('no main frame.')
        context = await self.mainFrame.executionContext()
        if not context:
            raise PageError('No context.')
        return await context.evaluateHandle(pageFunction, *args)

    async def queryObjects(self, prototypeHandle: JSHandle) -> JSHandle:
        """Iterate js heap and finds all the objects with the handle."""
        if not self.mainFrame:
            raise PageError('no main frame.')
        context = await self.mainFrame.executionContext()
        if not context:
            raise PageError('No context.')
        return await context.queryObjects(prototypeHandle)

    async def querySelectorEval(self, selector: str, pageFunction: str,
                               *args: Any) -> Any:
        """Execute function with an element which matches selector."""
        frame = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelectorEval(selector, pageFunction, *args)

    async def querySelectorAllEval(self, selector: str, pageFunction: str,
                                  *args: Any) -> Any:
        """Execute function with all elements which matches selector."""
        frame = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelectorAllEval(selector, pageFunction, *args)

    async def querySelectorAll(self, selector: str) -> List[ElementHandle]:
        """Get all element which matches selector as a list."""
        frame = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelectorAll(selector)

    async def xpath(self, expression: str) -> List[ElementHandle]:
        """Evaluate the XPath expression."""
        frame = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.xpath(expression)

    J = querySelector
    Jeval = querySelectorEval
    JJ = querySelectorAll
    JJeval = querySelectorAllEval
    Jx = xpath

    async def cookies(self, *urls: str) -> Dict[str, Any]:
        """Get cookies."""
        if not urls:
            urls = (self.url,)
        resp = await self._client.send('Network.getCookies', {
            'urls': urls,
        })
        return resp.get('cookies', {})

    async def deleteCookie(self, *cookies: Dict[str, Any]) -> None:
        """Delete cookie."""
        pageURL = self.url
        for cookie in cookies:
            item = dict(**cookie)
            if not cookie.get('url') and pageURL.startswith('http'):
                item['url'] = pageURL
            await self._client.send('Network.deleteCookies', item)

    async def setCookie(self, *cookies: Dict[str, Any]) -> None:
        """Set cookies."""
        pageURL = self.url
        startsWithHTTP = pageURL.startswith('http')
        items = []
        for cookie in cookies:
            item = dict(**cookie)
            if 'url' not in item and startsWithHTTP:
                item['url'] = pageURL
            if item.get('url') == 'about:blank':
                name = item.get('name', '')
                raise PageError(f'Blank page can not have cookie "{name}"')
            if item.get('url', '').startswith('data:'):
                name = item.get('name', '')
                raise PageError(f'Data URL page can not have cookie "{name}"')
            items.append(item)
        await self.deleteCookie(*items)
        if items:
            await self._client.send('Network.setCookies', {
                'cookies': items,
            })

    async def addScriptTag(self, options: Optional[Dict[str, Any]] = None, **kwargs: str
                          ) -> ElementHandle:
        """Add script tag to this page."""
        frame = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        options = merge_dict(options, kwargs)
        return await frame.addScriptTag(options)

    async def addStyleTag(self, options: Optional[Dict[str, Any]] = None, **kwargs: str
                         ) -> ElementHandle:
        """Add style or link tag to this page."""
        frame = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        options = merge_dict(options, kwargs)
        return await frame.addStyleTag(options)

    async def injectFile(self, filePath: str) -> str:
        """[Deprecated] Inject file to this page."""
        frame = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.injectFile(filePath)

    async def exposeFunction(self, name: str,
                            pyppeteerFunction: Callable[..., Any]
                            ) -> None:
        """Add python function to the browser's window object as name."""
        if self._pageBindings.get(name):
            raise PageError(f'Failed to add page binding with name {name}: '
                           f'window["{name}"] already exists!')
        self._pageBindings[name] = pyppeteerFunction

        addPageBinding = '''
function addPageBinding(bindingName) {
  const binding = window[bindingName];
  window[bindingName] = async(...args) => {
    const me = window[bindingName];
    let callbacks = me['callbacks'];
    if (!callbacks) {
      callbacks = new Map();
      me['callbacks'] = callbacks;
    }
    const seq = (me['lastSeq'] || 0) + 1;
    me['lastSeq'] = seq;
    const promise = new Promise(fulfill => callbacks.set(seq, fulfill));
    binding(JSON.stringify({name: bindingName, seq, args}));
    return promise;
  };
}
        '''
        expression = helper.evaluationString(addPage