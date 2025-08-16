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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
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
    """Page class.

    This class provides methods to interact with a single tab of chrome. One
    :class:`~pyppeteer.browser.Browser` object might have multiple Page object.

    The :class:`Page` class emits various :attr:`~Page.Events` which can be
    handled by using ``on`` or ``once`` method, which is inherited from
    `pyee <https://pyee.readthedocs.io/en/latest/>`_'s ``EventEmitter`` class.
    """

    #: Available events.
    Events = SimpleNamespace(
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

    PaperFormats: Dict[str, Dict[str, float]] = dict(
        letter={'width': 8.5, 'height': 11},
        legal={'width': 8.5, 'height': 14},
        tabloid={'width': 11, 'height': 17},
        ledger={'width': 17, 'height': 11},
        a0={'width': 33.1, 'height': 46.8},
        a1={'width': 23.4, 'height': 33.1},
        a2={'width': 16.5, 'height': 23.4},
        a3={'width': 11.7, 'height': 16.5},
        a4={'width': 8.27, 'height': 11.7},
        a5={'width': 5.83, 'height': 8.27},
    )

    @staticmethod
    async def create(client: CDPSession, target: 'Target',
                     ignoreHTTPSErrors: bool, defaultViewport: Optional[Dict[str, Any]],
                     screenshotTaskQueue: Optional[List[Any]] = None) -> 'Page':
        """Async function which makes new page object."""
        await client.send('Page.enable')
        frameTree: Dict[str, Any] = (await client.send('Page.getFrameTree'))['frameTree']
        page: Page = Page(client, target, frameTree, ignoreHTTPSErrors, screenshotTaskQueue)

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
            await client.send('Security.setOverrideCertificateErrors', {'override': True})
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
        self._pageBindings: Dict[str, Callable[..., Any]] = dict()
        self._ignoreHTTPSErrors: bool = ignoreHTTPSErrors
        self._defaultNavigationTimeout: int = 30000  # milliseconds
        self._javascriptEnabled: bool = True
        self._coverage: Coverage = Coverage(client)
        self._viewport: Optional[Dict[str, Any]] = None

        if screenshotTaskQueue is None:
            screenshotTaskQueue = list()
        self._screenshotTaskQueue: List[Any] = screenshotTaskQueue

        self._workers: Dict[str, Worker] = dict()

        def _onTargetAttached(event: Dict[str, Any]) -> None:
            targetInfo: Dict[str, Any] = event['targetInfo']
            if targetInfo['type'] != 'worker':
                try:
                    client.send('Target.detachFromTarget', {
                        'sessionId': event['sessionId'],
                    })
                except Exception as e:
                    debugError(logger, e)
                return
            sessionId: str = event['sessionId']
            session = client._createSession(targetInfo['type'], sessionId)
            worker: Worker = Worker(
                session,
                targetInfo['url'],
                self._addConsoleMessage,
                self._handleException,
            )
            self._workers[sessionId] = worker
            self.emit(Page.Events.WorkerCreated, worker)

        def _onTargetDetached(event: Dict[str, Any]) -> None:
            sessionId: str = event['sessionId']
            worker = self._workers.get(sessionId)
            if worker is None:
                return
            self.emit(Page.Events.WorkerDestroyed, worker)
            del self._workers[sessionId]

        client.on('Target.attachedToTarget', _onTargetAttached)
        client.on('Target.detachedFromTarget', _onTargetDetached)

        _fm: FrameManager = self._frameManager
        _fm.on(FrameManager.Events.FrameAttached,
               lambda event: self.emit(Page.Events.FrameAttached, event))
        _fm.on(FrameManager.Events.FrameDetached,
               lambda event: self.emit(Page.Events.FrameDetached, event))
        _fm.on(FrameManager.Events.FrameNavigated,
               lambda event: self.emit(Page.Events.FrameNavigated, event))

        _nm: NetworkManager = self._networkManager
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
                  lambda exception: self._handleException(exception.get('exceptionDetails')))
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
        entry: Dict[str, Any] = event.get('entry', {})
        level: str = entry.get('level', '')
        text: str = entry.get('text', '')
        args_list: List[Any] = entry.get('args', [])
        source: str = entry.get('source', '')
        for arg in args_list:
            helper.releaseObject(self._client, arg)

        if source != 'worker':
            self.emit(Page.Events.Console, ConsoleMessage(level, text))

    @property
    def mainFrame(self) -> Optional['Frame']:
        """Get main :class:`~pyppeteer.frame_manager.Frame` of this page."""
        return self._frameManager._mainFrame

    @property
    def keyboard(self) -> Keyboard:
        """Get :class:`~pyppeteer.input.Keyboard` object."""
        return self._keyboard

    @property
    def touchscreen(self) -> Touchscreen:
        """Get :class:`~pyppeteer.input.Touchscreen` object."""
        return self._touchscreen

    @property
    def coverage(self) -> Coverage:
        """Return :class:`~pyppeteer.coverage.Coverage`."""
        return self._coverage

    async def tap(self, selector: str) -> None:
        """Tap the element which matches the ``selector``.

        :arg str selector: A selector to search element to touch.
        """
        frame: Optional['Frame'] = self.mainFrame
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
        """Enable/disable request interception.

        Activating request interception enables
        :class:`~pyppeteer.network_manager.Request` class's
        :meth:`~pyppeteer.network_manager.Request.abort`,
        :meth:`~pyppeteer.network_manager.Request.continue_`, and
        :meth:`~pyppeteer.network_manager.Request.response` methods.
        This provides the capability to modify network requests that are made
        by a page.

        Once request interception is enabled, every request will stall unless
        it's continued, responded or aborted.
        """
        await self._networkManager.setRequestInterception(value)

    async def setOfflineMode(self, enabled: bool) -> None:
        """Set offline mode enable/disable."""
        await self._networkManager.setOfflineMode(enabled)

    def setDefaultNavigationTimeout(self, timeout: int) -> None:
        """Change the default maximum navigation timeout.

        This method changes the default timeout of 30 seconds for the following
        methods:

        * :meth:`goto`
        * :meth:`goBack`
        * :meth:`goForward`
        * :meth:`reload`
        * :meth:`waitForNavigation`

        :arg int timeout: Maximum navigation time in milliseconds. Pass ``0``
                          to disable timeout.
        """
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
        """Get an Element which matches ``selector``.

        :arg str selector: A selector to search element.
        :return Optional[ElementHandle]: If element which matches the
            ``selector`` is found, return its
            :class:`~pyppeteer.element_handle.ElementHandle`. If not found,
            returns ``None``.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelector(selector)

    async def evaluateHandle(self, pageFunction: str, *args: Any
                             ) -> JSHandle:
        """Execute function on this page.

        Difference between :meth:`~pyppeteer.page.Page.evaluate` and
        :meth:`~pyppeteer.page.Page.evaluateHandle` is that
        ``evaluateHandle`` returns JSHandle object (not value).

        :arg str pageFunction: JavaScript function to be executed.
        """
        if not self.mainFrame:
            raise PageError('no main frame.')
        context = await self.mainFrame.executionContext()
        if not context:
            raise PageError('No context.')
        return await context.evaluateHandle(pageFunction, *args)

    async def queryObjects(self, prototypeHandle: JSHandle) -> JSHandle:
        """Iterate js heap and finds all the objects with the handle.

        :arg JSHandle prototypeHandle: JSHandle of prototype object.
        """
        if not self.mainFrame:
            raise PageError('no main frame.')
        context = await self.mainFrame.executionContext()
        if not context:
            raise PageError('No context.')
        return await context.queryObjects(prototypeHandle)

    async def querySelectorEval(self, selector: str, pageFunction: str,
                                *args: Any) -> Any:
        """Execute function with an element which matches ``selector``.

        :arg str selector: A selector to query page for.
        :arg str pageFunction: String of JavaScript function to be evaluated on
                               browser. This function takes an element which
                               matches the selector as a first argument.
        :arg Any args: Arguments to pass to ``pageFunction``.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelectorEval(selector, pageFunction, *args)

    async def querySelectorAllEval(self, selector: str, pageFunction: str,
                                   *args: Any) -> Any:
        """Execute function with all elements which matches ``selector``.

        :arg str selector: A selector to query page for.
        :arg str pageFunction: String of JavaScript function to be evaluated on
                               browser. This function takes Array of the
                               matched elements as the first argument.
        :arg Any args: Arguments to pass to ``pageFunction``.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelectorAllEval(selector, pageFunction, *args)

    async def querySelectorAll(self, selector: str) -> List[ElementHandle]:
        """Get all element which matches ``selector`` as a list.

        :arg str selector: A selector to search element.
        :return List[ElementHandle]: List of
            :class:`~pyppeteer.element_handle.ElementHandle` which matches the
            ``selector``. If no element is matched to the ``selector``, return
            empty list.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelectorAll(selector)

    async def xpath(self, expression: str) -> List[ElementHandle]:
        """Evaluate the XPath expression.

        :arg str expression: XPath string to be evaluated.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.xpath(expression)

    #: alias to :meth:`querySelector`
    J = querySelector
    #: alias to :meth:`querySelectorEval`
    Jeval = querySelectorEval
    #: alias to :meth:`querySelectorAll`
    JJ = querySelectorAll
    #: alias to :meth:`querySelectorAllEval`
    JJeval = querySelectorAllEval
    #: alias to :meth:`xpath`
    Jx = xpath

    async def cookies(self, *urls: str) -> dict:
        """Get cookies.

        If no URLs are specified, this method returns cookies for the current
        page URL. If URLs are specified, only cookies for those URLs are
        returned.
        """
        if not urls:
            urls = (self.url, )
        resp: Dict[str, Any] = await self._client.send('Network.getCookies', {
            'urls': urls,
        })
        return resp.get('cookies', {})

    async def deleteCookie(self, *cookies: dict) -> None:
        """Delete cookie.

        ``cookies`` should be dictionaries which contain these fields.
        """
        pageURL: str = self.url
        for cookie in cookies:
            item: Dict[str, Any] = dict(**cookie)
            if not cookie.get('url') and pageURL.startswith('http'):
                item['url'] = pageURL
            await self._client.send('Network.deleteCookies', item)

    async def setCookie(self, *cookies: dict) -> None:
        """Set cookies.

        ``cookies`` should be dictionaries which contain these fields.
        """
        pageURL: str = self.url
        startsWithHTTP: bool = pageURL.startswith('http')
        items: List[Dict[str, Any]] = []
        for cookie in cookies:
            item: Dict[str, Any] = dict(**cookie)
            if 'url' not in item and startsWithHTTP:
                item['url'] = pageURL
            if item.get('url') == 'about:blank':
                name: str = item.get('name', '')
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

    async def addScriptTag(self, options: Optional[Dict[str, Any]] = None, **kwargs: str) -> ElementHandle:
        """Add script tag to this page.

        One of ``url``, ``path`` or ``content`` option is necessary.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        options = merge_dict(options, kwargs)
        return await frame.addScriptTag(options)

    async def addStyleTag(self, options: Optional[Dict[str, Any]] = None, **kwargs: str) -> ElementHandle:
        """Add style or link tag to this page.

        One of ``url``, ``path`` or ``content`` option is necessary.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        options = merge_dict(options, kwargs)
        return await frame.addStyleTag(options)

    async def injectFile(self, filePath: str) -> str:
        """[Deprecated] Inject file to this page.

        This method is deprecated. Use :meth:`addScriptTag` instead.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.injectFile(filePath)

    async def exposeFunction(self, name: str,
                             pyppeteerFunction: Callable[..., Any]
                             ) -> None:
        """Add python function to the browser's ``window`` object as ``name``.
        """
        if self._pageBindings.get(name):
            raise PageError(f'Failed to add page binding with name {name}: '
                            f'window["{name}"] already exists!')
        self._pageBindings[name] = pyppeteerFunction

        addPageBinding: str = '''
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
        expression: str = helper.evaluationString(addPageBinding, name)
        await self._client.send('Runtime.addBinding', {'name': name})
        await self._client.send('Page.addScriptToEvaluateOnNewDocument',
                                {'source': expression})

        async def _evaluate(frame: 'Frame', expression: str) -> None:
            try:
                await frame.evaluate(expression, force_expr=True)
            except Exception as e:
                debugError(logger, e)

        await asyncio.wait([_evaluate(frame, expression) for frame in self.frames])

    async def authenticate(self, credentials: Dict[str, str]) -> Any:
        """Provide credentials for http authentication.

        ``credentials`` should be ``None`` or dict which has ``username`` and
        ``password`` field.
        """
        return await self._networkManager.authenticate(credentials)

    async def setExtraHTTPHeaders(self, headers: Dict[str, str]) -> None:
        """Set extra HTTP headers.

        The extra HTTP headers will be sent with every request the page
        initiates.
        """
        await self._networkManager.setExtraHTTPHeaders(headers)

    async def setUserAgent(self, userAgent: str) -> None:
        """Set user agent to use in this page.
        """
        await self._networkManager.setUserAgent(userAgent)

    async def metrics(self) -> Dict[str, Any]:
        """Get metrics.
        """
        response: Dict[str, Any] = await self._client.send('Performance.getMetrics')
        return self._buildMetricsObject(response['metrics'])

    def _emitMetrics(self, event: Dict[str, Any]) -> None:
        self.emit(Page.Events.Metrics, {
            'title': event['title'],
            'metrics': self._buildMetricsObject(event['metrics']),
        })

    def _buildMetricsObject(self, metrics: List[Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for metric in metrics or []:
            if metric['name'] in supportedMetrics:
                result[metric['name']] = metric['value']
        return result

    def _handleException(self, exceptionDetails: Dict[str, Any]) -> None:
        message: str = helper.getExceptionMessage(exceptionDetails)
        self.emit(Page.Events.PageError, PageError(message))

    def _onConsoleAPI(self, event: Dict[str, Any]) -> None:
        _id: int = event['executionContextId']
        context = self._frameManager.executionContextById(_id)
        values: List[JSHandle] = []
        for arg in event.get('args', []):
            values.append(self._frameManager.createJSHandle(context, arg))
        self._addConsoleMessage(event['type'], values)

    def _onBindingCalled(self, event: Dict[str, Any]) -> None:
        obj: Dict[str, Any] = json.loads(event['payload'])
        name: str = obj['name']
        seq: int = obj['seq']
        args_list: List[Any] = obj['args']
        result: Any = self._pageBindings[name](*args_list)

        deliverResult: str = '''
            function deliverResult(name, seq, result) {
                window[name]['callbacks'].get(seq)(result);
                window[name]['callbacks'].delete(seq);
            }
        '''

        expression: str = helper.evaluationString(deliverResult, name, seq, result)
        try:
            self._client.send('Runtime.evaluate', {
                'expression': expression,
                'contextId': event['executionContextId'],
            })
        except Exception as e:
            helper.debugError(logger, e)

    def _addConsoleMessage(self, type: str, args: List[JSHandle]) -> None:
        if not self.listeners(Page.Events.Console):
            for arg in args:
                self._client._loop.create_task(arg.dispose())
            return

        textTokens: List[str] = []
        for arg in args:
            remoteObject: Dict[str, Any] = arg._remoteObject
            if remoteObject.get('objectId'):
                textTokens.append(arg.toString())
            else:
                textTokens.append(str(helper.valueFromRemoteObject(remoteObject)))

        message: ConsoleMessage = ConsoleMessage(type, ' '.join(textTokens), args)
        self.emit(Page.Events.Console, message)

    def _onDialog(self, event: Dict[str, Any]) -> None:
        dialogType: str = ''
        _type: str = event.get('type')
        if _type == 'alert':
            dialogType = Dialog.Type.Alert
        elif _type == 'confirm':
            dialogType = Dialog.Type.Confirm
        elif _type == 'prompt':
            dialogType = Dialog.Type.Prompt
        elif _type == 'beforeunload':
            dialogType = Dialog.Type.BeforeUnload
        dialog: Dialog = Dialog(self._client, dialogType, event.get('message'),
                        event.get('defaultPrompt'))
        self.emit(Page.Events.Dialog, dialog)

    @property
    def url(self) -> str:
        """Get URL of this page."""
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.url

    async def content(self) -> str:
        """Get the full HTML contents of the page.

        Returns HTML including the doctype.
        """
        frame: Optional['Frame'] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        return await frame.content()

    async def setContent(self, html: str) -> None:
        """Set content to this page.

        :arg str html: HTML markup to assign to the page.
        """
        frame: Optional['Frame'] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        await frame.setContent(html)

    async def goto(self, url: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Optional[Response]:
        """Go to the ``url``.

        :arg string url: URL to navigate page to.
        """
        options = merge_dict(options, kwargs)
        mainFrame: Optional['Frame'] = self._frameManager.mainFrame
        if mainFrame is None:
            raise PageError('No main frame.')

        referrer: str = self._networkManager.extraHTTPHeaders().get('referer', '')
        requests: Dict[str, Request] = {}

        def set_request(req: Request) -> None:
            if req.url not in requests:
                requests[req.url] = req

        eventListeners: List[Any] = [helper.addEventListener(
            self._networkManager,
            NetworkManager.Events.Request,
            set_request,
        )]

        timeout: int = options.get('timeout', self._defaultNavigationTimeout)
        watcher: NavigatorWatcher = NavigatorWatcher(self._frameManager, mainFrame, timeout, options)

        result: Optional[str] = await self._navigate(url, referrer)
        if result is not None:
            raise PageError(result)
        result = await watcher.navigationPromise()
        watcher.cancel()
        helper.removeEventListeners(eventListeners)
        error = result[0].pop().exception()  # type: ignore
        if error:
            raise error

        request: Optional[Request] = requests.get(mainFrame._navigationURL)
        return request.response if request else None

    async def _navigate(self, url: str, referrer: str) -> Optional[str]:
        response: Dict[str, Any] = await self._client.send(
            'Page.navigate', {'url': url, 'referrer': referrer})
        if response.get('errorText'):
            return f'{response["errorText"]} at {url}'
        return None

    async def reload(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Optional[Response]:
        """Reload this page.
        """
        options = merge_dict(options, kwargs)
        response: Optional[Response] = (await asyncio.gather(
            self.waitForNavigation(options),
            self._client.send('Page.reload'),
        ))[0]
        return response

    async def waitForNavigation(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Optional[Response]:
        """Wait for navigation.
        """
        options = merge_dict(options, kwargs)
        mainFrame: Optional['Frame'] = self._frameManager.mainFrame
        if mainFrame is None:
            raise PageError('No main frame.')
        timeout: int = options.get('timeout', self._defaultNavigationTimeout)
        watcher: NavigatorWatcher = NavigatorWatcher(self._frameManager, mainFrame, timeout, options)
        responses: Dict[str, Response] = {}
        listener = helper.addEventListener(
            self._networkManager,
            NetworkManager.Events.Response,
            lambda response: responses.__setitem__(response.url, response)
        )
        result = await watcher.navigationPromise()
        helper.removeEventListeners([listener])
        error = result[0].pop().exception()
        if error:
            raise error

        response: Optional[Response] = responses.get(self.url, None)
        return response

    async def waitForRequest(self, urlOrPredicate: Union[str, Callable[[Request], bool]],
                             options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Request:
        """Wait for request.
        """
        options = merge_dict(options, kwargs)
        timeout: Union[int, float] = options.get('timeout', 30000)

        def predicate(request: Request) -> bool:
            if isinstance(urlOrPredicate, str):
                return urlOrPredicate == request.url
            if callable(urlOrPredicate):
                return bool(urlOrPredicate(request))
            return False

        return await helper.waitForEvent(
            self._networkManager,
            NetworkManager.Events.Request,
            predicate,
            timeout,
            self._client._loop,
        )

    async def waitForResponse(self, urlOrPredicate: Union[str, Callable[[Response], bool]],
                              options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Response:
        """Wait for response.
        """
        options = merge_dict(options, kwargs)
        timeout: Union[int, float] = options.get('timeout', 30000)

        def predicate(response: Response) -> bool:
            if isinstance(urlOrPredicate, str):
                return urlOrPredicate == response.url
            if callable(urlOrPredicate):
                return bool(urlOrPredicate(response))
            return False

        return await helper.waitForEvent(
            self._networkManager,
            NetworkManager.Events.Response,
            predicate,
            timeout,
            self._client._loop,
        )

    async def goBack(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Optional[Response]:
        """Navigate to the previous page in history.
        """
        options = merge_dict(options, kwargs)
        return await self._go(-1, options)

    async def goForward(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Optional[Response]:
        """Navigate to the next page in history.
        """
        options = merge_dict(options, kwargs)
        return await self._go(+1, options)

    async def _go(self, delta: int, options: Dict[str, Any]) -> Optional[Response]:
        history: Dict[str, Any] = await self._client.send('Page.getNavigationHistory')
        _count: int = history.get('currentIndex', 0) + delta
        entries: List[Any] = history.get('entries', [])
        if len(entries) <= _count:
            return None
        entry: Dict[str, Any] = entries[_count]
        response: Optional[Response] = (await asyncio.gather(
            self.waitForNavigation(options),
            self._client.send('Page.navigateToHistoryEntry', {
                'entryId': entry.get('id')
            })
        ))[0]
        return response

    async def bringToFront(self) -> None:
        """Bring page to front (activate tab)."""
        await self._client.send('Page.bringToFront')

    async def emulate(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Emulate given device metrics and user agent.
        """
        options = merge_dict(options, kwargs)
        await self.setViewport(options.get('viewport', {}))
        await self.setUserAgent(options.get('userAgent', ''))

    async def setJavaScriptEnabled(self, enabled: bool) -> None:
        """Set JavaScript enable/disable."""
        if self._javascriptEnabled == enabled:
            return
        self._javascriptEnabled = enabled
        await self._client.send('Emulation.setScriptExecutionDisabled', {
            'value': not enabled,
        })

    async def setBypassCSP(self, enabled: bool) -> None:
        """Toggles bypassing page's Content-Security-Policy.
        """
        await self._client.send('Page.setBypassCSP', {'enabled': enabled})

    async def emulateMedia(self, mediaType: Optional[str] = None) -> None:
        """Emulate css media type of the page.
        """
        if mediaType not in ['screen', 'print', None, '']:
            raise ValueError(f'Unsupported media type: {mediaType}')
        await self._client.send('Emulation.setEmulatedMedia', {
            'media': mediaType or '',
        })

    async def setViewport(self, viewport: Dict[str, Any]) -> None:
        """Set viewport.
        """
        needsReload: bool = await self._emulationManager.emulateViewport(viewport)
        self._viewport = viewport
        if needsReload:
            await self.reload()

    @property
    def viewport(self) -> Optional[Dict[str, Any]]:
        """Get viewport as a dictionary or None.
        """
        return self._viewport

    async def evaluate(self, pageFunction: str, *args: Any,
                       force_expr: bool = False) -> Any:
        """Execute js-function or js-expression on browser and get result.
        """
        frame: Optional['Frame'] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        return await frame.evaluate(pageFunction, *args, force_expr=force_expr)

    async def evaluateOnNewDocument(self, pageFunction: str, *args: str) -> None:
        """Add a JavaScript function to the document.
        """
        source: str = helper.evaluationString(pageFunction, *args)
        await self._client.send('Page.addScriptToEvaluateOnNewDocument', {
            'source': source,
        })

    async def setCacheEnabled(self, enabled: bool = True) -> None:
        """Enable/Disable cache for each request.
        """
        await self._client.send('Network.setCacheDisabled',
                                {'cacheDisabled': not enabled})

    async def screenshot(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Union[bytes, str]:
        """Take a screen shot.
        """
        options = merge_dict(options, kwargs)
        screenshotType: Optional[str] = None
        if 'type' in options:
            screenshotType = options['type']
            if screenshotType not in ['png', 'jpeg']:
                raise ValueError(f'Unknown type value: {screenshotType}')
        elif 'path' in options:
            mimeType, _ = mimetypes.guess_type(options['path'])
            if mimeType == 'image/png':
                screenshotType = 'png'
            elif mimeType == 'image/jpeg':
                screenshotType = 'jpeg'
            else:
                raise ValueError('Unsupported screenshot mime type: ' + str(mimeType))
        if not screenshotType:
            screenshotType = 'png'
        return await self._screenshotTask(screenshotType, options)

    async def _screenshotTask(self, format: str, options: Dict[str, Any]) -> Union[bytes, str]:
        await self._client.send('Target.activateTarget', {
            'targetId': self._target._targetId,
        })
        clip: Optional[Dict[str, Any]] = options.get('clip')
        if clip:
            clip['scale'] = 1

        if options.get('fullPage'):
            metrics: Dict[str, Any] = await self._client.send('Page.getLayoutMetrics')
            width: int = math.ceil(metrics['contentSize']['width'])
            height: int = math.ceil(metrics['contentSize']['height'])

            clip = dict(x=0, y=0, width=width, height=height, scale=1)
            if self._viewport is not None:
                mobile: bool = self._viewport.get('isMobile', False)
                deviceScaleFactor: float = self._viewport.get('deviceScaleFactor', 1)
                landscape: bool = self._viewport.get('isLandscape', False)
            else:
                mobile = False
                deviceScaleFactor = 1
                landscape = False

            if landscape:
                screenOrientation: Dict[str, Any] = dict(angle=90, type='landscapePrimary')
            else:
                screenOrientation = dict(angle=0, type='portraitPrimary')
            await self._client.send('Emulation.setDeviceMetricsOverride', {
                'mobile': mobile,
                'width': width,
                'height': height,
                'deviceScaleFactor': deviceScaleFactor,
                'screenOrientation': screenOrientation,
            })

        if options.get('omitBackground'):
            await self._client.send(
                'Emulation.setDefaultBackgroundColorOverride',
                {'color': {'r': 0, 'g': 0, 'b': 0, 'a': 0}},
            )
        opt: Dict[str, Any] = {'format': format}
        if clip:
            opt['clip'] = clip
        result: Dict[str, Any] = await self._client.send('Page.captureScreenshot', opt)

        if options.get('omitBackground'):
            await self._client.send(
                'Emulation.setDefaultBackgroundColorOverride')

        if options.get('fullPage') and self._viewport is not None:
            await self.setViewport(self._viewport)

        if options.get('encoding') == 'base64':
            buffer: bytes = result.get('data', b'')
        else:
            buffer = base64.b64decode(result.get('data', b''))
        _path: Optional[str] = options.get('path')
        if _path:
            with open(_path, 'wb') as f:
                f.write(buffer)
        return buffer

    async def pdf(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bytes:
        """Generate a pdf of the page.
        """
        options = merge_dict(options, kwargs)
        scale: float = options.get('scale', 1)
        displayHeaderFooter: bool = bool(options.get('displayHeaderFooter'))
        headerTemplate: str = options.get('headerTemplate', '')
        footerTemplate: str = options.get('footerTemplate', '')
        printBackground: bool = bool(options.get('printBackground'))
        landscape: bool = bool(options.get('landscape'))
        pageRanges: str = options.get('pageRanges', '')

        paperWidth: float = 8.5
        paperHeight: float = 11.0
        if 'format' in options:
            fmt: Optional[Dict[str, float]] = Page.PaperFormats.get(options['format'].lower())
            if not fmt:
                raise ValueError('Unknown paper format: ' + options['format'])
            paperWidth = fmt['width']
            paperHeight = fmt['height']
        else:
            paperWidth = convertPrintParameterToInches(options.get('width')) or paperWidth
            paperHeight = convertPrintParameterToInches(options.get('height')) or paperHeight

        marginOptions: Dict[str, Any] = options.get('margin', {})
        marginTop: float = convertPrintParameterToInches(marginOptions.get('top')) or 0
        marginLeft: float = convertPrintParameterToInches(marginOptions.get('left')) or 0
        marginBottom: float = convertPrintParameterToInches(marginOptions.get('bottom')) or 0
        marginRight: float = convertPrintParameterToInches(marginOptions.get('right')) or 0
        preferCSSPageSize: bool = options.get('preferCSSPageSize', False)

        result: Dict[str, Any] = await self._client.send('Page.printToPDF', dict(
            landscape=landscape,
            displayHeaderFooter=displayHeaderFooter,
            headerTemplate=headerTemplate,
            footerTemplate=footerTemplate,
            printBackground=printBackground,
            scale=scale,
            paperWidth=paperWidth,
            paperHeight=paperHeight,
            marginTop=marginTop,
            marginBottom=marginBottom,
            marginLeft=marginLeft,
            marginRight=marginRight,
            pageRanges=pageRanges,
            preferCSSPageSize=preferCSSPageSize,
        ))
        buffer: bytes = base64.b64decode(result.get('data', b''))
        if 'path' in options:
            with open(options['path'], 'wb') as f:
                f.write(buffer)
        return buffer

    async def plainText(self) -> str:
        """[Deprecated] Get page content as plain text."""
        logger.warning('`Page.plainText` is deprecated.')
        return await self.evaluate('() => document.body.innerText')

    async def title(self) -> str:
        """Get page's title."""
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.title()

    async def close(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Close this page.
        """
        options = merge_dict(options, kwargs)
        conn = self._client._connection
        if conn is None:
            raise PageError('Protocol Error: Connectoin Closed. Most likely the page has been closed.')
        runBeforeUnload: bool = bool(options.get('runBeforeUnload'))
        if runBeforeUnload:
            await self._client.send('Page.close')
        else:
            await conn.send('Target.closeTarget', {'targetId': self._target._targetId})
            await self._target._isClosedPromise

    def isClosed(self) -> bool:
        """Indicate that the page has been closed."""
        return self._closed

    @property
    def mouse(self) -> Mouse:
        """Get :class:`~pyppeteer.input.Mouse` object."""
        return self._mouse

    async def click(self, selector: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Click element which matches ``selector``.
        """
        frame: Optional['Frame'] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        await frame.click(selector, options, **kwargs)

    async def hover(self, selector: str) -> None:
        """Mouse hover the element which matches ``selector``.
        """
        frame: Optional['Frame'] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        await frame.hover(selector)

    async def focus(self, selector: str) -> None:
        """Focus the element which matches ``selector``.
        """
        frame: Optional['Frame'] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        await frame.focus(selector)

    async def select(self, selector: str, *values: str) -> List[str]:
        """Select options and return selected values.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.select(selector, *values)

    async def type(self, selector: str, text: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Type ``text`` on the element which matches ``selector``.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.type(selector, text, options, **kwargs)

    def waitFor(self, selectorOrFunctionOrTimeout: Union[str, int, float],
                options: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any) -> Awaitable:
        """Wait for function, timeout, or element which matches on page.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.waitFor(selectorOrFunctionOrTimeout, options, *args, **kwargs)

    def waitForSelector(self, selector: str, options: Optional[Dict[str, Any]] = None,
                        **kwargs: Any) -> Awaitable:
        """Wait until element which matches ``selector`` appears on page.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.waitForSelector(selector, options, **kwargs)

    def waitForXPath(self, xpath: str, options: Optional[Dict[str, Any]] = None,
                     **kwargs: Any) -> Awaitable:
        """Wait until element which matches ``xpath`` appears on page.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.waitForXPath(xpath, options, **kwargs)

    def waitForFunction(self, pageFunction: str, options: Optional[Dict[str, Any]] = None,
                        *args: str, **kwargs: Any) -> Awaitable:
        """Wait until the function completes and returns a truthy value.
        """
        frame: Optional['Frame'] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.waitForFunction(pageFunction, options, *args, **kwargs)

supportedMetrics = (
    'Timestamp',
    'Documents',
    'Frames',
    'JSEventListeners',
    'Nodes',
    'LayoutCount',
    'RecalcStyleCount',
    'LayoutDuration',
    'RecalcStyleDuration',
    'ScriptDuration',
    'TaskDuration',
    'JSHeapUsedSize',
    'JSHeapTotalSize',
)

unitToPixels: Dict[str, float] = {
    'px': 1,
    'in': 96,
    'cm': 37.8,
    'mm': 3.78
}

def convertPrintParameterToInches(parameter: Union[None, int, float, str]) -> Optional[float]:
    """Convert print parameter to inches."""
    if parameter is None:
        return None
    if isinstance(parameter, (int, float)):
        pixels = parameter
    elif isinstance(parameter, str):
        text: str = parameter
        unit: str = text[-2:].lower()
        if unit in unitToPixels:
            valueText: str = text[:-2]
        else:
            unit = 'px'
            valueText = text
        try:
            value: float = float(valueText)
        except ValueError:
            raise ValueError('Failed to parse parameter value: ' + text)
        pixels = value * unitToPixels[unit]
    else:
        raise TypeError('page.pdf() Cannot handle parameter type: ' + str(type(parameter)))
    return pixels / 96

class ConsoleMessage(object):
    """Console message class.
    """

    def __init__(self, type: str, text: str, args: Optional[List[JSHandle]] = None) -> None:
        self._type: str = type
        self._text: str = text
        self._args: List[JSHandle] = args if args is not None else []

    @property
    def type(self) -> str:
        """Return type of this message."""
        return self._type

    @property
    def text(self) -> str:
        """Return text representation of this message."""
        return self._text

    @property
    def args(self) -> List[JSHandle]:
        """Return list of args (JSHandle) of this message."""
        return self._args

async def craete(*args: Any, **kwargs: Any) -> Page:
    """[Deprecated] miss-spelled function.
    """
    logger.warning(
        '`craete` function is deprecated and will be removed in future. '
        'Use `Page.create` instead.'
    )
    return await Page.create(*args, **kwargs)