"""Page module."""
import asyncio
import base64
import json
import logging
import math
import mimetypes
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, TYPE_CHECKING
from pyee import EventEmitter
from pyppeteer import helper
from pyppeteer.connection import CDPSession
from pyppeteer.coverage import Coverage
from pyppeteer.dialog import Dialog
from pyppeteer.element_handle import ElementHandle
from pyppeteer.emulation_manager import EmulationManager
from pyppeteer.errors import PageError
from pyppeteer.execution_context import JSHandle
from pyppeteer.frame_manager import Frame
from pyppeteer.frame_manager import FrameManager
from pyppeteer.helper import debugError
from pyppeteer.input import Keyboard, Mouse, Touchscreen
from pyppeteer.navigator_watcher import NavigatorWatcher
from pyppeteer.network_manager import NetworkManager, Response, Request
from pyppeteer.tracing import Tracing
from pyppeteer.util import merge_dict
from pyppeteer.worker import Worker

if TYPE_CHECKING:
    from pyppeteer.browser import Browser, Target

logger = logging.getLogger(__name__)


class Page(EventEmitter):
    """Page class.

    This class provides methods to interact with a single tab of chrome. One
    :class:`~pyppeteer.browser.Browser` object might have multiple Page object.

    The :class:`Page` class emits various :attr:`~Page.Events` which can be
    handled by using ``on`` or ``once`` method, which is inherited from
    `pyee <https://pyee.readthedocs.io/en/latest/>`_'s ``EventEmitter`` class.
    """
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
        WorkerDestroyed='workerdestroyed'
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
        a5={'width': 5.83, 'height': 8.27}
    )

    @staticmethod
    async def create(
        client: CDPSession,
        target: 'Target',
        ignoreHTTPSErrors: bool,
        defaultViewport: Optional[Dict[str, Any]],
        screenshotTaskQueue: Optional[List[Any]] = None
    ) -> 'Page':
        """Async function which makes new page object."""
        await client.send('Page.enable')
        frameTree: Dict[str, Any] = (await client.send('Page.getFrameTree'))['frameTree']
        page: 'Page' = Page(client, target, frameTree, ignoreHTTPSErrors, screenshotTaskQueue)
        await asyncio.gather(
            client.send('Target.setAutoAttach', {'autoAttach': True, 'waitForDebuggerOnStart': False}),
            client.send('Page.setLifecycleEventsEnabled', {'enabled': True}),
            client.send('Network.enable', {}),
            client.send('Runtime.enable', {}),
            client.send('Security.enable', {}),
            client.send('Performance.enable', {}),
            client.send('Log.enable', {})
        )
        if ignoreHTTPSErrors:
            await client.send('Security.setOverrideCertificateErrors', {'override': True})
        if defaultViewport:
            await page.setViewport(defaultViewport)
        return page

    def __init__(
        self,
        client: CDPSession,
        target: 'Target',
        frameTree: Dict[str, Any],
        ignoreHTTPSErrors: bool,
        screenshotTaskQueue: Optional[List[Any]] = None
    ) -> None:
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
        self._defaultNavigationTimeout: int = 30000
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
                    client.send('Target.detachFromTarget', {'sessionId': event['sessionId']})
                except Exception as e:
                    debugError(logger, e)
                return
            sessionId: str = event['sessionId']
            session: CDPSession = client._createSession(targetInfo['type'], sessionId)
            worker: Worker = Worker(
                session,
                targetInfo['url'],
                self._addConsoleMessage,
                self._handleException
            )
            self._workers[sessionId] = worker
            self.emit(Page.Events.WorkerCreated, worker)

        def _onTargetDetached(event: Dict[str, Any]) -> None:
            sessionId: str = event['sessionId']
            worker: Optional[Worker] = self._workers.get(sessionId)
            if worker is None:
                return
            self.emit(Page.Events.WorkerDestroyed, worker)
            del self._workers[sessionId]

        client.on('Target.attachedToTarget', _onTargetAttached)
        client.on('Target.detachedFromTarget', _onTargetDetached)

        _fm: FrameManager = self._frameManager
        _fm.on(FrameManager.Events.FrameAttached, lambda event: self.emit(Page.Events.FrameAttached, event))
        _fm.on(FrameManager.Events.FrameDetached, lambda event: self.emit(Page.Events.FrameDetached, event))
        _fm.on(FrameManager.Events.FrameNavigated, lambda event: self.emit(Page.Events.FrameNavigated, event))

        _nm: NetworkManager = self._networkManager
        _nm.on(NetworkManager.Events.Request, lambda event: self.emit(Page.Events.Request, event))
        _nm.on(NetworkManager.Events.Response, lambda event: self.emit(Page.Events.Response, event))
        _nm.on(NetworkManager.Events.RequestFailed, lambda event: self.emit(Page.Events.RequestFailed, event))
        _nm.on(NetworkManager.Events.RequestFinished, lambda event: self.emit(Page.Events.RequestFinished, event))

        client.on('Page.domContentEventFired', lambda event: self.emit(Page.Events.DOMContentLoaded))
        client.on('Page.loadEventFired', lambda event: self.emit(Page.Events.Load))
        client.on('Runtime.consoleAPICalled', lambda event: self._onConsoleAPI(event))
        client.on('Runtime.bindingCalled', lambda event: self._onBindingCalled(event))
        client.on('Page.javascriptDialogOpening', lambda event: self._onDialog(event))
        client.on('Runtime.exceptionThrown', lambda exception: self._handleException(exception.get('exceptionDetails')))

        client.on('Security.certificateError', lambda event: self._onCertificateError(event))
        client.on('Inspector.targetCrashed', lambda event: self._onTargetCrashed())
        client.on('Performance.metrics', lambda event: self._emitMetrics(event))
        client.on('Log.entryAdded', lambda event: self._onLogEntryAdded(event))

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
        args: List[Dict[str, Any]] = entry.get('args', [])
        source: str = entry.get('source', '')
        for arg in args:
            helper.releaseObject(self._client, arg)
        if source != 'worker':
            self.emit(Page.Events.Console, ConsoleMessage(level, text))

    @property
    def mainFrame(self) -> Frame:
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
        frame: Optional[Frame] = self.mainFrame
        if frame is None:
            raise PageError('no main frame')
        await frame.tap(selector)

    @property
    def tracing(self) -> Tracing:
        """Get tracing object."""
        return self._tracing

    @property
    def frames(self) -> List[Frame]:
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

        An example of a native request interceptor that aborts all image
        requests:

        .. code:: python

            browser = await launch()
            page = await browser.newPage()
            await page.setRequestInterception(True)

            async def intercept(request):
                if request.url.endswith('.png') or request.url.endswith('.jpg'):
                    await request.abort()
                else:
                    await request.continue_()

            page.on('request', lambda req: asyncio.ensure_future(intercept(req)))
            await page.goto('https://example.com')
            await browser.close()
        """
        return await self._networkManager.setRequestInterception(value)

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

    def _onCertificateError(self, event: Dict[str, Any]) -> None:
        if not self._ignoreHTTPSErrors:
            return
        asyncio.create_task(
            self._send(
                'Security.handleCertificateError',
                {
                    'eventId': event.get('eventId'),
                    'action': 'continue'
                }
            )
        )

    async def querySelector(self, selector: str) -> Optional[ElementHandle]:
        """Get an Element which matches ``selector``.

        :arg str selector: A selector to search element.
        :return Optional[ElementHandle]: If element which matches the
            ``selector`` is found, return its
            :class:`~pyppeteer.element_handle.ElementHandle`. If not found,
            returns ``None``.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelector(selector)

    async def evaluateHandle(
        self,
        pageFunction: Union[str, Callable[..., Any]],
        *args: Any
    ) -> JSHandle:
        """Execute function on this page.

        Difference between :meth:`~pyppeteer.page.Page.evaluate` and
        :meth:`~pyppeteer.page.Page.evaluateHandle` is that
        ``evaluateHandle`` returns JSHandle object (not value).

        :arg str pageFunction: JavaScript function to be executed.
        """
        if not self.mainFrame:
            raise PageError('no main frame.')
        context: Optional['ExecutionContext'] = await self.mainFrame.executionContext()
        if not context:
            raise PageError('No context.')
        return await context.evaluateHandle(pageFunction, *args)

    async def queryObjects(self, prototypeHandle: JSHandle) -> JSHandle:
        """Iterate js heap and finds all the objects with the handle.

        :arg JSHandle prototypeHandle: JSHandle of prototype object.
        """
        if not self.mainFrame:
            raise PageError('no main frame.')
        context: Optional['ExecutionContext'] = await self.mainFrame.executionContext()
        if not context:
            raise PageError('No context.')
        return await context.queryObjects(prototypeHandle)

    async def querySelectorEval(
        self,
        selector: str,
        pageFunction: Union[str, Callable[..., Any]],
        *args: Any
    ) -> Any:
        """Execute function with an element which matches ``selector``.

        :arg str selector: A selector to query page for.
        :arg str pageFunction: String of JavaScript function to be evaluated on
                               browser. This function takes an element which
                               matches the selector as a first argument.
        :arg Any args: Arguments to pass to ``pageFunction``.
        :raises PageError: If no element matches the selector.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelectorEval(selector, pageFunction, *args)

    async def querySelectorAllEval(
        self,
        selector: str,
        pageFunction: Union[str, Callable[..., Any]],
        *args: Any
    ) -> Any:
        """Execute function with all elements which matches ``selector``.

        :arg str selector: A selector to query page for.
        :arg str pageFunction: String of JavaScript function to be evaluated on
                               browser. This function takes Array of the
                               matched elements as the first argument.
        :arg Any args: Arguments to pass to ``pageFunction``.
        """
        frame: Optional[Frame] = self.mainFrame
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
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.querySelectorAll(selector)

    async def xpath(self, expression: str) -> List[ElementHandle]:
        """Evaluate the XPath expression.

        If there are no such elements in this page, return an empty list.

        :arg str expression: XPath string to be evaluated.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.xpath(expression)

    J: Callable[[str], Awaitable[Optional[ElementHandle]]] = querySelector
    Jeval: Callable[[str, Union[str, Callable[..., Any]], Any], Awaitable[Any]] = querySelectorEval
    JJ: Callable[[str], Awaitable[List[ElementHandle]]] = querySelectorAll
    JJeval: Callable[[str, Union[str, Callable[..., Any]], Any], Awaitable[Any]] = querySelectorAllEval
    Jx: Callable[[str], Awaitable[List[ElementHandle]]] = xpath

    async def cookies(self, *urls: str) -> List[Dict[str, Any]]:
        """Get cookies.

        If no URLs are specified, this method returns cookies for the current
        page URL. If URLs are specified, only cookies for those URLs are
        returned.

        Returned cookies are list of dictionaries which contain these fields:

        * ``name`` (str)
        * ``value`` (str)
        * ``url`` (str)
        * ``domain`` (str)
        * ``path`` (str)
        * ``expires`` (number): Unix time in seconds
        * ``httpOnly`` (bool)
        * ``secure`` (bool)
        * ``session`` (bool)
        * ``sameSite`` (str): ``'Strict'`` or ``'Lax'``
        """
        if not urls:
            urls = (self.url,)
        resp: Dict[str, Any] = await self._client.send('Network.getCookies', {'urls': urls})
        return resp.get('cookies', [])

    async def deleteCookie(self, *cookies: Dict[str, Any]) -> None:
        """Delete cookie.

        ``cookies`` should be dictionaries which contain these fields:

        * ``name`` (str): **required**
        * ``url`` (str)
        * ``domain`` (str)
        * ``path`` (str)
        * ``secure`` (bool)
        """
        pageURL: str = self.url
        for cookie in cookies:
            item: Dict[str, Any] = dict(**cookie)
            if not cookie.get('url') and pageURL.startswith('http'):
                item['url'] = pageURL
            await self._client.send('Network.deleteCookies', item)

    async def setCookie(self, *cookies: Dict[str, Any]) -> None:
        """Set cookies.

        ``cookies`` should be dictionaries which contain these fields:

        * ``name`` (str): **required**
        * ``value`` (str): **required**
        * ``url`` (str)
        * ``domain`` (str)
        * ``path`` (str)
        * ``expires`` (number): Unix time in seconds
        * ``httpOnly`` (bool)
        * ``secure`` (bool)
        * ``sameSite`` (str): ``'Strict'`` or ``'Lax'``
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
                name: str = item.get('name', '')
                raise PageError(f'Data URL page can not have cookie "{name}"')
            items.append(item)
        await self.deleteCookie(*items)
        if items:
            await self._client.send('Network.setCookies', {'cookies': items})

    async def addScriptTag(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> ElementHandle:
        """Add script tag to this page.

        One of ``url``, ``path`` or ``content`` option is necessary.
            * ``url`` (string): URL of a script to add.
            * ``path`` (string): Path to the local JavaScript file to add.
            * ``content`` (string): JavaScript string to add.
            * ``type`` (string): Script type. Use ``module`` in order to load a
              JavaScript ES6 module.

        :return ElementHandle: :class:`~pyppeteer.element_handle.ElementHandle`
                               of added tag.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        options = merge_dict(options, kwargs)
        return await frame.addScriptTag(options)

    async def addStyleTag(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> ElementHandle:
        """Add style or link tag to this page.

        One of ``url``, ``path`` or ``content`` option is necessary.
            * ``url`` (string): URL of the link tag to add.
            * ``path`` (string): Path to the local CSS file to add.
            * ``content`` (string): CSS string to add.

        :return ElementHandle: :class:`~pyppeteer.element_handle.ElementHandle`
                               of added tag.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        options = merge_dict(options, kwargs)
        return await frame.addStyleTag(options)

    async def injectFile(self, filePath: str) -> ElementHandle:
        """[Deprecated] Inject file to this page.

        This method is deprecated. Use :meth:`addScriptTag` instead.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.injectFile(filePath)

    async def exposeFunction(
        self,
        name: str,
        pyppeteerFunction: Callable[..., Any]
    ) -> None:
        """Add python function to the browser's ``window`` object as ``name``.

        Registered function can be called from chrome process.

        :arg string name: Name of the function on the window object.
        :arg Callable pyppeteerFunction: Function which will be called on
                                         python process. This function should
                                         not be asynchronous function.
        """
        if self._pageBindings.get(name):
            raise PageError(f'Failed to add page binding with name {name}: window["{name}"] already exists!')
        self._pageBindings[name] = pyppeteerFunction
        addPageBinding: str = "\nfunction addPageBinding(bindingName) {\n  const binding = window[bindingName];\n  window[bindingName] = async(...args) => {\n    const me = window[bindingName];\n    let callbacks = me['callbacks'];\n    if (!callbacks) {\n      callbacks = new Map();\n      me['callbacks'] = callbacks;\n    }\n    const seq = (me['lastSeq'] || 0) + 1;\n    me['lastSeq'] = seq;\n    const promise = new Promise(fulfill => callbacks.set(seq, fulfill));\n    binding(JSON.stringify({name: bindingName, seq, args}));\n    return promise;\n  };\n}\n        "
        expression: str = helper.evaluationString(addPageBinding, name)
        await self._client.send('Runtime.addBinding', {'name': name})
        await self._client.send('Page.addScriptToEvaluateOnNewDocument', {'source': expression})

        async def _evaluate(frame: Frame, expression: str) -> None:
            try:
                await frame.evaluate(expression, force_expr=True)
            except Exception as e:
                debugError(logger, e)

        await asyncio.wait([_evaluate(frame, expression) for frame in self.frames])

    async def authenticate(self, credentials: Optional[Dict[str, str]]) -> None:
        """Provide credentials for http authentication.

        ``credentials`` should be ``None`` or dict which has ``username`` and
        ``password`` field.
        """
        return await self._networkManager.authenticate(credentials)

    async def setExtraHTTPHeaders(self, headers: Dict[str, str]) -> None:
        """Set extra HTTP headers.

        The extra HTTP headers will be sent with every request the page
        initiates.

        .. note::
            ``page.setExtraHTTPHeaders`` does not guarantee the order of
            headers in the outgoing requests.

        :arg Dict headers: A dictionary containing additional http headers to
                           be sent with every requests. All header values must
                           be string.
        """
        return await self._networkManager.setExtraHTTPHeaders(headers)

    async def setUserAgent(self, userAgent: str) -> None:
        """Set user agent to use in this page.

        :arg str userAgent: Specific user agent to use in this page
        """
        return await self._networkManager.setUserAgent(userAgent)

    async def metrics(self) -> Dict[str, Union[int, float]]:
        """Get metrics.

        Returns dictionary containing metrics as key/value pairs:

        * ``Timestamp`` (number): The timestamp when the metrics sample was
          taken.
        * ``Documents`` (int): Number of documents in the page.
        * ``Frames`` (int): Number of frames in the page.
        * ``JSEventListeners`` (int): Number of events in the page.
        * ``Nodes`` (int): Number of DOM nodes in the page.
        * ``LayoutCount`` (int): Total number of full partial page layout.
        * ``RecalcStyleCount`` (int): Total number of page style
          recalculations.
        * ``LayoutDuration`` (int): Combined duration of page duration.
        * ``RecalcStyleDuration`` (int): Combined duration of all page style
          recalculations.
        * ``ScriptDuration`` (int): Combined duration of JavaScript
          execution.
        * ``TaskDuration`` (int): Combined duration of all tasks performed by
          the browser.
        * ``JSHeapUsedSize`` (float): Used JavaScript heap size.
        * ``JSHeapTotalSize`` (float): Total JavaScript heap size.
        """
        response: Dict[str, Any] = await self._client.send('Performance.getMetrics')
        return self._buildMetricsObject(response['metrics'])

    def _emitMetrics(self, event: Dict[str, Any]) -> None:
        self.emit(Page.Events.Metrics, {'title': event['title'], 'metrics': self._buildMetricsObject(event['metrics'])})

    def _buildMetricsObject(self, metrics: List[Dict[str, Any]]) -> Dict[str, Union[int, float]]:
        result: Dict[str, Union[int, float]] = {}
        for metric in metrics or []:
            if metric['name'] in supportedMetrics:
                result[metric['name']] = metric['value']
        return result

    def _handleException(self, exceptionDetails: Optional[Dict[str, Any]]) -> None:
        if exceptionDetails is None:
            return
        message: str = helper.getExceptionMessage(exceptionDetails)
        self.emit(Page.Events.PageError, PageError(message))

    def _onConsoleAPI(self, event: Dict[str, Any]) -> None:
        _id: int = event['executionContextId']
        context: Optional['ExecutionContext'] = self._frameManager.executionContextById(_id)
        values: List[JSHandle] = []
        for arg in event.get('args', []):
            values.append(self._frameManager.createJSHandle(context, arg))
        self._addConsoleMessage(event['type'], values)

    def _onBindingCalled(self, event: Dict[str, Any]) -> None:
        obj: Dict[str, Any] = json.loads(event['payload'])
        name: str = obj['name']
        seq: int = obj['seq']
        args: List[Any] = obj['args']
        result: Any = self._pageBindings[name](*args)
        deliverResult: str = "\n            function deliverResult(name, seq, result) {\n                window[name]['callbacks'].get(seq)(result);\n                window[name]['callbacks'].delete(seq);\n            }\n        "
        expression: str = helper.evaluationString(deliverResult, name, seq, result)
        try:
            self._client.send('Runtime.evaluate', {'expression': expression, 'contextId': event['executionContextId']})
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
        dialog: Dialog = Dialog(
            self._client,
            dialogType,
            event.get('message', ''),
            event.get('defaultPrompt', '')
        )
        self.emit(Page.Events.Dialog, dialog)

    @property
    def url(self) -> str:
        """Get URL of this page."""
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.url

    async def content(self) -> str:
        """Get the full HTML contents of the page.

        Returns HTML including the doctype.
        """
        frame: Optional[Frame] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        return await frame.content()

    async def setContent(self, html: str) -> None:
        """Set content to this page.

        :arg str html: HTML markup to assign to the page.
        """
        frame: Optional[Frame] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        await frame.setContent(html)

    async def goto(
        self,
        url: str,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[Response]:
        """Go to the ``url``.

        :arg string url: URL to navigate page to. The url should include
                         scheme, e.g. ``https://``.

        Available options are:

        * ``timeout`` (int): Maximum navigation time in milliseconds, defaults
          to 30 seconds, pass ``0`` to disable timeout. The default value can
          be changed by using the :meth:`setDefaultNavigationTimeout` method.
        * ``waitUntil`` (str|List[str]): When to consider navigation succeeded,
          defaults to ``load``. Given a list of event strings, navigation is
          considered to be successful after all events have been fired. Events
          can be either:

          * ``load``: when ``load`` event is fired.
          * ``domcontentloaded``: when the ``DOMContentLoaded`` event is fired.
          * ``networkidle0``: when there are no more than 0 network connections
            for at least 500 ms.
          * ``networkidle2``: when there are no more than 2 network connections
            for at least 500 ms.

        The ``Page.goto`` will raise errors if:

        * there's an SSL error (e.g. in case of self-signed certificates)
        * target URL is invalid
        * the ``timeout`` is exceeded during navigation
        * then main resource failed to load

        .. note::
            :meth:`goto` either raise error or return a main resource response.
            The only exceptions are navigation to ``about:blank`` or navigation
            to the same URL with a different hash, which would succeed and
            return ``None``.

        .. note::
            Headless mode doesn't support navigation to a PDF document.
        """
        options = merge_dict(options, kwargs)
        mainFrame: Optional[Frame] = self._frameManager.mainFrame
        if mainFrame is None:
            raise PageError('No main frame.')
        referrer: str = self._networkManager.extraHTTPHeaders().get('referer', '')
        requests: Dict[str, Request] = dict()

        def set_request(req: Request) -> None:
            if req.url not in requests:
                requests[req.url] = req

        eventListeners: List[Callable[[], None]] = [
            helper.addEventListener(self._networkManager, NetworkManager.Events.Request, set_request)
        ]
        timeout: int = options.get('timeout', self._defaultNavigationTimeout)
        watcher: NavigatorWatcher = NavigatorWatcher(self._frameManager, mainFrame, timeout, options)
        result: Optional[str] = await self._navigate(url, referrer)
        if result is not None:
            raise PageError(result)
        result = await watcher.navigationPromise()
        watcher.cancel()
        helper.removeEventListeners(eventListeners)
        error: Optional[PageError] = result[0].pop().exception() if result and result[0] else None
        if error:
            raise error
        request: Optional[Request] = requests.get(mainFrame._navigationURL)
        return request.response if request else None

    async def _navigate(self, url: str, referrer: str) -> Optional[str]:
        response: Dict[str, Any] = await self._client.send('Page.navigate', {'url': url, 'referrer': referrer})
        if response.get('errorText'):
            return f"{response['errorText']} at {url}"
        return None

    async def reload(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[Response]:
        """Reload this page.

        Available options are same as :meth:`goto` method.
        """
        options = merge_dict(options, kwargs)
        response: Optional[Response] = (await asyncio.gather(
            self.waitForNavigation(options),
            self._client.send('Page.reload')
        ))[0]
        return response

    async def waitForNavigation(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[Response]:
        """Wait for navigation.

        Available options are same as :meth:`goto` method.

        This returns :class:`~pyppeteer.network_manager.Response` when the page
        navigates to a new URL or reloads. It is useful for when you run code
        which will indirectly cause the page to navigate. In case of navigation
        to a different anchor or navigation due to
        `History API <https://developer.mozilla.org/en-US/docs/Web/API/History_API>`_
        usage, the navigation will return ``None``.

        Consider this example:

        .. code::

            navigationPromise = async.ensure_future(page.waitForNavigation())
            await page.click('a.my-link')  # indirectly cause a navigation
            await navigationPromise  # wait until navigation finishes

        or,

        .. code::

            await asyncio.wait([
                page.click('a.my-link'),
                page.waitForNavigation(),
            ])

        .. note::
            Usage of the History API to change the URL is considered a
            navigation.
        """
        options = merge_dict(options, kwargs)
        mainFrame: Optional[Frame] = self._frameManager.mainFrame
        if mainFrame is None:
            raise PageError('No main frame.')
        timeout: int = options.get('timeout', self._defaultNavigationTimeout)
        watcher: NavigatorWatcher = NavigatorWatcher(self._frameManager, mainFrame, timeout, options)
        responses: Dict[str, Response] = dict()

        def listener(response: Response) -> None:
            responses[response.url] = response

        listener_id = helper.addEventListener(
            self._networkManager,
            NetworkManager.Events.Response,
            listener
        )

        result: Optional[List[List[Any]]] = await watcher.navigationPromise()
        helper.removeEventListeners([listener_id])
        error: Optional[PageError] = result[0].pop().exception() if result and result[0] else None
        if error:
            raise error
        response: Optional[Response] = responses.get(self.url, None)
        return response

    async def waitForRequest(
        self,
        urlOrPredicate: Union[str, Callable[[Request], bool]],
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Request:
        """Wait for request.

        :arg urlOrPredicate: A URL or function to wait for.

        This method accepts below options:

        * ``timeout`` (int|float): Maximum wait time in milliseconds, defaults
          to 30 seconds, pass ``0`` to disable the timeout.

        Example:

        .. code::

            firstRequest = await page.waitForRequest('http://example.com/resource')
            finalRequest = await page.waitForRequest(lambda req: req.url == 'http://example.com' and req.method == 'GET')
            return firstRequest.url
        """
        options = merge_dict(options, kwargs)
        timeout: int = options.get('timeout', 30000)

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
            self._client._loop
        )

    async def waitForResponse(
        self,
        urlOrPredicate: Union[str, Callable[[Response], bool]],
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Response:
        """Wait for response.

        :arg urlOrPredicate: A URL or function to wait for.

        This method accepts below options:

        * ``timeout`` (int|float): Maximum wait time in milliseconds, defaults
          to 30 seconds, pass ``0`` to disable the timeout.

        Example:

        .. code::

            firstResponse = await page.waitForResponse('http://example.com/resource')
            finalResponse = await page.waitForResponse(lambda res: res.url == 'http://example.com' and res.status == 200)
            return finalResponse.ok
        """
        options = merge_dict(options, kwargs)
        timeout: int = options.get('timeout', 30000)

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
            self._client._loop
        )

    async def goBack(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[Response]:
        """Navigate to the previous page in history.

        Available options are same as :meth:`goto` method.

        If cannot go back, return ``None``.
        """
        options = merge_dict(options, kwargs)
        return await self._go(-1, options)

    async def goForward(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[Response]:
        """Navigate to the next page in history.

        Available options are same as :meth:`goto` method.

        If cannot go forward, return ``None``.
        """
        options = merge_dict(options, kwargs)
        return await self._go(+1, options)

    async def _go(self, delta: int, options: Dict[str, Any]) -> Optional[Response]:
        history: Dict[str, Any] = await self._client.send('Page.getNavigationHistory')
        _count: int = history.get('currentIndex', 0) + delta
        entries: List[Dict[str, Any]] = history.get('entries', [])
        if len(entries) <= _count:
            return None
        entry: Dict[str, Any] = entries[_count]
        response: Optional[Response] = (await asyncio.gather(
            self.waitForNavigation(options),
            self._client.send('Page.navigateToHistoryEntry', {'entryId': entry.get('id', '')})
        ))[0]
        return response

    async def bringToFront(self) -> None:
        """Bring page to front (activate tab)."""
        await self._client.send('Page.bringToFront')

    async def emulate(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Emulate given device metrics and user agent.

        This method is a shortcut for calling two methods:

        * :meth:`setUserAgent`
        * :meth:`setViewport`

        ``options`` is a dictionary containing these fields:

        * ``viewport`` (dict)

          * ``width`` (int): page width in pixels.
          * ``height`` (int): page width in pixels.
          * ``deviceScaleFactor`` (float): Specify device scale factor (can be
            thought as dpr). Defaults to 1.
          * ``isMobile`` (bool): Whether the ``meta viewport`` tag is taken
            into account. Defaults to ``False``.
          * ``hasTouch`` (bool): Specifies if viewport supports touch events.
            Defaults to ``False``.
          * ``isLandscape`` (bool): Specifies if viewport is in landscape mode.
            Defaults to ``False``.

        * ``userAgent`` (str): user agent string.
        """
        options = merge_dict(options, kwargs)
        await self.setViewport(options.get('viewport', {}))
        await self.setUserAgent(options.get('userAgent', ''))

    async def setJavaScriptEnabled(self, enabled: bool) -> None:
        """Set JavaScript enable/disable."""
        if self._javascriptEnabled == enabled:
            return
        self._javascriptEnabled = enabled
        await self._client.send('Emulation.setScriptExecutionDisabled', {'value': not enabled})

    async def setBypassCSP(self, enabled: bool) -> None:
        """Toggles bypassing page's Content-Security-Policy.

        .. note::
            CSP bypassing happens at the moment of CSP initialization rather
            then evaluation. Usually this means that ``page.setBypassCSP``
            should be called before navigating to the domain.
        """
        await self._client.send('Page.setBypassCSP', {'enabled': enabled})

    async def emulateMedia(self, mediaType: Optional[str] = None) -> None:
        """Emulate css media type of the page.

        :arg str mediaType: Changes the CSS media type of the page. The only
                            allowed values are ``'screen'``, ``'print'``, and
                            ``None``. Passing ``None`` disables media
                            emulation.
        """
        if mediaType not in ['screen', 'print', None, '']:
            raise ValueError(f'Unsupported media type: {mediaType}')
        await self._client.send('Emulation.setEmulatedMedia', {'media': mediaType or ''})

    async def setViewport(self, viewport: Dict[str, Any]) -> None:
        """Set viewport.

        Available options are:
            * ``width`` (int): page width in pixel.
            * ``height`` (int): page height in pixel.
            * ``deviceScaleFactor`` (float): Default to 1.0.
            * ``isMobile`` (bool): Default to ``False``.
            * ``hasTouch`` (bool): Default to ``False``.
            * ``isLandscape`` (bool): Default to ``False``.
        """
        needsReload: bool = await self._emulationManager.emulateViewport(viewport)
        self._viewport = viewport
        if needsReload:
            await self.reload()

    @property
    def viewport(self) -> Optional[Dict[str, Any]]:
        """Get viewport as a dictionary or None.

        Fields of returned dictionary is same as :meth:`setViewport`.
        """
        return self._viewport

    async def evaluate(
        self,
        pageFunction: Union[str, Callable[..., Any]],
        *args: Any,
        force_expr: bool = False
    ) -> Any:
        """Execute js-function or js-expression on browser and get result.

        :arg str pageFunction: String of js-function/expression to be executed
                               on the browser.
        :arg bool force_expr: If True, evaluate `pageFunction` as expression.
                              If False (default), try to automatically detect
                              function or expression.

        note: ``force_expr`` option is a keyword only argument.
        """
        frame: Optional[Frame] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        return await frame.evaluate(pageFunction, *args, force_expr=force_expr)

    async def evaluateOnNewDocument(
        self,
        pageFunction: Union[str, Callable[..., Any]],
        *args: Any
    ) -> None:
        """Add a JavaScript function to the document.

        This function would be invoked in one of the following scenarios:

        * whenever the page is navigated
        * whenever the child frame is attached or navigated. In this case, the
          function is invoked in the context of the newly attached frame.
        """
        source: str = helper.evaluationString(pageFunction, *args)
        await self._client.send('Page.addScriptToEvaluateOnNewDocument', {'source': source})

    async def setCacheEnabled(self, enabled: bool = True) -> None:
        """Enable/Disable cache for each request.

        By default, caching is enabled.
        """
        await self._client.send('Network.setCacheDisabled', {'cacheDisabled': not enabled})

    async def screenshot(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Union[bytes, str]:
        """Take a screen shot.

        The following options are available:

        * ``path`` (str): The file path to save the image to. The screenshot
          type will be inferred from the file extension.
        * ``type`` (str): Specify screenshot type, can be either ``jpeg`` or
          ``png``. Defaults to ``png``.
        * ``quality`` (int): The quality of the image, between 0-100. Not
          applicable to ``png`` image.
        * ``fullPage`` (bool): When true, take a screenshot of the full
          scrollable page. Defaults to ``False``.
        * ``clip`` (dict): An object which specifies clipping region of the
          page. This option should have the following fields:

          * ``x`` (int): x-coordinate of top-left corner of clip area.
          * ``y`` (int): y-coordinate of top-left corner of clip area.
          * ``width`` (int): width of clipping area.
          * ``height`` (int): height of clipping area.

        * ``omitBackground`` (bool): Hide default white background and allow
          capturing screenshot with transparency.
        * ``encoding`` (str): The encoding of the image, can be either
          ``'base64'`` or ``'binary'``. Defaults to ``'binary'``.
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
                raise ValueError(f'Unsupported screenshot mime type: {mimeType}')
        if not screenshotType:
            screenshotType = 'png'
        return await self._screenshotTask(screenshotType, options)

    async def _screenshotTask(
        self,
        format: str,
        options: Dict[str, Any]
    ) -> Union[bytes, str]:
        await self._client.send('Target.activateTarget', {'targetId': self._target._targetId})
        clip: Optional[Dict[str, Any]] = options.get('clip')
        if clip:
            clip['scale'] = 1
        if options.get('fullPage'):
            metrics: Dict[str, Any] = await self._client.send('Page.getLayoutMetrics')
            width: float = math.ceil(metrics['contentSize']['width'])
            height: float = math.ceil(metrics['contentSize']['height'])
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
                'screenOrientation': screenOrientation
            })
        if options.get('omitBackground'):
            await self._client.send('Emulation.setDefaultBackgroundColorOverride', {'color': {'r': 0, 'g': 0, 'b': 0, 'a': 0}})
        opt: Dict[str, Any] = {'format': format}
        if clip:
            opt['clip'] = clip
        result: Dict[str, Any] = await self._client.send('Page.captureScreenshot', opt)
        if options.get('omitBackground'):
            await self._client.send('Emulation.setDefaultBackgroundColorOverride')
        if options.get('fullPage') and self._viewport is not None:
            await self.setViewport(self._viewport)
        if options.get('encoding') == 'base64':
            buffer: Union[bytes, str] = result.get('data', b'')
        else:
            buffer = base64.b64decode(result.get('data', b''))
        _path: Optional[str] = options.get('path')
        if _path:
            with open(_path, 'wb') as f:
                f.write(buffer)
        return buffer

    async def pdf(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bytes:
        """Generate a pdf of the page.

        Options:

        * ``path`` (str): The file path to save the PDF.
        * ``scale`` (float): Scale of the webpage rendering, defaults to ``1``.
        * ``displayHeaderFooter`` (bool): Display header and footer.
          Defaults to ``False``.
        * ``headerTemplate`` (str): HTML template for the print header. Should
          be valid HTML markup with following classes.

          * ``date``: formatted print date
          * ``title``: document title
          * ``url``: document location
          * ``pageNumber``: current page number
          * ``totalPages``: total pages in the document

        * ``footerTemplate`` (str): HTML template for the print footer. Should
          use the same template as ``headerTemplate``.
        * ``printBackground`` (bool): Print background graphics. Defaults to
          ``False``.
        * ``landscape`` (bool): Paper orientation. Defaults to ``False``.
        * ``pageRanges`` (string): Paper ranges to print, e.g., '1-5,8,11-13'.
          Defaults to empty string, which means all pages.
        * ``format`` (str): Paper format. If set, takes priority over
          ``width`` or ``height``. Defaults to ``Letter``.
        * ``width`` (str): Paper width, accepts values labeled with units.
        * ``height`` (str): Paper height, accepts values labeled with units.
        * ``margin`` (dict): Paper margins, defaults to ``None``.

          * ``top`` (str): Top margin, accepts values labeled with units.
          * ``right`` (str): Right margin, accepts values labeled with units.
          * ``bottom`` (str): Bottom margin, accepts values labeled with units.
          * ``left`` (str): Left margin, accepts values labeled with units.

        * ``preferCSSPageSize``: Give any CSS ``@page`` size declared in the
          page priority over what is declared in ``width`` and ``height`` or
          ``format`` options. Defaults to ``False``, which will scale the
          content to fit the paper size.

        :return: Return generated PDF ``bytes`` object.

        .. note::
            Generating a pdf is currently only supported in headless mode.

        :meth:`pdf` generates a pdf of the page with ``print`` css media. To
        generate a pdf with ``screen`` media, call
        ``page.emulateMedia('screen')`` before calling :meth:`pdf`.

        .. note::
            By default, :meth:`pdf` generates a pdf with modified colors for
            printing. Use the ``--webkit-print-color-adjust`` property to force
            rendering of exact colors.

        .. code::

            await page.emulateMedia('screen')
            await page.pdf({'path': 'page.pdf'})

        The ``width``, ``height``, and ``margin`` options accept values labeled
        with units. Unlabeled values are treated as pixels.

        A few examples:

        - ``page.pdf({'width': 100})``: prints with width set to 100 pixels.
        - ``page.pdf({'width': '100px'})``: prints with width set to 100 pixels.
        - ``page.pdf({'width': '10cm'})``: prints with width set to 100 centimeters.

        All available units are:

        - ``px``: pixel
        - ``in``: inch
        - ``cm``: centimeter
        - ``mm``: millimeter

        The format options are:

        - ``Letter``: 8.5in x 11in
        - ``Legal``: 8.5in x 14in
        - ``Tabloid``: 11in x 17in
        - ``Ledger``: 17in x 11in
        - ``A0``: 33.1in x 46.8in
        - ``A1``: 23.4in x 33.1in
        - ``A2``: 16.5in x 23.4in
        - ``A3``: 11.7in x 16.5in
        - ``A4``: 8.27in x 11.7in
        - ``A5``: 5.83in x 8.27in
        - ``A6``: 4.13in x 5.83in

        .. note::
            ``headerTemplate`` and ``footerTemplate`` markup have the following
            limitations:

            1. Script tags inside templates are not evaluated.
            2. Page styles are not visible inside templates.
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
            preferCSSPageSize=preferCSSPageSize
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
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.title()

    async def close(
        self,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Close this page.

        Available options:

        * ``runBeforeUnload`` (bool): Defaults to ``False``. Whether to run the
          `before unload <https://developer.mozilla.org/en-US/docs/Web/Events/beforeunload>`_
          page handlers.

        By defaults, :meth:`close` **does not** run beforeunload handlers.

        .. note::
           If ``runBeforeUnload`` is passed as ``True``, a ``beforeunload``
           dialog might be summoned and should be handled manually via page's
           ``dialog`` event.
        """
        options = merge_dict(options, kwargs)
        conn: Optional[Any] = self._client._connection
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

    async def click(
        self,
        selector: str,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Click element which matches ``selector``.

        This method fetches an element with ``selector``, scrolls it into view
        if needed, and then uses :attr:`mouse` to click in the center of the
        element. If there's no element matching ``selector``, the method raises
        ``PageError``.

        Available options are:

        * ``button`` (str): ``left``, ``right``, or ``middle``, defaults to
          ``left``.
        * ``clickCount`` (int): defaults to 1.
        * ``delay`` (int|float): Time to wait between ``mousedown`` and
          ``mouseup`` in milliseconds. defaults to 0.

        .. note:: If this method triggers a navigation event and there's a
            separate :meth:`waitForNavigation`, you may end up with a race
            condition that yields unexpected results. The correct pattern for
            click and wait for navigation is the following::

                await asyncio.gather(
                    page.waitForNavigation(waitOptions),
                    page.click(selector, clickOptions),
                )
        """
        frame: Optional[Frame] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        await frame.click(selector, options, **kwargs)

    async def hover(self, selector: str) -> None:
        """Mouse hover the element which matches ``selector``.

        If no element matched the ``selector``, raise ``PageError``.
        """
        frame: Optional[Frame] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        await frame.hover(selector)

    async def focus(self, selector: str) -> None:
        """Focus the element which matches ``selector``.

        If no element matched the ``selector``, raise ``PageError``.
        """
        frame: Optional[Frame] = self.mainFrame
        if frame is None:
            raise PageError('No main frame.')
        await frame.focus(selector)

    async def select(self, selector: str, *values: str) -> List[str]:
        """Select options and return selected values.

        If no element matched the ``selector``, raise ``ElementHandleError``.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.select(selector, *values)

    async def type(
        self,
        selector: str,
        text: str,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Type ``text`` on the element which matches ``selector``.

        If no element matched the ``selector``, raise ``PageError``.

        Details see :meth:`pyppeteer.input.Keyboard.type`.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return await frame.type(selector, text, options, **kwargs)

    def waitFor(
        self,
        selectorOrFunctionOrTimeout: Union[str, Callable[..., Any], int, float],
        options: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> Awaitable[JSHandle]:
        """Wait for function, timeout, or element which matches on page.

        This method behaves differently with respect to the first argument:

        * If ``selectorOrFunctionOrTimeout`` is number (int or float), then it
          is treated as a timeout in milliseconds and this returns future which
          will be done after the timeout.
        * If ``selectorOrFunctionOrTimeout`` is a string of JavaScript
          function, this method is a shortcut to :meth:`waitForFunction`.
        * If ``selectorOrFunctionOrTimeout`` is a selector string or xpath
          string, this method is a shortcut to :meth:`waitForSelector` or
          :meth:`waitForXPath`. If the string starts with ``//``, the string is
          treated as xpath.

        Pyppeteer tries to automatically detect function or selector, but
        sometimes miss-detects. If not work as you expected, use
        :meth:`waitForFunction` or :meth:`waitForSelector` directly.

        :arg selectorOrFunctionOrTimeout: A selector, xpath, or function
                                          string, or timeout (milliseconds).
        :arg Any args: Arguments to pass the function.
        :return: Return awaitable object which resolves to a JSHandle of the
                 success value.

        Available options: see :meth:`waitForFunction` or
        :meth:`waitForSelector`
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.waitFor(selectorOrFunctionOrTimeout, options, *args, **kwargs)

    def waitForSelector(
        self,
        selector: str,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Awaitable[ElementHandle]:
        """Wait until element which matches ``selector`` appears on page.

        Wait for the ``selector`` to appear in page. If at the moment of
        calling the method the ``selector`` already exists, the method will
        return immediately. If the selector doesn't appear after the
        ``timeout`` milliseconds of waiting, the function will raise error.

        :arg str selector: A selector of an element to wait for.
        :return: Return awaitable object which resolves when element specified
                 by selector string is added to DOM.

        This method accepts the following options:

        * ``visible`` (bool): Wait for element to be present in DOM and to be
          visible; i.e. to not have ``display: none`` or ``visibility: hidden``
          CSS properties. Defaults to ``False``.
        * ``hidden`` (bool): Wait for element to not be found in the DOM or to
          be hidden, i.e. have ``display: none`` or ``visibility: hidden`` CSS
          properties. Defaults to ``False``.
        * ``timeout`` (int|float): Maximum time to wait for in milliseconds.
          Defaults to 30000 (30 seconds). Pass ``0`` to disable timeout.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.waitForSelector(selector, options, **kwargs)

    def waitForXPath(
        self,
        xpath: str,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Awaitable[ElementHandle]:
        """Wait until element which matches ``xpath`` appears on page.

        Wait for the ``xpath`` to appear in page. If the moment of calling the
        method the ``xpath`` already exists, the method will return
        immediately. If the xpath doesn't appear after ``timeout`` milliseconds
        of waiting, the function will raise exception.

        :arg str xpath: A [xpath] of an element to wait for.
        :return: Return awaitable object which resolves when element specified
                 by xpath string is added to DOM.

        Available options are:

        * ``visible`` (bool): wait for element to be present in DOM and to be
          visible, i.e. to not have ``display: none`` or ``visibility: hidden``
          CSS properties. Defaults to ``False``.
        * ``hidden`` (bool): wait for element to not be found in the DOM or to
          be hidden, i.e. have ``display: none`` or ``visibility: hidden`` CSS
          properties. Defaults to ``False``.
        * ``timeout`` (int|float): maximum time to wait for in milliseconds.
          Defaults to 30000 (30 seconds). Pass ``0`` to disable timeout.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.waitForXPath(xpath, options, **kwargs)

    def waitForFunction(
        self,
        pageFunction: Union[str, Callable[..., Any]],
        options: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> Awaitable[JSHandle]:
        """Wait until the function completes and returns a truthy value.

        :arg Any args: Arguments to pass to ``pageFunction``.
        :return: Return awaitable object which resolves when the
                 ``pageFunction`` returns a truthy value. It resolves to a
                 :class:`~pyppeteer.execution_context.JSHandle` of the truthy
                 value.

        This method accepts the following options:

        * ``polling`` (str|number): An interval at which the ``pageFunction``
          is executed, defaults to ``raf``. If ``polling`` is a number, then
          it is treated as an interval in milliseconds at which the function
          would be executed. If ``polling`` is a string, then it can be one of
          the following values:

          * ``raf``: to constantly execute ``pageFunction`` in
            ``requestAnimationFrame`` callback. This is the tightest polling
            mode which is suitable to observe styling changes.
          * ``mutation``: to execute ``pageFunction`` on every DOM mutation.

        * ``timeout`` (int|float): maximum time to wait for in milliseconds.
          Defaults to 30000 (30 seconds). Pass ``0`` to disable timeout.
        """
        frame: Optional[Frame] = self.mainFrame
        if not frame:
            raise PageError('no main frame.')
        return frame.waitForFunction(pageFunction, options, *args, **kwargs)


supportedMetrics: tuple = (
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
    'JSHeapTotalSize'
)
unitToPixels: Dict[str, float] = {'px': 1, 'in': 96, 'cm': 37.8, 'mm': 3.78}


def convertPrintParameterToInches(parameter: Optional[Union[int, float, str]]) -> Optional[float]:
    """Convert print parameter to inches."""
    if parameter is None:
        return None
    if isinstance(parameter, (int, float)):
        pixels: float = parameter
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


class ConsoleMessage:
    """Console message class.

    ConsoleMessage objects are dispatched by page via the ``console`` event.
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


async def craete(*args: Any, **kwargs: Any) -> 'Page':
    """[Deprecated] miss-spelled function.

    This function is undocumented and will be removed in future release.
    """
    logger.warning('`craete` function is deprecated and will be removed in future. Use `Page.create` instead.')
    return await Page.create(*args, **kwargs)
