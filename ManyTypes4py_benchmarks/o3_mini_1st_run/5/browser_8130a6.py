import logging
from subprocess import Popen
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional
from pyee import EventEmitter
from pyppeteer.connection import Connection
from pyppeteer.errors import BrowserError
from pyppeteer.page import Page
from pyppeteer.target import Target

logger = logging.getLogger(__name__)


class Browser(EventEmitter):
    Events = SimpleNamespace(
        TargetCreated='targetcreated',
        TargetDestroyed='targetdestroyed',
        TargetChanged='targetchanged',
        Disconnected='disconnected'
    )

    def __init__(
        self,
        connection: Connection,
        contextIds: List[str],
        ignoreHTTPSErrors: bool,
        defaultViewport: Optional[Dict[str, Any]],
        process: Optional[Popen] = None,
        closeCallback: Optional[Callable[[], Awaitable[None]]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self._ignoreHTTPSErrors: bool = ignoreHTTPSErrors
        self._defaultViewport: Optional[Dict[str, Any]] = defaultViewport
        self._process: Optional[Popen] = process
        self._screenshotTaskQueue: List[Any] = []
        self._connection: Connection = connection
        loop = self._connection._loop

        def _dummy_callback() -> Awaitable[None]:
            fut = loop.create_future()
            fut.set_result(None)
            return fut

        if closeCallback:
            self._closeCallback: Callable[[], Awaitable[None]] = closeCallback
        else:
            self._closeCallback = _dummy_callback
        self._defaultContext = BrowserContext(self, None)
        self._contexts: Dict[str, BrowserContext] = {}
        for contextId in contextIds:
            self._contexts[contextId] = BrowserContext(self, contextId)
        self._targets: Dict[str, Target] = {}
        self._connection.setClosedCallback(lambda: self.emit(Browser.Events.Disconnected))
        self._connection.on('Target.targetCreated', lambda event: loop.create_task(self._targetCreated(event)))
        self._connection.on('Target.targetDestroyed', lambda event: loop.create_task(self._targetDestroyed(event)))
        self._connection.on('Target.targetInfoChanged', lambda event: loop.create_task(self._targetInfoChanged(event)))

    @property
    def process(self) -> Optional[Popen]:
        """Return process of this browser.
        If browser instance is created by :func:`pyppeteer.launcher.connect`,
        return ``None``.
        """
        return self._process

    async def createIncogniteBrowserContext(self) -> BrowserContext:
        logger.warning('createIncogniteBrowserContext is deprecated. Use createIncognitoBrowserContext instead.')
        return await self.createIncognitoBrowserContext()

    async def createIncognitoBrowserContext(self) -> BrowserContext:
        obj: Dict[str, Any] = await self._connection.send('Target.createBrowserContext')
        browserContextId: str = obj['browserContextId']
        context: BrowserContext = BrowserContext(self, browserContextId)
        self._contexts[browserContextId] = context
        return context

    @property
    def browserContexts(self) -> List["BrowserContext"]:
        return [self._defaultContext] + [context for context in self._contexts.values()]

    async def _disposeContext(self, contextId: str) -> None:
        await self._connection.send('Target.disposeBrowserContext', {'browserContextId': contextId})
        self._contexts.pop(contextId, None)

    @staticmethod
    async def create(
        connection: Connection,
        contextIds: List[str],
        ignoreHTTPSErrors: bool,
        defaultViewport: Optional[Dict[str, Any]],
        process: Optional[Popen] = None,
        closeCallback: Optional[Callable[[], Awaitable[None]]] = None,
        **kwargs: Any
    ) -> "Browser":
        browser: Browser = Browser(connection, contextIds, ignoreHTTPSErrors, defaultViewport, process, closeCallback)
        await connection.send('Target.setDiscoverTargets', {'discover': True})
        return browser

    async def _targetCreated(self, event: Dict[str, Any]) -> None:
        targetInfo: Dict[str, Any] = event['targetInfo']
        browserContextId: Optional[str] = targetInfo.get('browserContextId')
        if browserContextId and browserContextId in self._contexts:
            context: BrowserContext = self._contexts[browserContextId]
        else:
            context = self._defaultContext
        target: Target = Target(
            targetInfo,
            context,
            lambda: self._connection.createSession(targetInfo),
            self._ignoreHTTPSErrors,
            self._defaultViewport,
            self._screenshotTaskQueue,
            self._connection._loop
        )
        if targetInfo['targetId'] in self._targets:
            raise BrowserError('target should not exist before create.')
        self._targets[targetInfo['targetId']] = target
        if await target._initializedPromise:
            self.emit(Browser.Events.TargetCreated, target)
            context.emit(BrowserContext.Events.TargetCreated, target)

    async def _targetDestroyed(self, event: Dict[str, Any]) -> None:
        target: Target = self._targets[event['targetId']]
        del self._targets[event['targetId']]
        target._closedCallback()
        if await target._initializedPromise:
            self.emit(Browser.Events.TargetDestroyed, target)
            target.browserContext.emit(BrowserContext.Events.TargetDestroyed, target)
        target._initializedCallback(False)

    async def _targetInfoChanged(self, event: Dict[str, Any]) -> None:
        target: Optional[Target] = self._targets.get(event['targetInfo']['targetId'])
        if not target:
            raise BrowserError('target should exist before targetInfoChanged')
        previousURL: str = target.url
        wasInitialized: bool = target._isInitialized
        target._targetInfoChanged(event['targetInfo'])
        if wasInitialized and previousURL != target.url:
            self.emit(Browser.Events.TargetChanged, target)
            target.browserContext.emit(BrowserContext.Events.TargetChanged, target)

    @property
    def wsEndpoint(self) -> str:
        return self._connection.url

    async def newPage(self) -> Page:
        return await self._defaultContext.newPage()

    async def _createPageInContext(self, contextId: Optional[str]) -> Page:
        options: Dict[str, Any] = {'url': 'about:blank'}
        if contextId:
            options['browserContextId'] = contextId
        response: Dict[str, Any] = await self._connection.send('Target.createTarget', options)
        targetId: Optional[str] = response.get('targetId')
        if targetId is None:
            raise BrowserError('Failed to create target for page.')
        target: Optional[Target] = self._targets.get(targetId)
        if target is None:
            raise BrowserError('Failed to create target for page.')
        if not await target._initializedPromise:
            raise BrowserError('Failed to create target for page.')
        page: Optional[Page] = await target.page()
        if page is None:
            raise BrowserError('Failed to create page.')
        return page

    def targets(self) -> List[Target]:
        return [target for target in self._targets.values() if target._isInitialized]

    async def pages(self) -> List[Page]:
        pages: List[Page] = []
        for context in self.browserContexts:
            pages.extend(await context.pages())
        return pages

    async def version(self) -> str:
        version_info: Dict[str, Any] = await self._getVersion()
        return version_info['product']

    async def userAgent(self) -> str:
        version_info: Dict[str, Any] = await self._getVersion()
        return version_info.get('userAgent', '')

    async def close(self) -> None:
        await self._closeCallback()

    async def disconnect(self) -> None:
        await self._connection.dispose()

    def _getVersion(self) -> Awaitable[Dict[str, Any]]:
        return self._connection.send('Browser.getVersion')


class BrowserContext(EventEmitter):
    Events = SimpleNamespace(
        TargetCreated='targetcreated',
        TargetDestroyed='targetdestroyed',
        TargetChanged='targetchanged'
    )

    def __init__(self, browser: Browser, contextId: Optional[str]) -> None:
        super().__init__()
        self._browser: Browser = browser
        self._id: Optional[str] = contextId

    def targets(self) -> List[Target]:
        targets: List[Target] = []
        for target in self._browser.targets():
            if target.browserContext == self:
                targets.append(target)
        return targets

    async def pages(self) -> List[Page]:
        pages: List[Page] = []
        for target in self.targets():
            if target.type == 'page':
                page: Optional[Page] = await target.page()
                if page:
                    pages.append(page)
        return pages

    def isIncognite(self) -> bool:
        logger.warning('isIncognite is deprecated. Use isIncognito instead.')
        return self.isIncognito()

    def isIncognito(self) -> bool:
        return bool(self._id)

    async def newPage(self) -> Page:
        return await self._browser._createPageInContext(self._id)

    @property
    def browser(self) -> Browser:
        return self._browser

    async def close(self) -> None:
        if self._id is None:
            raise BrowserError('Non-incognito profile cannot be closed')
        await self._browser._disposeContext(self._id)