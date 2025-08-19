from __future__ import annotations

"""Browser module."""
import logging
from asyncio import AbstractEventLoop
from subprocess import Popen
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pyee import EventEmitter
from pyppeteer.connection import Connection
from pyppeteer.errors import BrowserError
from pyppeteer.page import Page
from pyppeteer.target import Target

logger: logging.Logger = logging.getLogger(__name__)


class Browser(EventEmitter):
    """Browser class.

    A Browser object is created when pyppeteer connects to chrome, either
    through :func:`~pyppeteer.launcher.launch` or
    :func:`~pyppeteer.launcher.connect`.
    """

    Events: SimpleNamespace = SimpleNamespace(
        TargetCreated='targetcreated',
        TargetDestroyed='targetdestroyed',
        TargetChanged='targetchanged',
        Disconnected='disconnected',
    )

    def __init__(
        self,
        connection: Connection,
        contextIds: List[str],
        ignoreHTTPSErrors: bool,
        defaultViewport: Optional[Dict[str, Any]],
        process: Optional[Popen] = None,
        closeCallback: Optional[Callable[[], Awaitable[None]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._ignoreHTTPSErrors: bool = ignoreHTTPSErrors
        self._defaultViewport: Optional[Dict[str, Any]] = defaultViewport
        self._process: Optional[Popen] = process
        self._screenshotTaskQueue: List[Any] = []
        self._connection: Connection = connection
        loop: AbstractEventLoop = self._connection._loop

        def _dummy_callback() -> Awaitable[None]:
            fut = loop.create_future()
            fut.set_result(None)
            return fut

        if closeCallback:
            self._closeCallback: Callable[[], Awaitable[None]] = closeCallback
        else:
            self._closeCallback = _dummy_callback

        self._defaultContext: BrowserContext = BrowserContext(self, None)
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

    async def createIncogniteBrowserContext(self) -> 'BrowserContext':
        """[Deprecated] Miss spelled method.

        Use :meth:`createIncognitoBrowserContext` method instead.
        """
        logger.warning('createIncogniteBrowserContext is deprecated. Use createIncognitoBrowserContext instead.')
        return await self.createIncognitoBrowserContext()

    async def createIncognitoBrowserContext(self) -> 'BrowserContext':
        """Create a new incognito browser context.

        This won't share cookies/cache with other browser contexts.

        .. code::

            browser = await launch()
            # Create a new incognito browser context.
            context = await browser.createIncognitoBrowserContext()
            # Create a new page in a pristine context.
            page = await context.newPage()
            # Do stuff
            await page.goto('https://example.com')
            ...
        """
        obj: Dict[str, Any] = await self._connection.send('Target.createBrowserContext')
        browserContextId: str = obj['browserContextId']
        context = BrowserContext(self, browserContextId)
        self._contexts[browserContextId] = context
        return context

    @property
    def browserContexts(self) -> List['BrowserContext']:
        """Return a list of all open browser contexts.

        In a newly created browser, this will return a single instance of
        ``[BrowserContext]``
        """
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
        **kwargs: Any,
    ) -> 'Browser':
        """Create browser object."""
        browser = Browser(connection, contextIds, ignoreHTTPSErrors, defaultViewport, process, closeCallback)
        await connection.send('Target.setDiscoverTargets', {'discover': True})
        return browser

    async def _targetCreated(self, event: Dict[str, Any]) -> None:
        targetInfo: Dict[str, Any] = event['targetInfo']
        browserContextId: Optional[str] = targetInfo.get('browserContextId')
        if browserContextId and browserContextId in self._contexts:
            context = self._contexts[browserContextId]
        else:
            context = self._defaultContext
        target = Target(
            targetInfo,
            context,
            lambda: self._connection.createSession(targetInfo),
            self._ignoreHTTPSErrors,
            self._defaultViewport,
            self._screenshotTaskQueue,
            self._connection._loop,
        )
        if targetInfo['targetId'] in self._targets:
            raise BrowserError('target should not exist before create.')
        self._targets[targetInfo['targetId']] = target
        if await target._initializedPromise:  # type: ignore[attr-defined]
            self.emit(Browser.Events.TargetCreated, target)
            context.emit(BrowserContext.Events.TargetCreated, target)

    async def _targetDestroyed(self, event: Dict[str, Any]) -> None:
        target = self._targets[event['targetId']]
        del self._targets[event['targetId']]
        target._closedCallback()  # type: ignore[attr-defined]
        if await target._initializedPromise:  # type: ignore[attr-defined]
            self.emit(Browser.Events.TargetDestroyed, target)
            target.browserContext.emit(BrowserContext.Events.TargetDestroyed, target)  # type: ignore[attr-defined]
        target._initializedCallback(False)  # type: ignore[attr-defined]

    async def _targetInfoChanged(self, event: Dict[str, Any]) -> None:
        targetId: str = event['targetInfo']['targetId']
        target = self._targets.get(targetId)
        if not target:
            raise BrowserError('target should exist before targetInfoChanged')
        previousURL: str = target.url  # type: ignore[attr-defined]
        wasInitialized: bool = target._isInitialized  # type: ignore[attr-defined]
        target._targetInfoChanged(event['targetInfo'])  # type: ignore[attr-defined]
        if wasInitialized and previousURL != target.url:  # type: ignore[attr-defined]
            self.emit(Browser.Events.TargetChanged, target)
            target.browserContext.emit(BrowserContext.Events.TargetChanged, target)  # type: ignore[attr-defined]

    @property
    def wsEndpoint(self) -> str:
        """Return websocket end point url."""
        return self._connection.url

    async def newPage(self) -> Page:
        """Make new page on this browser and return its object."""
        return await self._defaultContext.newPage()

    async def _createPageInContext(self, contextId: Optional[str]) -> Page:
        options: Dict[str, Any] = {'url': 'about:blank'}
        if contextId:
            options['browserContextId'] = contextId
        targetId = (await self._connection.send('Target.createTarget', options)).get('targetId')
        target = self._targets.get(targetId)
        if target is None:
            raise BrowserError('Failed to create target for page.')
        if not await target._initializedPromise:  # type: ignore[attr-defined]
            raise BrowserError('Failed to create target for page.')
        page = await target.page()  # type: ignore[attr-defined]
        if page is None:
            raise BrowserError('Failed to create page.')
        return page

    def targets(self) -> List[Target]:
        """Get a list of all active targets inside the browser.

        In case of multiple browser contexts, the method will return a list
        with all the targets in all browser contexts.
        """
        return [target for target in self._targets.values() if target._isInitialized]  # type: ignore[attr-defined]

    async def pages(self) -> List[Page]:
        """Get all pages of this browser.

        Non visible pages, such as ``"background_page"``, will not be listed
        here. You can find then using :meth:`pyppeteer.target.Target.page`.

        In case of multiple browser contexts, this method will return a list
        with all the pages in all browser contexts.
        """
        pages: List[Page] = []
        for context in self.browserContexts:
            pages.extend(await context.pages())
        return pages

    async def version(self) -> str:
        """Get version of the browser."""
        version = await self._getVersion()
        return version['product']

    async def userAgent(self) -> str:
        """Return browser's original user agent.

        .. note::
            Pages can override browser user agent with
            :meth:`pyppeteer.page.Page.setUserAgent`.
        """
        version = await self._getVersion()
        return version.get('userAgent', '')

    async def close(self) -> None:
        """Close connections and terminate browser process."""
        await self._closeCallback()

    async def disconnect(self) -> None:
        """Disconnect browser."""
        await self._connection.dispose()

    def _getVersion(self) -> Awaitable[Dict[str, Any]]:
        return self._connection.send('Browser.getVersion')


class BrowserContext(EventEmitter):
    """BrowserContext provides multiple independent browser sessions.

    When a browser is launched, it has a single BrowserContext used by default.
    The method `browser.newPage()` creates a page in the default browser
    context.

    If a page opens another page, e.g. with a ``window.open`` call, the popup
    will belong to the parent page's browser context.

    Pyppeteer allows creation of "incognito" browser context with
    ``browser.createIncognitoBrowserContext()`` method.
    "incognito" browser contexts don't write any browser data to disk.

    .. code::

        # Create new incognito browser context
        context = await browser.createIncognitoBrowserContext()
        # Create a new page inside context
        page = await context.newPage()
        # ... do stuff with page ...
        await page.goto('https://example.com')
        # Dispose context once it's no longer needed
        await context.close()
    """

    Events: SimpleNamespace = SimpleNamespace(
        TargetCreated='targetcreated',
        TargetDestroyed='targetdestroyed',
        TargetChanged='targetchanged',
    )

    def __init__(self, browser: Browser, contextId: Optional[str]) -> None:
        super().__init__()
        self._browser: Browser = browser
        self._id: Optional[str] = contextId

    def targets(self) -> List[Target]:
        """Return a list of all active targets inside the browser context."""
        targets: List[Target] = []
        for target in self._browser.targets():
            if target.browserContext == self:  # type: ignore[attr-defined]
                targets.append(target)
        return targets

    async def pages(self) -> List[Page]:
        """Return list of all open pages.

        Non-visible pages, such as ``"background_page"``, will not be listed
        here. You can find them using :meth:`pyppeteer.target.Target.page`.
        """
        pages: List[Page] = []
        for target in self.targets():
            if target.type == 'page':  # type: ignore[attr-defined]
                page: Optional[Page] = await target.page()  # type: ignore[attr-defined]
                if page:
                    pages.append(page)
        return pages

    def isIncognite(self) -> bool:
        """[Deprecated] Miss spelled method.

        Use :meth:`isIncognito` method instead.
        """
        logger.warning('isIncognite is deprecated. Use isIncognito instead.')
        return self.isIncognito()

    def isIncognito(self) -> bool:
        """Return whether BrowserContext is incognito.

        The default browser context is the only non-incognito browser context.

        .. note::
            The default browser context cannot be closed.
        """
        return bool(self._id)

    async def newPage(self) -> Page:
        """Create a new page in the browser context."""
        return await self._browser._createPageInContext(self._id)

    @property
    def browser(self) -> Browser:
        """Return the browser this browser context belongs to."""
        return self._browser

    async def close(self) -> None:
        """Close the browser context.

        All the targets that belongs to the browser context will be closed.

        .. note::
            Only incognito browser context can be closed.
        """
        if self._id is None:
            raise BrowserError('Non-incognito profile cannot be closed')
        await self._browser._disposeContext(self._id)