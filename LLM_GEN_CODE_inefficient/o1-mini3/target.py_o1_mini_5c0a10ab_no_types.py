"""Target module."""
import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional
from typing import TYPE_CHECKING
from pyppeteer.connection import CDPSession
from pyppeteer.page import Page
if TYPE_CHECKING:
    from pyppeteer.browser import Browser, BrowserContext

class Target:
    """Browser's target class."""

    def __init__(self, targetInfo, browserContext, sessionFactory, ignoreHTTPSErrors, defaultViewport, screenshotTaskQueue, loop):
        self._targetInfo: Dict[str, Any] = targetInfo
        self._browserContext: 'BrowserContext' = browserContext
        self._targetId: str = targetInfo.get('targetId', '')
        self._sessionFactory: Callable[[], Coroutine[Any, Any, CDPSession]] = sessionFactory
        self._ignoreHTTPSErrors: bool = ignoreHTTPSErrors
        self._defaultViewport: Optional[Dict[str, Any]] = defaultViewport
        self._screenshotTaskQueue: List[Any] = screenshotTaskQueue
        self._loop: asyncio.AbstractEventLoop = loop
        self._page: Optional[Page] = None
        self._initializedPromise: asyncio.Future[bool] = self._loop.create_future()
        self._isClosedPromise: asyncio.Future[None] = self._loop.create_future()
        self._isInitialized: bool = self._targetInfo.get('type') != 'page' or self._targetInfo.get('url') != ''
        if self._isInitialized:
            self._initializedCallback(True)

    def _initializedCallback(self, bl):
        if self._initializedPromise.done():
            self._initializedPromise = self._loop.create_future()
        self._initializedPromise.set_result(bl)

    def _closedCallback(self):
        self._isClosedPromise.set_result(None)

    async def createCDPSession(self) -> CDPSession:
        """Create a Chrome Devtools Protocol session attached to the target."""
        return await self._sessionFactory()

    async def page(self) -> Optional[Page]:
        """Get page of this target.

        If the target is not of type "page" or "background_page", return
        ``None``.
        """
        target_type: str = self._targetInfo.get('type', '')
        if target_type in ['page', 'background_page'] and self._page is None:
            client: CDPSession = await self._sessionFactory()
            new_page: Page = await Page.create(client, self, self._ignoreHTTPSErrors, self._defaultViewport, self._screenshotTaskQueue)
            self._page = new_page
            return new_page
        return self._page

    @property
    def url(self):
        """Get url of this target."""
        return self._targetInfo.get('url', '')

    @property
    def type(self):
        """Get type of this target.

        Type can be ``'page'``, ``'background_page'``, ``'service_worker'``,
        ``'browser'``, or ``'other'``.
        """
        _type: str = self._targetInfo.get('type', '')
        if _type in ['page', 'background_page', 'service_worker', 'browser']:
            return _type
        return 'other'

    @property
    def browser(self):
        """Get the browser the target belongs to."""
        return self._browserContext.browser

    @property
    def browserContext(self):
        """Return the browser context the target belongs to."""
        return self._browserContext

    @property
    def opener(self):
        """Get the target that opened this target.

        Top-level targets return ``None``.
        """
        openerId: Optional[str] = self._targetInfo.get('openerId')
        if openerId is None:
            return None
        return self.browser._targets.get(openerId)

    def _targetInfoChanged(self, targetInfo):
        self._targetInfo = targetInfo
        if not self._isInitialized and (self._targetInfo.get('type') != 'page' or self._targetInfo.get('url') != ''):
            self._isInitialized = True
            self._initializedCallback(True)
            return