import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional
from typing import TYPE_CHECKING
from pyppeteer.connection import CDPSession
from pyppeteer.page import Page
if TYPE_CHECKING:
    from pyppeteer.browser import Browser, BrowserContext

class Target(object):
    def __init__(self, targetInfo: Dict[str, Any], browserContext: 'BrowserContext', sessionFactory: Callable[[], Coroutine[Any, Any, CDPSession]], ignoreHTTPSErrors: bool, defaultViewport: Dict[str, Any], screenshotTaskQueue: List[Any], loop: asyncio.AbstractEventLoop) -> None:
        self._targetInfo: Dict[str, Any] = targetInfo
        self._browserContext: 'BrowserContext' = browserContext
        self._targetId: str = targetInfo.get('targetId', '')
        self._sessionFactory: Callable[[], Coroutine[Any, Any, CDPSession]] = sessionFactory
        self._ignoreHTTPSErrors: bool = ignoreHTTPSErrors
        self._defaultViewport: Dict[str, Any] = defaultViewport
        self._screenshotTaskQueue: List[Any] = screenshotTaskQueue
        self._loop: asyncio.AbstractEventLoop = loop
        self._page: Optional[Page] = None
        self._initializedPromise: asyncio.Future = self._loop.create_future()
        self._isClosedPromise: asyncio.Future = self._loop.create_future()
        self._isInitialized: bool = self._targetInfo['type'] != 'page' or self._targetInfo['url'] != ''
        if self._isInitialized:
            self._initializedCallback(True)

    def _initializedCallback(self, bl: bool) -> None:
        if self._initializedPromise.done():
            self._initializedPromise = self._loop.create_future()
        self._initializedPromise.set_result(bl)

    def _closedCallback(self) -> None:
        self._isClosedPromise.set_result(None)

    async def createCDPSession(self) -> CDPSession:
        return await self._sessionFactory()

    async def page(self) -> Optional[Page]:
        if self._targetInfo['type'] in ['page', 'background_page'] and self._page is None:
            client = await self._sessionFactory()
            new_page = await Page.create(client, self, self._ignoreHTTPSErrors, self._defaultViewport, self._screenshotTaskQueue)
            self._page = new_page
            return new_page
        return self._page

    @property
    def url(self) -> str:
        return self._targetInfo['url']

    @property
    def type(self) -> str:
        _type: str = self._targetInfo['type']
        if _type in ['page', 'background_page', 'service_worker', 'browser']:
            return _type
        return 'other'

    @property
    def browser(self) -> 'Browser':
        return self._browserContext.browser

    @property
    def browserContext(self) -> 'BrowserContext':
        return self._browserContext

    @property
    def opener(self) -> Optional['Target']:
        openerId = self._targetInfo.get('openerId')
        if openerId is None:
            return None
        return self.browser._targets.get(openerId)

    def _targetInfoChanged(self, targetInfo: Dict[str, Any]) -> None:
        self._targetInfo = targetInfo
        if not self._isInitialized and (self._targetInfo['type'] != 'page' or self._targetInfo['url'] != ''):
            self._isInitialized = True
            self._initializedCallback(True)
            return
