class Target:
    """Browser's target class."""

    def __init__(self, targetInfo: Dict[str, Any], browserContext: 'BrowserContext', sessionFactory: Callable[[], Coroutine[Any, Any, CDPSession]], ignoreHTTPSErrors: bool, defaultViewport: Any, screenshotTaskQueue: Any, loop: asyncio.AbstractEventLoop):
        self._targetInfo: Dict[str, Any] = targetInfo
        self._browserContext: 'BrowserContext' = browserContext
        self._targetId: str = targetInfo.get('targetId', '')
        self._sessionFactory: Callable[[], Coroutine[Any, Any, CDPSession]] = sessionFactory
        self._ignoreHTTPSErrors: bool = ignoreHTTPSErrors
        self._defaultViewport: Any = defaultViewport
        self._screenshotTaskQueue: Any = screenshotTaskQueue
        self._loop: asyncio.AbstractEventLoop = loop
        self._page: Optional[Page] = None
        self._initializedPromise: asyncio.Future[bool] = self._loop.create_future()
        self._isClosedPromise: asyncio.Future[None] = self._loop.create_future()
        self._isInitialized: bool = self._targetInfo['type'] != 'page' or self._targetInfo['url'] != ''
        if self._isInitialized:
            self._initializedCallback(True)

    # ... rest of the code
