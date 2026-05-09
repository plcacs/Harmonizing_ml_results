from typing import Any, Awaitable, Callable, Dict, List, Optional
from pyee import EventEmitter
from pyppeteer.connection import Connection
from pyppeteer.errors import BrowserError
from pyppeteer.page import Page
from pyppeteer.target import Target
from pyppeteer.browser import Browser
from pyppeteer.browser_context import BrowserContext

class Browser(EventEmitter):
    """Browser class.

    A Browser object is created when pyppeteer connects to chrome, either
    through :func:`~pyppeteer.launcher.launch` or
    :func:`~pyppeteer.launcher.connect`.
    """
    Events: SimpleNamespace = SimpleNamespace(TargetCreated='targetcreated', TargetDestroyed='targetdestroyed', TargetChanged='targetchanged', Disconnected='disconnected')

    def __init__(self, connection: Connection, contextIds: List[str], ignoreHTTPSErrors: bool, defaultViewport: Any, process: Optional[Popen] = None, closeCallback: Optional[Callable[[], Awaitable[None]]] = None, **kwargs: Any) -> None:
        super().__init__()
        self._ignoreHTTPSErrors: bool = ignoreHTTPSErrors
        self._defaultViewport: Any = defaultViewport
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
            self._closeCallback: Callable[[], Awaitable[None]] = _dummy_callback
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

    # ... rest of the class ...

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
    Events: SimpleNamespace = SimpleNamespace(TargetCreated='targetcreated', TargetDestroyed='targetdestroyed', TargetChanged='targetchanged')

    def __init__(self, browser: Browser, contextId: str) -> None:
        super().__init__()
        self._browser: Browser = browser
        self._id: str = contextId

    # ... rest of the class ...
