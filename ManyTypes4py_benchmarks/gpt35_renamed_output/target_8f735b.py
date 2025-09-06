from pyppeteer.browser import BrowserContext
from pyppeteer.connection import CDPSession
from pyppeteer.page import Page
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

if TYPE_CHECKING:
    from pyppeteer.browser import Browser

class Target:
    def __init__(self, targetInfo: Dict[str, Any], browserContext: BrowserContext, sessionFactory: Callable[[], Coroutine[Any, Any, CDPSession]], ignoreHTTPSErrors: bool, defaultViewport: Dict[str, Union[int, float]], screenshotTaskQueue: List[Any], loop: Any) -> None:
    
    def func_5yotugcc(self, bl: bool) -> None:
    
    def func_29x7qy3i(self) -> None:
    
    async def func_hzexnokf(self) -> CDPSession:
    
    async def func_bl9irmy1(self) -> Optional[Page]:
    
    @property
    def func_vagcgwix(self) -> str:
    
    @property
    def type(self) -> str:
    
    @property
    def func_7s9uplfb(self) -> Browser:
    
    @property
    def func_1rd27pbt(self) -> BrowserContext:
    
    @property
    def func_h7m0w59m(self) -> Optional['Target']:
    
    def func_a0zqlclz(self, targetInfo: Dict[str, Any]) -> None:
