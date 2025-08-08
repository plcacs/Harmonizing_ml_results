from pyppeteer.element_handle import ElementHandle
from pyppeteer.frame_manager import Frame

class ExecutionContext:
    def __init__(self, client: CDPSession, contextPayload: Dict[str, Any], objectHandleFactory: Any, frame: Optional[Frame] = None) -> None:
    def frame(self) -> Optional[Frame]:
    async def evaluate(self, pageFunction: str, *args: Any, force_expr: bool = False) -> Any:
    async def evaluateHandle(self, pageFunction: str, *args: Any, force_expr: bool = False) -> Any:
    def _convertArgument(self, arg: Any) -> Dict[str, Any]:
    async def queryObjects(self, prototypeHandle: ElementHandle) -> Any:

class JSHandle:
    def __init__(self, context: ExecutionContext, client: CDPSession, remoteObject: Dict[str, Any]) -> None:
    def executionContext(self) -> ExecutionContext:
    async def getProperty(self, propertyName: str) -> Any:
    async def getProperties(self) -> Dict[str, Any]:
    async def jsonValue(self) -> Any:
    def asElement(self) -> Optional[ElementHandle]:
    async def dispose(self) -> None:
    def toString(self) -> str:

def _rewriteError(error: Exception) -> None:
