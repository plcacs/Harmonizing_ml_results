"""Network Manager module."""

from asyncio import Future
from collections import OrderedDict
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from urllib.parse import unquote
from pyee import EventEmitter
from pyppeteer.connection import CDPSession
from pyppeteer.errors import NetworkError
from pyppeteer.frame_manager import Frame, FrameManager
from pyppeteer.multimap import Multimap
from types import SimpleNamespace

logger: logging.Logger = ...

class NetworkManager(EventEmitter):
    Events: SimpleNamespace = ...

    def __init__(self, client: CDPSession, frameManager: FrameManager) -> None:
        ...

    async def authenticate(self, credentials: dict) -> Awaitable[None]:
        ...

    async def setExtraHTTPHeaders(self, extraHTTPHeaders: dict) -> Awaitable[None]:
        ...

    def extraHTTPHeaders(self) -> dict:
        ...

    async def setOfflineMode(self, value: bool) -> Awaitable[None]:
        ...

    async def setUserAgent(self, userAgent: str) -> Awaitable[None]:
        ...

    async def setRequestInterception(self, value: bool) -> Awaitable[None]:
        ...

    async def _updateProtocolRequestInterception(self) -> None:
        ...

    async def _onRequestWillBeSent(self, event: dict) -> None:
        ...

    async def _send(self, method: str, msg: dict) -> None:
        ...

    def _onRequestIntercepted(self, event: dict) -> None:
        ...

    def _onRequest(self, event: dict, interceptionId: Optional[str]) -> None:
        ...

    def _onRequestServedFromCache(self, event: dict) -> None:
        ...

    def _handleRequestRedirect(
        self,
        request: Request,
        redirectStatus: int,
        redirectHeaders: dict,
        fromDiskCache: bool,
        fromServiceWorker: bool,
        securityDetails: Optional[dict] = None,
    ) -> None:
        ...

    def _handleRequestStart(
        self,
        requestId: str,
        interceptionId: Optional[str],
        url: str,
        isNavigationRequest: bool,
        resourceType: str,
        requestPayload: dict,
        frameId: str,
        redirectChain: List[Request],
    ) -> None:
        ...

    def _onResponseReceived(self, event: dict) -> None:
        ...

    def _onLoadingFinished(self, event: dict) -> None:
        ...

    def _onLoadingFailed(self, event: dict) -> None:
        ...

class Request:
    def __init__(
        self,
        client: CDPSession,
        requestId: str,
        interceptionId: Optional[str],
        isNavigationRequest: bool,
        allowInterception: bool,
        url: str,
        resourceType: str,
        payload: dict,
        frame: Optional[Frame],
        redirectChain: List[Request],
    ) -> None:
        ...

    @property
    def url(self) -> str:
        ...

    @property
    def resourceType(self) -> str:
        ...

    @property
    def method(self) -> str:
        ...

    @property
    def postData(self) -> Optional[str]:
        ...

    @property
    def headers(self) -> dict:
        ...

    @property
    def response(self) -> Optional[Response]:
        ...

    @property
    def frame(self) -> Optional[Frame]:
        ...

    def isNavigationRequest(self) -> bool:
        ...

    @property
    def redirectChain(self) -> List[Request]:
        ...

    @property
    def failure(self) -> Optional[dict]:
        ...

    async def continue_(self, overrides: Optional[dict] = None) -> Awaitable[None]:
        ...

    async def respond(self, response: dict) -> Awaitable[None]:
        ...

    async def abort(self, errorCode: str = "failed") -> Awaitable[None]:
        ...

class Response:
    def __init__(
        self,
        client: CDPSession,
        request: Request,
        status: int,
        headers: dict,
        fromDiskCache: bool,
        fromServiceWorker: bool,
        securityDetails: Optional[Union[SecurityDetails, dict]] = None,
    ) -> None:
        ...

    @property
    def url(self) -> str:
        ...

    @property
    def ok(self) -> bool:
        ...

    @property
    def status(self) -> int:
        ...

    @property
    def headers(self) -> dict:
        ...

    @property
    def securityDetails(self) -> Optional[SecurityDetails]:
        ...

    @property
    def request(self) -> Request:
        ...

    @property
    def fromCache(self) -> bool:
        ...

    @property
    def fromServiceWorker(self) -> bool:
        ...

    async def buffer(self) -> Future[bytes]:
        ...

    async def text(self) -> str:
        ...

    async def json(self) -> Any:
        ...

class SecurityDetails:
    def __init__(
        self,
        subjectName: str,
        issuer: str,
        validFrom: str,
        validTo: str,
        protocol: str,
    ) -> None:
        ...

    @property
    def subjectName(self) -> str:
        ...

    @property
    def issuer(self) -> str:
        ...

    @property
    def validFrom(self) -> str:
        ...

    @property
    def validTo(self) -> str:
        ...

    @property
    def protocol(self) -> str:
        ...

def generateRequestHash(request: dict) -> str:
    ...

statusTexts: Dict[str, str] = ...
errorReasons: Dict[str, str] = ...