"""Network Manager module."""
import asyncio
from collections import OrderedDict
from typing import (
    Awaitable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    Any,
    Tuple,
    FrozenSet,
    TypeVar,
    overload,
)
from urllib.parse import unquote
from pyee import EventEmitter
from pyppeteer.connection import CDPSession
from pyppeteer.frame_manager import FrameManager, Frame
from pyppeteer.errors import NetworkError
from pyppeteer.helper import debugError
from pyppeteer.multimap import Multimap

_T = TypeVar('_T')

class NetworkManager(EventEmitter):
    class Events:
        Request: str
        Response: str
        RequestFailed: str
        RequestFinished: str

    def __init__(self, client: CDPSession, frameManager: FrameManager) -> None:
        ...

    async def authenticate(self, credentials: Dict[str, str]) -> None:
        ...

    async def setExtraHTTPHeaders(self, extraHTTPHeaders: Dict[str, str]) -> None:
        ...

    def extraHTTPHeaders(self) -> Dict[str, str]:
        ...

    async def setOfflineMode(self, value: bool) -> None:
        ...

    async def setUserAgent(self, userAgent: str) -> None:
        ...

    async def setRequestInterception(self, value: bool) -> None:
        ...

    async def _updateProtocolRequestInterception(self) -> None:
        ...

    async def _onRequestWillBeSent(self, event: Dict[str, Any]) -> None:
        ...

    async def _send(self, method: str, msg: Dict[str, Any]) -> None:
        ...

    def _onRequestIntercepted(self, event: Dict[str, Any]) -> None:
        ...

    def _onRequest(self, event: Dict[str, Any], interceptionId: Optional[str]) -> None:
        ...

    def _onRequestServedFromCache(self, event: Dict[str, Any]) -> None:
        ...

    def _handleRequestRedirect(
        self,
        request: 'Request',
        redirectStatus: int,
        redirectHeaders: Dict[str, str],
        fromDiskCache: bool,
        fromServiceWorker: bool,
        securityDetails: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    def _handleRequestStart(
        self,
        requestId: str,
        interceptionId: Optional[str],
        url: str,
        isNavigationRequest: bool,
        resourceType: str,
        requestPayload: Dict[str, Any],
        frameId: Optional[str],
        redirectChain: List['Request'],
    ) -> None:
        ...

    def _onResponseReceived(self, event: Dict[str, Any]) -> None:
        ...

    def _onLoadingFinished(self, event: Dict[str, Any]) -> None:
        ...

    def _onLoadingFailed(self, event: Dict[str, Any]) -> None:
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
        payload: Dict[str, Any],
        frame: Optional[Frame],
        redirectChain: List['Request'],
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
    def headers(self) -> Dict[str, str]:
        ...

    @property
    def response(self) -> Optional['Response']:
        ...

    @property
    def frame(self) -> Optional[Frame]:
        ...

    def isNavigationRequest(self) -> bool:
        ...

    @property
    def redirectChain(self) -> List['Request']:
        ...

    def failure(self) -> Optional[Dict[str, str]]:
        ...

    async def continue_(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        ...

    async def respond(self, response: Dict[str, Any]) -> None:
        ...

    async def abort(self, errorCode: str = 'failed') -> None:
        ...

class Response:
    def __init__(
        self,
        client: CDPSession,
        request: Request,
        status: int,
        headers: Dict[str, str],
        fromDiskCache: bool,
        fromServiceWorker: bool,
        securityDetails: Optional['SecurityDetails'] = None,
    ) -> None:
        ...

    def _bodyLoadedPromiseFulfill(self, value: Optional[Exception]) -> None:
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
    def headers(self) -> Dict[str, str]:
        ...

    @property
    def securityDetails(self) -> Optional['SecurityDetails']:
        ...

    async def _bufread(self) -> bytes:
        ...

    def buffer(self) -> Awaitable[bytes]:
        ...

    async def text(self) -> str:
        ...

    async def json(self) -> Dict[str, Any]:
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

def generateRequestHash(request: Dict[str, Any]) -> str:
    ...

class SecurityDetails:
    def __init__(
        self,
        subjectName: str,
        issuer: str,
        validFrom: int,
        validTo: int,
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
    def validFrom(self) -> int:
        ...

    @property
    def validTo(self) -> int:
        ...

    @property
    def protocol(self) -> str:
        ...

errorReasons: Dict[str, str] = ...
statusTexts: Dict[str, str] = ...