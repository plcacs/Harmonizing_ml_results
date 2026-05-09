import asyncio
import base64
from collections import OrderedDict
import copy
import json
import logging
from types import SimpleNamespace
from typing import Awaitable, Dict, List, Optional, Union, TYPE_CHECKING
from urllib.parse import unquote
from pyee import EventEmitter
from pyppeteer.connection import CDPSession
from pyppeteer.errors import NetworkError
from pyppeteer.frame_manager import FrameManager, Frame
from pyppeteer.helper import debugError
from pyppeteer.multimap import Multimap

class NetworkManager(EventEmitter):
    """NetworkManager class."""
    Events: SimpleNamespace(Request='request', Response='response', RequestFailed='requestfailed', RequestFinished='requestfinished')

    def __init__(self, client: CDPSession, frameManager: FrameManager) -> None:
        """Make new NetworkManager."""
        super().__init__()
        self._client: CDPSession = client
        self._frameManager: FrameManager = frameManager
        self._requestIdToRequest: Dict[str, Request] = {}
        self._requestIdToResponseWillBeSent: Dict[str, Awaitable] = {}
        self._extraHTTPHeaders: OrderedDict = OrderedDict()
        self._offline: bool = False
        self._credentials: Optional[Dict] = None
        self._attemptedAuthentications: Set = set()
        self._userRequestInterceptionEnabled: bool = False
        self._protocolRequestInterceptionEnabled: bool = False
        self._requestHashToRequestIds: Multimap = Multimap()
        self._requestHashToInterceptionIds: Multimap = Multimap()
        self._client.on('Network.requestWillBeSent', lambda event: self._client._loop.create_task(self._onRequestWillBeSent(event)))
        self._client.on('Network.requestIntercepted', self._onRequestIntercepted)
        self._client.on('Network.requestServedFromCache', self._onRequestServedFromCache)
        self._client.on('Network.responseReceived', self._onResponseReceived)
        self._client.on('Network.loadingFinished', self._onLoadingFinished)
        self._client.on('Network.loadingFailed', self._onLoadingFailed)

    # ... rest of the code ...

class Request(object):
    """Request class."""

    def __init__(self, client: CDPSession, requestId: str, interceptionId: str, isNavigationRequest: bool, allowInterception: bool, url: str, resourceType: str, payload: Dict, frame: Frame, redirectChain: List[Request]) -> None:
        # ... rest of the code ...

class Response(object):
    """Response class represents responses which are received by ``Page``."""

    def __init__(self, client: CDPSession, request: Request, status: int, headers: Dict, fromDiskCache: bool, fromServiceWorker: bool, securityDetails: Optional[Dict] = None) -> None:
        # ... rest of the code ...

class SecurityDetails(object):
    """Class represents responses which are received by page."""

    def __init__(self, subjectName: str, issuer: str, validFrom: int, validTo: int, protocol: str) -> None:
        # ... rest of the code ...
