from pyppeteer.connection import CDPSession
from pyppeteer.errors import NetworkError
from pyppeteer.frame_manager import FrameManager, Frame
from pyppeteer.helper import debugError
from pyppeteer.multimap import Multimap

class NetworkManager(EventEmitter):
    Events: SimpleNamespace = SimpleNamespace(Request='request', Response='response', RequestFailed='requestfailed', RequestFinished='requestfinished')

    def __init__(self, client: CDPSession, frameManager: FrameManager) -> None:
        super().__init__()
        self._client: CDPSession = client
        self._frameManager: FrameManager = frameManager
        self._requestIdToRequest: Dict[str, Request] = dict()
        self._requestIdToResponseWillBeSent: Dict[str, dict] = dict()
        self._extraHTTPHeaders: OrderedDict = OrderedDict()
        self._offline: bool = False
        self._credentials: Optional[Dict[str, str]] = None
        self._attemptedAuthentications: Set[str] = set()
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

    async def authenticate(self, credentials: Dict[str, str]) -> None:
        self._credentials = credentials
        await self._updateProtocolRequestInterception()

    async def setExtraHTTPHeaders(self, extraHTTPHeaders: Dict[str, str]) -> None:
        self._extraHTTPHeaders = OrderedDict()
        for k, v in extraHTTPHeaders.items():
            if not isinstance(v, str):
                raise TypeError(f'Expected value of header "{k}" to be string, but {type(v)} is found.')
            self._extraHTTPHeaders[k.lower()] = v
        await self._client.send('Network.setExtraHTTPHeaders', {'headers': self._extraHTTPHeaders})

    def extraHTTPHeaders(self) -> Dict[str, str]:
        return dict(**self._extraHTTPHeaders)

    async def setOfflineMode(self, value: bool) -> None:
        if self._offline == value:
            return
        self._offline = value
        await self._client.send('Network.emulateNetworkConditions', {'offline': self._offline, 'latency': 0, 'downloadThroughput': -1, 'uploadThroughput': -1})

    async def setUserAgent(self, userAgent: str) -> None:
        await self._client.send('Network.setUserAgentOverride', {'userAgent': userAgent})

    async def setRequestInterception(self, value: bool) -> None:
        self._userRequestInterceptionEnabled = value
        await self._updateProtocolRequestInterception()

    async def _updateProtocolRequestInterception(self) -> None:
        enabled: bool = self._userRequestInterceptionEnabled or bool(self._credentials)
        if enabled == self._protocolRequestInterceptionEnabled:
            return
        self._protocolRequestInterceptionEnabled = enabled
        patterns = [{'urlPattern': '*'}] if enabled else []
        await asyncio.gather(self._client.send('Network.setCacheDisabled', {'cacheDisabled': enabled}), self._client.send('Network.setRequestInterception', {'patterns': patterns}))

    async def _onRequestWillBeSent(self, event: dict) -> None:
        if self._protocolRequestInterceptionEnabled:
            requestHash = generateRequestHash(event.get('request', {}))
            interceptionId = self._requestHashToInterceptionIds.firstValue(requestHash)
            if interceptionId:
                self._onRequest(event, interceptionId)
                self._requestHashToInterceptionIds.delete(requestHash, interceptionId)
            else:
                self._requestHashToRequestIds.set(requestHash, event.get('requestId'))
                self._requestIdToResponseWillBeSent[event.get('requestId')] = event
            return
        self._onRequest(event, None)

    async def _send(self, method: str, msg: dict) -> None:
        try:
            await self._client.send(method, msg)
        except Exception as e:
            debugError(logger, e)

    def _onRequestIntercepted(self, event: dict) -> None:
        if event.get('authChallenge'):
            response = 'Default'
            if event['interceptionId'] in self._attemptedAuthentications:
                response = 'CancelAuth'
            elif self._credentials:
                response = 'ProvideCredentials'
                self._attemptedAuthentications.add(event['interceptionId'])
            username = getattr(self, '_credentials', {}).get('username')
            password = getattr(self, '_credentials', {}).get('password')
            self._client._loop.create_task(self._send('Network.continueInterceptedRequest', {'interceptionId': event['interceptionId'], 'authChallengeResponse': {'response': response, 'username': username, 'password': password}}))
            return
        if not self._userRequestInterceptionEnabled and self._protocolRequestInterceptionEnabled:
            self._client._loop.create_task(self._send('Network.continueInterceptedRequest', {'interceptionId': event['interceptionId']}))
        requestHash = generateRequestHash(event['request'])
        requestId = self._requestHashToRequestIds.firstValue(requestHash)
        if requestId:
            requestWillBeSentEvent = self._requestIdToResponseWillBeSent[requestId]
            self._onRequest(requestWillBeSentEvent, event.get('interceptionId'))
            self._requestHashToRequestIds.delete(requestHash, requestId)
            self._requestIdToResponseWillBeSent.pop(requestId, None)
        else:
            self._requestHashToInterceptionIds.set(requestHash, event['interceptionId'])

    def _onRequest(self, event: dict, interceptionId: Optional[str]) -> None:
        redirectChain: List[Request] = list()
        if event.get('redirectResponse'):
            request = self._requestIdToRequest.get(event['requestId'])
            if request:
                redirectResponse = event['redirectResponse']
                self._handleRequestRedirect(request, redirectResponse.get('status'), redirectResponse.get('headers'), redirectResponse.get('fromDiskCache'), redirectResponse.get('fromServiceWorker'), redirectResponse.get('SecurityDetails'))
                redirectChain = request._redirectChain
        isNavigationRequest: bool = bool(event.get('requestId') == event.get('loaderId') and event.get('type') == 'Document')
        self._handleRequestStart(event['requestId'], interceptionId, event.get('request', {}).get('url'), isNavigationRequest, event.get('type', ''), event.get('request', {}), event.get('frameId'), redirectChain)

    def _onRequestServedFromCache(self, event: dict) -> None:
        request = self._requestIdToRequest.get(event.get('requestId'))
        if request:
            request._fromMemoryCache = True

    def _handleRequestRedirect(self, request: Request, redirectStatus: int, redirectHeaders: dict, fromDiskCache: bool, fromServiceWorker: bool, securityDetails: Optional[dict] = None) -> None:
        response = Response(self._client, request, redirectStatus, redirectHeaders, fromDiskCache, fromServiceWorker, securityDetails)
        request._response = response
        request._redirectChain.append(request)
        response._bodyLoadedPromiseFulfill(NetworkError('Response body is unavailable for redirect response'))
        self._requestIdToRequest.pop(request._requestId, None)
        self._attemptedAuthentications.discard(request._interceptionId)
        self.emit(NetworkManager.Events.Response, response)
        self.emit(NetworkManager.Events.RequestFinished, request)

    def _handleRequestStart(self, requestId: str, interceptionId: Optional[str], url: str, isNavigationRequest: bool, resourceType: str, requestPayload: dict, frameId: str, redirectChain: List[Request]) -> None:
        frame = None
        if frameId and self._frameManager is not None:
            frame = self._frameManager.frame(frameId)
        request = Request(self._client, requestId, interceptionId, isNavigationRequest, self._userRequestInterceptionEnabled, url, resourceType, requestPayload, frame, redirectChain)
        self._requestIdToRequest[requestId] = request
        self.emit(NetworkManager.Events.Request, request)

    def _onResponseReceived(self, event: dict) -> None:
        request = self._requestIdToRequest.get(event['requestId'])
        if not request:
            return
        _resp = event.get('response', {})
        response = Response(self._client, request, _resp.get('status', 0), _resp.get('headers', {}), _resp.get('fromDiskCache'), _resp.get('fromServiceWorker'), _resp.get('securityDetails'))
        request._response = response
        self.emit(NetworkManager.Events.Response, response)

    def _onLoadingFinished(self, event: dict) -> None:
        request = self._requestIdToRequest.get(event['requestId'])
        if not request:
            return
        response = request.response
        if response:
            response._bodyLoadedPromiseFulfill(None)
        self._requestIdToRequest.pop(request._requestId, None)
        self._attemptedAuthentications.discard(request._interceptionId)
        self.emit(NetworkManager.Events.RequestFinished, request)

    def _onLoadingFailed(self, event: dict) -> None:
        request = self._requestIdToRequest.get(event['requestId'])
        if not request:
            return
        request._failureText = event.get('errorText')
        response = request.response
        if response:
            response._bodyLoadedPromiseFulfill(None)
        self._requestIdToRequest.pop(request._requestId, None)
        self._attemptedAuthentications.discard(request._interceptionId)
        self.emit(NetworkManager.Events.RequestFailed, request)

class Request:
    def __init__(self, client: CDPSession, requestId: str, interceptionId: Optional[str], isNavigationRequest: bool, allowInterception: bool, url: str, resourceType: str, payload: dict, frame: Optional[Frame], redirectChain: List[Request]) -> None:
        self._client: CDPSession = client
        self._requestId: str = requestId
        self._isNavigationRequest: bool = isNavigationRequest
        self._interceptionId: Optional[str] = interceptionId
        self._allowInterception: bool = allowInterception
        self._interceptionHandled: bool = False
        self._response: Optional[Response] = None
        self._failureText: Optional[str] = None
        self._url: str = url
        self._resourceType: str = resourceType.lower()
        self._method: Optional[str] = payload.get('method')
        self._postData: Optional[str] = payload.get('postData')
        headers: dict = payload.get('headers', {})
        self._headers: Dict[str, str] = {k.lower(): v for k, v in headers.items()}
        self._frame: Optional[Frame] = frame
        self._redirectChain: List[Request] = redirectChain
        self._fromMemoryCache: bool = False

    def isNavigationRequest(self) -> bool:
        return self._isNavigationRequest

class Response:
    def __init__(self, client: CDPSession, request: Request, status: int, headers: dict, fromDiskCache: bool, fromServiceWorker: bool, securityDetails: Optional[dict] = None) -> None:
        self._client: CDPSession = client
        self._request: Request = request
        self._status: int = status
        self._contentPromise: Awaitable = self._client._loop.create_future()
        self._bodyLoadedPromise: Awaitable = self._client._loop.create_future()
        self._url: str = request.url
        self._fromDiskCache: bool = fromDiskCache
        self._fromServiceWorker: bool = fromServiceWorker
        self._headers: Dict[str, str] = {k.lower(): v for k, v in headers.items()}
        self._securityDetails: SecurityDetails = SecurityDetails(securityDetails['subjectName'], securityDetails['issuer'], securityDetails['validFrom'], securityDetails['validTo'], securityDetails['protocol'])

    async def _bufread(self) -> bytes:
        result = await self._bodyLoadedPromise
        if isinstance(result, Exception):
            raise result
        response = await self._client.send('Network.getResponseBody', {'requestId': self._request._requestId})
        body = response.get('body', b'')
        if response.get('base64Encoded'):
            return base64.b64decode(body)
        return body

    def buffer(self) -> Awaitable:
        if not self._contentPromise.done():
            return self._client._loop.create_task(self._bufread())
        return self._contentPromise

    async def text(self) -> str:
        content = await self.buffer()
        if isinstance(content, str):
            return content
        else:
            return content.decode('utf-8')

    async def json(self) -> dict:
        content = await self.text()
        return json.loads(content)

class SecurityDetails:
    def __init__(self, subjectName: str, issuer: str, validFrom: float, validTo: float, protocol: str) -> None:
        self._subjectName: str = subjectName
        self._issuer: str = issuer
        self._validFrom: float = validFrom
        self._validTo: float = validTo
        self._protocol: str = protocol
