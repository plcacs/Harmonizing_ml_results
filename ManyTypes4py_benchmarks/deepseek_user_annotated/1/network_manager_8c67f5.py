#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Network Manager module."""

import asyncio
import base64
from collections import OrderedDict
import copy
import json
import logging
from types import SimpleNamespace
from typing import Awaitable, Dict, List, Optional, Union, TYPE_CHECKING, Any, Set, cast
from urllib.parse import unquote

from pyee import EventEmitter

from pyppeteer.connection import CDPSession
from pyppeteer.errors import NetworkError
from pyppeteer.frame_manager import FrameManager, Frame
from pyppeteer.helper import debugError
from pyppeteer.multimap import Multimap

if TYPE_CHECKING:
    from typing_extensions import TypedDict

    class RequestWillBeSentEvent(TypedDict, total=False):
        requestId: str
        request: Dict[str, Any]
        loaderId: str
        type: str
        frameId: str
        redirectResponse: Dict[str, Any]

    class ResponseReceivedEvent(TypedDict, total=False):
        requestId: str
        response: Dict[str, Any]

    class LoadingFinishedEvent(TypedDict, total=False):
        requestId: str
        errorText: str

    class LoadingFailedEvent(TypedDict, total=False):
        requestId: str
        errorText: str

    class RequestInterceptedEvent(TypedDict, total=False):
        interceptionId: str
        request: Dict[str, Any]
        authChallenge: Dict[str, Any]

    class RequestServedFromCacheEvent(TypedDict, total=False):
        requestId: str

logger = logging.getLogger(__name__)


class NetworkManager(EventEmitter):
    """NetworkManager class."""

    Events = SimpleNamespace(
        Request='request',
        Response='response',
        RequestFailed='requestfailed',
        RequestFinished='requestfinished',
    )

    def __init__(self, client: CDPSession, frameManager: FrameManager) -> None:
        """Make new NetworkManager."""
        super().__init__()
        self._client: CDPSession = client
        self._frameManager: FrameManager = frameManager
        self._requestIdToRequest: Dict[str, 'Request'] = dict()
        self._requestIdToResponseWillBeSent: Dict[str, Dict[str, Any]] = dict()
        self._extraHTTPHeaders: OrderedDict[str, str] = OrderedDict()
        self._offline: bool = False
        self._credentials: Optional[Dict[str, str]] = None
        self._attemptedAuthentications: Set[str] = set()
        self._userRequestInterceptionEnabled: bool = False
        self._protocolRequestInterceptionEnabled: bool = False
        self._requestHashToRequestIds: Multimap[str, str] = Multimap()
        self._requestHashToInterceptionIds: Multimap[str, str] = Multimap()

        self._client.on(
            'Network.requestWillBeSent',
            lambda event: self._client._loop.create_task(
                self._onRequestWillBeSent(event)
            ),
        )
        self._client.on('Network.requestIntercepted', self._onRequestIntercepted)
        self._client.on('Network.requestServedFromCache', self._onRequestServedFromCache)
        self._client.on('Network.responseReceived', self._onResponseReceived)
        self._client.on('Network.loadingFinished', self._onLoadingFinished)
        self._client.on('Network.loadingFailed', self._onLoadingFailed)

    async def authenticate(self, credentials: Dict[str, str]) -> None:
        """Provide credentials for http auth."""
        self._credentials = credentials
        await self._updateProtocolRequestInterception()

    async def setExtraHTTPHeaders(self, extraHTTPHeaders: Dict[str, str]) -> None:
        """Set extra http headers."""
        self._extraHTTPHeaders = OrderedDict()
        for k, v in extraHTTPHeaders.items():
            if not isinstance(v, str):
                raise TypeError(
                    f'Expected value of header "{k}" to be string, '
                    f'but {type(v)} is found.')
            self._extraHTTPHeaders[k.lower()] = v
        await self._client.send('Network.setExtraHTTPHeaders',
                              {'headers': self._extraHTTPHeaders})

    def extraHTTPHeaders(self) -> Dict[str, str]:
        """Get extra http headers."""
        return dict(**self._extraHTTPHeaders)

    async def setOfflineMode(self, value: bool) -> None:
        """Change offline mode enable/disable."""
        if self._offline == value:
            return
        self._offline = value
        await self._client.send('Network.emulateNetworkConditions', {
            'offline': self._offline,
            'latency': 0,
            'downloadThroughput': -1,
            'uploadThroughput': -1,
        })

    async def setUserAgent(self, userAgent: str) -> None:
        """Set user agent."""
        await self._client.send('Network.setUserAgentOverride',
                              {'userAgent': userAgent})

    async def setRequestInterception(self, value: bool) -> None:
        """Enable request interception."""
        self._userRequestInterceptionEnabled = value
        await self._updateProtocolRequestInterception()

    async def _updateProtocolRequestInterception(self) -> None:
        enabled = (self._userRequestInterceptionEnabled or
                  bool(self._credentials))
        if enabled == self._protocolRequestInterceptionEnabled:
            return
        self._protocolRequestInterceptionEnabled = enabled
        patterns = [{'urlPattern': '*'}] if enabled else []
        await asyncio.gather(
            self._client.send(
                'Network.setCacheDisabled',
                {'cacheDisabled': enabled},
            ),
            self._client.send(
                'Network.setRequestInterception',
                {'patterns': patterns},
            )
        )

    async def _onRequestWillBeSent(self, event: RequestWillBeSentEvent) -> None:
        if self._protocolRequestInterceptionEnabled:
            requestHash = generateRequestHash(event.get('request', {}))
            interceptionId = self._requestHashToInterceptionIds.firstValue(requestHash)
            if interceptionId:
                self._onRequest(event, interceptionId)
                self._requestHashToInterceptionIds.delete(requestHash, interceptionId)
            else:
                requestId = event.get('requestId', '')
                self._requestHashToRequestIds.set(requestHash, requestId)
                self._requestIdToResponseWillBeSent[requestId] = event
            return
        self._onRequest(event, None)

    async def _send(self, method: str, msg: dict) -> None:
        try:
            await self._client.send(method, msg)
        except Exception as e:
            debugError(logger, e)

    def _onRequestIntercepted(self, event: RequestInterceptedEvent) -> None:
        if event.get('authChallenge'):
            response = 'Default'
            if event['interceptionId'] in self._attemptedAuthentications:
                response = 'CancelAuth'
            elif self._credentials:
                response = 'ProvideCredentials'
                self._attemptedAuthentications.add(event['interceptionId'])
            username = self._credentials.get('username') if self._credentials else None
            password = self._credentials.get('password') if self._credentials else None

            self._client._loop.create_task(self._send(
                'Network.continueInterceptedRequest', {
                    'interceptionId': event['interceptionId'],
                    'authChallengeResponse': {
                        'response': response,
                        'username': username,
                        'password': password,
                    }
                }
            ))
            return

        if (not self._userRequestInterceptionEnabled and
                self._protocolRequestInterceptionEnabled):
            self._client._loop.create_task(self._send(
                'Network.continueInterceptedRequest', {
                    'interceptionId': event['interceptionId'],
                }
            ))

        requestHash = generateRequestHash(event['request'])
        requestId = self._requestHashToRequestIds.firstValue(requestHash)
        if requestId:
            requestWillBeSentEvent = self._requestIdToResponseWillBeSent[requestId]
            self._onRequest(requestWillBeSentEvent, event.get('interceptionId'))
            self._requestHashToRequestIds.delete(requestHash, requestId)
            self._requestIdToResponseWillBeSent.pop(requestId, None)
        else:
            self._requestHashToInterceptionIds.set(requestHash, event['interceptionId'])

    def _onRequest(self, event: RequestWillBeSentEvent, interceptionId: Optional[str]) -> None:
        redirectChain: List['Request'] = list()
        if event.get('redirectResponse'):
            request = self._requestIdToRequest.get(event['requestId'])
            if request:
                redirectResponse = event['redirectResponse']
                self._handleRequestRedirect(
                    request,
                    redirectResponse.get('status'),
                    redirectResponse.get('headers'),
                    redirectResponse.get('fromDiskCache'),
                    redirectResponse.get('fromServiceWorker'),
                    redirectResponse.get('SecurityDetails'),
                )
                redirectChain = request._redirectChain

        isNavigationRequest = bool(
            event.get('requestId') == event.get('loaderId') and
            event.get('type') == 'Document'
        )
        requestId = event.get('requestId', '')
        frameId = event.get('frameId')
        self._handleRequestStart(
            requestId,
            interceptionId,
            event.get('request', {}).get('url', ''),
            isNavigationRequest,
            event.get('type', ''),
            event.get('request', {}),
            frameId,
            redirectChain,
        )

    def _onRequestServedFromCache(self, event: RequestServedFromCacheEvent) -> None:
        request = self._requestIdToRequest.get(event.get('requestId', ''))
        if request:
            request._fromMemoryCache = True

    def _handleRequestRedirect(self, request: 'Request', redirectStatus: Optional[int],
                             redirectHeaders: Optional[Dict[str, str]], fromDiskCache: Optional[bool],
                             fromServiceWorker: Optional[bool],
                             securityDetails: Optional[Dict[str, Any]] = None) -> None:
        response = Response(
            self._client, 
            request, 
            redirectStatus or 0,
            redirectHeaders or {},
            fromDiskCache or False,
            fromServiceWorker or False,
            securityDetails
        )
        request._response = response
        request._redirectChain.append(request)
        response._bodyLoadedPromiseFulfill(
            NetworkError('Response body is unavailable for redirect response')
        )
        self._requestIdToRequest.pop(request._requestId, None)
        self._attemptedAuthentications.discard(request._interceptionId)
        self.emit(NetworkManager.Events.Response, response)
        self.emit(NetworkManager.Events.RequestFinished, request)

    def _handleRequestStart(self, requestId: str,
                          interceptionId: Optional[str], url: str,
                          isNavigationRequest: bool, resourceType: str,
                          requestPayload: Dict[str, Any], frameId: Optional[str],
                          redirectChain: List['Request']) -> None:
        frame = None
        if frameId and self._frameManager is not None:
            frame = self._frameManager.frame(frameId)

        request = Request(
            self._client, 
            requestId, 
            interceptionId,
            isNavigationRequest,
            self._userRequestInterceptionEnabled, 
            url,
            resourceType, 
            requestPayload, 
            frame, 
            redirectChain
        )
        self._requestIdToRequest[requestId] = request
        self.emit(NetworkManager.Events.Request, request)

    def _onResponseReceived(self, event: ResponseReceivedEvent) -> None:
        request = self._requestIdToRequest.get(event.get('requestId', ''))
        if not request:
            return
        _resp = event.get('response', {})
        response = Response(
            self._client, 
            request,
            _resp.get('status', 0),
            _resp.get('headers', {}),
            _resp.get('fromDiskCache', False),
            _resp.get('fromServiceWorker', False),
            _resp.get('securityDetails')
        )
        request._response = response
        self.emit(NetworkManager.Events.Response, response)

    def _onLoadingFinished(self, event: LoadingFinishedEvent) -> None:
        request = self._requestIdToRequest.get(event.get('requestId', ''))
        if not request:
            return
        response = request.response
        if response:
            response._bodyLoadedPromiseFulfill(None)
        self._requestIdToRequest.pop(request._requestId, None)
        self._attemptedAuthentications.discard(request._interceptionId)
        self.emit(NetworkManager.Events.RequestFinished, request)

    def _onLoadingFailed(self, event: LoadingFailedEvent) -> None:
        request = self._requestIdToRequest.get(event.get('requestId', ''))
        if not request:
            return
        request._failureText = event.get('errorText')
        response = request.response
        if response:
            response._bodyLoadedPromiseFulfill(None)
        self._requestIdToRequest.pop(request._requestId, None)
        self._attemptedAuthentications.discard(request._interceptionId)
        self.emit(NetworkManager.Events.RequestFailed, request)


class Request(object):
    """Request class."""

    def __init__(self, client: CDPSession, requestId: str,
                 interceptionId: Optional[str], isNavigationRequest: bool,
                 allowInterception: bool, url: str, resourceType: str,
                 payload: Dict[str, Any], frame: Optional[Frame],
                 redirectChain: List['Request']) -> None:
        self._client: CDPSession = client
        self._requestId: str = requestId
        self._isNavigationRequest: bool = isNavigationRequest
        self._interceptionId: Optional[str] = interceptionId
        self._allowInterception: bool = allowInterception
        self._interceptionHandled: bool = False
        self._response: Optional['Response'] = None
        self._failureText: Optional[str] = None

        self._url: str = url
        self._resourceType: str = resourceType.lower()
        self._method: Optional[str] = payload.get('method')
        self._postData: Optional[str] = payload.get('postData')
        headers = payload.get('headers', {})
        self._headers: Dict[str, str] = {k.lower(): v for k, v in headers.items()}
        self._frame: Optional[Frame] = frame
        self._redirectChain: List['Request'] = redirectChain

        self._fromMemoryCache: bool = False

    @property
    def url(self) -> str:
        """URL of this request."""
        return self._url

    @property
    def resourceType(self) -> str:
        """Resource type of this request."""
        return self._resourceType

    @property
    def method(self) -> Optional[str]:
        """Return this request's method (GET, POST, etc.)."""
        return self._method

    @property
    def postData(self) -> Optional[str]:
        """Return post body of this request."""
        return self._postData

    @property
    def headers(self) -> Dict[str, str]:
        """Return a dictionary of HTTP headers of this request."""
        return self._headers

    @property
    def response(self) -> Optional['Response']:
        """Return matching Response object, or None."""
        return self._response

    @property
    def frame(self) -> Optional[Frame]:
        """Return a matching Frame object."""
        return self._frame

    def isNavigationRequest(self) -> bool:
        """Whether this request is driving frame's navigation."""
        return self._isNavigationRequest

    @property
    def redirectChain(self) -> List['Request']:
        """Return chain of requests initiated to fetch a resource."""
        return copy.copy(self._redirectChain)

    def failure(self) -> Optional[Dict[str, str]]:
        """Return error text."""
        if not self._failureText:
            return None
        return {'errorText': self._failureText}

    async def continue_(self, overrides: Dict[str, Any] = None) -> None:
        """Continue request with optional request overrides."""
        if overrides is None:
            overrides = {}

        if not self._allowInterception:
            raise NetworkError('Request interception is not enabled.')
        if self._interceptionHandled:
            raise NetworkError('Request is already handled.')

        self._interceptionHandled = True
        opt: Dict[str, Any] = {'interceptionId': self._interceptionId}
        opt.update(overrides)
        try:
            await self._client.send('Network.continueInterceptedRequest', opt)
        except Exception as e:
            debugError(logger, e)

    async def respond(self, response: Dict[str, Any]) -> None:
        """Fulfills request with given response."""
        if self._url.startswith('data:'):
            return
        if not self._allowInterception:
            raise NetworkError('Request interception is not enabled.')
        if self._interceptionHandled:
            raise NetworkError('Request is already handled.')
        self._interceptionHandled = True

        responseBody: Optional[bytes]
        if response.get('body') and isinstance(response['body'], str):
            responseBody = response['body'].encode('utf-8')
        else:
            responseBody = response.get('body')

        responseHeaders: Dict[str, str] = {}
        if response.get('headers'):
            for header in response['headers']:
                responseHeaders[header.lower()] = response['headers'][header]
        if response.get('contentType'):
            responseHeaders['content-type'] = response['contentType']
        if responseBody and 'content-length' not in responseHeaders:
            responseHeaders['content-length'] = str(len(responseBody))

        statusCode = response.get('status', 200)
        statusText = statusTexts.get(str(statusCode), '')
        statusLine = f'HTTP/1.1 {statusCode} {statusText}'

        CRLF = '\r\n'
        text = statusLine + CRLF
        for header in responseHeaders:
            text = f'{text}{header}: {responseHeaders[header]}{CRLF}'
        text = text + CRLF
        responseBuffer = text.encode('utf-8')
        if responseBody:
            responseBuffer = responseBuffer + responseBody

        rawResponse = base64.b64encode(responseBuffer).decode('ascii')
        try:
            await self._client.send('Network.continueInterceptedRequest', {
                'interceptionId': self._interceptionId,
                'rawResponse': rawResponse,
            })
        except Exception as