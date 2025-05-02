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
from typing import Awaitable, Dict, List, Optional, Union, TYPE_CHECKING, Set
from urllib.parse import unquote

from pyee import EventEmitter

from pyppeteer.connection import CDPSession
from pyppeteer.errors import NetworkError
from pyppeteer.frame_manager import FrameManager, Frame
from pyppeteer.helper import debugError
from pyppeteer.multimap import Multimap

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
        self._requestIdToRequest: Dict[Optional[str], 'Request'] = dict()
        self._requestIdToResponseWillBeSent: Dict[Optional[str], Dict] = dict()
        self._extraHTTPHeaders: OrderedDict[str, str] = OrderedDict()
        self._offline: bool = False
        self._credentials: Optional[Dict[str, str]] = None
        self._attemptedAuthentications: Set[Optional[str]] = set()
        self._userRequestInterceptionEnabled: bool = False
        self._protocolRequestInterceptionEnabled: bool = False
        self._requestHashToRequestIds: Multimap = Multimap()
        self._requestHashToInterceptionIds: Multimap = Multimap()

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

    async def _onRequestWillBeSent(self, event: Dict) -> None:
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

    def _onRequest(self, event: Dict, interceptionId: Optional[str]) -> None:
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
        self._handleRequestStart(
            event['requestId'],
            interceptionId,
            event.get('request', {}).get('url'),
            isNavigationRequest,
            event.get('type', ''),
            event.get('request', {}),
            event.get('frameId'),
            redirectChain,
        )

    def _onRequestServedFromCache(self, event: Dict) -> None:
        request = self._requestIdToRequest.get(event.get('requestId'))
        if request:
            request._fromMemoryCache = True

    def _handleRequestRedirect(self, request: 'Request', redirectStatus: int,
                               redirectHeaders: Dict, fromDiskCache: bool,
                               fromServiceWorker: bool,
                               securityDetails: Dict = None) -> None:
        response = Response(self._client, request, redirectStatus,
                            redirectHeaders, fromDiskCache, fromServiceWorker,
                            securityDetails)
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
                            requestPayload: Dict, frameId: Optional[str],
                            redirectChain: List['Request']
                            ) -> None:
        frame = None
        if frameId and self._frameManager is not None:
            frame = self._frameManager.frame(frameId)

        request = Request(self._client, requestId, interceptionId,
                          isNavigationRequest,
                          self._userRequestInterceptionEnabled, url,
                          resourceType, requestPayload, frame, redirectChain)
        self._requestIdToRequest[requestId] = request
        self.emit(NetworkManager.Events.Request, request)

    def _onResponseReceived(self, event: dict) -> None:
        request = self._requestIdToRequest.get(event['requestId'])
        if not request:
            return
        _resp = event.get('response', {})
        response = Response(self._client, request,
                            _resp.get('status', 0),
                            _resp.get('headers', {}),
                            _resp.get('fromDiskCache'),
                            _resp.get('fromServiceWorker'),
                            _resp.get('securityDetails'))
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
    """Request class."""

    def __init__(self, client: CDPSession, requestId: Optional[str],
                 interceptionId: Optional[str], isNavigationRequest: bool,
                 allowInterception: bool, url: str, resourceType: str,
                 payload: dict, frame: Optional[Frame],
                 redirectChain: List['Request']
                 ) -> None:
        self._client: CDPSession = client
        self._requestId: Optional[str] = requestId
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
        """Resource type of this request perceived by the rendering engine."""
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
        """Return matching :class:`Response` object, or ``None``."""
        return self._response

    @property
    def frame(self) -> Optional[Frame]:
        """Return a matching :class:`~pyppeteer.frame_manager.frame` object."""
        return self._frame

    def isNavigationRequest(self) -> bool:
        """Whether this request is driving frame's navigation."""
        return self._isNavigationRequest

    @property
    def redirectChain(self) -> List['Request']:
        """Return chain of requests initiated to fetch a resource."""
        return copy.copy(self._redirectChain)

    def failure(self) -> Optional[Dict]:
        """Return error text."""
        if not self._failureText:
            return None
        return {'errorText': self._failureText}

    async def continue_(self, overrides: Dict = None) -> None:
        """Continue request with optional request overrides."""
        if overrides is None:
            overrides = {}

        if not self._allowInterception:
            raise NetworkError('Request interception is not enabled.')
        if self._interceptionHandled:
            raise NetworkError('Request is already handled.')

        self._interceptionHandled = True
        opt = {'interceptionId': self._interceptionId}
        opt.update(overrides)
        try:
            await self._client.send('Network.continueInterceptedRequest', opt)
        except Exception as e:
            debugError(logger, e)

    async def respond(self, response: Dict) -> None:
        """Fulfills request with given response."""
        if self._url.startswith('data:'):
            return
        if not self._allowInterception:
            raise NetworkError('Request interception is not enabled.')
        if self._interceptionHandled:
            raise NetworkError('Request is already handled.')
        self._interceptionHandled = True

        if response.get('body') and isinstance(response['body'], str):
            responseBody: Optional[bytes] = response['body'].encode('utf-8')
        else:
            responseBody = response.get('body')

        responseHeaders = {}
        if response.get('headers'):
            for header in response['headers']:
                responseHeaders[header.lower()] = response['headers'][header]
        if response.get('contentType'):
            responseHeaders['content-type'] = response['contentType']
        if responseBody and 'content-length' not in responseHeaders:
            responseHeaders['content-length'] = len(responseBody)

        statusCode = response.get('status', 200)
        statusText = statusTexts.get(statusCode, '')
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
        except Exception as e:
            debugError(logger, e)

    async def abort(self, errorCode: str = 'failed') -> None:
        """Abort request."""
        errorReason = errorReasons[errorCode]
        if not errorReason:
            raise NetworkError('Unknown error code: {}'.format(errorCode))
        if not self._allowInterception:
            raise NetworkError('Request interception is not enabled.')
        if self._interceptionHandled:
            raise NetworkError('Request is already handled.')
        self._interceptionHandled = True
        try:
            await self._client.send('Network.continueInterceptedRequest', dict(
                interceptionId=self._interceptionId,
                errorReason=errorReason,
            ))
        except Exception as e:
            debugError(logger, e)


errorReasons = {
    'aborted': 'Aborted',
    'accessdenied': 'AccessDenied',
    'addressunreachable': 'AddressUnreachable',
    'blockedbyclient': 'BlockedByClient',
    'blockedbyresponse': 'BlockedByResponse',
    'connectionaborted': 'ConnectionAborted',
    'connectionclosed': 'ConnectionClosed',
    'connectionfailed': 'ConnectionFailed',
    'connectionrefused': 'ConnectionRefused',
    'connectionreset': 'ConnectionReset',
    'internetdisconnected': 'InternetDisconnected',
    'namenotresolved': 'NameNotResolved',
    'timedout': 'TimedOut',
    'failed': 'Failed',
}


class Response:
    """Response class represents responses which are received by ``Page``."""

    def __init__(self, client: CDPSession, request: Request, status: int,
                 headers: Dict[str, str], fromDiskCache: bool,
                 fromServiceWorker: bool, securityDetails: Dict = None
                 ) -> None:
        self._client: CDPSession = client
        self._request: Request = request
        self._status: int = status
        self._contentPromise = self._client._loop.create_future()
        self._bodyLoadedPromise = self._client._loop.create_future()

        self._url: str = request.url
        self._fromDiskCache: bool = fromDiskCache
        self._fromServiceWorker: bool = fromServiceWorker
        self._headers: Dict[str, str] = {k.lower(): v for k, v in headers.items()}
        self._securityDetails: Union[Dict, 'SecurityDetails'] = {}
        if securityDetails:
            self._securityDetails = SecurityDetails(
                securityDetails['subjectName'],
                securityDetails['issuer'],
                securityDetails['validFrom'],
                securityDetails['validTo'],
                securityDetails['protocol'],
            )

    def _bodyLoadedPromiseFulfill(self, value: Optional[Exception]) -> None:
        self._bodyLoadedPromise.set_result(value)

    @property
    def url(self) -> str:
        """URL of the response."""
        return self._url

    @property
    def ok(self) -> bool:
        """Return bool whether this request is successful (200-299) or not."""
        return self._status == 0 or 200 <= self._status <= 299

    @property
    def status(self) -> int:
        """Status code of the response."""
        return self._status

    @property
    def headers(self) -> Dict[str, str]:
        """Return dictionary of HTTP headers of this response."""
        return self._headers

    @property
    def securityDetails(self) -> Union[Dict, 'SecurityDetails']:
        """Return security details associated with this response."""
        return self._securityDetails

    async def _bufread(self) -> bytes:
        result = await self._bodyLoadedPromise
        if isinstance(result, Exception):
            raise result
        response = await self._client.send('Network.getResponseBody', {
            'requestId': self._request._requestId
        })
        body = response.get('body', b'')
        if response.get('base64Encoded'):
            return base64.b64decode(body)
        return body

    def buffer(self) -> Awaitable[bytes]:
        """Return awaitable which resolves to bytes with response body."""
        if not self._contentPromise.done():
            return self._client._loop.create_task(self._bufread())
        return self._contentPromise

    async def text(self) -> str:
        """Get text representation of response body."""
        content = await self.buffer()
        if isinstance(content, str):
            return content
        else:
            return content.decode('utf-8')

    async def json(self) -> dict:
        """Get JSON representation of response body."""
        content = await self.text()
        return json.loads(content)

    @property
    def request(self) -> Request:
        """Get matching :class:`Request` object."""
        return self._request

    @property
    def fromCache(self) -> bool:
        """Return ``True`` if the response was served from cache."""
        return self._fromDiskCache or self._request._fromMemoryCache

    @property
    def fromServiceWorker(self) -> bool:
        """Return ``True`` if the response was served by a service worker."""
        return self._fromServiceWorker


def generateRequestHash(request: dict) -> str:
    """Generate request hash."""
    normalizedURL = request.get('url', '')
    try:
        normalizedURL = unquote(normalizedURL)
    except Exception:
        pass

    _hash = {
        'url': normalizedURL,
        'method': request.get('method'),
        'postData': request.get('postData'),
        'headers': {},
    }

    if not normalizedURL.startswith('data:'):
        headers = list(request['headers'].keys())
        headers.sort()
        for header in headers:
            headerValue = request['headers'][header]
            header = header.lower()
            if header in [
                'accept',
                'referer',
                'x-devtools-emulate-network-conditions-client-id',
                'cookie',
            ]:
                continue
            _hash['headers'][header] = headerValue
    return json.dumps(_hash)


class SecurityDetails:
    """Class represents responses which are received by page."""

    def __init__(self, subjectName: str, issuer: str, validFrom: int,
                 validTo: int, protocol: str) -> None:
        self._subjectName: str = subjectName
        self._issuer: str = issuer
        self._validFrom: int = validFrom
        self._validTo: int = validTo
        self._protocol: str = protocol

    @property
    def subjectName(self) -> str:
        """Return the subject to which the certificate was issued to."""
        return self._subjectName

    @property
    def issuer(self) -> str:
        """Return a string with the name of issuer of the certificate."""
        return self._issuer

    @property
    def validFrom(self) -> int:
        """Return UnixTime of the start of validity of the certificate."""
        return self._validFrom

    @property
    def validTo(self) -> int:
        """Return UnixTime of the end of validity of the certificate."""
        return self._validTo

    @property
    def protocol(self) -> str:
        """Return string of with the security protocol, e.g. "TLS1.2"."""
        return self._protocol


statusTexts = {
    '100': 'Continue',
    '101': 'Switching Protocols',
    '102': 'Processing',
    '200': 'OK',
    '201': 'Created',
    '202': 'Accepted',
    '203': 'Non-Authoritative Information',
    '204': 'No Content',
    '206': 'Partial Content',
    '207': 'Multi-Status',
    '208': 'Already Reported',
    '209': 'IM Used',
    '300': 'Multiple Choices',
    '301': 'Moved Permanently',
    '302': 'Found',
    '303': 'See Other',
    '304': 'Not Modified',
    '305': 'Use Proxy',
    '306': 'Switch Proxy',
    '307': 'Temporary Redirect',
    '308': 'Permanent Redirect',
    '400': 'Bad Request',
    '401': 'Unauthorized',
    '402': 'Payment Required',
    '403': 'Forbidden',
    '404': 'Not Found',
    '405': 'Method Not Allowed',
    '406': 'Not Acceptable',
    '407': 'Proxy Authentication Required',
    '408': 'Request Timeout',
    '409': 'Conflict',
    '410': 'Gone',
    '411': 'Length Required',
    '412': 'Precondition Failed',
    '413': 'Payload Too Large',
    '414': 'URI Too Long',
    '415': 'Unsupported Media Type',
    '416': 'Range Not Satisfiable',
    '417': 'Expectation Failed',
    '418': 'I\'m a teapot',
    '421': 'Misdirected Request',
    '422': 'Unprocessable Entity',
    '423': 'Locked',
    '424': 'Failed Dependency',
    '426': 'Upgrade Required',
    '428': 'Precondition Required',
    '429': 'Too Many Requests',
    '431': 'Request Header Fields Too Large',
    '451': 'Unavailable For Legal Reasons',
    '500': 'Internal Server Error',
    '501': 'Not Implemented',
    '502': 'Bad Gateway',
    '503': 'Service Unavailable',
    '504': 'Gateway Timeout',
    '505': 'HTTP Version Not Supported',
    '506': 'Variant Also Negotiates',
    '507': 'Insufficient Storage',
    '508': 'Loop Detected',
    '510': 'Not Extended',
    '511': 'Network Authentication Required',
}
