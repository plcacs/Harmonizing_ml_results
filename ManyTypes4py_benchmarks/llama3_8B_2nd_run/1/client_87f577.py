import asyncio
import base64
import dataclasses
import hashlib
import json
import os
import sys
import traceback
import warnings
from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Collection, Coroutine, Generic, Iterable, List, Mapping, Optional, Set, Tuple, Type, TypedDict, TypeVar, Union, final
from multidict import CIMultiDict, MultiDict, MultiDictProxy, istr
from yarl import URL
from . import hdrs, http, payload
from ._websocket.reader import WebSocketDataQueue
from . import AbstractCookieJar
from .client_exceptions import ClientConnectionError, ClientConnectionResetError, ClientConnectorCertificateError, ClientConnectorDNSError, ClientConnectorError, ClientConnectorSSLError, ClientError, ClientHttpProxyError, ClientOSError, ClientPayloadError, ClientProxyConnectionError, ClientResponseError, ClientSSLError, ConnectionTimeoutError, ContentTypeError, InvalidURL, InvalidUrlClientError, InvalidUrlRedirectClientError, NonHttpUrlClientError, NonHttpUrlRedirectClientError, RedirectClientError, ServerConnectionError, ServerDisconnectedError, ServerFingerprintMismatch, ServerTimeoutError, SocketTimeoutError, TooManyRedirects, WSMessageTypeError, WSServerHandshakeError
from .client_reqrep import SSL_ALLOWED_TYPES, ClientRequest, ClientResponse, Fingerprint, RequestInfo
from .client_ws import DEFAULT_WS_CLIENT_TIMEOUT, ClientWebSocketResponse, ClientWSTimeout
from .connector import HTTP_AND_EMPTY_SCHEMA_SET, BaseConnector, NamedPipeConnector, TCPConnector, UnixConnector
from .cookiejar import CookieJar
from .helpers import _SENTINEL, EMPTY_BODY_METHODS, BasicAuth, TimeoutHandle, frozen_dataclass_decorator, get_env_proxy_for_url, sentinel, strip_auth_from_url
from .http import WS_KEY, HttpVersion, WebSocketReader, WebSocketWriter
from .http_websocket import WSHandshakeError, ws_ext_gen, ws_ext_parse
from .tracing import Trace, TraceConfig
from .typedefs import JSONEncoder, LooseCookies, LooseHeaders, Query, StrOrURL
__all__ = ('ClientConnectionError', 'ClientConnectionResetError', 'ClientConnectorCertificateError', 'ClientConnectorDNSError', 'ClientConnectorError', 'ClientConnectorSSLError', 'ClientError', 'ClientHttpProxyError', 'ClientOSError', 'ClientPayloadError', 'ClientProxyConnectionError', 'ClientResponseError', 'ClientSSLError', 'ConnectionTimeoutError', 'ContentTypeError', 'InvalidURL', 'InvalidUrlClientError', 'RedirectClientError', 'NonHttpUrlClientError', 'InvalidUrlRedirectClientError', 'NonHttpUrlRedirectClientError', 'ServerConnectionError', 'ServerDisconnectedError', 'ServerFingerprintMismatch', 'ServerTimeoutError', 'SocketTimeoutError', 'TooManyRedirects', 'WSServerHandshakeError', 'ClientRequest', 'ClientResponse', 'Fingerprint', 'RequestInfo', 'BaseConnector', 'TCPConnector', 'UnixConnector', 'ClientWebSocketResponse', 'ClientSession', 'ClientTimeout', 'WSMessageTypeError')

@dataclasses.dataclass(frozen=True)
class _RequestOptions(TypedDict, total=False):
    pass

class ClientSession:
    __slots__ = ('_base_url', '_base_url_origin', '_source_traceback', '_connector', '_loop', '_cookie_jar', '_connector_owner', '_default_auth', '_version', '_json_serialize', '_raise_for_status', '_auto_decompress', '_trust_env', '_requote_redirect_url', '_read_bufsize', '_max_line_size', '_max_field_size', '_resolve_charset', '_default_proxy', '_default_proxy_auth', '_retry_connection')
    ...

class _BaseRequestContextManager(Coroutine[Any, Any, _RetType], Generic[_RetType]):
    __slots__ = ('_coro', '_resp')

    def __init__(self, coro):
        self._coro = coro

    def send(self, arg):
        return self._coro.send(arg)

    def throw(self, *args, **kwargs):
        return self._coro.throw(*args, **kwargs)

    def close(self):
        return self._coro.close()

    def __await__(self):
        ret = self._coro.__await__()
        return ret

    def __iter__(self):
        return self.__await__()

    async def __aenter__(self):
        self._resp = await self._coro
        return await self._resp.__aenter__()

    async def __aexit__(self, exc_type,