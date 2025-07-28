#!/usr/bin/env python3
import copy
import sys
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable, MutableMapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, TypeVar, Union
from typing_extensions import Self, Protocol, runtime_checkable

import anyio
import httpx
from asgi_lifespan import LifespanManager
from httpx import HTTPStatusError, Request, Response
from starlette import status

import prefect
from prefect.client import constants
from prefect.client.schemas.objects import CsrfToken
from prefect.exceptions import PrefectHTTPStatusError
from prefect.logging import get_logger
from prefect.settings import (
    PREFECT_API_URL,
    PREFECT_CLIENT_MAX_RETRIES,
    PREFECT_CLIENT_RETRY_EXTRA_CODES,
    PREFECT_CLIENT_RETRY_JITTER_FACTOR,
    PREFECT_CLOUD_API_URL,
    PREFECT_SERVER_ALLOW_EPHEMERAL_MODE,
)
from prefect.utilities.collections import AutoEnum
from prefect.utilities.math import bounded_poisson_interval, clamped_poisson_interval

# Type aliases
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
T = TypeVar("T", bound="PrefectResponse")

# Global lifespan caches
APP_LIFESPANS: Dict[Tuple[int, int], LifespanManager] = {}
APP_LIFESPANS_REF_COUNTS: Dict[Tuple[int, int], int] = {}
APP_LIFESPANS_LOCKS: defaultdict[int, anyio.Lock] = defaultdict(anyio.Lock)

logger: Logger = get_logger("client")


@runtime_checkable
class ASGIApp(Protocol):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        ...


@asynccontextmanager
async def app_lifespan_context(app: ASGIApp) -> AsyncGenerator[None, None]:
    """
    A context manager that calls startup/shutdown hooks for the given application.
    """
    thread_id: int = threading.get_ident()
    key: Tuple[int, int] = (thread_id, id(app))
    exc_info: Tuple[Optional[Any], Optional[Any], Optional[Any]] = (None, None, None)
    lock: anyio.Lock = APP_LIFESPANS_LOCKS[thread_id]
    async with lock:
        if key in APP_LIFESPANS:
            APP_LIFESPANS_REF_COUNTS[key] += 1
        else:
            context: LifespanManager = LifespanManager(app, startup_timeout=30, shutdown_timeout=30)
            APP_LIFESPANS[key] = context
            APP_LIFESPANS_REF_COUNTS[key] = 1
            await context.__aenter__()
    try:
        yield
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        with anyio.CancelScope(shield=True):
            async with lock:
                APP_LIFESPANS_REF_COUNTS[key] -= 1
                if APP_LIFESPANS_REF_COUNTS[key] <= 0:
                    APP_LIFESPANS_REF_COUNTS.pop(key)
                    context = APP_LIFESPANS.pop(key)
                    await context.__aexit__(*exc_info)


class PrefectResponse(httpx.Response):
    """
    A Prefect wrapper for the `httpx.Response` class.
    Provides more informative error messages.
    """

    def raise_for_status(self) -> None:
        """
        Raise an exception if the response contains an HTTPStatusError.
        """
        try:
            return super().raise_for_status()
        except HTTPStatusError as exc:
            raise PrefectHTTPStatusError.from_httpx_error(exc) from exc.__cause__

    @classmethod
    def from_httpx_response(cls: Type[T], response: httpx.Response) -> T:
        """
        Create a `PrefectResponse` from an `httpx.Response`.
        """
        new_response = copy.copy(response)
        new_response.__class__ = cls
        return new_response


class PrefectHttpxAsyncClient(httpx.AsyncClient):
    """
    A Prefect wrapper for the async httpx client with support for retry-after headers.
    """

    def __init__(
        self,
        *args: Any,
        enable_csrf_support: bool = False,
        raise_on_all_errors: bool = True,
        **kwargs: Any,
    ) -> None:
        self.enable_csrf_support: bool = enable_csrf_support
        self.csrf_token: Optional[str] = None
        self.csrf_token_expiration: Optional[datetime] = None
        self.csrf_client_id: uuid.UUID = uuid.uuid4()
        self.raise_on_all_errors: bool = raise_on_all_errors
        super().__init__(*args, **kwargs)
        user_agent: str = f"prefect/{prefect.__version__} (API {constants.SERVER_API_VERSION})"
        self.headers["User-Agent"] = user_agent

    async def _send_with_retry(
        self,
        request: httpx.Request,
        send: Callable[[httpx.Request, Any, Any], Awaitable[httpx.Response]],
        send_args: Any,
        send_kwargs: Any,
        retry_codes: Set[int] = set(),
        retry_exceptions: Tuple[Type[BaseException], ...] = (),
    ) -> httpx.Response:
        """
        Send a request and retry it if it fails.
        """
        try_count: int = 0
        response: Optional[httpx.Response] = None
        is_change_request: bool = request.method.lower() in {"post", "put", "patch", "delete"}
        if self.enable_csrf_support and is_change_request:
            await self._add_csrf_headers(request=request)
        while try_count <= PREFECT_CLIENT_MAX_RETRIES.value():
            try_count += 1
            retry_seconds: Optional[float] = None
            exc_info: Optional[Tuple[Any, Any, Any]] = None
            try:
                response = await send(request, *send_args, **send_kwargs)
            except retry_exceptions:
                if try_count > PREFECT_CLIENT_MAX_RETRIES.value():
                    raise
                exc_info = sys.exc_info()
            else:
                if response.status_code == status.HTTP_403_FORBIDDEN and "Invalid CSRF token" in response.text:
                    self.csrf_token = None
                    await self._add_csrf_headers(request)
                elif response.status_code not in retry_codes:
                    return response
                if "Retry-After" in response.headers:
                    retry_seconds = float(response.headers["Retry-After"])
            if retry_seconds is None:
                retry_seconds = 2 ** try_count
            jitter_factor: float = PREFECT_CLIENT_RETRY_JITTER_FACTOR.value()
            if retry_seconds > 0 and jitter_factor > 0:
                if response is not None and "Retry-After" in response.headers:
                    retry_seconds = bounded_poisson_interval(retry_seconds, retry_seconds * (1 + jitter_factor))
                else:
                    retry_seconds = clamped_poisson_interval(retry_seconds, jitter_factor)
            logger.debug(
                (
                    "Encountered retryable exception during request. "
                    if exc_info
                    else f"Received response with retryable status code {(response.status_code if response else 'unknown')}. "
                )
                + f"Another attempt will be made in {retry_seconds}s. This is attempt {try_count}/{PREFECT_CLIENT_MAX_RETRIES.value() + 1}.",
                exc_info=exc_info,
            )
            await anyio.sleep(retry_seconds)
        assert response is not None, "Retry handling ended without response or exception"
        return response

    async def send(self, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        """
        Send a request with automatic retry behavior for specified status codes.
        """
        super_send = super().send
        retry_status_codes: Set[int] = {
            status.HTTP_429_TOO_MANY_REQUESTS,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_502_BAD_GATEWAY,
            status.HTTP_408_REQUEST_TIMEOUT,
            *PREFECT_CLIENT_RETRY_EXTRA_CODES.value(),
        }
        response: httpx.Response = await self._send_with_retry(
            request=request,
            send=super_send,
            send_args=args,
            send_kwargs=kwargs,
            retry_codes=retry_status_codes,
            retry_exceptions=(
                httpx.ReadTimeout,
                httpx.PoolTimeout,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.LocalProtocolError,
            ),
        )
        response = PrefectResponse.from_httpx_response(response)
        if self.raise_on_all_errors:
            response.raise_for_status()
        return response

    async def _add_csrf_headers(self, request: httpx.Request) -> None:
        now: datetime = datetime.now(timezone.utc)
        if not self.enable_csrf_support:
            return
        if not self.csrf_token or (self.csrf_token_expiration and now > self.csrf_token_expiration):
            token_request: httpx.Request = self.build_request("GET", f"/csrf-token?client={self.csrf_client_id}")
            try:
                token_response: httpx.Response = await self.send(token_request)
            except PrefectHTTPStatusError as exc:
                old_server: bool = exc.response.status_code == status.HTTP_404_NOT_FOUND
                unconfigured_server: bool = (
                    exc.response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
                    and "CSRF protection is disabled." in exc.response.text
                )
                if old_server or unconfigured_server:
                    self.enable_csrf_support = False
                    return
                raise
            token: CsrfToken = CsrfToken.model_validate(token_response.json())
            self.csrf_token = token.token
            self.csrf_token_expiration = token.expiration
        request.headers["Prefect-Csrf-Token"] = self.csrf_token  # type: ignore
        request.headers["Prefect-Csrf-Client"] = str(self.csrf_client_id)


class PrefectHttpxSyncClient(httpx.Client):
    """
    A Prefect wrapper for the sync httpx client with support for retry-after headers.
    """

    def __init__(
        self,
        *args: Any,
        enable_csrf_support: bool = False,
        raise_on_all_errors: bool = True,
        **kwargs: Any,
    ) -> None:
        self.enable_csrf_support: bool = enable_csrf_support
        self.csrf_token: Optional[str] = None
        self.csrf_token_expiration: Optional[datetime] = None
        self.csrf_client_id: uuid.UUID = uuid.uuid4()
        self.raise_on_all_errors: bool = raise_on_all_errors
        super().__init__(*args, **kwargs)
        user_agent: str = f"prefect/{prefect.__version__} (API {constants.SERVER_API_VERSION})"
        self.headers["User-Agent"] = user_agent

    def _send_with_retry(
        self,
        request: httpx.Request,
        send: Callable[[httpx.Request, Any, Any], httpx.Response],
        send_args: Any,
        send_kwargs: Any,
        retry_codes: Set[int] = set(),
        retry_exceptions: Tuple[Type[BaseException], ...] = (),
    ) -> httpx.Response:
        """
        Send a request and retry it if it fails.
        """
        try_count: int = 0
        response: Optional[httpx.Response] = None
        is_change_request: bool = request.method.lower() in {"post", "put", "patch", "delete"}
        if self.enable_csrf_support and is_change_request:
            self._add_csrf_headers(request=request)
        while try_count <= PREFECT_CLIENT_MAX_RETRIES.value():
            try_count += 1
            retry_seconds: Optional[float] = None
            exc_info: Optional[Tuple[Any, Any, Any]] = None
            try:
                response = send(request, *send_args, **send_kwargs)
            except retry_exceptions:
                if try_count > PREFECT_CLIENT_MAX_RETRIES.value():
                    raise
                exc_info = sys.exc_info()
            else:
                if response.status_code == status.HTTP_403_FORBIDDEN and "Invalid CSRF token" in response.text:
                    self.csrf_token = None
                    self._add_csrf_headers(request)
                elif response.status_code not in retry_codes:
                    return response
                if "Retry-After" in response.headers:
                    retry_seconds = float(response.headers["Retry-After"])
            if retry_seconds is None:
                retry_seconds = 2 ** try_count
            jitter_factor: float = PREFECT_CLIENT_RETRY_JITTER_FACTOR.value()
            if retry_seconds > 0 and jitter_factor > 0:
                if response is not None and "Retry-After" in response.headers:
                    retry_seconds = bounded_poisson_interval(retry_seconds, retry_seconds * (1 + jitter_factor))
                else:
                    retry_seconds = clamped_poisson_interval(retry_seconds, jitter_factor)
            logger.debug(
                (
                    "Encountered retryable exception during request. "
                    if exc_info
                    else f"Received response with retryable status code {(response.status_code if response else 'unknown')}. "
                )
                + f"Another attempt will be made in {retry_seconds}s. This is attempt {try_count}/{PREFECT_CLIENT_MAX_RETRIES.value() + 1}.",
                exc_info=exc_info,
            )
            time.sleep(retry_seconds)
        assert response is not None, "Retry handling ended without response or exception"
        return response

    def send(self, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        """
        Send a request with automatic retry behavior for specified status codes.
        """
        super_send = super().send
        retry_status_codes: Set[int] = {
            status.HTTP_429_TOO_MANY_REQUESTS,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_502_BAD_GATEWAY,
            status.HTTP_408_REQUEST_TIMEOUT,
            *PREFECT_CLIENT_RETRY_EXTRA_CODES.value(),
        }
        response: httpx.Response = self._send_with_retry(
            request=request,
            send=super_send,
            send_args=args,
            send_kwargs=kwargs,
            retry_codes=retry_status_codes,
            retry_exceptions=(
                httpx.ReadTimeout,
                httpx.PoolTimeout,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.LocalProtocolError,
            ),
        )
        response = PrefectResponse.from_httpx_response(response)
        if self.raise_on_all_errors:
            response.raise_for_status()
        return response

    def _add_csrf_headers(self, request: httpx.Request) -> None:
        now: datetime = datetime.now(timezone.utc)
        if not self.enable_csrf_support:
            return
        if not self.csrf_token or (self.csrf_token_expiration and now > self.csrf_token_expiration):
            token_request: httpx.Request = self.build_request("GET", f"/csrf-token?client={self.csrf_client_id}")
            try:
                token_response: httpx.Response = self.send(token_request)
            except PrefectHTTPStatusError as exc:
                old_server: bool = exc.response.status_code == status.HTTP_404_NOT_FOUND
                unconfigured_server: bool = (
                    exc.response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
                    and "CSRF protection is disabled." in exc.response.text
                )
                if old_server or unconfigured_server:
                    self.enable_csrf_support = False
                    return
                raise
            token: CsrfToken = CsrfToken.model_validate(token_response.json())
            self.csrf_token = token.token
            self.csrf_token_expiration = token.expiration
        request.headers["Prefect-Csrf-Token"] = self.csrf_token  # type: ignore
        request.headers["Prefect-Csrf-Client"] = str(self.csrf_client_id)


class ServerType(AutoEnum):
    EPHEMERAL = AutoEnum.auto()
    SERVER = AutoEnum.auto()
    CLOUD = AutoEnum.auto()
    UNCONFIGURED = AutoEnum.auto()


def determine_server_type() -> ServerType:
    """
    Determine the server type based on the current settings.
    """
    api_url: Optional[str] = PREFECT_API_URL.value()
    if api_url is None:
        if PREFECT_SERVER_ALLOW_EPHEMERAL_MODE.value():
            return ServerType.EPHEMERAL
        else:
            return ServerType.UNCONFIGURED
    if api_url.startswith(PREFECT_CLOUD_API_URL.value()):
        return ServerType.CLOUD
    else:
        return ServerType.SERVER
