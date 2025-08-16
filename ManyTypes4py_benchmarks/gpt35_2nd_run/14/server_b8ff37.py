from __future__ import annotations
import asyncio
import atexit
import base64
import contextlib
import gc
import mimetypes
import os
import random
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from functools import wraps
from hashlib import sha256
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Optional
import anyio
import asyncpg
import httpx
import sqlalchemy as sa
import sqlalchemy.exc
import sqlalchemy.orm.exc
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from typing_extensions import Self
import prefect
import prefect.server.api as api
import prefect.settings
from prefect.client.constants import SERVER_API_VERSION
from prefect.logging import get_logger
from prefect.server.api.dependencies import EnforceMinimumAPIVersion
from prefect.server.exceptions import ObjectNotFoundError
from prefect.server.services.base import RunInAllServers, Service
from prefect.server.utilities.database import get_dialect
from prefect.settings import PREFECT_API_DATABASE_CONNECTION_URL, PREFECT_API_LOG_RETRYABLE_ERRORS, PREFECT_DEBUG_MODE, PREFECT_MEMO_STORE_PATH, PREFECT_MEMOIZE_BLOCK_AUTO_REGISTRATION, PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS, PREFECT_UI_SERVE_BASE, get_current_settings
from prefect.utilities.hashing import hash_objects

if TYPE_CHECKING:
    import logging

TITLE: str = 'Prefect Server'
API_TITLE: str = 'Prefect Prefect REST API'
UI_TITLE: str = 'Prefect Prefect REST API UI'
API_VERSION: str = prefect.__version__
LIFESPAN_RAN_FOR_APP: set = set()
logger: logging.Logger = get_logger('server')
enforce_minimum_version: EnforceMinimumAPIVersion = EnforceMinimumAPIVersion(minimum_api_version='0.8.0', logger=logger)
API_ROUTERS: tuple = (api.flows.router, api.flow_runs.router, api.task_runs.router, api.flow_run_states.router, api.task_run_states.router, api.flow_run_notification_policies.router, api.deployments.router, api.saved_searches.router, api.logs.router, api.concurrency_limits.router, api.concurrency_limits_v2.router, api.block_types.router, api.block_documents.router, api.workers.router, api.task_workers.router, api.work_queues.router, api.artifacts.router, api.block_schemas.router, api.block_capabilities.router, api.collections.router, api.variables.router, api.csrf_token.router, api.events.router, api.automations.router, api.templates.router, api.ui.flows.router, api.ui.flow_runs.router, api.ui.schemas.router, api.ui.task_runs.router, api.admin.router, api.root.router)
SQLITE_LOCKED_MSG: str = 'database is locked'

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict) -> Any:
        try:
            return await super().get_response(path, scope)
        except HTTPException:
            return await super().get_response('./index.html', scope)

class RequestLimitMiddleware:
    def __init__(self, app: FastAPI, limit: int) -> None:
        self.app = app
        self._limiter = anyio.CapacityLimiter(limit)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        async with self._limiter:
            await self.app(scope, receive, send)

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=jsonable_encoder({'exception_message': 'Invalid request received.', 'exception_detail': exc.errors(), 'request_body': exc.body}))

async def integrity_exception_handler(request: Request, exc: sa.exc.IntegrityError) -> JSONResponse:
    logger.error('Encountered exception in request:', exc_info=True)
    return JSONResponse(content={'detail': 'Data integrity conflict. This usually means a unique or foreign key constraint was violated. See server logs for details.'}, status_code=status.HTTP_409_CONFLICT)

def is_client_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, sqlalchemy.exc.OperationalError) and isinstance(exc.orig, sqlite3.OperationalError):
        if getattr(exc.orig, 'sqlite_errorname', None) in {'SQLITE_BUSY', 'SQLITE_BUSY_SNAPSHOT'} or SQLITE_LOCKED_MSG in getattr(exc.orig, 'args', []):
            return True
        else:
            return False
    if isinstance(exc, (sqlalchemy.exc.DBAPIError, asyncpg.exceptions.QueryCanceledError, asyncpg.exceptions.ConnectionDoesNotExistError, asyncpg.exceptions.CannotConnectNowError, sqlalchemy.exc.InvalidRequestError, sqlalchemy.orm.exc.DetachedInstanceError)):
        return True
    return False

def replace_placeholder_string_in_files(directory: str, placeholder: str, replacement: str, allowed_extensions: Optional[list] = None) -> None:
    ...

def copy_directory(directory: str, path: str) -> None:
    ...

async def custom_internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    ...

async def prefect_object_not_found_exception_handler(request: Request, exc: ObjectNotFoundError) -> JSONResponse:
    ...

def create_api_app(dependencies: Optional[list] = None, health_check_path: str = '/health', version_check_path: str = '/version', fast_api_app_kwargs: Optional[dict] = None, final: bool = False) -> FastAPI:
    ...

def create_ui_app(ephemeral: bool) -> FastAPI:
    ...

def _memoize_block_auto_registration(fn: Callable) -> Callable:
    ...

def create_app(settings: Optional[Any] = None, ephemeral: bool = False, webserver_only: bool = False, final: bool = False, ignore_cache: bool = False) -> FastAPI:
    ...

subprocess_server_logger: logging.Logger = get_logger()

class SubprocessASGIServer:
    _instances: dict = {}
    _port_range: range = range(8000, 9000)

    def __new__(cls, port: Optional[int] = None, *args, **kwargs) -> SubprocessASGIServer:
        ...

    def __init__(self, port: Optional[int] = None) -> None:
        ...

    def find_available_port(self) -> int:
        ...

    @staticmethod
    def is_port_available(port: int) -> bool:
        ...

    @property
    def address(self) -> str:
        ...

    @property
    def api_url(self) -> str:
        ...

    def start(self, timeout: Optional[int] = None) -> None:
        ...

    def _run_uvicorn_command(self) -> subprocess.Popen:
        ...

    def stop(self) -> None:
        ...

    def __enter__(self) -> SubprocessASGIServer:
        ...

    def __exit__(self, *args) -> None:
        ...
