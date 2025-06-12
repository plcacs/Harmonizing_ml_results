"""
Defines the Prefect REST API FastAPI app.
"""
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
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type, Union
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
from starlette.types import ASGIApp, Message, Receive, Scope, Send
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
    from pathlib import Path
    from sqlalchemy.orm import Session
    from prefect.server.database.interface import PrefectDBInterface

TITLE: str = 'Prefect Server'
API_TITLE: str = 'Prefect Prefect REST API'
UI_TITLE: str = 'Prefect Prefect REST API UI'
API_VERSION: str = prefect.__version__
LIFESPAN_RAN_FOR_APP: Set[FastAPI] = set()
logger: logging.Logger = get_logger('server')
enforce_minimum_version: EnforceMinimumAPIVersion = EnforceMinimumAPIVersion(minimum_api_version='0.8.0', logger=logger)
API_ROUTERS: Tuple[fastapi.routing.APIRouter, ...] = (
    api.flows.router, api.flow_runs.router, api.task_runs.router, 
    api.flow_run_states.router, api.task_run_states.router, 
    api.flow_run_notification_policies.router, api.deployments.router, 
    api.saved_searches.router, api.logs.router, api.concurrency_limits.router, 
    api.concurrency_limits_v2.router, api.block_types.router, 
    api.block_documents.router, api.workers.router, api.task_workers.router, 
    api.work_queues.router, api.artifacts.router, api.block_schemas.router, 
    api.block_capabilities.router, api.collections.router, api.variables.router, 
    api.csrf_token.router, api.events.router, api.automations.router, 
    api.templates.router, api.ui.flows.router, api.ui.flow_runs.router, 
    api.ui.schemas.router, api.ui.task_runs.router, api.admin.router, 
    api.root.router
)
SQLITE_LOCKED_MSG: str = 'database is locked'

class SPAStaticFiles(StaticFiles):
    """
    Implementation of `StaticFiles` for serving single page applications.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException:
            return await super().get_response('./index.html', scope)

class RequestLimitMiddleware:
    """
    A middleware that limits the number of concurrent requests handled by the API.
    """

    def __init__(self, app: ASGIApp, limit: int):
        self.app = app
        self._limiter: anyio.CapacityLimiter = anyio.CapacityLimiter(limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with self._limiter:
            await self.app(scope, receive, send)

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Provide a detailed message for request validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({
            'exception_message': 'Invalid request received.',
            'exception_detail': exc.errors(),
            'request_body': exc.body
        })
    )

async def integrity_exception_handler(request: Request, exc: sa.exc.IntegrityError) -> JSONResponse:
    """Capture database integrity errors."""
    logger.error('Encountered exception in request:', exc_info=True)
    return JSONResponse(
        content={'detail': 'Data integrity conflict. This usually means a unique or foreign key constraint was violated. See server logs for details.'},
        status_code=status.HTTP_409_CONFLICT
    )

def is_client_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, sqlalchemy.exc.OperationalError) and isinstance(exc.orig, sqlite3.OperationalError):
        if getattr(exc.orig, 'sqlite_errorname', None) in {'SQLITE_BUSY', 'SQLITE_BUSY_SNAPSHOT'} or SQLITE_LOCKED_MSG in getattr(exc.orig, 'args', []):
            return True
        else:
            return False
    if isinstance(exc, (sqlalchemy.exc.DBAPIError, asyncpg.exceptions.QueryCanceledError, 
                      asyncpg.exceptions.ConnectionDoesNotExistError, 
                      asyncpg.exceptions.CannotConnectNowError, 
                      sqlalchemy.exc.InvalidRequestError, 
                      sqlalchemy.orm.exc.DetachedInstanceError)):
        return True
    return False

def replace_placeholder_string_in_files(
    directory: str, 
    placeholder: str, 
    replacement: str, 
    allowed_extensions: Optional[List[str]] = None
) -> None:
    """
    Recursively loops through all files in the given directory and replaces
    a placeholder string.
    """
    if allowed_extensions is None:
        allowed_extensions = ['.txt', '.html', '.css', '.js', '.json', '.txt']
    for root, _, files in os.walk(directory):
        for file in files:
            if any((file.endswith(ext) for ext in allowed_extensions):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_data = file.read()
                file_data = file_data.replace(placeholder, replacement)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(file_data)

def copy_directory(directory: str, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for item in os.listdir(directory):
        source = os.path.join(directory, item)
        destination = os.path.join(path, item)
        if os.path.isdir(source):
            if os.path.exists(destination):
                shutil.rmtree(destination)
            shutil.copytree(source, destination, symlinks=True)
        else:
            shutil.copy2(source, destination)

async def custom_internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Log a detailed exception for internal server errors before returning.
    Send 503 for errors clients can retry on.
    """
    if is_client_retryable_exception(exc):
        if PREFECT_API_LOG_RETRYABLE_ERRORS.value():
            logger.error('Encountered retryable exception in request:', exc_info=True)
        return JSONResponse(
            content={'exception_message': 'Service Unavailable'},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    logger.error('Encountered exception in request:', exc_info=True)
    return JSONResponse(
        content={'exception_message': 'Internal Server Error'},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )

async def prefect_object_not_found_exception_handler(request: Request, exc: ObjectNotFoundError) -> JSONResponse:
    """Return 404 status code on object not found exceptions."""
    return JSONResponse(
        content={'exception_message': str(exc)},
        status_code=status.HTTP_404_NOT_FOUND
    )

def create_api_app(
    dependencies: Optional[List[Depends]] = None,
    health_check_path: str = '/health',
    version_check_path: str = '/version',
    fast_api_app_kwargs: Optional[Dict[str, Any]] = None,
    final: bool = False
) -> FastAPI:
    """
    Create a FastAPI app that includes the Prefect REST API
    """
    fast_api_app_kwargs = fast_api_app_kwargs or {}
    api_app = FastAPI(title=API_TITLE, **fast_api_app_kwargs)
    api_app.add_middleware(GZipMiddleware)

    @api_app.get(health_check_path, tags=['Root'])
    async def health_check() -> bool:
        return True

    @api_app.get(version_check_path, tags=['Root'])
    async def server_version() -> str:
        return SERVER_API_VERSION

    if dependencies is None:
        dependencies = [Depends(enforce_minimum_version)]
    else:
        dependencies.append(Depends(enforce_minimum_version))

    for router in API_ROUTERS:
        api_app.include_router(router, dependencies=dependencies)
        if final:
            del router.routes
    if final:
        gc.collect()

    auth_string = prefect.settings.PREFECT_SERVER_API_AUTH_STRING.value()
    if auth_string is not None:

        @api_app.middleware('http')
        async def token_validation(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            header_token = request.headers.get('Authorization')
            if request.url.path in ['/api/health', '/api/ready']:
                return await call_next(request)
            try:
                if header_token is None:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={'exception_message': 'Unauthorized'}
                    )
                scheme, creds = header_token.split()
                assert scheme == 'Basic'
                decoded = base64.b64decode(creds).decode('utf-8')
            except Exception:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={'exception_message': 'Unauthorized'}
                )
            if decoded != auth_string:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={'exception_message': 'Unauthorized'}
                )
            return await call_next(request)
    return api_app

def create_ui_app(ephemeral: bool) -> FastAPI:
    ui_app = FastAPI(title=UI_TITLE)
    base_url = prefect.settings.PREFECT_UI_SERVE_BASE.value()
    cache_key = f'{prefect.__version__}:{base_url}'
    stripped_base_url = base_url.rstrip('/')
    static_dir = prefect.settings.PREFECT_UI_STATIC_DIRECTORY.value() or prefect.__ui_static_subpath__
    reference_file_name = 'UI_SERVE_BASE'
    if os.name == 'nt':
        mimetypes.init()
        mimetypes.add_type('application/javascript', '.js')

    @ui_app.get(f'{stripped_base_url}/ui-settings')
    def ui_settings() -> Dict[str, Any]:
        return {
            'api_url': prefect.settings.PREFECT_UI_API_URL.value(),
            'csrf_enabled': prefect.settings.PREFECT_SERVER_CSRF_PROTECTION_ENABLED.value(),
            'auth': 'BASIC' if prefect.settings.PREFECT_SERVER_API_AUTH_STRING.value() else None,
            'flags': []
        }

    def reference_file_matches_base_url() -> bool:
        reference_file_path = os.path.join(static_dir, reference_file_name)
        if os.path.exists(static_dir):
            try:
                with open(reference_file_path, 'r') as f:
                    return f.read() == cache_key
            except FileNotFoundError:
                return False
        else:
            return False

    def create_ui_static_subpath() -> None:
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        copy_directory(str(prefect.__ui_static_path__), str(static_dir))
        replace_placeholder_string_in_files(
            str(static_dir),
            '/PREFECT_UI_SERVE_BASE_REPLACE_PLACEHOLDER',
            stripped_base_url
        )
        with open(os.path.join(static_dir, reference_file_name), 'w') as f:
            f.write(cache_key)

    ui_app.add_middleware(GZipMiddleware)
    if os.path.exists(prefect.__ui_static_path__) and prefect.settings.PREFECT_UI_ENABLED.value() and (not ephemeral):
        if not reference_file_matches_base_url():
            create_ui_static_subpath()
        ui_app.mount(
            PREFECT_UI_SERVE_BASE.value(),
            SPAStaticFiles(directory=static_dir),
            name='ui_root'
        )
    return ui_app

APP_CACHE: Dict[Tuple[str, bool, bool], FastAPI] = {}

def _memoize_block_auto_registration(fn: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    """
    Decorator to handle skipping the wrapped function if the block registry has
    not changed since the last invocation
    """
    import toml
    import prefect.plugins
    from prefect.blocks.core import Block
    from prefect.server.models.block_registration import _load_collection_blocks_data
    from prefect.utilities.dispatch import get_registry_for_type

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> None:
        if not PREFECT_MEMOIZE_BLOCK_AUTO_REGISTRATION.value():
            await fn(*args, **kwargs)
            return
        prefect.plugins.load_prefect_collections()
        blocks_registry = get_registry_for_type(Block)
        collection_blocks_data = await _load_collection_blocks_data()
        current_blocks_loading_hash = hash_objects(
            blocks_registry,
            collection_blocks_data,
            PREFECT_API_DATABASE_CONNECTION_URL.value(),
            hash_algo=sha256
        )
        memo_store_path = PREFECT_MEMO_STORE_PATH.value()
        try:
            if memo_store_path.exists():
                saved_blocks_loading_hash = toml.load(memo_store_path).get('block_auto_registration')
                if saved_blocks_loading_hash is not None and current_blocks_loading_hash == saved_blocks_loading_hash:
                    if PREFECT_DEBUG_MODE.value():
                        logger.debug('Skipping block loading due to matching hash for block auto-registration found in memo store.')
                    return
        except Exception as exc:
            logger.warning(f'Unable to read memo_store.toml from {PREFECT_MEMO_STORE_PATH} during block auto-registration: {exc!r}.\nAll blocks will be registered.')
        await fn(*args, **kwargs)
        if current_blocks_loading_hash is not None:
            try:
                if not memo_store_path.exists():
                    memo_store_path.touch(mode=384)
                memo_store_path.write_text(toml.dumps({'block_auto_registration': current_blocks_loading_hash}))
            except Exception as exc:
                logger.warning(f'Unable to write to memo_store.toml at {PREFECT_MEMO_STORE_PATH} after block auto-registration: {exc!r}.\n Subsequent server start ups will perform block auto-registration, which may result in slower server startup.')
    return wrapper

def create_app(
    settings: Optional[prefect.settings.Settings] = None,
    ephemeral: bool = False,
    webserver_only: bool = False,
    final: bool = False,
    ignore_cache: bool = False
) -> FastAPI:
    """
    Create a FastAPI app that includes the Prefect REST API and UI
    """
    settings = settings or prefect.settings.get_current_settings()
    cache_key = (settings.hash_key(), ephemeral, webserver_only)
    ephemeral = ephemeral or bool(os.getenv('PREFECT__SERVER_EPHEMERAL'))
    webserver_only = webserver_only or bool(os.getenv('PREFECT__SERVER_WEBSERVER_ONLY'))
    final = final or bool(os.getenv('PREFECT__SERVER_FINAL'))
    from prefect.logging.configuration import setup_logging
    setup_logging()

    if cache_key in APP_CACHE and (not ignore_cache):
        return APP_CACHE[cache_key]

    async def run_migrations() -> None:
        """Ensure the database is created and up to date with the current migrations"""
        if prefect.settings.PREFECT_API_DATABASE_MIGRATE_ON_START:
            from prefect.server.database import provide_database_interface
            db: PrefectDBInterface = provide_database_interface()
            await db.create_db()

    @_memoize_block_auto_registration
    async def add_block_types() -> None:
        """Add all registered blocks to the database"""
