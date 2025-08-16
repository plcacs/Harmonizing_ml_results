from enum import Enum
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Sequence, Type, TypeVar, Union
from fastapi import routing
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler, websocket_request_validation_exception_handler
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.logger import logger
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.openapi.utils import get_openapi
from fastapi.params import Depends
from fastapi.types import DecoratedCallable, IncEx
from fastapi.utils import generate_unique_id
from starlette.applications import Starlette
from starlette.datastructures import State
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import BaseRoute
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from typing_extensions import Annotated, Doc, deprecated

AppType = TypeVar('AppType', bound='FastAPI')

class FastAPI(Starlette):
    def __init__(self, *, debug: bool = False, routes: Optional[List[BaseRoute]] = None, title: str = 'FastAPI', summary: Optional[str] = None, description: str = '', version: str = '0.1.0', openapi_url: str = '/openapi.json', openapi_tags: Optional[List[Dict[str, Any]]] = None, servers: Optional[List[Dict[str, Any]]] = None, dependencies: Optional[List[Depends]] = None, default_response_class: Type[Response] = Default(JSONResponse), redirect_slashes: bool = True, docs_url: str = '/docs', redoc_url: str = '/redoc', swagger_ui_oauth2_redirect_url: str = '/docs/oauth2-redirect', swagger_ui_init_oauth: Optional[Dict[str, Any]] = None, middleware: Optional[List[Middleware]] = None, exception_handlers: Optional[Dict[Union[Type[Exception], int], Callable]] = None, on_startup: Optional[List[Callable]] = None, on_shutdown: Optional[List[Callable]] = None, lifespan: Optional[Lifespan] = None, terms_of_service: Optional[str] = None, contact: Optional[Dict[str, Any]] = None, license_info: Optional[Dict[str, Any]] = None, openapi_prefix: str = '', root_path: str = '', root_path_in_servers: bool = True, responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, callbacks: Optional[Dict[str, Callable]] = None, webhooks: Optional[routing.APIRouter] = None, deprecated: Optional[Dict[str, Any]] = None, include_in_schema: bool = True, swagger_ui_parameters: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id), separate_input_output_schemas: bool = True, **extra: Any) -> None:
        ...

    def openapi(self) -> Dict[str, Any]:
        ...

    def setup(self) -> None:
        ...

    def add_api_route(self, path: str, endpoint: Callable, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, methods: Optional[List[str]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> None:
        ...

    def api_route(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, methods: Optional[List[str]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def add_api_websocket_route(self, path: str, endpoint: Callable, name: Optional[str] = None, *, dependencies: Optional[List[Depends]] = None) -> None:
        ...

    def websocket(self, path: str, name: Optional[str] = None, *, dependencies: Optional[List[Depends]] = None) -> Callable:
        ...

    def include_router(self, router: routing.APIRouter, *, prefix: str = '', tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, include_in_schema: bool = True, default_response_class: Type[Response] = Default(JSONResponse), callbacks: Optional[Dict[str, Callable]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> None:
        ...

    def get(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[Dict[str, Callable]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def put(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[Dict[str, Callable]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def post(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[Dict[str, Callable]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def delete(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[Dict[str, Callable]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def options(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[Dict[str, Callable]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def head(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[Dict[str, Callable]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def patch(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[Dict[str, Callable]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def trace(self, path: str, *, response_model: Any = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[List[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[Dict[str, Any]] = None, operation_id: Optional[str] = None, response_model_include: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_exclude: Optional[Union[Set[str], Dict[str, Union[Set[str], Dict[str, Any]]]]] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[Response] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[Dict[str, Callable]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default = Default(generate_unique_id)) -> Callable:
        ...

    def websocket_route(self, path: str, name: Optional[str] = None) -> Callable:
        ...

    @deprecated('\n        on_event is deprecated, use lifespan event handlers instead.\n\n        Read more about it in the\n        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).\n        ')
    def on_event(self, event_type: str) -> Callable:
        ...

    def middleware(self, middleware_type: str) -> Callable:
        ...

    def exception_handler(self, exc_class_or_status_code: Union[Type[Exception], int]) -> Callable:
        ...
