from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union
from fastapi import params
from fastapi._compat import ModelField, Undefined, _get_model_config, _model_dump, _normalize_errors, lenient_issubclass
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import _should_embed_body_fields, get_body_field, get_dependant, get_flat_dependant, get_parameterless_sub_dependant, get_typed_return_annotation, solve_dependencies
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import FastAPIError, RequestValidationError, ResponseValidationError, WebSocketRequestValidationError
from fastapi.types import DecoratedCallable, IncEx
from fastapi.utils import create_cloned_field, create_model_field, generate_unique_id, get_value_or_default, is_body_allowed_for_status_code
from pydantic import BaseModel
from starlette import routing
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import BaseRoute, Match, compile_path, get_name, request_response, websocket_session
from starlette.routing import Mount as Mount
from starlette.types import AppType, ASGIApp, Lifespan, Scope
from starlette.websockets import WebSocket
from typing_extensions import Annotated, Doc, deprecated

def _prepare_response_content(res: Any, *, exclude_unset: bool, exclude_defaults: bool = False, exclude_none: bool = False) -> Any:
    ...

def _merge_lifespan_context(original_context, nested_context) -> Callable:
    ...

async def serialize_response(*, field=None, response_content, include=None, exclude=None, by_alias=True, exclude_unset=False, exclude_defaults=False, exclude_none=False, is_coroutine=True) -> Any:
    ...

async def run_endpoint_function(*, dependant, values, is_coroutine) -> Any:
    ...

def get_request_handler(dependant, body_field=None, status_code=None, response_class=Default(JSONResponse), response_field=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, dependency_overrides_provider=None, embed_body_fields=False) -> Callable:
    ...

def get_websocket_app(dependant, dependency_overrides_provider=None, embed_body_fields=False) -> Callable:
    ...

class APIWebSocketRoute(routing.WebSocketRoute):
    ...

class APIRoute(routing.Route):
    ...

class APIRouter(routing.Router):
    ...

    def route(self, path, methods=None, name=None, include_in_schema=True) -> Callable:
        ...

    def add_api_route(self, path, endpoint, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, methods=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, route_class_override=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> None:
        ...

    def api_route(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, methods=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    def add_api_websocket_route(self, path, endpoint, name=None, *, dependencies=None) -> None:
        ...

    def websocket(self, path, name=None, *, dependencies=None) -> Callable:
        ...

    def websocket_route(self, path, name=None) -> Callable:
        ...

    def include_router(self, router, *, prefix='', tags=None, dependencies=None, default_response_class=Default(JSONResponse), responses=None, callbacks=None, deprecated=None, include_in_schema=True, generate_unique_id_function=Default(generate_unique_id)) -> None:
        ...

    def get(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    def put(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    def post(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    def delete(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    def options(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    def head(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    def patch(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    def trace(self, path, *, response_model=Default(None), status_code=None, tags=None, dependencies=None, summary=None, description=None, response_description='Successful Response', responses=None, deprecated=None, operation_id=None, response_model_include=None, response_model_exclude=None, response_model_by_alias=True, response_model_exclude_unset=False, response_model_exclude_defaults=False, response_model_exclude_none=False, include_in_schema=True, response_class=Default(JSONResponse), name=None, callbacks=None, openapi_extra=None, generate_unique_id_function=Default(generate_unique_id)) -> Callable:
        ...

    @deprecated('\n        on_event is deprecated, use lifespan event handlers instead.\n\n        Read more about it in the\n        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).\n        ')
    def on_event(self, event_type) -> Callable:
        ...
