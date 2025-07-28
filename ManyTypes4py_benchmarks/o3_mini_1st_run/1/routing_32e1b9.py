#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import dataclasses
import email.message
import inspect
import json
from contextlib import AsyncExitStack, asynccontextmanager
from enum import Enum, IntEnum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from fastapi import params
from fastapi._compat import ModelField, Undefined, _get_model_config, _model_dump, _normalize_errors, lenient_issubclass
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import (
    _should_embed_body_fields,
    get_body_field,
    get_dependant,
    get_flat_dependant,
    get_parameterless_sub_dependant,
    get_typed_return_annotation,
    solve_dependencies,
)
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


def _prepare_response_content(
    res: Any, *, exclude_unset: bool, exclude_defaults: bool = False, exclude_none: bool = False
) -> Any:
    if isinstance(res, BaseModel):
        read_with_orm_mode = getattr(_get_model_config(res), 'read_with_orm_mode', None)
        if read_with_orm_mode:
            return res
        return _model_dump(
            res,
            by_alias=True,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
    elif isinstance(res, list):
        return [
            _prepare_response_content(
                item, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none
            )
            for item in res
        ]
    elif isinstance(res, dict):
        return {
            k: _prepare_response_content(
                v, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none
            )
            for k, v in res.items()
        }
    elif dataclasses.is_dataclass(res):
        return dataclasses.asdict(res)
    return res


def _merge_lifespan_context(
    original_context: Callable[[AppType], AsyncIterator[Any]],
    nested_context: Callable[[AppType], AsyncIterator[Any]],
) -> Callable[[AppType], AsyncIterator[Any]]:
    @asynccontextmanager
    async def merged_lifespan(app: AppType) -> AsyncIterator[Optional[Dict[Any, Any]]]:
        async with original_context(app) as maybe_original_state:
            async with nested_context(app) as maybe_nested_state:
                if maybe_nested_state is None and maybe_original_state is None:
                    yield None
                else:
                    yield {**(maybe_nested_state or {}), **(maybe_original_state or {})}

    return merged_lifespan


async def serialize_response(
    *,
    field: Optional[ModelField] = None,
    response_content: Any,
    include: Optional[Any] = None,
    exclude: Optional[Any] = None,
    by_alias: bool = True,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    is_coroutine: bool = True,
) -> Any:
    if field:
        errors: List[Any] = []
        if not hasattr(field, "serialize"):
            response_content = _prepare_response_content(
                response_content, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none
            )
        if is_coroutine:
            value, errors_ = field.validate(response_content, {}, loc=("response",))
        else:
            value, errors_ = await run_in_threadpool(field.validate, response_content, {}, loc=("response",))
        if isinstance(errors_, list):
            errors.extend(errors_)
        elif errors_:
            errors.append(errors_)
        if errors:
            raise ResponseValidationError(errors=_normalize_errors(errors), body=response_content)
        if hasattr(field, "serialize"):
            return field.serialize(
                value,
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            )
        return jsonable_encoder(
            value,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
    else:
        return jsonable_encoder(response_content)


async def run_endpoint_function(
    *, dependant: Dependant, values: Mapping[str, Any], is_coroutine: bool
) -> Any:
    assert dependant.call is not None, "dependant.call must be a function"
    if is_coroutine:
        return await dependant.call(**values)
    else:
        return await run_in_threadpool(dependant.call, **values)


def get_request_handler(
    dependant: Dependant,
    body_field: Optional[Any] = None,
    status_code: Optional[int] = None,
    response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
    response_field: Optional[ModelField] = None,
    response_model_include: Optional[Any] = None,
    response_model_exclude: Optional[Any] = None,
    response_model_by_alias: bool = True,
    response_model_exclude_unset: bool = False,
    response_model_exclude_defaults: bool = False,
    response_model_exclude_none: bool = False,
    dependency_overrides_provider: Optional[Any] = None,
    embed_body_fields: bool = False,
) -> Callable[[Request], Coroutine[Any, Any, Response]]:
    assert dependant.call is not None, "dependant.call must be a function"
    is_coroutine: bool = asyncio.iscoroutinefunction(dependant.call)
    is_body_form: bool = bool(body_field and isinstance(body_field.field_info, params.Form))
    if isinstance(response_class, DefaultPlaceholder):
        actual_response_class: Type[Response] = response_class.value
    else:
        actual_response_class = response_class

    async def app(request: Request) -> Response:
        response: Optional[Response] = None
        async with AsyncExitStack() as file_stack:
            try:
                body: Any = None
                if body_field:
                    if is_body_form:
                        body = await request.form()
                        file_stack.push_async_callback(body.close)
                    else:
                        body_bytes: bytes = await request.body()
                        if body_bytes:
                            json_body: Any = Undefined
                            content_type_value: Optional[str] = request.headers.get("content-type")
                            if not content_type_value:
                                json_body = await request.json()
                            else:
                                message = email.message.Message()
                                message["content-type"] = content_type_value
                                if message.get_content_maintype() == "application":
                                    subtype: str = message.get_content_subtype()
                                    if subtype == "json" or subtype.endswith("+json"):
                                        json_body = await request.json()
                            if json_body != Undefined:
                                body = json_body
                            else:
                                body = body_bytes
            except json.JSONDecodeError as e:
                validation_error = RequestValidationError(
                    [
                        {
                            "type": "json_invalid",
                            "loc": ("body", e.pos),
                            "msg": "JSON decode error",
                            "input": {},
                            "ctx": {"error": e.msg},
                        }
                    ],
                    body=e.doc,
                )
                raise validation_error from e
            except HTTPException:
                raise
            except Exception as e:
                http_error = HTTPException(status_code=400, detail="There was an error parsing the body")
                raise http_error from e
            errors: List[Any] = []
            async with AsyncExitStack() as async_exit_stack:
                solved_result = await solve_dependencies(
                    request=request,
                    dependant=dependant,
                    body=body,
                    dependency_overrides_provider=dependency_overrides_provider,
                    async_exit_stack=async_exit_stack,
                    embed_body_fields=embed_body_fields,
                )
                errors = solved_result.errors
                if not errors:
                    raw_response = await run_endpoint_function(
                        dependant=dependant, values=solved_result.values, is_coroutine=is_coroutine
                    )
                    if isinstance(raw_response, Response):
                        if raw_response.background is None:
                            raw_response.background = solved_result.background_tasks
                        response = raw_response
                    else:
                        response_args: Dict[str, Any] = {"background": solved_result.background_tasks}
                        current_status_code = status_code if status_code else solved_result.response.status_code
                        if current_status_code is not None:
                            response_args["status_code"] = current_status_code
                        if solved_result.response.status_code:
                            response_args["status_code"] = solved_result.response.status_code
                        content = await serialize_response(
                            field=response_field,
                            response_content=raw_response,
                            include=response_model_include,
                            exclude=response_model_exclude,
                            by_alias=response_model_by_alias,
                            exclude_unset=response_model_exclude_unset,
                            exclude_defaults=response_model_exclude_defaults,
                            exclude_none=response_model_exclude_none,
                            is_coroutine=is_coroutine,
                        )
                        response = actual_response_class(content, **response_args)
                        if not is_body_allowed_for_status_code(response.status_code):
                            response.body = b""
                        response.headers.raw.extend(solved_result.response.headers.raw)
            if errors:
                validation_error = RequestValidationError(_normalize_errors(errors), body=body)
                raise validation_error
        if response is None:
            raise FastAPIError(
                "No response object was returned. There's a high chance that the application code is raising an "
                "exception and a dependency with yield has a block with a bare except, or a block with except Exception, "
                "and is not raising the exception again. Read more about it in the docs: "
                "https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#dependencies-with-yield-and-except"
            )
        return response

    return app


def get_websocket_app(
    dependant: Dependant,
    dependency_overrides_provider: Optional[Any] = None,
    embed_body_fields: bool = False,
) -> Callable[[WebSocket], Coroutine[Any, Any, None]]:
    async def app(websocket: WebSocket) -> None:
        async with AsyncExitStack() as async_exit_stack:
            websocket.scope["fastapi_astack"] = async_exit_stack
            solved_result = await solve_dependencies(
                request=websocket,
                dependant=dependant,
                dependency_overrides_provider=dependency_overrides_provider,
                async_exit_stack=async_exit_stack,
                embed_body_fields=embed_body_fields,
            )
            if solved_result.errors:
                raise WebSocketRequestValidationError(_normalize_errors(solved_result.errors))
            assert dependant.call is not None, "dependant.call must be a function"
            await dependant.call(**solved_result.values)

    return app


class APIWebSocketRoute(routing.WebSocketRoute):
    def __init__(
        self,
        path: str,
        endpoint: Callable[..., Any],
        *,
        name: Optional[str] = None,
        dependencies: Optional[Sequence[Any]] = None,
        dependency_overrides_provider: Optional[Any] = None,
    ) -> None:
        self.path: str = path
        self.endpoint: Callable[..., Any] = endpoint
        self.name: str = get_name(endpoint) if name is None else name
        self.dependencies: List[Any] = list(dependencies or [])
        self.path_regex, self.path_format, self.param_convertors = compile_path(path)
        self.dependant: Dependant = get_dependant(path=self.path_format, call=self.endpoint)
        for depends in self.dependencies[::-1]:
            self.dependant.dependencies.insert(
                0, get_parameterless_sub_dependant(depends=depends, path=self.path_format)
            )
        self._flat_dependant = get_flat_dependant(self.dependant)
        self._embed_body_fields = _should_embed_body_fields(self._flat_dependant.body_params)
        self.app: ASGIApp = websocket_session(
            get_websocket_app(
                dependant=self.dependant,
                dependency_overrides_provider=dependency_overrides_provider,
                embed_body_fields=self._embed_body_fields,
            )
        )

    def matches(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        match, child_scope = super().matches(scope)
        if match != Match.NONE:
            child_scope["route"] = self
        return (match, child_scope)


class APIRoute(routing.Route):
    def __init__(
        self,
        path: str,
        endpoint: Callable[..., Any],
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        name: Optional[str] = None,
        methods: Optional[Sequence[str]] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        dependency_overrides_provider: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> None:
        self.path: str = path
        self.endpoint: Callable[..., Any] = endpoint
        if isinstance(response_model, DefaultPlaceholder):
            return_annotation = get_typed_return_annotation(endpoint)
            if lenient_issubclass(return_annotation, Response):
                response_model = None
            else:
                response_model = return_annotation
        self.response_model: Any = response_model
        self.summary: Optional[str] = summary
        self.response_description: str = response_description
        self.deprecated: Optional[bool] = deprecated
        self.operation_id: Optional[str] = operation_id
        self.response_model_include: Optional[Any] = response_model_include
        self.response_model_exclude: Optional[Any] = response_model_exclude
        self.response_model_by_alias: bool = response_model_by_alias
        self.response_model_exclude_unset: bool = response_model_exclude_unset
        self.response_model_exclude_defaults: bool = response_model_exclude_defaults
        self.response_model_exclude_none: bool = response_model_exclude_none
        self.include_in_schema: bool = include_in_schema
        self.response_class: Union[DefaultPlaceholder, Type[Response]] = response_class
        self.dependency_overrides_provider: Optional[Any] = dependency_overrides_provider
        self.callbacks: Any = callbacks
        self.openapi_extra: Optional[Any] = openapi_extra
        self.generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = generate_unique_id_function
        self.tags: List[Any] = tags.copy() if tags else []
        self.responses: Dict[Any, Any] = responses or {}
        self.name: str = get_name(endpoint) if name is None else name
        self.path_regex, self.path_format, self.param_convertors = compile_path(path)
        if methods is None:
            methods = ["GET"]
        self.methods: Set[str] = {method.upper() for method in methods}
        if isinstance(generate_unique_id_function, DefaultPlaceholder):
            current_generate_unique_id = generate_unique_id_function.value
        else:
            current_generate_unique_id = generate_unique_id_function
        self.unique_id: str = self.operation_id or current_generate_unique_id(self)
        if isinstance(status_code, IntEnum):
            status_code = int(status_code)
        self.status_code: Optional[int] = status_code
        if self.response_model:
            assert is_body_allowed_for_status_code(status_code), f"Status code {status_code} must not have a response body"
            response_name: str = "Response_" + self.unique_id
            self.response_field: ModelField = create_model_field(name=response_name, type_=self.response_model, mode="serialization")
            self.secure_cloned_response_field: ModelField = create_cloned_field(self.response_field)
        else:
            self.response_field = None
            self.secure_cloned_response_field = None
        self.dependencies: List[Any] = list(dependencies or [])
        self.description: str = description or inspect.cleandoc(self.endpoint.__doc__ or "")
        self.description = self.description.split("\x0c")[0].strip()
        response_fields: Dict[Any, ModelField] = {}
        for additional_status_code, response in self.responses.items():
            assert isinstance(response, dict), "An additional response must be a dict"
            model = response.get("model")
            if model:
                assert is_body_allowed_for_status_code(additional_status_code), f"Status code {additional_status_code} must not have a response body"
                response_name = f"Response_{additional_status_code}_{self.unique_id}"
                response_field = create_model_field(name=response_name, type_=model, mode="serialization")
                response_fields[additional_status_code] = response_field
        if response_fields:
            self.response_fields: Dict[Any, ModelField] = response_fields
        else:
            self.response_fields = {}
        assert callable(endpoint), "An endpoint must be a callable"
        self.dependant: Dependant = get_dependant(path=self.path_format, call=self.endpoint)
        for depends in self.dependencies[::-1]:
            self.dependant.dependencies.insert(
                0, get_parameterless_sub_dependant(depends=depends, path=self.path_format)
            )
        self._flat_dependant = get_flat_dependant(self.dependant)
        self._embed_body_fields: bool = _should_embed_body_fields(self._flat_dependant.body_params)
        self.body_field = get_body_field(flat_dependant=self._flat_dependant, name=self.unique_id, embed_body_fields=self._embed_body_fields)
        self.app: ASGIApp = request_response(self.get_route_handler())

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        return get_request_handler(
            dependant=self.dependant,
            body_field=self.body_field,
            status_code=self.status_code,
            response_class=self.response_class,
            response_field=self.secure_cloned_response_field,
            response_model_include=self.response_model_include,
            response_model_exclude=self.response_model_exclude,
            response_model_by_alias=self.response_model_by_alias,
            response_model_exclude_unset=self.response_model_exclude_unset,
            response_model_exclude_defaults=self.response_model_exclude_defaults,
            response_model_exclude_none=self.response_model_exclude_none,
            dependency_overrides_provider=self.dependency_overrides_provider,
            embed_body_fields=self._embed_body_fields,
        )

    def matches(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        match, child_scope = super().matches(scope)
        if match != Match.NONE:
            child_scope["route"] = self
        return (match, child_scope)


class APIRouter(routing.Router):
    def __init__(
        self,
        *,
        prefix: str = "",
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        default_response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        responses: Optional[Dict[Any, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        routes: Optional[List[BaseRoute]] = None,
        redirect_slashes: bool = True,
        default: Optional[Any] = None,
        dependency_overrides_provider: Optional[Any] = None,
        route_class: Type[APIRoute] = APIRoute,
        on_startup: Optional[Sequence[Callable[..., Any]]] = None,
        on_shutdown: Optional[Sequence[Callable[..., Any]]] = None,
        lifespan: Optional[Callable[..., Any]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> None:
        super().__init__(
            routes=routes,
            redirect_slashes=redirect_slashes,
            default=default,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )
        if prefix:
            assert prefix.startswith("/"), "A path prefix must start with '/'"
            assert not prefix.endswith("/"), "A path prefix must not end with '/', as the routes will start with '/'"
        self.prefix: str = prefix
        self.tags: List[Any] = list(tags or [])
        self.dependencies: List[Any] = list(dependencies or [])
        self.deprecated: Optional[bool] = deprecated
        self.include_in_schema: bool = include_in_schema
        self.responses: Dict[Any, Any] = responses or {}
        self.callbacks: List[Any] = callbacks or []
        self.dependency_overrides_provider: Optional[Any] = dependency_overrides_provider
        self.route_class: Type[APIRoute] = route_class
        self.default_response_class: Union[DefaultPlaceholder, Type[Response]] = default_response_class
        self.generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = generate_unique_id_function

    def route(self, path: str, methods: Optional[Sequence[str]] = None, name: Optional[str] = None, include_in_schema: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_route(path, func, methods=methods, name=name, include_in_schema=include_in_schema)
            return func
        return decorator

    def add_api_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        methods: Optional[Sequence[str]] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        route_class_override: Optional[Type[APIRoute]] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> None:
        route_class: Type[APIRoute] = route_class_override or self.route_class
        responses = responses or {}
        combined_responses: Dict[Any, Any] = {**self.responses, **responses}
        current_response_class: Union[DefaultPlaceholder, Type[Response]] = get_value_or_default(response_class, self.default_response_class)
        current_tags: List[Any] = self.tags.copy()
        if tags:
            current_tags.extend(tags)
        current_dependencies: List[Any] = self.dependencies.copy()
        if dependencies:
            current_dependencies.extend(dependencies)
        current_callbacks: List[Any] = self.callbacks.copy()
        if callbacks:
            current_callbacks.extend(callbacks)
        current_generate_unique_id: Union[DefaultPlaceholder, Callable[[Any], str]] = get_value_or_default(generate_unique_id_function, self.generate_unique_id_function)
        route = route_class(
            self.prefix + path,
            endpoint=endpoint,
            response_model=response_model,
            status_code=status_code,
            tags=current_tags,
            dependencies=current_dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=combined_responses,
            deprecated=deprecated or self.deprecated,
            methods=methods,
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema and self.include_in_schema,
            response_class=current_response_class,
            name=name,
            dependency_overrides_provider=self.dependency_overrides_provider,
            callbacks=current_callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=current_generate_unique_id,
        )
        self.routes.append(route)

    def api_route(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(
                path,
                func,
                response_model=response_model,
                status_code=status_code,
                tags=tags,
                dependencies=dependencies,
                summary=summary,
                description=description,
                response_description=response_description,
                responses=responses,
                deprecated=deprecated,
                methods=["GET"],
                operation_id=operation_id,
                response_model_include=response_model_include,
                response_model_exclude=response_model_exclude,
                response_model_by_alias=response_model_by_alias,
                response_model_exclude_unset=response_model_exclude_unset,
                response_model_exclude_defaults=response_model_exclude_defaults,
                response_model_exclude_none=response_model_exclude_none,
                include_in_schema=include_in_schema,
                response_class=response_class,
                name=name,
                callbacks=callbacks,
                openapi_extra=openapi_extra,
                generate_unique_id_function=generate_unique_id_function,
            )
            return func
        return decorator

    def add_api_websocket_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        name: Optional[str] = None,
        *,
        dependencies: Optional[Sequence[Any]] = None,
    ) -> None:
        current_dependencies: List[Any] = self.dependencies.copy()
        if dependencies:
            current_dependencies.extend(dependencies)
        route = APIWebSocketRoute(
            self.prefix + path,
            endpoint=endpoint,
            name=name,
            dependencies=current_dependencies,
            dependency_overrides_provider=self.dependency_overrides_provider,
        )
        self.routes.append(route)

    def websocket(self, path: str, name: Optional[str] = None, *, dependencies: Optional[Sequence[Any]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_websocket_route(path, func, name=name, dependencies=dependencies)
            return func
        return decorator

    def websocket_route(self, path: str, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_websocket_route(self.prefix + path, func, name=name)
            return func
        return decorator

    def include_router(
        self,
        router: APIRouter,
        *,
        prefix: str = "",
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        default_response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        responses: Optional[Dict[Any, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> None:
        if prefix:
            assert prefix.startswith("/"), "A path prefix must start with '/'"
            assert not prefix.endswith("/"), "A path prefix must not end with '/', as the routes will start with '/'"
        else:
            for r in router.routes:
                path_value = getattr(r, "path", None)
                name_value = getattr(r, "name", "unknown")
                if path_value is not None and (not path_value):
                    raise FastAPIError(f"Prefix and path cannot be both empty (path operation: {name_value})")
        if responses is None:
            responses = {}
        for route in router.routes:
            if isinstance(route, APIRoute):
                combined_responses: Dict[Any, Any] = {**responses, **route.responses}
                use_response_class: Union[DefaultPlaceholder, Type[Response]] = get_value_or_default(
                    route.response_class, router.default_response_class, default_response_class, self.default_response_class
                )
                current_tags: List[Any] = []
                if tags:
                    current_tags.extend(tags)
                if route.tags:
                    current_tags.extend(route.tags)
                current_dependencies: List[Any] = []
                if dependencies:
                    current_dependencies.extend(dependencies)
                if route.dependencies:
                    current_dependencies.extend(route.dependencies)
                current_callbacks: List[Any] = []
                if callbacks:
                    current_callbacks.extend(callbacks)
                if route.callbacks:
                    current_callbacks.extend(route.callbacks)
                current_generate_unique_id: Union[DefaultPlaceholder, Callable[[Any], str]] = get_value_or_default(
                    route.generate_unique_id_function, router.generate_unique_id_function, generate_unique_id_function, self.generate_unique_id_function
                )
                self.add_api_route(
                    prefix + route.path,
                    route.endpoint,
                    response_model=route.response_model,
                    status_code=route.status_code,
                    tags=current_tags,
                    dependencies=current_dependencies,
                    summary=route.summary,
                    description=route.description,
                    response_description=route.response_description,
                    responses=combined_responses,
                    deprecated=route.deprecated or deprecated or self.deprecated,
                    methods=route.methods,
                    operation_id=route.operation_id,
                    response_model_include=route.response_model_include,
                    response_model_exclude=route.response_model_exclude,
                    response_model_by_alias=route.response_model_by_alias,
                    response_model_exclude_unset=route.response_model_exclude_unset,
                    response_model_exclude_defaults=route.response_model_exclude_defaults,
                    response_model_exclude_none=route.response_model_exclude_none,
                    include_in_schema=route.include_in_schema and self.include_in_schema and include_in_schema,
                    response_class=use_response_class,
                    name=route.name,
                    route_class_override=type(route),
                    callbacks=current_callbacks,
                    openapi_extra=route.openapi_extra,
                    generate_unique_id_function=current_generate_unique_id,
                )
            elif isinstance(route, routing.Route):
                methods = list(route.methods or [])
                self.add_route(prefix + route.path, route.endpoint, methods=methods, include_in_schema=route.include_in_schema, name=route.name)
            elif isinstance(route, APIWebSocketRoute):
                current_dependencies = []
                if dependencies:
                    current_dependencies.extend(dependencies)
                if route.dependencies:
                    current_dependencies.extend(route.dependencies)
                self.add_api_websocket_route(prefix + route.path, route.endpoint, dependencies=current_dependencies, name=route.name)
            elif isinstance(route, routing.WebSocketRoute):
                self.add_websocket_route(prefix + route.path, route.endpoint, name=route.name)
        for handler in router.on_startup:
            self.add_event_handler("startup", handler)
        for handler in router.on_shutdown:
            self.add_event_handler("shutdown", handler)
        self.lifespan_context = _merge_lifespan_context(self.lifespan_context, router.lifespan_context)

    def get(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api_route(
            path=path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=["GET"],
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            response_class=response_class,
            name=name,
            callbacks=callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    def put(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api_route(
            path=path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=["PUT"],
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            response_class=response_class,
            name=name,
            callbacks=callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    def post(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api_route(
            path=path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=["POST"],
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            response_class=response_class,
            name=name,
            callbacks=callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    def delete(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api_route(
            path=path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=["DELETE"],
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            response_class=response_class,
            name=name,
            callbacks=callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    def options(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api_route(
            path=path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=["OPTIONS"],
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            response_class=response_class,
            name=name,
            callbacks=callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    def head(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api_route(
            path=path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=["HEAD"],
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            response_class=response_class,
            name=name,
            callbacks=callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    def patch(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api_route(
            path=path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=["PATCH"],
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            response_class=response_class,
            name=name,
            callbacks=callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    def trace(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[Union[int, IntEnum]] = None,
        tags: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Any, Any]] = None,
        deprecated: Optional[bool] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[Any] = None,
        response_model_exclude: Optional[Any] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[DefaultPlaceholder, Type[Response]] = Default(JSONResponse),
        name: Optional[str] = None,
        callbacks: Optional[Any] = None,
        openapi_extra: Optional[Any] = None,
        generate_unique_id_function: Union[DefaultPlaceholder, Callable[[Any], str]] = Default(generate_unique_id),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api_route(
            path=path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=["TRACE"],
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            response_class=response_class,
            name=name,
            callbacks=callbacks,
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    @deprecated(
        "\n        on_event is deprecated, use lifespan event handlers instead.\n\n        Read more about it in the\n        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).\n        "
    )
    def on_event(self, event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_event_handler(event_type, func)
            return func
        return decorator
