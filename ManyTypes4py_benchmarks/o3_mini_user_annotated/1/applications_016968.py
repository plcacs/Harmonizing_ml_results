from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import Annotated, Literal

from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
    websocket_request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.logger import logger
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
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

AppType = TypeVar("AppType", bound="FastAPI")


class FastAPI(Starlette):
    """
    FastAPI app class, the main entrypoint to use FastAPI.
    """

    def __init__(
        self: AppType,
        *,
        debug: Annotated[
            bool,
            "Boolean indicating if debug tracebacks should be returned on server errors."
        ] = False,
        routes: Annotated[
            Optional[List[BaseRoute]],
            "A list of routes to serve incoming HTTP and WebSocket requests."
        ] = None,
        title: Annotated[
            str,
            "The title of the API, visible in the generated OpenAPI docs."
        ] = "FastAPI",
        summary: Annotated[
            Optional[str],
            "A short summary of the API, visible in the generated OpenAPI docs."
        ] = None,
        description: Annotated[
            str,
            "A description of the API. Supports Markdown."
        ] = "",
        version: Annotated[
            str,
            "The version of the API."
        ] = "0.1.0",
        openapi_url: Annotated[
            Optional[str],
            "The URL where the OpenAPI schema will be served."
        ] = "/openapi.json",
        openapi_tags: Annotated[
            Optional[List[Dict[str, Any]]],
            "A list of tags used by OpenAPI."
        ] = None,
        servers: Annotated[
            Optional[List[Dict[str, Union[str, Any]]]],
            "A list of server information for the API."
        ] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]],
            "A list of global dependencies to be applied to every path operation."
        ] = None,
        default_response_class: Annotated[
            Type[Response],
            "The default response class to be used."
        ] = Default(JSONResponse),
        redirect_slashes: Annotated[
            bool,
            "Whether to detect and redirect slashes in URLs when needed."
        ] = True,
        docs_url: Annotated[
            Optional[str],
            "The path to the Swagger UI docs."
        ] = "/docs",
        redoc_url: Annotated[
            Optional[str],
            "The path to the ReDoc docs."
        ] = "/redoc",
        swagger_ui_oauth2_redirect_url: Annotated[
            Optional[str],
            "The OAuth2 redirect endpoint for Swagger UI."
        ] = "/docs/oauth2-redirect",
        swagger_ui_init_oauth: Annotated[
            Optional[Dict[str, Any]],
            "OAuth2 configuration for Swagger UI."
        ] = None,
        middleware: Annotated[
            Optional[Sequence[Middleware]],
            "List of middleware to be added when creating the application."
        ] = None,
        exception_handlers: Annotated[
            Optional[
                Dict[
                    Union[int, Type[Exception]],
                    Callable[[Request, Any], Coroutine[Any, Any, Response]],
                ]
            ],
            "A dictionary with handlers for exceptions."
        ] = None,
        on_startup: Annotated[
            Optional[Sequence[Callable[[], Any]]],
            "A list of startup event handler functions."
        ] = None,
        on_shutdown: Annotated[
            Optional[Sequence[Callable[[], Any]]],
            "A list of shutdown event handler functions."
        ] = None,
        lifespan: Annotated[
            Optional[Lifespan[AppType]],
            "A Lifespan context manager handler, replacing startup and shutdown functions."
        ] = None,
        terms_of_service: Annotated[
            Optional[str],
            "A URL to the Terms of Service for your API."
        ] = None,
        contact: Annotated[
            Optional[Dict[str, Union[str, Any]]],
            "A dictionary with the contact information for the API."
        ] = None,
        license_info: Annotated[
            Optional[Dict[str, Union[str, Any]]],
            "A dictionary with the license information for the API."
        ] = None,
        openapi_prefix: Annotated[
            str,
            "A URL prefix for the OpenAPI URL."
        ] = "",
        root_path: Annotated[
            str,
            "A path prefix handled by a proxy."
        ] = "",
        root_path_in_servers: Annotated[
            bool,
            "To disable automatically generating URLs in the servers field using the root_path."
        ] = True,
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]],
            "Additional responses to be added in OpenAPI."
        ] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]],
            "OpenAPI callbacks for all path operations."
        ] = None,
        webhooks: Annotated[
            Optional[Any],
            "OpenAPI webhooks for the API."
        ] = None,
        deprecated: Annotated[
            Optional[bool],
            "Mark all path operations as deprecated."
        ] = None,
        include_in_schema: Annotated[
            bool,
            "Include all path operations in the generated OpenAPI schema."
        ] = True,
        swagger_ui_parameters: Annotated[
            Optional[Dict[str, Any]],
            "Parameters to configure Swagger UI."
        ] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str],
            "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
        separate_input_output_schemas: Annotated[
            bool,
            "Whether to generate separate OpenAPI schemas for request and response."
        ] = True,
        **extra: Annotated[Any, "Extra keyword arguments to store in the app."]
    ) -> None:
        self.debug: bool = debug
        self.title: str = title
        self.summary: Optional[str] = summary
        self.description: str = description
        self.version: str = version
        self.terms_of_service: Optional[str] = terms_of_service
        self.contact: Optional[Dict[str, Union[str, Any]]] = contact
        self.license_info: Optional[Dict[str, Union[str, Any]]] = license_info
        self.openapi_url: Optional[str] = openapi_url
        self.openapi_tags: Optional[List[Dict[str, Any]]] = openapi_tags
        self.root_path_in_servers: bool = root_path_in_servers
        self.docs_url: Optional[str] = docs_url
        self.redoc_url: Optional[str] = redoc_url
        self.swagger_ui_oauth2_redirect_url: Optional[str] = swagger_ui_oauth2_redirect_url
        self.swagger_ui_init_oauth: Optional[Dict[str, Any]] = swagger_ui_init_oauth
        self.swagger_ui_parameters: Optional[Dict[str, Any]] = swagger_ui_parameters
        self.servers: List[Dict[str, Union[str, Any]]] = servers or []
        self.separate_input_output_schemas: bool = separate_input_output_schemas
        self.extra: Dict[str, Any] = extra
        self.openapi_version: str = "3.1.0"
        self.openapi_schema: Optional[Dict[str, Any]] = None
        if self.openapi_url:
            assert self.title, "A title must be provided for OpenAPI, e.g.: 'My API'"
            assert self.version, "A version must be provided for OpenAPI, e.g.: '2.1.0'"
        if openapi_prefix:
            logger.warning(
                '"openapi_prefix" has been deprecated in favor of "root_path".'
            )
        self.webhooks: Any = webhooks or None
        self.root_path: str = root_path or openapi_prefix
        self.state: State = State()
        self.dependency_overrides: Dict[Callable[..., Any], Callable[..., Any]] = {}
        from fastapi import routing  # local import for typing
        self.router: routing.APIRouter = routing.APIRouter(
            routes=routes,
            redirect_slashes=redirect_slashes,
            dependency_overrides_provider=self,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
            default_response_class=default_response_class,
            dependencies=dependencies,
            callbacks=callbacks,
            deprecated=deprecated,
            include_in_schema=include_in_schema,
            responses=responses,
            generate_unique_id_function=generate_unique_id_function,
        )
        self.exception_handlers: Dict[
            Any, Callable[[Request, Any], Union[Response, Awaitable[Response]]]
        ] = {} if exception_handlers is None else dict(exception_handlers)
        self.exception_handlers.setdefault(HTTPException, http_exception_handler)
        self.exception_handlers.setdefault(
            RequestValidationError, request_validation_exception_handler
        )
        self.exception_handlers.setdefault(
            WebSocketRequestValidationError,
            websocket_request_validation_exception_handler,  # type: ignore
        )
        self.user_middleware: List[Middleware] = [] if middleware is None else list(middleware)
        self.middleware_stack: Optional[ASGIApp] = None
        self.setup()

    def openapi(self) -> Dict[str, Any]:
        if not self.openapi_schema:
            self.openapi_schema = get_openapi(
                title=self.title,
                version=self.version,
                openapi_version=self.openapi_version,
                summary=self.summary,
                description=self.description,
                terms_of_service=self.terms_of_service,
                contact=self.contact,
                license_info=self.license_info,
                routes=self.router.routes,
                webhooks=self.webhooks.routes if self.webhooks else None,
                tags=self.openapi_tags,
                servers=self.servers,
                separate_input_output_schemas=self.separate_input_output_schemas,
            )
        return self.openapi_schema

    def setup(self) -> None:
        if self.openapi_url:
            urls = (server_data.get("url") for server_data in self.servers)
            server_urls = {url for url in urls if url}
            async def openapi(req: Request) -> JSONResponse:
                root_path: str = req.scope.get("root_path", "").rstrip("/")
                if root_path not in server_urls:
                    if root_path and self.root_path_in_servers:
                        self.servers.insert(0, {"url": root_path})
                        server_urls.add(root_path)
                return JSONResponse(self.openapi())
            self.add_route(self.openapi_url, openapi, include_in_schema=False)
        if self.openapi_url and self.docs_url:
            async def swagger_ui_html(req: Request) -> HTMLResponse:
                root_path: str = req.scope.get("root_path", "").rstrip("/")
                openapi_url: str = root_path + self.openapi_url
                oauth2_redirect_url: Optional[str] = self.swagger_ui_oauth2_redirect_url
                if oauth2_redirect_url:
                    oauth2_redirect_url = root_path + oauth2_redirect_url
                return get_swagger_ui_html(
                    openapi_url=openapi_url,
                    title=f"{self.title} - Swagger UI",
                    oauth2_redirect_url=oauth2_redirect_url,
                    init_oauth=self.swagger_ui_init_oauth,
                    swagger_ui_parameters=self.swagger_ui_parameters,
                )
            self.add_route(self.docs_url, swagger_ui_html, include_in_schema=False)
            if self.swagger_ui_oauth2_redirect_url:
                async def swagger_ui_redirect(req: Request) -> HTMLResponse:
                    return get_swagger_ui_oauth2_redirect_html()
                self.add_route(
                    self.swagger_ui_oauth2_redirect_url,
                    swagger_ui_redirect,
                    include_in_schema=False,
                )
        if self.openapi_url and self.redoc_url:
            async def redoc_html(req: Request) -> HTMLResponse:
                root_path: str = req.scope.get("root_path", "").rstrip("/")
                openapi_url: str = root_path + self.openapi_url
                return get_redoc_html(
                    openapi_url=openapi_url, title=f"{self.title} - ReDoc"
                )
            self.add_route(self.redoc_url, redoc_html, include_in_schema=False)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.root_path:
            scope["root_path"] = self.root_path
        await super().__call__(scope, receive, send)

    def add_api_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        *,
        response_model: Any = Default(None),
        status_code: Optional[int] = None,
        tags: Optional[List[Union[str, Enum]]] = None,
        dependencies: Optional[Sequence[Depends]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
        deprecated: Optional[bool] = None,
        methods: Optional[List[str]] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[IncEx] = None,
        response_model_exclude: Optional[IncEx] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Union[Type[Response], DefaultPlaceholder] = Default(JSONResponse),
        name: Optional[str] = None,
        openapi_extra: Optional[Dict[str, Any]] = None,
        generate_unique_id_function: Callable[[BaseRoute], str] = Default(generate_unique_id),
    ) -> None:
        self.router.add_api_route(
            path,
            endpoint=endpoint,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=methods,
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
            openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function,
        )

    def api_route(
        self,
        path: str,
        *,
        response_model: Any = Default(None),
        status_code: Optional[int] = None,
        tags: Optional[List[Union[str, Enum]]] = None,
        dependencies: Optional[Sequence[Depends]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
        deprecated: Optional[bool] = None,
        methods: Optional[List[str]] = None,
        operation_id: Optional[str] = None,
        response_model_include: Optional[IncEx] = None,
        response_model_exclude: Optional[IncEx] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        response_class: Type[Response] = Default(JSONResponse),
        name: Optional[str] = None,
        openapi_extra: Optional[Dict[str, Any]] = None,
        generate_unique_id_function: Callable[[BaseRoute], str] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.router.add_api_route(
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
                methods=methods,
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
        dependencies: Optional[Sequence[Depends]] = None,
    ) -> None:
        self.router.add_api_websocket_route(
            path,
            endpoint,
            name=name,
            dependencies=dependencies,
        )

    def websocket(
        self,
        path: Annotated[str, "WebSocket path."],
        name: Annotated[Optional[str], "A name for the WebSocket. Only used internally."] = None,
        *,
        dependencies: Annotated[
            Optional[Sequence[Depends]],
            "A list of dependencies to be used for this WebSocket."
        ] = None,
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.add_api_websocket_route(
                path,
                func,
                name=name,
                dependencies=dependencies,
            )
            return func
        return decorator

    def include_router(
        self,
        router: Annotated[Any, "The APIRouter to include."],
        *,
        prefix: Annotated[str, "An optional path prefix for the router."] = "",
        tags: Annotated[
            Optional[List[Union[str, Enum]]],
            "A list of tags to be applied to all the path operations in this router."
        ] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]],
            "A list of dependencies applied to all the path operations in this router."
        ] = None,
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]],
            "Additional responses to be added in OpenAPI."
        ] = None,
        deprecated: Annotated[
            Optional[bool],
            "Mark all the path operations in this router as deprecated."
        ] = None,
        include_in_schema: Annotated[
            bool,
            "Include (or not) all the path operations in the generated OpenAPI schema."
        ] = True,
        default_response_class: Annotated[
            Type[Response],
            "Default response class to be used for the route operations in this router."
        ] = Default(JSONResponse),
        callbacks: Annotated[
            Optional[List[BaseRoute]],
            "List of path operations that will be used as OpenAPI callbacks."
        ] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str],
            "Customize function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> None:
        self.router.include_router(
            router,
            prefix=prefix,
            tags=tags,
            dependencies=dependencies,
            responses=responses,
            deprecated=deprecated,
            include_in_schema=include_in_schema,
            default_response_class=default_response_class,
            callbacks=callbacks,
            generate_unique_id_function=generate_unique_id_function,
        )

    def get(
        self,
        path: Annotated[str, "The URL path to be used for this GET operation."],
        *,
        response_model: Annotated[Any, "The type to use for the response."] = Default(None),
        status_code: Annotated[Optional[int], "The default status code to be used for the response."] = None,
        tags: Annotated[Optional[List[Union[str, Enum]]], "A list of tags to be applied to this GET operation."] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]], "A list of dependencies to be applied to this GET operation."
        ] = None,
        summary: Annotated[Optional[str], "A summary for the GET operation."] = None,
        description: Annotated[Optional[str], "A description for the GET operation."] = None,
        response_description: Annotated[str, "The description for the default response."] = "Successful Response",
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]], "Additional responses for this GET operation."
        ] = None,
        deprecated: Annotated[Optional[bool], "Mark this GET operation as deprecated."] = None,
        operation_id: Annotated[Optional[str], "Custom operation ID for this GET operation."] = None,
        response_model_include: Annotated[Optional[IncEx], "Fields to include in the response model."] = None,
        response_model_exclude: Annotated[Optional[IncEx], "Fields to exclude in the response model."] = None,
        response_model_by_alias: Annotated[bool, "Serialize the response model by alias."] = True,
        response_model_exclude_unset: Annotated[bool, "Exclude unset values from the response model."] = False,
        response_model_exclude_defaults: Annotated[bool, "Exclude fields with default values."] = False,
        response_model_exclude_none: Annotated[bool, "Exclude fields that are None."] = False,
        include_in_schema: Annotated[bool, "Include this GET operation in the OpenAPI schema."] = True,
        response_class: Annotated[Type[Response], "Response class to be used for this GET operation."] = Default(JSONResponse),
        name: Annotated[Optional[str], "Name for this GET operation."] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]], "List of callbacks for this GET operation."
        ] = None,
        openapi_extra: Annotated[Optional[Dict[str, Any]], "Extra metadata for the OpenAPI schema."] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str], "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.get(
            path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
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
        path: Annotated[str, "The URL path to be used for this PUT operation."],
        *,
        response_model: Annotated[Any, "The type to use for the response."] = Default(None),
        status_code: Annotated[Optional[int], "The default status code to be used for the response."] = None,
        tags: Annotated[Optional[List[Union[str, Enum]]], "A list of tags to be applied to this PUT operation."] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]], "A list of dependencies to be applied to this PUT operation."
        ] = None,
        summary: Annotated[Optional[str], "A summary for the PUT operation."] = None,
        description: Annotated[Optional[str], "A description for the PUT operation."] = None,
        response_description: Annotated[str, "The description for the default response."] = "Successful Response",
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]], "Additional responses for this PUT operation."
        ] = None,
        deprecated: Annotated[Optional[bool], "Mark this PUT operation as deprecated."] = None,
        operation_id: Annotated[Optional[str], "Custom operation ID for this PUT operation."] = None,
        response_model_include: Annotated[Optional[IncEx], "Fields to include in the response model."] = None,
        response_model_exclude: Annotated[Optional[IncEx], "Fields to exclude in the response model."] = None,
        response_model_by_alias: Annotated[bool, "Serialize the response model by alias."] = True,
        response_model_exclude_unset: Annotated[bool, "Exclude unset values from the response model."] = False,
        response_model_exclude_defaults: Annotated[bool, "Exclude fields with default values."] = False,
        response_model_exclude_none: Annotated[bool, "Exclude fields that are None."] = False,
        include_in_schema: Annotated[bool, "Include this PUT operation in the OpenAPI schema."] = True,
        response_class: Annotated[Type[Response], "Response class to be used for this PUT operation."] = Default(JSONResponse),
        name: Annotated[Optional[str], "Name for this PUT operation."] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]], "List of callbacks for this PUT operation."
        ] = None,
        openapi_extra: Annotated[Optional[Dict[str, Any]], "Extra metadata for the OpenAPI schema."] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str], "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.put(
            path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
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
        path: Annotated[str, "The URL path to be used for this POST operation."],
        *,
        response_model: Annotated[Any, "The type to use for the response."] = Default(None),
        status_code: Annotated[Optional[int], "The default status code to be used for the response."] = None,
        tags: Annotated[Optional[List[Union[str, Enum]]], "A list of tags to be applied to this POST operation."] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]], "A list of dependencies to be applied to this POST operation."
        ] = None,
        summary: Annotated[Optional[str], "A summary for the POST operation."] = None,
        description: Annotated[Optional[str], "A description for the POST operation."] = None,
        response_description: Annotated[str, "The description for the default response."] = "Successful Response",
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]], "Additional responses for this POST operation."
        ] = None,
        deprecated: Annotated[Optional[bool], "Mark this POST operation as deprecated."] = None,
        operation_id: Annotated[Optional[str], "Custom operation ID for this POST operation."] = None,
        response_model_include: Annotated[Optional[IncEx], "Fields to include in the response model."] = None,
        response_model_exclude: Annotated[Optional[IncEx], "Fields to exclude in the response model."] = None,
        response_model_by_alias: Annotated[bool, "Serialize the response model by alias."] = True,
        response_model_exclude_unset: Annotated[bool, "Exclude unset values from the response model."] = False,
        response_model_exclude_defaults: Annotated[bool, "Exclude fields with default values."] = False,
        response_model_exclude_none: Annotated[bool, "Exclude fields that are None."] = False,
        include_in_schema: Annotated[bool, "Include this POST operation in the OpenAPI schema."] = True,
        response_class: Annotated[Type[Response], "Response class to be used for this POST operation."] = Default(JSONResponse),
        name: Annotated[Optional[str], "Name for this POST operation."] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]], "List of callbacks for this POST operation."
        ] = None,
        openapi_extra: Annotated[Optional[Dict[str, Any]], "Extra metadata for the OpenAPI schema."] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str], "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.post(
            path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
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
        path: Annotated[str, "The URL path to be used for this DELETE operation."],
        *,
        response_model: Annotated[Any, "The type to use for the response."] = Default(None),
        status_code: Annotated[Optional[int], "The default status code to be used for the response."] = None,
        tags: Annotated[Optional[List[Union[str, Enum]]], "A list of tags to be applied to this DELETE operation."] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]], "A list of dependencies to be applied to this DELETE operation."
        ] = None,
        summary: Annotated[Optional[str], "A summary for the DELETE operation."] = None,
        description: Annotated[Optional[str], "A description for the DELETE operation."] = None,
        response_description: Annotated[str, "The description for the default response."] = "Successful Response",
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]], "Additional responses for this DELETE operation."
        ] = None,
        deprecated: Annotated[Optional[bool], "Mark this DELETE operation as deprecated."] = None,
        operation_id: Annotated[Optional[str], "Custom operation ID for this DELETE operation."] = None,
        response_model_include: Annotated[Optional[IncEx], "Fields to include in the response model."] = None,
        response_model_exclude: Annotated[Optional[IncEx], "Fields to exclude in the response model."] = None,
        response_model_by_alias: Annotated[bool, "Serialize the response model by alias."] = True,
        response_model_exclude_unset: Annotated[bool, "Exclude unset values from the response model."] = False,
        response_model_exclude_defaults: Annotated[bool, "Exclude fields with default values."] = False,
        response_model_exclude_none: Annotated[bool, "Exclude fields that are None."] = False,
        include_in_schema: Annotated[bool, "Include this DELETE operation in the OpenAPI schema."] = True,
        response_class: Annotated[Type[Response], "Response class to be used for this DELETE operation."] = Default(JSONResponse),
        name: Annotated[Optional[str], "Name for this DELETE operation."] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]], "List of callbacks for this DELETE operation."
        ] = None,
        openapi_extra: Annotated[Optional[Dict[str, Any]], "Extra metadata for the OpenAPI schema."] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str], "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.delete(
            path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
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
        path: Annotated[str, "The URL path to be used for this OPTIONS operation."],
        *,
        response_model: Annotated[Any, "The type to use for the response."] = Default(None),
        status_code: Annotated[Optional[int], "The default status code to be used for the response."] = None,
        tags: Annotated[Optional[List[Union[str, Enum]]], "A list of tags for this OPTIONS operation."] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]], "A list of dependencies to be applied to this OPTIONS operation."
        ] = None,
        summary: Annotated[Optional[str], "A summary for the OPTIONS operation."] = None,
        description: Annotated[Optional[str], "A description for the OPTIONS operation."] = None,
        response_description: Annotated[str, "The description for the default response."] = "Successful Response",
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]], "Additional responses for this OPTIONS operation."
        ] = None,
        deprecated: Annotated[Optional[bool], "Mark this OPTIONS operation as deprecated."] = None,
        operation_id: Annotated[Optional[str], "Custom operation ID for this OPTIONS operation."] = None,
        response_model_include: Annotated[Optional[IncEx], "Fields to include in the response model."] = None,
        response_model_exclude: Annotated[Optional[IncEx], "Fields to exclude in the response model."] = None,
        response_model_by_alias: Annotated[bool, "Serialize the response model by alias."] = True,
        response_model_exclude_unset: Annotated[bool, "Exclude unset values from the response model."] = False,
        response_model_exclude_defaults: Annotated[bool, "Exclude fields with default values."] = False,
        response_model_exclude_none: Annotated[bool, "Exclude fields that are None."] = False,
        include_in_schema: Annotated[bool, "Include this OPTIONS operation in the OpenAPI schema."] = True,
        response_class: Annotated[Type[Response], "Response class to be used for this OPTIONS operation."] = Default(JSONResponse),
        name: Annotated[Optional[str], "Name for this OPTIONS operation."] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]], "List of callbacks for this OPTIONS operation."
        ] = None,
        openapi_extra: Annotated[Optional[Dict[str, Any]], "Extra metadata for the OpenAPI schema."] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str], "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.options(
            path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
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
        path: Annotated[str, "The URL path to be used for this HEAD operation."],
        *,
        response_model: Annotated[Any, "The type to use for the response."] = Default(None),
        status_code: Annotated[Optional[int], "The default status code for the HEAD operation."] = None,
        tags: Annotated[Optional[List[Union[str, Enum]]], "Tags for this HEAD operation."] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]], "Dependencies for this HEAD operation."
        ] = None,
        summary: Annotated[Optional[str], "A summary for the HEAD operation."] = None,
        description: Annotated[Optional[str], "A description for the HEAD operation."] = None,
        response_description: Annotated[str, "Description for the default response."] = "Successful Response",
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]], "Additional responses for this HEAD operation."
        ] = None,
        deprecated: Annotated[Optional[bool], "Mark this HEAD operation as deprecated."] = None,
        operation_id: Annotated[Optional[str], "Custom operation ID for this HEAD operation."] = None,
        response_model_include: Annotated[Optional[IncEx], "Fields to include in the response model."] = None,
        response_model_exclude: Annotated[Optional[IncEx], "Fields to exclude from the response model."] = None,
        response_model_by_alias: Annotated[bool, "Serialize the response model by alias."] = True,
        response_model_exclude_unset: Annotated[bool, "Exclude unset values from the response model."] = False,
        response_model_exclude_defaults: Annotated[bool, "Exclude fields with default values."] = False,
        response_model_exclude_none: Annotated[bool, "Exclude fields that are None."] = False,
        include_in_schema: Annotated[bool, "Include this HEAD operation in the OpenAPI schema."] = True,
        response_class: Annotated[Type[Response], "Response class to be used for this HEAD operation."] = Default(JSONResponse),
        name: Annotated[Optional[str], "Name for this HEAD operation."] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]], "List of callbacks for this HEAD operation."
        ] = None,
        openapi_extra: Annotated[Optional[Dict[str, Any]], "Extra metadata for the OpenAPI schema."] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str], "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.head(
            path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
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
        path: Annotated[str, "The URL path to be used for this PATCH operation."],
        *,
        response_model: Annotated[Any, "The type to use for the response."] = Default(None),
        status_code: Annotated[Optional[int], "The default status code for the PATCH operation."] = None,
        tags: Annotated[Optional[List[Union[str, Enum]]], "Tags for this PATCH operation."] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]], "Dependencies for this PATCH operation."
        ] = None,
        summary: Annotated[Optional[str], "A summary for the PATCH operation."] = None,
        description: Annotated[Optional[str], "A description for the PATCH operation."] = None,
        response_description: Annotated[str, "Description for the default response."] = "Successful Response",
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]], "Additional responses for this PATCH operation."
        ] = None,
        deprecated: Annotated[Optional[bool], "Mark this PATCH operation as deprecated."] = None,
        operation_id: Annotated[Optional[str], "Custom operation ID for this PATCH operation."] = None,
        response_model_include: Annotated[Optional[IncEx], "Fields to include in the response model."] = None,
        response_model_exclude: Annotated[Optional[IncEx], "Fields to exclude from the response model."] = None,
        response_model_by_alias: Annotated[bool, "Serialize the response model by alias."] = True,
        response_model_exclude_unset: Annotated[bool, "Exclude unset values from the response model."] = False,
        response_model_exclude_defaults: Annotated[bool, "Exclude fields with default values."] = False,
        response_model_exclude_none: Annotated[bool, "Exclude fields that are None."] = False,
        include_in_schema: Annotated[bool, "Include this PATCH operation in the OpenAPI schema."] = True,
        response_class: Annotated[Type[Response], "Response class to be used for this PATCH operation."] = Default(JSONResponse),
        name: Annotated[Optional[str], "Name for this PATCH operation."] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]], "List of callbacks for this PATCH operation."
        ] = None,
        openapi_extra: Annotated[Optional[Dict[str, Any]], "Extra metadata for the OpenAPI schema."] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str], "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.patch(
            path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
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
        path: Annotated[str, "The URL path to be used for this TRACE operation."],
        *,
        response_model: Annotated[Any, "The type to use for the response."] = Default(None),
        status_code: Annotated[Optional[int], "The default status code for the TRACE operation."] = None,
        tags: Annotated[Optional[List[Union[str, Enum]]], "Tags for this TRACE operation."] = None,
        dependencies: Annotated[
            Optional[Sequence[Depends]], "Dependencies for this TRACE operation."
        ] = None,
        summary: Annotated[Optional[str], "A summary for the TRACE operation."] = None,
        description: Annotated[Optional[str], "A description for the TRACE operation."] = None,
        response_description: Annotated[str, "Description for the default response."] = "Successful Response",
        responses: Annotated[
            Optional[Dict[Union[int, str], Dict[str, Any]]], "Additional responses for this TRACE operation."
        ] = None,
        deprecated: Annotated[Optional[bool], "Mark this TRACE operation as deprecated."] = None,
        operation_id: Annotated[Optional[str], "Custom operation ID for this TRACE operation."] = None,
        response_model_include: Annotated[Optional[IncEx], "Fields to include in the response model."] = None,
        response_model_exclude: Annotated[Optional[IncEx], "Fields to exclude from the response model."] = None,
        response_model_by_alias: Annotated[bool, "Serialize the response model by alias."] = True,
        response_model_exclude_unset: Annotated[bool, "Exclude unset values from the response model."] = False,
        response_model_exclude_defaults: Annotated[bool, "Exclude fields with default values."] = False,
        response_model_exclude_none: Annotated[bool, "Exclude fields that are None."] = False,
        include_in_schema: Annotated[bool, "Include this TRACE operation in the OpenAPI schema."] = True,
        response_class: Annotated[Type[Response], "Response class to be used for this TRACE operation."] = Default(JSONResponse),
        name: Annotated[Optional[str], "Name for this TRACE operation."] = None,
        callbacks: Annotated[
            Optional[List[BaseRoute]], "List of callbacks for this TRACE operation."
        ] = None,
        openapi_extra: Annotated[Optional[Dict[str, Any]], "Extra metadata for the OpenAPI schema."] = None,
        generate_unique_id_function: Annotated[
            Callable[[BaseRoute], str], "Function to generate unique IDs for path operations."
        ] = Default(generate_unique_id),
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.trace(
            path,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
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

    def websocket_route(
        self, path: str, name: Union[str, None] = None
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.router.add_websocket_route(path, func, name=name)
            return func
        return decorator

    def on_event(
        self,
        event_type: Annotated[str, "The type of event. 'startup' or 'shutdown'."],
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        return self.router.on_event(event_type)

    def middleware(
        self,
        middleware_type: Annotated[str, "The type of middleware. Currently only supports 'http'."],
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.add_middleware(BaseHTTPMiddleware, dispatch=func)
            return func
        return decorator

    def exception_handler(
        self,
        exc_class_or_status_code: Annotated[
            Union[int, Type[Exception]], "The Exception class or status code to handle."
        ],
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.add_exception_handler(exc_class_or_status_code, func)
            return func
        return decorator