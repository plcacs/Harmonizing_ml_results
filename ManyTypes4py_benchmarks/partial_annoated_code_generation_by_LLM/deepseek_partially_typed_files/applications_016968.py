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
    def __init__(
        self,
        *,
        debug: Annotated[bool, Doc("Boolean indicating if debug tracebacks should be returned on server errors.")] = False,
        routes: Annotated[Optional[List[BaseRoute]], Doc("A list of routes to serve incoming HTTP and WebSocket requests."), deprecated("You normally wouldn't use this parameter with FastAPI.")] = None,
        title: Annotated[str, Doc("The title of the API.")] = 'FastAPI',
        summary: Annotated[Optional[str], Doc("A short summary of the API.")] = None,
        description: Annotated[str, Doc("A description of the API.")] = '',
        version: Annotated[str, Doc("The version of the API.")] = '0.1.0',
        openapi_url: Annotated[Optional[str], Doc("The URL where the OpenAPI schema will be served from.")] = '/openapi.json',
        openapi_tags: Annotated[Optional[List[Dict[str, Any]]], Doc("A list of tags used by OpenAPI.")] = None,
        servers: Annotated[Optional[List[Dict[str, Union[str, Any]]]], Doc("A list of dicts with connectivity information to a target server.")] = None,
        dependencies: Annotated[Optional[Sequence[Depends]], Doc("A list of global dependencies.")] = None,
        default_response_class: Annotated[Type[Response], Doc("The default response class to be used.")] = Default(JSONResponse),
        redirect_slashes: Annotated[bool, Doc("Whether to detect and redirect slashes in URLs.")] = True,
        docs_url: Annotated[Optional[str], Doc("The path to the automatic interactive API documentation.")] = '/docs',
        redoc_url: Annotated[Optional[str], Doc("The path to the alternative automatic interactive API documentation.")] = '/redoc',
        swagger_ui_oauth2_redirect_url: Annotated[Optional[str], Doc("The OAuth2 redirect endpoint for the Swagger UI.")] = '/docs/oauth2-redirect',
        swagger_ui_init_oauth: Annotated[Optional[Dict[str, Any]], Doc("OAuth2 configuration for the Swagger UI.")] = None,
        middleware: Annotated[Optional[Sequence[Middleware]], Doc("List of middleware to be added when creating the application.")] = None,
        exception_handlers: Annotated[Optional[Dict[Union[int, Type[Exception]], Callable[[Request, Any], Coroutine[Any, Any, Response]]]], Doc("A dictionary with handlers for exceptions.")] = None,
        on_startup: Annotated[Optional[Sequence[Callable[[], Any]]], Doc("A list of startup event handler functions.")] = None,
        on_shutdown: Annotated[Optional[Sequence[Callable[[], Any]]], Doc("A list of shutdown event handler functions.")] = None,
        lifespan: Annotated[Optional[Lifespan[AppType]], Doc("A Lifespan context manager handler.")] = None,
        terms_of_service: Annotated[Optional[str], Doc("A URL to the Terms of Service for your API.")] = None,
        contact: Annotated[Optional[Dict[str, Union[str, Any]]], Doc("A dictionary with the contact information for the exposed API.")] = None,
        license_info: Annotated[Optional[Dict[str, Union[str, Any]]], Doc("A dictionary with the license information for the exposed API.")] = None,
        openapi_prefix: Annotated[str, Doc("A URL prefix for the OpenAPI URL."), deprecated("openapi_prefix has been deprecated in favor of root_path")] = '',
        root_path: Annotated[str, Doc("A path prefix handled by a proxy.")] = '',
        root_path_in_servers: Annotated[bool, Doc("To disable automatically generating the URLs in the servers field.")] = True,
        responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc("Additional responses to be shown in OpenAPI.")] = None,
        callbacks: Annotated[Optional[List[BaseRoute]], Doc("OpenAPI callbacks that should apply to all path operations.")] = None,
        webhooks: Annotated[Optional[routing.APIRouter], Doc("Add OpenAPI webhooks.")] = None,
        deprecated: Annotated[Optional[bool], Doc("Mark all path operations as deprecated.")] = None,
        include_in_schema: Annotated[bool, Doc("To include (or not) all the path operations in the generated OpenAPI.")] = True,
        swagger_ui_parameters: Annotated[Optional[Dict[str, Any]], Doc("Parameters to configure Swagger UI.")] = None,
        generate_unique_id_function: Annotated[Callable[[routing.APIRoute], str], Doc("Customize the function used to generate unique IDs.")] = Default(generate_unique_id),
        separate_input_output_schemas: Annotated[bool, Doc("Whether to generate separate OpenAPI schemas for request body and response body.")] = True,
        **extra: Annotated[Any, Doc("Extra keyword arguments to be stored in the app.")]
    ) -> None:
        self.debug = debug
        self.title = title
        self.summary = summary
        self.description = description
        self.version = version
        self.terms_of_service = terms_of_service
        self.contact = contact
        self.license_info = license_info
        self.openapi_url = openapi_url
        self.openapi_tags = openapi_tags
        self.root_path_in_servers = root_path_in_servers
        self.docs_url = docs_url
        self.redoc_url = redoc_url
        self.swagger_ui_oauth2_redirect_url = swagger_ui_oauth2_redirect_url
        self.swagger_ui_init_oauth = swagger_ui_init_oauth
        self.swagger_ui_parameters = swagger_ui_parameters
        self.servers = servers or []
        self.separate_input_output_schemas = separate_input_output_schemas
        self.extra = extra
        self.openapi_version: Annotated[str, Doc("The version string of OpenAPI.")] = '3.1.0'
        self.openapi_schema: Optional[Dict[str, Any]] = None
        if self.openapi_url:
            assert self.title, "A title must be provided for OpenAPI, e.g.: 'My API'"
            assert self.version, "A version must be provided for OpenAPI, e.g.: '2.1.0'"
        if openapi_prefix:
            logger.warning('"openapi_prefix" has been deprecated in favor of "root_path"')
        self.webhooks: Annotated[routing.APIRouter, Doc("The app.webhooks attribute is an APIRouter.")] = webhooks or routing.APIRouter()
        self.root_path = root_path or openapi_prefix
        self.state: Annotated[State, Doc("A state object for the application.")] = State()
        self.dependency_overrides: Annotated[Dict[Callable[..., Any], Callable[..., Any]], Doc("A dictionary with overrides for the dependencies.")] = {}
        self.router: routing.APIRouter = routing.APIRouter(
            routes=routes, redirect_slashes=redirect_slashes, dependency_overrides_provider=self,
            on_startup=on_startup, on_shutdown=on_shutdown, lifespan=lifespan,
            default_response_class=default_response_class, dependencies=dependencies,
            callbacks=callbacks, deprecated=deprecated, include_in_schema=include_in_schema,
            responses=responses, generate_unique_id_function=generate_unique_id_function
        )
        self.exception_handlers: Dict[Any, Callable[[Request, Any], Union[Response, Awaitable[Response]]]] = {} if exception_handlers is None else dict(exception_handlers)
        self.exception_handlers.setdefault(HTTPException, http_exception_handler)
        self.exception_handlers.setdefault(RequestValidationError, request_validation_exception_handler)
        self.exception_handlers.setdefault(WebSocketRequestValidationError, websocket_request_validation_exception_handler)
        self.user_middleware: List[Middleware] = [] if middleware is None else list(middleware)
        self.middleware_stack: Union[ASGIApp, None] = None
        self.setup()

    def openapi(self) -> Dict[str, Any]:
        if not self.openapi_schema:
            self.openapi_schema = get_openapi(
                title=self.title, version=self.version, openapi_version=self.openapi_version,
                summary=self.summary, description=self.description, terms_of_service=self.terms_of_service,
                contact=self.contact, license_info=self.license_info, routes=self.routes,
                webhooks=self.webhooks.routes, tags=self.openapi_tags, servers=self.servers,
                separate_input_output_schemas=self.separate_input_output_schemas
            )
        return self.openapi_schema

    def setup(self) -> None:
        if self.openapi_url:
            urls = (server_data.get('url') for server_data in self.servers)
            server_urls = {url for url in urls if url}

            async def openapi(req: Request) -> JSONResponse:
                root_path = req.scope.get('root_path', '').rstrip('/')
                if root_path not in server_urls:
                    if root_path and self.root_path_in_servers:
                        self.servers.insert(0, {'url': root_path})
                        server_urls.add(root_path)
                return JSONResponse(self.openapi())
            self.add_route(self.openapi_url, openapi, include_in_schema=False)
        
        if self.openapi_url and self.docs_url:
            async def swagger_ui_html(req: Request) -> HTMLResponse:
                root_path = req.scope.get('root_path', '').rstrip('/')
                openapi_url = root_path + self.openapi_url
                oauth2_redirect_url = self.swagger_ui_oauth2_redirect_url
                if oauth2_redirect_url:
                    oauth2_redirect_url = root_path + oauth2_redirect_url
                return get_swagger_ui_html(
                    openapi_url=openapi_url, title=f'{self.title} - Swagger UI',
                    oauth2_redirect_url=oauth2_redirect_url, init_oauth=self.swagger_ui_init_oauth,
                    swagger_ui_parameters=self.swagger_ui_parameters
                )
            self.add_route(self.docs_url, swagger_ui_html, include_in_schema=False)
            
            if self.swagger_ui_oauth2_redirect_url:
                async def swagger_ui_redirect(req: Request) -> HTMLResponse:
                    return get_swagger_ui_oauth2_redirect_html()
                self.add_route(self.swagger_ui_oauth2_redirect_url, swagger_ui_redirect, include_in_schema=False)
        
        if self.openapi_url and self.redoc_url:
            async def redoc_html(req: Request) -> HTMLResponse:
                root_path = req.scope.get('root_path', '').rstrip('/')
                openapi_url = root_path + self.openapi_url
                return get_redoc_html(openapi_url=openapi_url, title=f'{self.title} - ReDoc')
            self.add_route(self.redoc_url, redoc_html, include_in_schema=False)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.root_path:
            scope['root_path'] = self.root_path
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
        response_description: str = 'Successful Response',
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
        generate_unique_id_function: Callable[[routing.APIRoute], str] = Default(generate_unique_id)
    ) -> None:
        self.router.add_api_route(
            path, endpoint=endpoint, response_model=response_model, status_code=status_code,
            tags=tags, dependencies=dependencies, summary=summary, description=description,
            response_description=response_description, responses=responses, deprecated=deprecated,
            methods=methods, operation_id=operation_id, response_model_include=response_model_include,
            response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema,
            response_class=response_class, name=name, openapi_extra=openapi_extra,
            generate_unique_id_function=generate_unique_id_function
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
        response_description: str = 'Successful Response',
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
        generate_unique_id_function: Callable[[routing.APIRoute], str] = Default(generate_unique_id)
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.router.add_api_route(
                path, func, response_model=response_model, status_code=status_code,
                tags=tags, dependencies=dependencies, summary=summary, description=description,
                response_description=response_description, responses=responses, deprecated=deprecated,
                methods=methods, operation_id=operation_id, response_model_include=response_model_include,
                response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias,
                response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults,
                response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema,
                response_class=response_class, name=name, openapi_extra=openapi_extra,
                generate_unique_id_function=generate_unique_id_function
            )
            return func
        return decorator

    def add_api_websocket_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        name: Optional[str] = None,
        *,
        dependencies: Optional[Sequence[Depends]] = None
    ) -> None:
        self.router.add_api_websocket_route(path, endpoint, name=name, dependencies=dependencies)

    def websocket(
        self,
        path: str,
        name: Annotated[Optional[str], Doc("A name for the WebSocket.")] = None,
        *,
        dependencies: Annotated[Optional[Sequence[Depends]], Doc("A list of