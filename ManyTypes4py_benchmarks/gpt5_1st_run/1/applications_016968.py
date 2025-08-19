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
    """
    `FastAPI` app class, the main entrypoint to use FastAPI.

    Read more in the
    [FastAPI docs for First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/).

    ## Example

    ```python
    from fastapi import FastAPI

    app = FastAPI()
    ```
    """

    def __init__(self, *, debug: bool = False, routes: Optional[Sequence[BaseRoute]] = None, title: str = 'FastAPI', summary: Optional[str] = None, description: str = '', version: str = '0.1.0', openapi_url: Optional[str] = '/openapi.json', openapi_tags: Optional[List[Dict[str, Any]]] = None, servers: Optional[List[Dict[str, Any]]] = None, dependencies: Optional[Sequence[Depends]] = None, default_response_class: Default[Type[Response]] = Default(JSONResponse), redirect_slashes: bool = True, docs_url: Optional[str] = '/docs', redoc_url: Optional[str] = '/redoc', swagger_ui_oauth2_redirect_url: Optional[str] = '/docs/oauth2-redirect', swagger_ui_init_oauth: Optional[Dict[str, Any]] = None, middleware: Optional[Sequence[Middleware]] = None, exception_handlers: Optional[Dict[Union[int, Type[Exception]], Callable[[Request, Exception], Awaitable[Response]]]] = None, on_startup: Optional[Sequence[Callable[[], Any]]] = None, on_shutdown: Optional[Sequence[Callable[[], Any]]] = None, lifespan: Optional[Lifespan[State]] = None, terms_of_service: Optional[str] = None, contact: Optional[Dict[str, Any]] = None, license_info: Optional[Dict[str, Any]] = None, openapi_prefix: str = '', root_path: str = '', root_path_in_servers: bool = True, responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, callbacks: Optional[List[routing.APIRouter]] = None, webhooks: Optional[routing.APIRouter] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, swagger_ui_parameters: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id), separate_input_output_schemas: bool = True, **extra: Any) -> None:
        self.debug: bool = debug
        self.title: str = title
        self.summary: Optional[str] = summary
        self.description: str = description
        self.version: str = version
        self.terms_of_service: Optional[str] = terms_of_service
        self.contact: Optional[Dict[str, Any]] = contact
        self.license_info: Optional[Dict[str, Any]] = license_info
        self.openapi_url: Optional[str] = openapi_url
        self.openapi_tags: Optional[List[Dict[str, Any]]] = openapi_tags
        self.root_path_in_servers: bool = root_path_in_servers
        self.docs_url: Optional[str] = docs_url
        self.redoc_url: Optional[str] = redoc_url
        self.swagger_ui_oauth2_redirect_url: Optional[str] = swagger_ui_oauth2_redirect_url
        self.swagger_ui_init_oauth: Optional[Dict[str, Any]] = swagger_ui_init_oauth
        self.swagger_ui_parameters: Optional[Dict[str, Any]] = swagger_ui_parameters
        self.servers: List[Dict[str, Any]] = servers or []
        self.separate_input_output_schemas: bool = separate_input_output_schemas
        self.extra: Dict[str, Any] = extra
        self.openapi_version: str = '3.1.0'
        self.openapi_schema: Optional[Dict[str, Any]] = None
        if self.openapi_url:
            assert self.title, "A title must be provided for OpenAPI, e.g.: 'My API'"
            assert self.version, "A version must be provided for OpenAPI, e.g.: '2.1.0'"
        if openapi_prefix:
            logger.warning('"openapi_prefix" has been deprecated in favor of "root_path", which follows more closely the ASGI standard, is simpler, and more automatic. Check the docs at https://fastapi.tiangolo.com/advanced/sub-applications/')
        self.webhooks: routing.APIRouter = webhooks or routing.APIRouter()
        self.root_path: str = root_path or openapi_prefix
        self.state: State = State()
        self.dependency_overrides: Dict[Any, Any] = {}
        self.router: routing.APIRouter = routing.APIRouter(routes=routes, redirect_slashes=redirect_slashes, dependency_overrides_provider=self, on_startup=on_startup, on_shutdown=on_shutdown, lifespan=lifespan, default_response_class=default_response_class, dependencies=dependencies, callbacks=callbacks, deprecated=deprecated, include_in_schema=include_in_schema, responses=responses, generate_unique_id_function=generate_unique_id_function)
        self.exception_handlers: Dict[Union[int, Type[Exception]], Callable[[Request, Exception], Awaitable[Response]]] = {} if exception_handlers is None else dict(exception_handlers)
        self.exception_handlers.setdefault(HTTPException, http_exception_handler)
        self.exception_handlers.setdefault(RequestValidationError, request_validation_exception_handler)
        self.exception_handlers.setdefault(WebSocketRequestValidationError, websocket_request_validation_exception_handler)
        self.user_middleware: List[Middleware] = [] if middleware is None else list(middleware)
        self.middleware_stack: Optional[ASGIApp] = None
        self.setup()

    def openapi(self) -> Dict[str, Any]:
        """
        Generate the OpenAPI schema of the application. This is called by FastAPI
        internally.

        The first time it is called it stores the result in the attribute
        `app.openapi_schema`, and next times it is called, it just returns that same
        result. To avoid the cost of generating the schema every time.

        If you need to modify the generated OpenAPI schema, you could modify it.

        Read more in the
        [FastAPI docs for OpenAPI](https://fastapi.tiangolo.com/how-to/extending-openapi/).
        """
        if not self.openapi_schema:
            self.openapi_schema = get_openapi(title=self.title, version=self.version, openapi_version=self.openapi_version, summary=self.summary, description=self.description, terms_of_service=self.terms_of_service, contact=self.contact, license_info=self.license_info, routes=self.routes, webhooks=self.webhooks.routes, tags=self.openapi_tags, servers=self.servers, separate_input_output_schemas=self.separate_input_output_schemas)
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
                return get_swagger_ui_html(openapi_url=openapi_url, title=f'{self.title} - Swagger UI', oauth2_redirect_url=oauth2_redirect_url, init_oauth=self.swagger_ui_init_oauth, swagger_ui_parameters=self.swagger_ui_parameters)
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

    def add_api_route(self, path: str, endpoint: DecoratedCallable, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, methods: Optional[List[str]] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> None:
        self.router.add_api_route(path, endpoint=endpoint, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, methods=methods, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def api_route(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, methods: Optional[List[str]] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:

        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.router.add_api_route(path, func, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, methods=methods, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)
            return func
        return decorator

    def add_api_websocket_route(self, path: str, endpoint: DecoratedCallable, name: Optional[str] = None, *, dependencies: Optional[Sequence[Depends]] = None) -> None:
        self.router.add_api_websocket_route(path, endpoint, name=name, dependencies=dependencies)

    def websocket(self, path: str, name: Optional[str] = None, *, dependencies: Optional[Sequence[Depends]] = None) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Decorate a WebSocket function.

        Read more about it in the
        [FastAPI docs for WebSockets](https://fastapi.tiangolo.com/advanced/websockets/).

        **Example**

        ```python
        from fastapi import FastAPI, WebSocket

        app = FastAPI()

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(f"Message text was: {data}")
        ```
        """

        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.add_api_websocket_route(path, func, name=name, dependencies=dependencies)
            return func
        return decorator

    def include_router(self, router: routing.APIRouter, *, prefix: str = '', tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, default_response_class: Default[Type[Response]] = Default(JSONResponse), callbacks: Optional[List[routing.APIRouter]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> None:
        """
        Include an `APIRouter` in the same app.

        Read more about it in the
        [FastAPI docs for Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/).

        ## Example

        ```python
        from fastapi import FastAPI

        from .users import users_router

        app = FastAPI()

        app.include_router(users_router)
        ```
        """
        self.router.include_router(router, prefix=prefix, tags=tags, dependencies=dependencies, responses=responses, deprecated=deprecated, include_in_schema=include_in_schema, default_response_class=default_response_class, callbacks=callbacks, generate_unique_id_function=generate_unique_id_function)

    def get(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[List[routing.APIRouter]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add a *path operation* using an HTTP GET operation.

        ## Example

        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/items/")
        def read_items():
            return [{"name": "Empanada"}, {"name": "Arepa"}]
        ```
        """
        return self.router.get(path, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def put(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[List[routing.APIRouter]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add a *path operation* using an HTTP PUT operation.

        ## Example

        ```python
        from fastapi import FastAPI
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            description: str | None = None

        app = FastAPI()

        @app.put("/items/{item_id}")
        def replace_item(item_id: str, item: Item):
            return {"message": "Item replaced", "id": item_id}
        ```
        """
        return self.router.put(path, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def post(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[List[routing.APIRouter]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add a *path operation* using an HTTP POST operation.

        ## Example

        ```python
        from fastapi import FastAPI
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            description: str | None = None

        app = FastAPI()

        @app.post("/items/")
        def create_item(item: Item):
            return {"message": "Item created"}
        ```
        """
        return self.router.post(path, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def delete(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[List[routing.APIRouter]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add a *path operation* using an HTTP DELETE operation.

        ## Example

        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/items/{item_id}")
        def delete_item(item_id: str):
            return {"message": "Item deleted"}
        ```
        """
        return self.router.delete(path, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def options(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[List[routing.APIRouter]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add a *path operation* using an HTTP OPTIONS operation.

        ## Example

        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.options("/items/")
        def get_item_options():
            return {"additions": ["Aji", "Guacamole"]}
        ```
        """
        return self.router.options(path, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def head(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[List[routing.APIRouter]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add a *path operation* using an HTTP HEAD operation.

        ## Example

        ```python
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.head("/items/", status_code=204)
        def get_items_headers(response: Response):
            response.headers["X-Cat-Dog"] = "Alone in the world"
        ```
        """
        return self.router.head(path, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def patch(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[List[routing.APIRouter]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add a *path operation* using an HTTP PATCH operation.

        ## Example

        ```python
        from fastapi import FastAPI
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            description: str | None = None

        app = FastAPI()

        @app.patch("/items/")
        def update_item(item: Item):
            return {"message": "Item updated in place"}
        ```
        """
        return self.router.patch(path, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def trace(self, path: str, *, response_model: Default[Any] = Default(None), status_code: Optional[int] = None, tags: Optional[List[str]] = None, dependencies: Optional[Sequence[Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, operation_id: Optional[str] = None, response_model_include: Optional[IncEx] = None, response_model_exclude: Optional[IncEx] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Default[Type[Response]] = Default(JSONResponse), name: Optional[str] = None, callbacks: Optional[List[routing.APIRouter]] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Default[Callable[[routing.APIRoute], str]] = Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add a *path operation* using an HTTP TRACE operation.

        ## Example

        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.put("/items/{item_id}")
        def trace_item(item_id: str):
            return None
        ```
        """
        return self.router.trace(path, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)

    def websocket_route(self, path: str, name: Optional[str] = None) -> Callable[[DecoratedCallable], DecoratedCallable]:

        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            self.router.add_websocket_route(path, func, name=name)
            return func
        return decorator

    @deprecated('\n        on_event is deprecated, use lifespan event handlers instead.\n\n        Read more about it in the\n        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).\n        ')
    def on_event(self, event_type: str) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """
        Add an event handler for the application.

        `on_event` is deprecated, use `lifespan` event handlers instead.

        Read more about it in the
        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/#alternative-events-deprecated).
        """
        return self.router.on_event(event_type)

    def middleware(self, middleware_type: str) -> Callable[[Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]], Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]]]:
        """
        Add a middleware to the application.

        Read more about it in the
        [FastAPI docs for Middleware](https://fastapi.tiangolo.com/tutorial/middleware/).

        ## Example

        ```python
        import time

        from fastapi import FastAPI, Request

        app = FastAPI()


        @app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        ```
        """

        def decorator(func: Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
            self.add_middleware(BaseHTTPMiddleware, dispatch=func)
            return func
        return decorator

    def exception_handler(self, exc_class_or_status_code: Union[int, Type[Exception]]) -> Callable[[Callable[[Request, Exception], Awaitable[Response]]], Callable[[Request, Exception], Awaitable[Response]]]:
        """
        Add an exception handler to the app.

        Read more about it in the
        [FastAPI docs for Handling Errors](https://fastapi.tiangolo.com/tutorial/handling-errors/).

        ## Example

        ```python
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse


        class UnicornException(Exception):
            def __init__(self, name: str):
                self.name = name


        app = FastAPI()


        @app.exception_handler(UnicornException)
        async def unicorn_exception_handler(request: Request, exc: UnicornException):
            return JSONResponse(
                status_code=418,
                content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
            )
        ```
        """

        def decorator(func: Callable[[Request, Exception], Awaitable[Response]]) -> Callable[[Request, Exception], Awaitable[Response]]:
            self.add_exception_handler(exc_class_or_status_code, func)
            return func
        return decorator