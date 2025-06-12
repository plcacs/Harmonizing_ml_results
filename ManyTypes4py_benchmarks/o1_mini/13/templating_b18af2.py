from __future__ import annotations
import typing
from typing import Any, Callable, Dict, Optional, Union
import warnings
from os import PathLike
from starlette.background import BackgroundTask
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import Receive, Scope, Send
try:
    import jinja2
    if hasattr(jinja2, 'pass_context'):
        pass_context: Callable[..., Any] = jinja2.pass_context
    else:
        pass_context = jinja2.contextfunction
except ModuleNotFoundError:
    jinja2 = None  # type: ignore
    pass_context = None  # type: ignore

class _TemplateResponse(HTMLResponse):

    template: jinja2.Template
    context: Dict[str, Any]

    def __init__(
        self,
        template: jinja2.Template,
        context: Dict[str, Any],
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None
    ) -> None:
        self.template = template
        self.context = context
        content: str = template.render(context)
        super().__init__(content, status_code, headers, media_type, background)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send
    ) -> None:
        request: Dict[str, Any] = self.context.get('request', {})
        extensions: Dict[str, Any] = request.get('extensions', {})
        if 'http.response.debug' in extensions:
            await send({
                'type': 'http.response.debug',
                'info': {
                    'template': self.template,
                    'context': self.context
                }
            })
        await super().__call__(scope, receive, send)

class Jinja2Templates:
    """
    templates = Jinja2Templates("templates")

    return templates.TemplateResponse("index.html", {"request": request})
    """

    def __init__(
        self,
        directory: Optional[str] = None,
        *,
        context_processors: Optional[typing.List[Callable[[Request], Dict[str, Any]]]] = None,
        env: Optional[jinja2.Environment] = None,
        **env_options: Any
    ) -> None:
        if env_options:
            warnings.warn(
                'Extra environment options are deprecated. Use a preconfigured jinja2.Environment instead.',
                DeprecationWarning
            )
        assert jinja2 is not None, 'jinja2 must be installed to use Jinja2Templates'
        assert bool(directory) ^ bool(env), "either 'directory' or 'env' arguments must be passed"
        self.context_processors: typing.List[Callable[[Request], Dict[str, Any]]] = context_processors or []
        if directory is not None:
            self.env: jinja2.Environment = self._create_env(directory, **env_options)
        elif env is not None:
            self.env = env
        self._setup_env_defaults(self.env)

    def _create_env(self, directory: str, **env_options: Any) -> jinja2.Environment:
        loader: jinja2.BaseLoader = jinja2.FileSystemLoader(directory)
        env_options.setdefault('loader', loader)
        env_options.setdefault('autoescape', True)
        return jinja2.Environment(**env_options)

    def _setup_env_defaults(self, env: jinja2.Environment) -> None:

        @pass_context
        def url_for(context: Dict[str, Any], name: str, /, **path_params: Any) -> str:
            request: Request = context['request']
            return request.url_for(name, **path_params)

        env.globals.setdefault('url_for', url_for)

    def get_template(self, name: str) -> jinja2.Template:
        return self.env.get_template(name)

    @typing.overload
    def TemplateResponse(
        self,
        request: Request,
        name: str,
        context: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None
    ) -> _TemplateResponse:
        ...

    @typing.overload
    def TemplateResponse(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None
    ) -> _TemplateResponse:
        ...

    def TemplateResponse(
        self,
        *args: Any,
        **kwargs: Any
    ) -> _TemplateResponse:
        if args:
            if isinstance(args[0], str):
                warnings.warn(
                    'The `name` is not the first parameter anymore. The first parameter should be the `Request` instance.\n'
                    'Replace `TemplateResponse(name, {"request": request})` by `TemplateResponse(request, name)`.',
                    DeprecationWarning
                )
                name: str = args[0]
                context: Dict[str, Any] = args[1] if len(args) > 1 else kwargs.get('context', {})
                status_code: int = args[2] if len(args) > 2 else kwargs.get('status_code', 200)
                headers: Optional[Dict[str, str]] = args[3] if len(args) > 3 else kwargs.get('headers')
                media_type: Optional[str] = args[4] if len(args) > 4 else kwargs.get('media_type')
                background: Optional[BackgroundTask] = args[5] if len(args) > 5 else kwargs.get('background')
                if 'request' not in context:
                    raise ValueError('context must include a "request" key')
                request: Request = context['request']
            else:
                request = args[0]
                name = args[1] if len(args) > 1 else kwargs['name']
                context = args[2] if len(args) > 2 else kwargs.get('context', {})
                status_code = args[3] if len(args) > 3 else kwargs.get('status_code', 200)
                headers = args[4] if len(args) > 4 else kwargs.get('headers')
                media_type = args[5] if len(args) > 5 else kwargs.get('media_type')
                background = args[6] if len(args) > 6 else kwargs.get('background')
        else:
            if 'request' not in kwargs:
                warnings.warn(
                    'The `TemplateResponse` now requires the `request` argument.\n'
                    'Replace `TemplateResponse(name, {"context": context})` by `TemplateResponse(request, name)`.',
                    DeprecationWarning
                )
                if 'request' not in kwargs.get('context', {}):
                    raise ValueError('context must include a "request" key')
            context = kwargs.get('context', {})
            request = kwargs.get('request', context.get('request'))
            name = typing.cast(str, kwargs['name'])
            status_code = kwargs.get('status_code', 200)
            headers = kwargs.get('headers')
            media_type = kwargs.get('media_type')
            background = kwargs.get('background')
        context.setdefault('request', request)
        for context_processor in self.context_processors:
            updated_context: Dict[str, Any] = context_processor(request)
            context.update(updated_context)
        template: jinja2.Template = self.get_template(name)
        return _TemplateResponse(
            template,
            context,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background
        )
