from __future__ import annotations
import typing
import warnings
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Union
from starlette.background import BackgroundTask
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import Receive, Scope, Send

try:
    import jinja2
    if hasattr(jinja2, 'pass_context'):
        pass_context: Callable = jinja2.pass_context
    else:
        pass_context: Callable = jinja2.contextfunction
except ModuleNotFoundError:
    jinja2 = None

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
        super().__init__(content, status_code=status_code, headers=headers, media_type=media_type, background=background)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request: Dict[str, Any] = self.context.get('request', {})
        extensions: Dict[str, Any] = request.get('extensions', {})
        if 'http.response.debug' in extensions:
            info: Dict[str, Any] = {'template': self.template, 'context': self.context}
            await send({'type': 'http.response.debug', 'info': info})
        await super().__call__(scope, receive, send)

class Jinja2Templates:
    """
    templates = Jinja2Templates("templates")

    return templates.TemplateResponse("index.html", {"request": request})
    """

    @typing.overload
    def __init__(
        self,
        directory: str,
        *,
        context_processors: Optional[List[Callable[[Request], Dict[str, Any]]]] = None,
        **env_options: Any
    ) -> None:
        ...

    @typing.overload
    def __init__(
        self,
        *,
        env: jinja2.Environment,
        context_processors: Optional[List[Callable[[Request], Dict[str, Any]]]] = None
    ) -> None:
        ...

    def __init__(
        self,
        directory: Optional[str] = None,
        *,
        context_processors: Optional[List[Callable[[Request], Dict[str, Any]]]] = None,
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
        self.context_processors: List[Callable[[Request], Dict[str, Any]]] = context_processors or []
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
                    'The `name` is not the first parameter anymore. The first parameter should be the `Request` instance.\nReplace `TemplateResponse(name, {"request": request})` by `TemplateResponse(request, name)`.',
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
                request: Request = args[0]
                name = args[1] if len(args) > 1 else kwargs['name']
                context = args[2] if len(args) > 2 else kwargs.get('context', {})
                status_code = args[3] if len(args) > 3 else kwargs.get('status_code', 200)
                headers = args[4] if len(args) > 4 else kwargs.get('headers')
                media_type = args[5] if len(args) > 5 else kwargs.get('media_type')
                background = args[6] if len(args) > 6 else kwargs.get('background')
        else:
            if 'request' not in kwargs:
                warnings.warn(
                    'The `TemplateResponse` now requires the `request` argument.\nReplace `TemplateResponse(name, {"context": context})` by `TemplateResponse(request, name)`.',
                    DeprecationWarning
                )
                if 'request' not in kwargs.get('context', {}):
                    raise ValueError('context must include a "request" key')
            context: Dict[str, Any] = kwargs.get('context', {})
            request: Optional[Request] = kwargs.get('request', context.get('request'))
            if request is None:
                raise ValueError('request must be provided either as a positional argument or in context')
            name: str = typing.cast(str, kwargs['name'])
            status_code: int = kwargs.get('status_code', 200)
            headers: Optional[Dict[str, str]] = kwargs.get('headers')
            media_type: Optional[str] = kwargs.get('media_type')
            background: Optional[BackgroundTask] = kwargs.get('background')
        context.setdefault('request', request)
        for context_processor in self.context_processors:
            context.update(context_processor(request))
        template: jinja2.Template = self.get_template(name)
        return _TemplateResponse(template, context, status_code=status_code, headers=headers, media_type=media_type, background=background)
