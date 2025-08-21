from __future__ import annotations
import typing
import warnings
from os import PathLike
from starlette.background import BackgroundTask
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import Receive, Scope, Send

pass_context: typing.Any
jinja2: typing.Any

try:
    import jinja2 as _jinja2
    jinja2 = _jinja2
    if hasattr(jinja2, 'pass_context'):
        pass_context = jinja2.pass_context
    else:
        pass_context = jinja2.contextfunction
except ModuleNotFoundError:
    jinja2 = None
    pass_context = typing.cast(typing.Any, lambda f: f)


class _TemplateResponse(HTMLResponse):
    def __init__(
        self,
        template: "jinja2.Template",
        context: dict[str, typing.Any],
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None,
    ) -> None:
        self.template: "jinja2.Template" = template
        self.context: dict[str, typing.Any] = context
        content: str = template.render(context)
        super().__init__(content, status_code, headers, media_type, background)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request: typing.Mapping[str, typing.Any] = self.context.get('request', {})
        extensions: typing.Mapping[str, typing.Any] = request.get('extensions', {})
        if 'http.response.debug' in extensions:
            await send({'type': 'http.response.debug', 'info': {'template': self.template, 'context': self.context}})
        await super().__call__(scope, receive, send)


class Jinja2Templates:
    """
    templates = Jinja2Templates("templates")

    return templates.TemplateResponse("index.html", {"request": request})
    """

    @typing.overload
    def __init__(
        self,
        directory: str | PathLike[str],
        *,
        context_processors: typing.Optional[typing.Sequence[typing.Callable[[Request], dict[str, typing.Any]]]] = None,
        **env_options: typing.Any,
    ) -> None:
        ...

    @typing.overload
    def __init__(
        self,
        *,
        env: "jinja2.Environment",
        context_processors: typing.Optional[typing.Sequence[typing.Callable[[Request], dict[str, typing.Any]]]] = None,
    ) -> None:
        ...

    def __init__(
        self,
        directory: typing.Optional[str | PathLike[str]] = None,
        *,
        context_processors: typing.Optional[typing.Sequence[typing.Callable[[Request], dict[str, typing.Any]]]] = None,
        env: typing.Optional["jinja2.Environment"] = None,
        **env_options: typing.Any,
    ) -> None:
        if env_options:
            warnings.warn('Extra environment options are deprecated. Use a preconfigured jinja2.Environment instead.', DeprecationWarning)
        assert jinja2 is not None, 'jinja2 must be installed to use Jinja2Templates'
        assert bool(directory) ^ bool(env), "either 'directory' or 'env' arguments must be passed"
        self.context_processors: list[typing.Callable[[Request], dict[str, typing.Any]]] = list(context_processors or [])
        if directory is not None:
            self.env: "jinja2.Environment" = self._create_env(directory, **env_options)
        elif env is not None:
            self.env = env
        self._setup_env_defaults(self.env)

    def _create_env(self, directory: str | PathLike[str], **env_options: typing.Any) -> "jinja2.Environment":
        loader = jinja2.FileSystemLoader(directory)
        env_options.setdefault('loader', loader)
        env_options.setdefault('autoescape', True)
        return jinja2.Environment(**env_options)

    def _setup_env_defaults(self, env: "jinja2.Environment") -> None:
        @pass_context
        def url_for(context: typing.Mapping[str, typing.Any], name: str, /, **path_params: typing.Any) -> URL:
            request: Request = typing.cast(Request, context['request'])
            return request.url_for(name, **path_params)

        env.globals.setdefault('url_for', url_for)

    def get_template(self, name: str) -> "jinja2.Template":
        return self.env.get_template(name)

    @typing.overload
    def TemplateResponse(
        self,
        request: Request,
        name: str,
        context: typing.Optional[dict[str, typing.Any]] = None,
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None,
    ) -> _TemplateResponse:
        ...

    @typing.overload
    def TemplateResponse(
        self,
        name: str,
        context: typing.Optional[dict[str, typing.Any]] = None,
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None,
    ) -> _TemplateResponse:
        ...

    def TemplateResponse(self, *args: typing.Any, **kwargs: typing.Any) -> _TemplateResponse:
        if args:
            if isinstance(args[0], str):
                warnings.warn('The `name` is not the first parameter anymore. The first parameter should be the `Request` instance.\nReplace `TemplateResponse(name, {"request": request})` by `TemplateResponse(request, name)`.', DeprecationWarning)
                name: str = typing.cast(str, args[0])
                context: dict[str, typing.Any] = typing.cast(dict[str, typing.Any], args[1] if len(args) > 1 else kwargs.get('context', {}))
                status_code: int = typing.cast(int, args[2] if len(args) > 2 else kwargs.get('status_code', 200))
                headers: typing.Optional[typing.Mapping[str, str]] = typing.cast(typing.Optional[typing.Mapping[str, str]], args[2] if len(args) > 2 else kwargs.get('headers'))
                media_type: typing.Optional[str] = typing.cast(typing.Optional[str], args[3] if len(args) > 3 else kwargs.get('media_type'))
                background: typing.Optional[BackgroundTask] = typing.cast(typing.Optional[BackgroundTask], args[4] if len(args) > 4 else kwargs.get('background'))
                if 'request' not in context:
                    raise ValueError('context must include a "request" key')
                request: Request = typing.cast(Request, context['request'])
            else:
                request = typing.cast(Request, args[0])
                name = typing.cast(str, args[1] if len(args) > 1 else kwargs['name'])
                context = typing.cast(dict[str, typing.Any], args[2] if len(args) > 2 else kwargs.get('context', {}))
                status_code = typing.cast(int, args[3] if len(args) > 3 else kwargs.get('status_code', 200))
                headers = typing.cast(typing.Optional[typing.Mapping[str, str]], args[4] if len(args) > 4 else kwargs.get('headers'))
                media_type = typing.cast(typing.Optional[str], args[5] if len(args) > 5 else kwargs.get('media_type'))
                background = typing.cast(typing.Optional[BackgroundTask], args[6] if len(args) > 6 else kwargs.get('background'))
        else:
            if 'request' not in kwargs:
                warnings.warn('The `TemplateResponse` now requires the `request` argument.\nReplace `TemplateResponse(name, {"context": context})` by `TemplateResponse(request, name)`.', DeprecationWarning)
                if 'request' not in kwargs.get('context', {}):
                    raise ValueError('context must include a "request" key')
            context = typing.cast(dict[str, typing.Any], kwargs.get('context', {}))
            request = typing.cast(Request, kwargs.get('request', context.get('request')))
            name = typing.cast(str, kwargs['name'])
            status_code = typing.cast(int, kwargs.get('status_code', 200))
            headers = typing.cast(typing.Optional[typing.Mapping[str, str]], kwargs.get('headers'))
            media_type = typing.cast(typing.Optional[str], kwargs.get('media_type'))
            background = typing.cast(typing.Optional[BackgroundTask], kwargs.get('background'))
        context.setdefault('request', request)
        for context_processor in self.context_processors:
            context.update(context_processor(request))
        template = self.get_template(name)
        return _TemplateResponse(template, context, status_code=status_code, headers=headers, media_type=media_type, background=background)