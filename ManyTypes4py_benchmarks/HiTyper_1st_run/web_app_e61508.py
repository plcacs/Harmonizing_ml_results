import asyncio
import logging
import warnings
from functools import lru_cache, partial, update_wrapper
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, cast, final, overload
from aiosignal import Signal
from frozenlist import FrozenList
from . import hdrs
from .helpers import AppKey
from .log import web_logger
from .typedefs import Handler, Middleware
from .web_exceptions import NotAppKeyWarning
from .web_middlewares import _fix_request_current_app
from .web_request import Request
from .web_response import StreamResponse
from .web_routedef import AbstractRouteDef
from .web_urldispatcher import AbstractResource, AbstractRoute, Domain, MaskDomain, MatchedSubAppResource, PrefixedSubAppResource, SystemRoute, UrlDispatcher
__all__ = ('Application', 'CleanupError')
if TYPE_CHECKING:
    _AppSignal = Signal[Callable[['Application'], Awaitable[None]]]
    _RespPrepareSignal = Signal[Callable[[Request, StreamResponse], Awaitable[None]]]
    _Middlewares = FrozenList[Middleware]
    _MiddlewaresHandlers = Sequence[Middleware]
    _Subapps = List['Application']
else:
    _AppSignal = Signal
    _RespPrepareSignal = Signal
    _Handler = Callable
    _Middlewares = FrozenList
    _MiddlewaresHandlers = Sequence
    _Subapps = List
_T = TypeVar('_T')
_U = TypeVar('_U')
_Resource = TypeVar('_Resource', bound=AbstractResource)

def _build_middlewares(handler: typing.Callable, apps: Union[asyncio.AbstractEventLoop, asyncio.Handle]) -> typing.Callable:
    """Apply middlewares to handler."""
    for app in apps[::-1]:
        assert app.pre_frozen, 'middleware handlers are not ready'
        for m in app._middlewares_handlers:
            handler = update_wrapper(partial(m, handler=handler), handler)
    return handler
_cached_build_middleware = lru_cache(maxsize=1024)(_build_middlewares)

@final
class Application(MutableMapping[Union[str, AppKey[Any]], Any]):
    __slots__ = ('logger', '_router', '_loop', '_handler_args', '_middlewares', '_middlewares_handlers', '_run_middlewares', '_state', '_frozen', '_pre_frozen', '_subapps', '_on_response_prepare', '_on_startup', '_on_shutdown', '_on_cleanup', '_client_max_size', '_cleanup_ctx')

    def __init__(self, *, logger=web_logger, middlewares=() -> None, handler_args=None, client_max_size=1024 ** 2, debug=...):
        if debug is not ...:
            warnings.warn('debug argument is no-op since 4.0 and scheduled for removal in 5.0', DeprecationWarning, stacklevel=2)
        self._router = UrlDispatcher()
        self._handler_args = handler_args
        self.logger = logger
        self._middlewares = FrozenList(middlewares)
        self._middlewares_handlers = tuple()
        self._run_middlewares = None
        self._state = {}
        self._frozen = False
        self._pre_frozen = False
        self._subapps = []
        self._on_response_prepare = Signal(self)
        self._on_startup = Signal(self)
        self._on_shutdown = Signal(self)
        self._on_cleanup = Signal(self)
        self._cleanup_ctx = CleanupContext()
        self._on_startup.append(self._cleanup_ctx._on_startup)
        self._on_cleanup.append(self._cleanup_ctx._on_cleanup)
        self._client_max_size = client_max_size

    def __init_subclass__(cls: Union[str, typing.Type]) -> None:
        raise TypeError('Inheritance class {} from web.Application is forbidden'.format(cls.__name__))

    def __eq__(self, other: typing.Iterable[T]) -> bool:
        return self is other

    @overload
    def __getitem__(self, key: Union[str, None, KT]) -> None:
        ...

    @overload
    def __getitem__(self, key: Union[str, None, KT]) -> None:
        ...

    def __getitem__(self, key: Union[str, None, KT]) -> None:
        return self._state[key]

    def _check_frozen(self) -> None:
        if self._frozen:
            raise RuntimeError('Changing state of started or joined application is forbidden')

    @overload
    def __setitem__(self, key: Union[T, None, str, bytes], value: Union[str, VT, int]) -> None:
        ...

    @overload
    def __setitem__(self, key: Union[T, None, str, bytes], value: Union[str, VT, int]) -> None:
        ...

    def __setitem__(self, key: Union[T, None, str, bytes], value: Union[str, VT, int]) -> None:
        self._check_frozen()
        if not isinstance(key, AppKey):
            warnings.warn('It is recommended to use web.AppKey instances for keys.\n' + 'https://docs.aiohttp.org/en/stable/web_advanced.html' + '#application-s-config', category=NotAppKeyWarning, stacklevel=2)
        self._state[key] = value

    def __delitem__(self, key: Union[str, T]) -> None:
        self._check_frozen()
        del self._state[key]

    def __len__(self) -> int:
        return len(self._state)

    def __iter__(self):
        return iter(self._state)

    def __hash__(self) -> int:
        return id(self)

    @overload
    def get(self, key: Union[str, typing.Any, None, typing.Hashable], default: Union[str, typing.Any, None, typing.Hashable]=...) -> None:
        ...

    @overload
    def get(self, key: Union[str, typing.Any, None, typing.Hashable], default: Union[str, typing.Any, None, typing.Hashable]) -> None:
        ...

    @overload
    def get(self, key: Union[str, typing.Any, None, typing.Hashable], default: Union[str, typing.Any, None, typing.Hashable]=...) -> None:
        ...

    def get(self, key: Union[str, typing.Any, None, typing.Hashable], default: Union[str, typing.Any, None, typing.Hashable]=None) -> None:
        return self._state.get(key, default)

    def _set_loop(self, loop: Union[asyncio.AbstractEventLoop, None, collections.abc.Coroutine]) -> None:
        warnings.warn('_set_loop() is no-op since 4.0 and scheduled for removal in 5.0', DeprecationWarning, stacklevel=2)

    @property
    def pre_frozen(self):
        return self._pre_frozen

    def pre_freeze(self) -> None:
        if self._pre_frozen:
            return
        self._pre_frozen = True
        self._middlewares.freeze()
        self._router.freeze()
        self._on_response_prepare.freeze()
        self._cleanup_ctx.freeze()
        self._on_startup.freeze()
        self._on_shutdown.freeze()
        self._on_cleanup.freeze()
        self._middlewares_handlers = tuple(self._prepare_middleware())
        self._run_middlewares = True if self.middlewares else False
        for subapp in self._subapps:
            subapp.pre_freeze()
            self._run_middlewares = self._run_middlewares or subapp._run_middlewares

    @property
    def frozen(self):
        return self._frozen

    def freeze(self) -> None:
        if self._frozen:
            return
        self.pre_freeze()
        self._frozen = True
        for subapp in self._subapps:
            subapp.freeze()

    @property
    def debug(self) -> Union[bool, str, None]:
        warnings.warn('debug property is deprecated since 4.0 and scheduled for removal in 5.0', DeprecationWarning, stacklevel=2)
        return asyncio.get_event_loop().get_debug()

    def _reg_subapp_signals(self, subapp: typing.Callable) -> None:

        def reg_handler(signame: Any) -> None:
            subsig = getattr(subapp, signame)

            async def handler(app):
                await subsig.send(subapp)
            appsig = getattr(self, signame)
            appsig.append(handler)
        reg_handler('on_startup')
        reg_handler('on_shutdown')
        reg_handler('on_cleanup')

    def add_subapp(self, prefix: str, subapp: Union[str, bool, dict[str, str]]):
        if not isinstance(prefix, str):
            raise TypeError('Prefix must be str')
        prefix = prefix.rstrip('/')
        if not prefix:
            raise ValueError('Prefix cannot be empty')
        factory = partial(PrefixedSubAppResource, prefix, subapp)
        return self._add_subapp(factory, subapp)

    def _add_subapp(self, resource_factory: Union[typing.Callable, T], subapp: typing.Callable) -> Union[list[dict[str, typing.Any]], str]:
        if self.frozen:
            raise RuntimeError('Cannot add sub application to frozen application')
        if subapp.frozen:
            raise RuntimeError('Cannot add frozen application')
        resource = resource_factory()
        self.router.register_resource(resource)
        self._reg_subapp_signals(subapp)
        self._subapps.append(subapp)
        subapp.pre_freeze()
        return resource

    def add_domain(self, domain: str, subapp: Union[str, typing.Mapping, None]):
        if not isinstance(domain, str):
            raise TypeError('Domain must be str')
        elif '*' in domain:
            rule = MaskDomain(domain)
        else:
            rule = Domain(domain)
        factory = partial(MatchedSubAppResource, rule, subapp)
        return self._add_subapp(factory, subapp)

    def add_routes(self, routes: dict) -> Union[typing.Callable, asyncio.Task, list]:
        return self.router.add_routes(routes)

    @property
    def on_response_prepare(self):
        return self._on_response_prepare

    @property
    def on_startup(self):
        return self._on_startup

    @property
    def on_shutdown(self):
        return self._on_shutdown

    @property
    def on_cleanup(self):
        return self._on_cleanup

    @property
    def cleanup_ctx(self):
        return self._cleanup_ctx

    @property
    def router(self):
        return self._router

    @property
    def middlewares(self):
        return self._middlewares

    async def startup(self):
        """Causes on_startup signal

        Should be called in the event loop along with the request handler.
        """
        await self.on_startup.send(self)

    async def shutdown(self):
        """Causes on_shutdown signal

        Should be called before cleanup()
        """
        await self.on_shutdown.send(self)

    async def cleanup(self):
        """Causes on_cleanup signal

        Should be called after shutdown()
        """
        if self.on_cleanup.frozen:
            await self.on_cleanup.send(self)
        else:
            await self._cleanup_ctx._on_cleanup(self)

    def _prepare_middleware(self) -> typing.Generator:
        yield from reversed(self._middlewares)
        yield _fix_request_current_app(self)

    async def _handle(self, request):
        match_info = await self._router.resolve(request)
        match_info.add_app(self)
        match_info.freeze()
        request._match_info = match_info
        if request.headers.get(hdrs.EXPECT):
            resp = await match_info.expect_handler(request)
            await request.writer.drain()
            if resp is not None:
                return resp
        handler = match_info.handler
        if self._run_middlewares:
            if isinstance(match_info.route, SystemRoute):
                handler = _build_middlewares(handler, match_info.apps)
            else:
                handler = _cached_build_middleware(handler, match_info.apps)
        return await handler(request)

    def __call__(self) -> Application:
        """gunicorn compatibility"""
        return self

    def __repr__(self) -> typing.Text:
        return f'<Application 0x{id(self):x}>'

    def __bool__(self) -> bool:
        return True

class CleanupError(RuntimeError):

    @property
    def exceptions(self) -> Union[list[str], int, tuple]:
        return cast(List[BaseException], self.args[1])
if TYPE_CHECKING:
    _CleanupContextBase = FrozenList[Callable[[Application], AsyncIterator[None]]]
else:
    _CleanupContextBase = FrozenList

class CleanupContext(_CleanupContextBase):

    def __init__(self) -> None:
        super().__init__()
        self._exits = []

    async def _on_startup(self, app):
        for cb in self:
            it = cb(app).__aiter__()
            await it.__anext__()
            self._exits.append(it)

    async def _on_cleanup(self, app):
        errors = []
        for it in reversed(self._exits):
            try:
                await it.__anext__()
            except StopAsyncIteration:
                pass
            except (Exception, asyncio.CancelledError) as exc:
                errors.append(exc)
            else:
                errors.append(RuntimeError(f"{it!r} has more than one 'yield'"))
        if errors:
            if len(errors) == 1:
                raise errors[0]
            else:
                raise CleanupError('Multiple errors on cleanup stage', errors)