import abc
import asyncio
import base64
import functools
import hashlib
import html
import keyword
import os
import re
import sys
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Container,
    Dict,
    Final,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Pattern,
    Set,
    Sized,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)
from yarl import URL
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import (
    HTTPException,
    HTTPExpectationFailed,
    HTTPForbidden,
    HTTPMethodNotAllowed,
    HTTPNotFound,
)
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef

__all__ = (
    "UrlDispatcher",
    "UrlMappingMatchInfo",
    "AbstractResource",
    "Resource",
    "PlainResource",
    "DynamicResource",
    "AbstractRoute",
    "ResourceRoute",
    "StaticResource",
    "View",
)

if TYPE_CHECKING:
    from .web_app import Application
    BaseDict = Dict[str, str]
else:
    BaseDict = dict

CIRCULAR_SYMLINK_ERROR: Tuple[type, ...]
if sys.version_info < (3, 10) and sys.platform.startswith("win32"):
    CIRCULAR_SYMLINK_ERROR = (OSError,)
elif sys.version_info < (3, 13):
    CIRCULAR_SYMLINK_ERROR = (RuntimeError,)
else:
    CIRCULAR_SYMLINK_ERROR = ()

HTTP_METHOD_RE: Pattern[str] = re.compile(r"^[0-9A-Za-z!#\$%&'\*\+\-\.\^_`\\|~]+$")
ROUTE_RE: Pattern[str] = re.compile(r"(\{[_a-zA-Z][^{}]*(?:\{[^{}]*\}[^{}]*)*\})")
PATH_SEP: str = re.escape("/")
_ExpectHandler = Callable[[Request], Awaitable[Optional[StreamResponse]]]
_Resolve = Tuple[Optional["UrlMappingMatchInfo"], Set[str]]
html_escape = functools.partial(html.escape, quote=True)


class _InfoDict(TypedDict, total=False):
    pass


class AbstractResource(Sized, Iterable["AbstractRoute"]):

    def __init__(self, *, name: Optional[str] = None) -> None:
        self._name = name

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    @abc.abstractmethod
    def canonical(self) -> str:
        """Exposes the resource's canonical path.

        For example '/foo/bar/{name}'

        """

    @abc.abstractmethod
    def url_for(self, **kwargs: Any) -> URL:
        """Construct url for resource with additional params."""

    @abc.abstractmethod
    async def resolve(self, request: Request) -> _Resolve:
        """Resolve resource.

        Return (UrlMappingMatchInfo, allowed_methods) pair.
        """

    @abc.abstractmethod
    def add_prefix(self, prefix: str) -> None:
        """Add a prefix to processed URLs.

        Required for subapplications support.
        """

    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return a dict with additional info useful for introspection"""

    def freeze(self) -> None:
        pass

    @abc.abstractmethod
    def raw_match(self, path: str) -> bool:
        """Perform a raw match against path"""


class AbstractRoute(abc.ABC):

    def __init__(
        self,
        method: str,
        handler: Handler,
        *,
        expect_handler: Optional[_ExpectHandler] = None,
        resource: Optional[AbstractResource] = None,
    ) -> None:
        if expect_handler is None:
            expect_handler = _default_expect_handler
        assert asyncio.iscoroutinefunction(expect_handler), (
            f"Coroutine is expected, got {expect_handler!r}"
        )
        method = method.upper()
        if not HTTP_METHOD_RE.match(method):
            raise ValueError(f"{method} is not allowed HTTP method")
        if asyncio.iscoroutinefunction(handler):
            pass
        elif isinstance(handler, type) and issubclass(handler, AbstractView):
            pass
        else:
            raise TypeError(
                "Only async functions are allowed as web-handlers , got {!r}".format(
                    handler
                )
            )
        self._method: str = method
        self._handler: Handler = handler
        self._expect_handler: _ExpectHandler = expect_handler
        self._resource: Optional[AbstractResource] = resource

    @property
    def method(self) -> str:
        return self._method

    @property
    def handler(self) -> Handler:
        return self._handler

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        """Optional route's name, always equals to resource's name."""

    @property
    def resource(self) -> Optional[AbstractResource]:
        return self._resource

    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return a dict with additional info useful for introspection"""

    @abc.abstractmethod
    def url_for(self, *args: Any, **kwargs: Any) -> URL:
        """Construct url for route with additional params."""

    async def handle_expect_header(self, request: Request) -> Optional[StreamResponse]:
        return await self._expect_handler(request)


class UrlMappingMatchInfo(BaseDict, AbstractMatchInfo):
    __slots__ = ("_route", "_apps", "_current_app", "_frozen")

    def __init__(self, match_dict: Dict[str, str], route: "AbstractRoute") -> None:
        super().__init__(**match_dict)
        self._route: AbstractRoute = route
        self._apps: List["Application"] = []
        self._current_app: Optional["Application"] = None
        self._frozen: bool = False

    @property
    def handler(self) -> Handler:
        return self._route.handler

    @property
    def route(self) -> "AbstractRoute":
        return self._route

    @property
    def expect_handler(self) -> Callable[[Request], Awaitable[Optional[StreamResponse]]]:
        return self._route.handle_expect_header

    @property
    def http_exception(self) -> Optional[HTTPException]:
        return None

    def get_info(self) -> Dict[str, Any]:
        return self._route.get_info()

    @property
    def apps(self) -> Tuple["Application", ...]:
        return tuple(self._apps)

    def add_app(self, app: "Application") -> None:
        if self._frozen:
            raise RuntimeError("Cannot change apps stack after .freeze() call")
        if self._current_app is None:
            self._current_app = app
        self._apps.insert(0, app)

    @property
    def current_app(self) -> "Application":
        app = self._current_app
        assert app is not None
        return app

    @current_app.setter
    def current_app(self, app: "Application") -> None:
        if DEBUG:
            if app not in self._apps:
                raise RuntimeError(
                    "Expected one of the following apps {!r}, got {!r}".format(
                        self._apps, app
                    )
                )
        self._current_app = app

    def freeze(self) -> None:
        self._frozen = True

    def __repr__(self) -> str:
        return f"<MatchInfo {super().__repr__()}: {self._route}>"


class MatchInfoError(UrlMappingMatchInfo):
    __slots__ = ("_exception",)

    def __init__(self, http_exception: HTTPException) -> None:
        self._exception: HTTPException = http_exception
        super().__init__({}, SystemRoute(self._exception))

    @property
    def http_exception(self) -> HTTPException:
        return self._exception

    def __repr__(self) -> str:
        return f"<MatchInfoError {self._exception.status}: {self._exception.reason}>"


async def _default_expect_handler(request: Request) -> Optional[StreamResponse]:
    """Default handler for Expect header.

    Just send "100 Continue" to client.
    raise HTTPExpectationFailed if value of header is not "100-continue"
    """
    expect = request.headers.get(hdrs.EXPECT, "")
    if request.version == HttpVersion11:
        if expect.lower() == "100-continue":
            await request.writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            request.writer.output_size = 0
        else:
            raise HTTPExpectationFailed(text=f"Unknown Expect: {expect}")


class Resource(AbstractResource):

    def __init__(self, *, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._routes: Dict[str, "ResourceRoute"] = {}
        self._any_route: Optional["ResourceRoute"] = None
        self._allowed_methods: Set[str] = set()

    def add_route(
        self,
        method: str,
        handler: Handler,
        *,
        expect_handler: Optional[_ExpectHandler] = None,
    ) -> "ResourceRoute":
        existing_route = self._routes.get(method, self._any_route)
        if existing_route:
            raise RuntimeError(
                f"Added route will never be executed, method {existing_route.method} is already registered"
            )
        route_obj = ResourceRoute(
            method, handler, self, expect_handler=expect_handler
        )
        self.register_route(route_obj)
        return route_obj

    def register_route(self, route: "ResourceRoute") -> None:
        assert isinstance(route, ResourceRoute), (
            f"Instance of Route class is required, got {route!r}"
        )
        if route.method == hdrs.METH_ANY:
            self._any_route = route
        self._allowed_methods.add(route.method)
        self._routes[route.method] = route

    async def resolve(self, request: Request) -> _Resolve:
        match_dict = self._match(request.rel_url.path_safe)
        if match_dict is None:
            return (None, set())
        route = self._routes.get(request.method, self._any_route)
        if route:
            return (UrlMappingMatchInfo(match_dict, route), self._allowed_methods)
        return (None, self._allowed_methods)

    @abc.abstractmethod
    def _match(self, path: str) -> Optional[Dict[str, str]]:
        """Return dict of path values if path matches this resource, otherwise None."""

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator["AbstractRoute"]:
        return iter(self._routes.values())


class PlainResource(Resource):

    def __init__(self, path: str, *, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        assert not path or path.startswith("/")
        self._path: str = path

    @property
    def canonical(self) -> str:
        return self._path

    def freeze(self) -> None:
        if not self._path:
            self._path = "/"

    def add_prefix(self, prefix: str) -> None:
        assert prefix.startswith("/")
        assert not prefix.endswith("/")
        assert len(prefix) > 1
        self._path = prefix + self._path

    def _match(self, path: str) -> Optional[Dict[str, str]]:
        if self._path == path:
            return {}
        return None

    def raw_match(self, path: str) -> bool:
        return self._path == path

    def get_info(self) -> Dict[str, Any]:
        return {"path": self._path}

    def url_for(self) -> URL:
        return URL.build(path=self._path, encoded=True)

    def __repr__(self) -> str:
        name = f"'{self.name}' " if self.name is not None else ""
        return f"<PlainResource {name} {self._path}>"


class DynamicResource(Resource):
    DYN: Pattern[str] = re.compile(r"\{(?P<var>[_a-zA-Z][_a-zA-Z0-9]*)\}")
    DYN_WITH_RE: Pattern[str] = re.compile(r"\{(?P<var>[_a-zA-Z][_a-zA-Z0-9]*):(?P<re>.+)\}")
    GOOD: str = "[^{}/]+"

    def __init__(self, path: str, *, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._orig_path: str = path
        pattern = ""
        formatter = ""
        for part in ROUTE_RE.split(path):
            match = self.DYN.fullmatch(part)
            if match:
                var = match.group("var")
                pattern += f"(?P<{var}>{self.GOOD})"
                formatter += f"{{{var}}}"
                continue
            match = self.DYN_WITH_RE.fullmatch(part)
            if match:
                var = match.group("var")
                regex = match.group("re")
                pattern += f"(?P<{var}>{regex})"
                formatter += f"{{{var}}}"
                continue
            if "{" in part or "}" in part:
                raise ValueError(f"Invalid path '{path}'['{part}']")
            part_quoted = _requote_path(part)
            formatter += part_quoted
            pattern += re.escape(part)
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Bad pattern '{pattern}': {exc}") from None
        assert compiled.pattern.startswith(PATH_SEP)
        assert formatter.startswith("/")
        self._pattern: Pattern[str] = compiled
        self._formatter: str = formatter

    @property
    def canonical(self) -> str:
        return self._formatter

    def add_prefix(self, prefix: str) -> None:
        assert prefix.startswith("/")
        assert not prefix.endswith("/")
        assert len(prefix) > 1
        self._pattern = re.compile(re.escape(prefix) + self._pattern.pattern)
        self._formatter = prefix + self._formatter

    def _match(self, path: str) -> Optional[Dict[str, str]]:
        match = self._pattern.fullmatch(path)
        if match is None:
            return None
        return {key: _unquote_path_safe(value) for key, value in match.groupdict().items()}

    def raw_match(self, path: str) -> bool:
        return self._orig_path == path

    def get_info(self) -> Dict[str, Any]:
        return {"formatter": self._formatter, "pattern": self._pattern}

    def url_for(self, **parts: Any) -> URL:
        url = self._formatter.format_map(
            {k: _quote_path(v) for k, v in parts.items()}
        )
        return URL.build(path=url, encoded=True)

    def __repr__(self) -> str:
        name = f"'{self.name}' " if self.name is not None else ""
        return f"<DynamicResource {name} {self._formatter}>"


class PrefixResource(AbstractResource):

    def __init__(self, prefix: str, *, name: Optional[str] = None) -> None:
        assert not prefix or prefix.startswith("/"), prefix
        assert prefix in ("", "/") or not prefix.endswith("/"), prefix
        super().__init__(name=name)
        self._prefix: str = _requote_path(prefix)
        self._prefix2: str = self._prefix + "/"

    @property
    def canonical(self) -> str:
        return self._prefix

    def add_prefix(self, prefix: str) -> None:
        assert prefix.startswith("/")
        assert not prefix.endswith("/")
        assert len(prefix) > 1
        self._prefix = prefix + self._prefix
        self._prefix2 = self._prefix + "/"

    def raw_match(self, prefix: str) -> bool:
        return False


class StaticResource(PrefixResource):
    VERSION_KEY: Final[str] = "v"

    def __init__(
        self,
        prefix: str,
        directory: PathLike,
        *,
        name: Optional[str] = None,
        expect_handler: Optional[_ExpectHandler] = None,
        chunk_size: int = 256 * 1024,
        show_index: bool = False,
        follow_symlinks: bool = False,
        append_version: bool = False,
    ) -> None:
        super().__init__(prefix, name=name)
        try:
            directory_path = Path(directory).expanduser().resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"'{directory}' does not exist") from error
        if not directory_path.is_dir():
            raise ValueError(f"'{directory}' is not a directory")
        self._directory: Path = directory_path
        self._show_index: bool = show_index
        self._chunk_size: int = chunk_size
        self._follow_symlinks: bool = follow_symlinks
        self._expect_handler: Optional[_ExpectHandler] = expect_handler
        self._append_version: bool = append_version
        self._routes: Dict[str, "ResourceRoute"] = {
            hdrs.METH_GET: ResourceRoute(
                hdrs.METH_GET, self._handle, self, expect_handler=expect_handler
            ),
            hdrs.METH_HEAD: ResourceRoute(
                hdrs.METH_HEAD, self._handle, self, expect_handler=expect_handler
            ),
        }
        self._allowed_methods: Set[str] = set(self._routes)

    def url_for(
        self,
        *,
        filename: str,
        append_version: Optional[bool] = None,
    ) -> URL:
        if append_version is None:
            append_version = self._append_version
        filename = str(filename).lstrip("/")
        url = URL.build(path=self._prefix, encoded=True)
        url = url / filename
        if append_version:
            unresolved_path = self._directory.joinpath(filename)
            try:
                if self._follow_symlinks:
                    normalized_path = Path(os.path.normpath(unresolved_path))
                    normalized_path.relative_to(self._directory)
                    filepath = normalized_path.resolve()
                else:
                    filepath = unresolved_path.resolve()
                    filepath.relative_to(self._directory)
            except (ValueError, FileNotFoundError):
                return url
            if filepath.is_file():
                with filepath.open("rb") as f:
                    file_bytes = f.read()
                h = self._get_file_hash(file_bytes)
                url = url.with_query({self.VERSION_KEY: h})
                return url
        return url

    @staticmethod
    def _get_file_hash(byte_array: bytes) -> str:
        m = hashlib.sha256()
        m.update(byte_array)
        b64 = base64.urlsafe_b64encode(m.digest())
        return b64.decode("ascii")

    def get_info(self) -> Dict[str, Any]:
        return {"directory": self._directory, "prefix": self._prefix, "routes": self._routes}

    def set_options_route(self, handler: Handler) -> None:
        if "OPTIONS" in self._routes:
            raise RuntimeError("OPTIONS route was set already")
        self._routes["OPTIONS"] = ResourceRoute(
            "OPTIONS", handler, self, expect_handler=self._expect_handler
        )
        self._allowed_methods.add("OPTIONS")

    async def resolve(self, request: Request) -> _Resolve:
        path = request.rel_url.path_safe
        method = request.method
        if not path.startswith(self._prefix2) and path != self._prefix:
            return (None, set())
        allowed_methods = self._allowed_methods.copy()
        if method not in allowed_methods:
            return (None, allowed_methods)
        match_dict: Dict[str, str] = {"filename": _unquote_path_safe(path[len(self._prefix) + 1 :])}
        return (UrlMappingMatchInfo(match_dict, self._routes[method]), allowed_methods)

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator["AbstractRoute"]:
        return iter(self._routes.values())

    async def _handle(self, request: Request) -> Response:
        rel_url: str = request.match_info["filename"]
        filename = Path(rel_url)
        if filename.anchor:
            raise HTTPForbidden()
        unresolved_path = self._directory.joinpath(filename)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._resolve_path_to_response, unresolved_path)

    def _resolve_path_to_response(self, unresolved_path: Path) -> Response:
        """Take the unresolved path and query the file system to form a response."""
        try:
            if self._follow_symlinks:
                normalized_path = Path(os.path.normpath(unresolved_path))
                normalized_path.relative_to(self._directory)
                file_path = normalized_path.resolve()
            else:
                file_path = unresolved_path.resolve()
                file_path.relative_to(self._directory)
        except (ValueError, *CIRCULAR_SYMLINK_ERROR) as error:
            raise HTTPNotFound() from error
        try:
            if file_path.is_dir():
                if self._show_index:
                    return Response(
                        text=self._directory_as_html(file_path), content_type="text/html"
                    )
                else:
                    raise HTTPForbidden()
        except PermissionError as error:
            raise HTTPForbidden() from error
        return FileResponse(file_path, chunk_size=self._chunk_size)

    def _directory_as_html(self, dir_path: Path) -> str:
        """returns directory's index as html."""
        assert dir_path.is_dir()
        relative_path_to_dir = dir_path.relative_to(self._directory).as_posix()
        index_of: str = f"Index of /{html_escape(relative_path_to_dir)}"
        h1: str = f"<h1>{index_of}</h1>"
        index_list: List[str] = []
        dir_index = dir_path.iterdir()
        for _file in sorted(dir_index):
            rel_path = _file.relative_to(self._directory).as_posix()
            quoted_file_url = _quote_path(f"{self._prefix}/{rel_path}")
            if _file.is_dir():
                file_name = f"{_file.name}/"
            else:
                file_name = _file.name
            index_list.append(
                f'<li><a href="{quoted_file_url}">{html_escape(file_name)}</a></li>'
            )
        ul: str = "<ul>\n{}\n</ul>".format("\n".join(index_list))
        body: str = f"<body>\n{h1}\n{ul}\n</body>"
        head_str: str = f"<head>\n<title>{index_of}</title>\n</head>"
        html_content: str = f"<html>\n{head_str}\n{body}\n</html>"
        return html_content

    def __repr__(self) -> str:
        name = f"'{self.name}'" if self.name is not None else ""
        return f"<StaticResource {name} {self._prefix} -> {self._directory!r}>"


class PrefixedSubAppResource(PrefixResource):

    def __init__(self, prefix: str, app: "Application") -> None:
        super().__init__(prefix)
        self._app: "Application" = app
        self._add_prefix_to_resources(prefix)

    def add_prefix(self, prefix: str) -> None:
        super().add_prefix(prefix)
        self._add_prefix_to_resources(prefix)

    def _add_prefix_to_resources(self, prefix: str) -> None:
        router = self._app.router
        for resource in router.resources():
            router.unindex_resource(resource)
            resource.add_prefix(prefix)
            router.index_resource(resource)

    def url_for(self, *args: Any, **kwargs: Any) -> URL:
        raise RuntimeError(".url_for() is not supported by sub-application root")

    def get_info(self) -> Dict[str, Any]:
        return {"app": self._app, "prefix": self._prefix}

    async def resolve(self, request: Request) -> _Resolve:
        match_info = await self._app.router.resolve(request)
        match_info.add_app(self._app)
        if isinstance(match_info.http_exception, HTTPMethodNotAllowed):
            methods = match_info.http_exception.allowed_methods
        else:
            methods = set()
        return (match_info, methods)

    def __len__(self) -> int:
        return len(self._app.router.routes())

    def __iter__(self) -> Iterator["AbstractRoute"]:
        return iter(self._app.router.routes())

    def __repr__(self) -> str:
        return f"<PrefixedSubAppResource {self._prefix} -> {self._app!r}>"


class AbstractRuleMatching(abc.ABC):

    @abc.abstractmethod
    async def match(self, request: Request) -> bool:
        """Return bool if the request satisfies the criteria"""

    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return a dict with additional info useful for introspection"""

    @property
    @abc.abstractmethod
    def canonical(self) -> str:
        """Return a str"""


class Domain(AbstractRuleMatching):
    re_part: ClassVar[Pattern[str]] = re.compile(r"(?!-)[a-z\d-]{1,63}(?<!-)")

    def __init__(self, domain: str) -> None:
        super().__init__()
        self._domain: str = self.validation(domain)

    @property
    def canonical(self) -> str:
        return self._domain

    def validation(self, domain: str) -> str:
        if not isinstance(domain, str):
            raise TypeError("Domain must be str")
        domain = domain.rstrip(".").lower()
        if not domain:
            raise ValueError("Domain cannot be empty")
        elif "://" in domain:
            raise ValueError("Scheme not supported")
        url = URL("http://" + domain)
        assert url.raw_host is not None
        host_parts = url.raw_host.split(".")
        if not all((self.re_part.fullmatch(x) for x in host_parts)):
            raise ValueError("Domain not valid")
        if url.port == 80 or url.port is None:
            return url.raw_host
        return f"{url.raw_host}:{url.port}"

    async def match(self, request: Request) -> bool:
        host = request.headers.get(hdrs.HOST)
        if not host:
            return False
        return self.match_domain(host)

    def match_domain(self, host: str) -> bool:
        return host.lower() == self._domain

    def get_info(self) -> Dict[str, Any]:
        return {"domain": self._domain}


class MaskDomain(Domain):
    re_part: ClassVar[Pattern[str]] = re.compile(r"(?!-)[a-z\d\*-]{1,63}(?<!-)")

    def __init__(self, domain: str) -> None:
        super().__init__(domain)
        mask = self._domain.replace(".", "\\.").replace("*", ".*")
        self._mask: Pattern[str] = re.compile(mask)

    @property
    def canonical(self) -> str:
        return self._mask.pattern

    def match_domain(self, host: str) -> bool:
        return self._mask.fullmatch(host) is not None


class MatchedSubAppResource(PrefixedSubAppResource):

    def __init__(self, rule: AbstractRuleMatching, app: "Application") -> None:
        AbstractResource.__init__(self)
        self._prefix: str = ""
        self._app: "Application" = app
        self._rule: AbstractRuleMatching = rule

    @property
    def canonical(self) -> str:
        return self._rule.canonical

    def get_info(self) -> Dict[str, Any]:
        return {"app": self._app, "rule": self._rule}

    async def resolve(self, request: Request) -> _Resolve:
        if not await self._rule.match(request):
            return (None, set())
        match_info = await self._app.router.resolve(request)
        match_info.add_app(self._app)
        if isinstance(match_info.http_exception, HTTPMethodNotAllowed):
            methods = match_info.http_exception.allowed_methods
        else:
            methods = set()
        return (match_info, methods)

    def __repr__(self) -> str:
        return f"<MatchedSubAppResource -> {self._app!r}>"


class ResourceRoute(AbstractRoute):
    """A route with resource"""

    def __init__(
        self,
        method: str,
        handler: Handler,
        resource: AbstractResource,
        *,
        expect_handler: Optional[_ExpectHandler] = None,
    ) -> None:
        super().__init__(method, handler, expect_handler=expect_handler, resource=resource)

    def __repr__(self) -> str:
        return (
            "<ResourceRoute [{method}] {resource} -> {handler!r}>".format(
                method=self.method, resource=self._resource, handler=self.handler
            )
        )

    @property
    def name(self) -> Optional[str]:
        if self._resource is None:
            return None
        return self._resource.name

    def url_for(self, *args: Any, **kwargs: Any) -> URL:
        """Construct url for route with additional params."""
        assert self._resource is not None
        return self._resource.url_for(*args, **kwargs)

    def get_info(self) -> Dict[str, Any]:
        assert self._resource is not None
        return self._resource.get_info()


class SystemRoute(AbstractRoute):

    def __init__(self, http_exception: HTTPException) -> None:
        super().__init__(hdrs.METH_ANY, self._handle)
        self._http_exception: HTTPException = http_exception

    def url_for(self, *args: Any, **kwargs: Any) -> URL:
        raise RuntimeError(".url_for() is not allowed for SystemRoute")

    @property
    def name(self) -> Optional[str]:
        return None

    def get_info(self) -> Dict[str, Any]:
        return {"http_exception": self._http_exception}

    async def _handle(self, request: Request) -> NoReturn:
        raise self._http_exception

    @property
    def status(self) -> int:
        return self._http_exception.status

    @property
    def reason(self) -> str:
        return self._http_exception.reason

    def __repr__(self) -> str:
        return (
            f"<SystemRoute {self.status}: {self.reason}>"
        )


class View(AbstractView):

    async def _iter(self) -> StreamResponse:
        if self.request.method not in hdrs.METH_ALL:
            self._raise_allowed_methods()
        method = getattr(self, self.request.method.lower(), None)
        if method is None:
            self._raise_allowed_methods()
        return await method()

    def __await__(self) -> Any:
        return self._iter().__await__()

    def _raise_allowed_methods(self) -> NoReturn:
        allowed_methods = {m for m in hdrs.METH_ALL if hasattr(self, m.lower())}
        raise HTTPMethodNotAllowed(self.request.method, allowed_methods)


class ResourcesView(Sized, Iterable[AbstractResource], Container[AbstractResource]):

    def __init__(self, resources: List[AbstractResource]) -> None:
        self._resources: List[AbstractResource] = resources

    def __len__(self) -> int:
        return len(self._resources)

    def __iter__(self) -> Iterator[AbstractResource]:
        return iter(self._resources)

    def __contains__(self, resource: Any) -> bool:
        return resource in self._resources


class RoutesView(Sized, Iterable[AbstractRoute], Container[AbstractRoute]):

    def __init__(self, resources: List[AbstractResource]) -> None:
        self._routes: List[AbstractRoute] = []
        for resource in resources:
            for route in resource:
                self._routes.append(route)

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator[AbstractRoute]:
        return iter(self._routes)

    def __contains__(self, route: Any) -> bool:
        return route in self._routes


class UrlDispatcher(AbstractRouter, Mapping[str, AbstractResource]):
    NAME_SPLIT_RE: Final[Pattern[str]] = re.compile(r"[.:-]")
    HTTP_NOT_FOUND: Final[HTTPNotFound] = HTTPNotFound()

    def __init__(self) -> None:
        super().__init__()
        self._resources: List[AbstractResource] = []
        self._named_resources: Dict[str, AbstractResource] = {}
        self._resource_index: Dict[str, List[AbstractResource]] = {}
        self._matched_sub_app_resources: List[MatchedSubAppResource] = []

    async def resolve(self, request: Request) -> "UrlMappingMatchInfo":
        resource_index = self._resource_index
        allowed_methods: Set[str] = set()
        url_part = request.rel_url.path_safe
        while url_part:
            candidates = resource_index.get(url_part, ())
            for candidate in candidates:
                match_dict, allowed = await candidate.resolve(request)
                if match_dict is not None:
                    return match_dict
                else:
                    allowed_methods |= allowed
            if url_part == "/":
                break
            url_part = url_part.rpartition("/")[0] or "/"
        for resource in self._matched_sub_app_resources:
            match_dict, allowed = await resource.resolve(request)
            if match_dict is not None:
                return match_dict
            else:
                allowed_methods |= allowed
        if allowed_methods:
            return MatchInfoError(
                HTTPMethodNotAllowed(request.method, allowed_methods)
            )
        return MatchInfoError(self.HTTP_NOT_FOUND)

    def __iter__(self) -> Iterator[str]:
        return iter(self._named_resources)

    def __len__(self) -> int:
        return len(self._named_resources)

    def __contains__(self, resource: Any) -> bool:
        return resource in self._named_resources.values()

    def __getitem__(self, name: str) -> AbstractResource:
        return self._named_resources[name]

    def resources(self) -> ResourcesView:
        return ResourcesView(self._resources)

    def routes(self) -> RoutesView:
        return RoutesView(self._resources)

    def named_resources(self) -> MappingProxyType:
        return MappingProxyType(self._named_resources)

    def register_resource(self, resource: AbstractResource) -> None:
        assert isinstance(resource, AbstractResource), (
            f"Instance of AbstractResource class is required, got {resource!r}"
        )
        if self.frozen:
            raise RuntimeError("Cannot register a resource into frozen router.")
        name = resource.name
        if name is not None:
            parts = self.NAME_SPLIT_RE.split(name)
            for part in parts:
                if keyword.iskeyword(part):
                    raise ValueError(
                        f"Incorrect route name {name!r}, python keywords cannot be used for route name"
                    )
                if not part.isidentifier():
                    raise ValueError(
                        "Incorrect route name {!r}, the name should be a sequence of python identifiers separated by dash, dot or column".format(
                            name
                        )
                    )
            if name in self._named_resources:
                raise ValueError(
                    f"Duplicate {name!r}, already handled by {self._named_resources[name]!r}"
                )
            self._named_resources[name] = resource
        self._resources.append(resource)
        if isinstance(resource, MatchedSubAppResource):
            self._matched_sub_app_resources.append(resource)
        else:
            self.index_resource(resource)

    def _get_resource_index_key(self, resource: AbstractResource) -> str:
        """Return a key to index the resource in the resource index."""
        index_key = resource.canonical
        if "{" in index_key:
            index_key = index_key.partition("{")[0].rpartition("/")[0]
        return index_key.rstrip("/") or "/"

    def index_resource(self, resource: AbstractResource) -> None:
        """Add a resource to the resource index."""
        resource_key = self._get_resource_index_key(resource)
        self._resource_index.setdefault(resource_key, []).append(resource)

    def unindex_resource(self, resource: AbstractResource) -> None:
        """Remove a resource from the resource index."""
        resource_key = self._get_resource_index_key(resource)
        self._resource_index[resource_key].remove(resource)

    def add_resource(
        self, path: str, *, name: Optional[str] = None
    ) -> AbstractResource:
        if path and not path.startswith("/"):
            raise ValueError("path should be started with / or be empty")
        if self._resources:
            resource = self._resources[-1]
            if resource.name == name and resource.raw_match(path):
                return cast(Resource, resource)
        if not ("{" in path or "}" in path or ROUTE_RE.search(path)):
            resource = PlainResource(path, name=name)
            self.register_resource(resource)
            return resource
        resource = DynamicResource(path, name=name)
        self.register_resource(resource)
        return resource

    def add_route(
        self,
        method: str,
        path: str,
        handler: Handler,
        *,
        name: Optional[str] = None,
        expect_handler: Optional[_ExpectHandler] = None,
    ) -> "ResourceRoute":
        resource = self.add_resource(path, name=name)
        return resource.add_route(method, handler, expect_handler=expect_handler)

    def add_static(
        self,
        prefix: str,
        path: PathLike,
        *,
        name: Optional[str] = None,
        expect_handler: Optional[_ExpectHandler] = None,
        chunk_size: int = 256 * 1024,
        show_index: bool = False,
        follow_symlinks: bool = False,
        append_version: bool = False,
    ) -> StaticResource:
        """Add static files view.

        prefix - url prefix
        path - folder with files

        """
        assert prefix.startswith("/")
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        resource = StaticResource(
            prefix,
            path,
            name=name,
            expect_handler=expect_handler,
            chunk_size=chunk_size,
            show_index=show_index,
            follow_symlinks=follow_symlinks,
            append_version=append_version,
        )
        self.register_resource(resource)
        return resource

    def add_head(
        self, path: str, handler: Handler, **kwargs: Any
    ) -> "ResourceRoute":
        """Shortcut for add_route with method HEAD."""
        return self.add_route(hdrs.METH_HEAD, path, handler, **kwargs)

    def add_options(
        self, path: str, handler: Handler, **kwargs: Any
    ) -> "ResourceRoute":
        """Shortcut for add_route with method OPTIONS."""
        return self.add_route(hdrs.METH_OPTIONS, path, handler, **kwargs)

    def add_get(
        self,
        path: str,
        handler: Handler,
        *,
        name: Optional[str] = None,
        allow_head: bool = True,
        **kwargs: Any,
    ) -> "ResourceRoute":
        """Shortcut for add_route with method GET.

        If allow_head is true, another
        route is added allowing head requests to the same endpoint.
        """
        resource = self.add_resource(path, name=name)
        if allow_head:
            resource.add_route(hdrs.METH_HEAD, handler, **kwargs)
        return resource.add_route(hdrs.METH_GET, handler, **kwargs)

    def add_post(
        self, path: str, handler: Handler, **kwargs: Any
    ) -> "ResourceRoute":
        """Shortcut for add_route with method POST."""
        return self.add_route(hdrs.METH_POST, path, handler, **kwargs)

    def add_put(
        self, path: str, handler: Handler, **kwargs: Any
    ) -> "ResourceRoute":
        """Shortcut for add_route with method PUT."""
        return self.add_route(hdrs.METH_PUT, path, handler, **kwargs)

    def add_patch(
        self, path: str, handler: Handler, **kwargs: Any
    ) -> "ResourceRoute":
        """Shortcut for add_route with method PATCH."""
        return self.add_route(hdrs.METH_PATCH, path, handler, **kwargs)

    def add_delete(
        self, path: str, handler: Handler, **kwargs: Any
    ) -> "ResourceRoute":
        """Shortcut for add_route with method DELETE."""
        return self.add_route(hdrs.METH_DELETE, path, handler, **kwargs)

    def add_view(
        self, path: str, handler: Type[AbstractView], **kwargs: Any
    ) -> "ResourceRoute":
        """Shortcut for add_route with ANY methods for a class-based view."""
        return self.add_route(hdrs.METH_ANY, path, handler, **kwargs)

    def freeze(self) -> None:
        super().freeze()
        for resource in self._resources:
            resource.freeze()

    def add_routes(self, routes: Iterable["AbstractRouteDef"]) -> List[AbstractRoute]:
        """Append routes to route table.

        Parameter should be a sequence of RouteDef objects.

        Returns a list of registered AbstractRoute instances.
        """
        registered_routes: List[AbstractRoute] = []
        for route_def in routes:
            registered_routes.extend(route_def.register(self))
        return registered_routes


def _quote_path(value: str) -> str:
    return URL.build(path=value, encoded=False).raw_path


def _unquote_path_safe(value: str) -> str:
    if "%" not in value:
        return value
    return value.replace("%2F", "/").replace("%25", "%")


def _requote_path(value: str) -> str:
    result = _quote_path(value)
    if "%" in value:
        result = result.replace("%25", "%")
    return result
