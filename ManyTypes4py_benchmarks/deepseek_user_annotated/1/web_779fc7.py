from typing import (
    Dict,
    Any,
    Union,
    Optional,
    Awaitable,
    Tuple,
    List,
    Callable,
    Iterable,
    Generator,
    Type,
    TypeVar,
    cast,
    overload,
)
from types import TracebackType
import typing

if typing.TYPE_CHECKING:
    from typing import Set  # noqa: F401


# The following types are accepted by RequestHandler.set_header
# and related methods.
_HeaderTypes = Union[bytes, str, int, numbers.Integral, datetime.datetime]

_CookieSecretTypes = Union[str, bytes, Dict[int, str], Dict[int, bytes]]


MIN_SUPPORTED_SIGNED_VALUE_VERSION = 1
MAX_SUPPORTED_SIGNED_VALUE_VERSION = 2
DEFAULT_SIGNED_VALUE_VERSION = 2
DEFAULT_SIGNED_VALUE_MIN_VERSION = 1


class _ArgDefaultMarker:
    pass


_ARG_DEFAULT = _ArgDefaultMarker()


class RequestHandler:
    SUPPORTED_METHODS: Tuple[str, ...] = (
        "GET",
        "HEAD",
        "POST",
        "DELETE",
        "PATCH",
        "PUT",
        "OPTIONS",
    )

    _template_loaders: Dict[str, template.BaseLoader] = {}
    _template_loader_lock: threading.Lock = threading.Lock()
    _remove_control_chars_regex: re.Pattern = re.compile(r"[\x00-\x08\x0e-\x1f]")

    _stream_request_body: bool = False

    def __init__(
        self,
        application: "Application",
        request: httputil.HTTPServerRequest,
        **kwargs: Any,
    ) -> None:
        self.application: Application = application
        self.request: httputil.HTTPServerRequest = request
        self._headers_written: bool = False
        self._finished: bool = False
        self._auto_finish: bool = True
        self._prepared_future: Optional[Future[None]] = None
        self.ui: ObjectDict = ObjectDict(
            (n, self._ui_method(m)) for n, m in application.ui_methods.items()
        )
        self.ui["_tt_modules"] = _UIModuleNamespace(self, application.ui_modules)
        self.ui["modules"] = self.ui["_tt_modules"]
        self.clear()
        assert self.request.connection is not None
        self.request.connection.set_close_callback(self.on_connection_close)
        self.initialize(**kwargs)

    def _initialize(self) -> None:
        pass

    initialize: Callable[..., None] = _initialize

    @property
    def settings(self) -> Dict[str, Any]:
        return self.application.settings

    def _unimplemented_method(self, *args: str, **kwargs: str) -> None:
        raise HTTPError(405)

    head: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    get: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    post: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    delete: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    patch: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    put: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    options: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method

    def prepare(self) -> Optional[Awaitable[None]]:
        pass

    def on_finish(self) -> None:
        pass

    def on_connection_close(self) -> None:
        if _has_stream_request_body(self.__class__):
            if not self.request._body_future.done():
                self.request._body_future.set_exception(iostream.StreamClosedError())
                self.request._body_future.exception()

    def clear(self) -> None:
        self._headers: httputil.HTTPHeaders = httputil.HTTPHeaders(
            {
                "Server": "TornadoServer/%s" % tornado.version,
                "Content-Type": "text/html; charset=UTF-8",
                "Date": httputil.format_timestamp(time.time()),
            }
        )
        self.set_default_headers()
        self._write_buffer: List[bytes] = []
        self._status_code: int = 200
        self._reason: str = httputil.responses[200]

    def set_default_headers(self) -> None:
        pass

    def set_status(self, status_code: int, reason: Optional[str] = None) -> None:
        self._status_code = status_code
        if reason is not None:
            self._reason = escape.native_str(reason)
        else:
            self._reason = httputil.responses.get(status_code, "Unknown")

    def get_status(self) -> int:
        return self._status_code

    def set_header(self, name: str, value: _HeaderTypes) -> None:
        self._headers[name] = self._convert_header_value(value)

    def add_header(self, name: str, value: _HeaderTypes) -> None:
        self._headers.add(name, self._convert_header_value(value))

    def clear_header(self, name: str) -> None:
        if name in self._headers:
            del self._headers[name]

    _INVALID_HEADER_CHAR_RE: re.Pattern = re.compile(r"[\x00-\x1f]")

    def _convert_header_value(self, value: _HeaderTypes) -> str:
        if isinstance(value, str):
            retval = value
        elif isinstance(value, bytes):
            retval = value.decode("latin1")
        elif isinstance(value, numbers.Integral):
            return str(value)
        elif isinstance(value, datetime.datetime):
            return httputil.format_timestamp(value)
        else:
            raise TypeError("Unsupported header value %r" % value)
        if RequestHandler._INVALID_HEADER_CHAR_RE.search(retval):
            raise ValueError("Unsafe header value %r", retval)
        return retval

    @overload
    def get_argument(self, name: str, default: str, strip: bool = True) -> str:
        pass

    @overload
    def get_argument(
        self, name: str, default: _ArgDefaultMarker = _ARG_DEFAULT, strip: bool = True
    ) -> str:
        pass

    @overload
    def get_argument(
        self, name: str, default: None, strip: bool = True
    ) -> Optional[str]:
        pass

    def get_argument(
        self,
        name: str,
        default: Union[None, str, _ArgDefaultMarker] = _ARG_DEFAULT,
        strip: bool = True,
    ) -> Optional[str]:
        return self._get_argument(name, default, self.request.arguments, strip)

    def get_arguments(self, name: str, strip: bool = True) -> List[str]:
        return self._get_arguments(name, self.request.arguments, strip)

    @overload
    def get_body_argument(self, name: str, default: str, strip: bool = True) -> str:
        pass

    @overload
    def get_body_argument(
        self, name: str, default: _ArgDefaultMarker = _ARG_DEFAULT, strip: bool = True
    ) -> str:
        pass

    @overload
    def get_body_argument(
        self, name: str, default: None, strip: bool = True
    ) -> Optional[str]:
        pass

    def get_body_argument(
        self,
        name: str,
        default: Union[None, str, _ArgDefaultMarker] = _ARG_DEFAULT,
        strip: bool = True,
    ) -> Optional[str]:
        return self._get_argument(name, default, self.request.body_arguments, strip)

    def get_body_arguments(self, name: str, strip: bool = True) -> List[str]:
        return self._get_arguments(name, self.request.body_arguments, strip)

    @overload
    def get_query_argument(self, name: str, default: str, strip: bool = True) -> str:
        pass

    @overload
    def get_query_argument(
        self, name: str, default: _ArgDefaultMarker = _ARG_DEFAULT, strip: bool = True
    ) -> str:
        pass

    @overload
    def get_query_argument(
        self, name: str, default: None, strip: bool = True
    ) -> Optional[str]:
        pass

    def get_query_argument(
        self,
        name: str,
        default: Union[None, str, _ArgDefaultMarker] = _ARG_DEFAULT,
        strip: bool = True,
    ) -> Optional[str]:
        return self._get_argument(name, default, self.request.query_arguments, strip)

    def get_query_arguments(self, name: str, strip: bool = True) -> List[str]:
        return self._get_arguments(name, self.request.query_arguments, strip)

    def _get_argument(
        self,
        name: str,
        default: Union[None, str, _ArgDefaultMarker],
        source: Dict[str, List[bytes]],
        strip: bool = True,
    ) -> Optional[str]:
        args = self._get_arguments(name, source, strip=strip)
        if not args:
            if isinstance(default, _ArgDefaultMarker):
                raise MissingArgumentError(name)
            return default
        return args[-1]

    def _get_arguments(
        self, name: str, source: Dict[str, List[bytes]], strip: bool = True
    ) -> List[str]:
        values = []
        for v in source.get(name, []):
            s = self.decode_argument(v, name=name)
            if isinstance(s, str):
                s = RequestHandler._remove_control_chars_regex.sub(" ", s)
            if strip:
                s = s.strip()
            values.append(s)
        return values

    def decode_argument(self, value: bytes, name: Optional[str] = None) -> str:
        try:
            return _unicode(value)
        except UnicodeDecodeError:
            raise HTTPError(
                400, "Invalid unicode in {}: {!r}".format(name or "url", value[:40])
            )

    @property
    def cookies(self) -> Dict[str, http.cookies.Morsel]:
        return self.request.cookies

    @overload
    def get_cookie(self, name: str, default: str) -> str:
        pass

    @overload
    def get_cookie(self, name: str, default: None = None) -> Optional[str]:
        pass

    def get_cookie(self, name: str, default: Optional[str] = None) -> Optional[str]:
        if self.request.cookies is not None and name in self.request.cookies:
            return self.request.cookies[name].value
        return default

    def set_cookie(
        self,
        name: str,
        value: Union[str, bytes],
        domain: Optional[str] = None,
        expires: Optional[Union[float, Tuple, datetime.datetime]] = None,
        path: str = "/",
        expires_days: Optional[float] = None,
        *,
        max_age: Optional[int] = None,
        httponly: bool = False,
        secure: bool = False,
        samesite: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        name = escape.native_str(name)
        value = escape.native_str(value)
        if re.search(r"[\x00-\x20]", name + value):
            raise ValueError(f"Invalid cookie {name!r}: {value!r}")
        if not hasattr(self, "_new_cookie"):
            self._new_cookie = http.cookies.SimpleCookie()
        if name in self._new_cookie:
            del self._new_cookie[name]
        self._new_cookie[name] = value
        morsel = self._new_cookie[name]
        if domain:
            morsel["domain"] = domain
        if expires_days is not None and not expires:
            expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
                days=expires_days
            )
        if expires:
            morsel["expires"] = httputil.format_timestamp(expires)
        if path:
            morsel["path"] = path
        if max_age:
            morsel["max-age"] = str(max_age)
        if httponly:
            morsel["httponly"] = True
        if secure:
            morsel["secure"] = True
        if samesite:
            morsel["samesite"] = samesite
        if kwargs:
            for k, v in kwargs.items():
                morsel[k] = v
            warnings.warn(
                f"Deprecated arguments to set_cookie: {set(kwargs.keys())} "
                "(should be lowercase)",
                DeprecationWarning,
            )

    def clear_cookie(self, name: str, **kwargs: Any) -> None:
        for excluded_arg in ["expires", "max_age"]:
            if excluded_arg in kwargs:
                raise TypeError(
                    f"clear_cookie() got an unexpected keyword argument '{excluded_arg}'"
                )
        expires = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=365
        )
        self.set_cookie(name, value="", expires=expires, **kwargs)

    def clear_all_cookies(self, **kwargs: Any) -> None:
        for name in self.request.cookies:
            self.clear_cookie(name, **kwargs)

    def set_signed_cookie(
        self,
        name: str,
        value: Union[str, bytes],
        expires_days: Optional[float] = 30,
        version: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.set_cookie(
            name,
            self.create_signed_value(name, value, version=version),
            expires_days=expires_days,
            **kwargs,
        )

    set_secure_cookie = set_signed_cookie

    def create_signed_value(
        self, name: str, value: Union[str, bytes], version: Optional[int] = None
    ) -> bytes:
        self.require_setting("cookie_secret", "secure cookies")
        secret = self.application.settings["cookie_secret"]
        key_version = None
        if isinstance(secret, dict):
            if self.application.settings.get("key_version") is None:
                raise Exception("key_version setting must be used for secret_key dicts")
            key_version = self.application.settings["key_version"]

        return create_signed_value(
            secret, name, value, version=version, key_version=key_version
        )

    def get_signed_cookie(
        self,
        name: str,
        value: Optional[str] = None,
        max_age_days: float = 31,
        min_version: Optional[int] = None,
    ) -> Optional[bytes]:
        self.require_setting("cookie_secret", "secure cookies")
        if value is None:
            value = self.get_cookie(name)
        return decode_signed_value(
            self.application.settings["cookie_secret"],
            name,
            value,
            max_age_days=max_age_days,
            min_version=min_version,
        )

    get_secure_cookie = get_signed_cookie

    def get_signed_cookie_key_version(
        self, name: str, value: Optional[str] = None
    ) -> Optional[int]:
        self.require_setting("cookie_secret", "secure cookies")
        if value is None:
            value = self.get_cookie(name)
        if value is None:
            return None
        return get_signature_key_version(value)

    get_secure_cookie_key_version = get_signed_cookie_key_version

    def redirect(
        self, url: str, permanent: bool = False, status: Optional[int] = None
    ) -> None:
        if self._headers_written:
            raise Exception("Cannot redirect after headers have been written")
        if status is None:
            status = 301 if permanent else 302
        else:
            assert isinstance(status, int) and 300 <= status <= 399
        self.set_status(status)
        self.set_header("Location", utf8(url))
        self.finish()

    def write(self, chunk: Union[str, bytes, dict]) -> None:
        if self._finished:
            raise RuntimeError("Cannot write() after finish()")
        if not isinstance(chunk, (bytes, str, dict)):
            message = "write() only accepts bytes, unicode, and dict objects"
            if isinstance(chunk, list):
                message += (
                    ". Lists not accepted for security reasons; see "
                    + "http://www.tornadoweb.org/en/stable/web.html#tornado.web.RequestHandler.write"
                )
            raise TypeError(message)
        if isinstance(chunk, dict):
            chunk = escape.json_encode(chunk)
            self.set_header("Content-Type", "application/json; charset=UTF-8")
        chunk = utf8(chunk)
        self._write_buffer.append(chunk)

    def render(self, template_name: str, **kwargs: Any) -> "Future[None]":
        if self._finished:
            raise RuntimeError("Cannot render() after finish()")
        html = self.render_string(template_name, **kwargs)

        js_embed = []
        js_files = []
        css_embed = []
        css_files = []
        html_heads = []
        html_bodies = []
        for module in getattr(self, "_active_modules", {}).values():
            embed_part = module.embedded_javascript()
            if embed_part:
                js_embed.append(utf8(embed_part))
            file_part = module.javascript_files()
            if file_part:
                if isinstance(file_part, (str, bytes)):
                    js_files.append(_unicode(file_part))
                else:
                    js_files.extend(file_part)
            embed_part = module.embedded_css()
            if embed_part:
                css_embed.append(utf8(embed_part))
            file_part = module.css_files()
            if file_part:
                if isinstance(file_part, (str, bytes)):
                    css_files.append(_unicode(file_part))
                else:
                    css_files.extend(file_part)
            head_part = module.html_head()
            if head_part:
                html_heads.append(utf8(head_part))
            body_part = module.html_body()
            if body_part:
                html_bodies.append(utf8(body_part))

        if js_files:
            js =