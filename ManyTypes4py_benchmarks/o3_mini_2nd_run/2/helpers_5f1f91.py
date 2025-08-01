#!/usr/bin/env python3
"""Various helper functions"""
import asyncio
import base64
import binascii
import contextlib
import dataclasses
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from http.cookies import SimpleCookie, Morsel
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    final,
    get_args,
    overload,
)
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass

from multidict import CIMultiDict, MultiDict, MultiDictProxy, MultiMapping
from propcache.api import under_cached_property as reify
from yarl import URL
from . import hdrs
from .log import client_logger
from .typedefs import PathLike

if sys.version_info >= (3, 11):
    import asyncio as async_timeout
else:
    import async_timeout

_T = TypeVar("_T")
_S = TypeVar("_S")
_SENTINEL = enum.Enum("_SENTINEL", "sentinel")
sentinel = _SENTINEL.sentinel
NO_EXTENSIONS: bool = bool(os.environ.get("AIOHTTP_NO_EXTENSIONS"))
EMPTY_BODY_STATUS_CODES = frozenset((204, 304, *range(100, 200)))
EMPTY_BODY_METHODS = hdrs.METH_HEAD_ALL
DEBUG: bool = sys.flags.dev_mode or (
    not sys.flags.ignore_environment and bool(os.environ.get("PYTHONASYNCIODEBUG"))
)
CHAR = {chr(i) for i in range(0, 128)}
CTL = {chr(i) for i in range(0, 32)} | {chr(127)}
SEPARATORS = {
    "(",
    ")",
    "<",
    ">",
    "@",
    ",",
    ";",
    ":",
    "\\",
    '"',
    "/",
    "[",
    "]",
    "?",
    "=",
    "{",
    "}",
    " ",
    chr(9),
}
TOKEN = CHAR ^ CTL ^ SEPARATORS
json_re = re.compile(
    "(?:application/|[\\w.-]+/[\\w.+-]+?\\+)json$", re.IGNORECASE
)

__all__ = ("BasicAuth", "ChainMapProxy", "ETag", "frozen_dataclass_decorator", "reify")
PY_310 = sys.version_info >= (3, 10)
COOKIE_MAX_LENGTH = 4096

# BasicAuth helper
class BasicAuth(namedtuple("BasicAuth", ["login", "password", "encoding"])):
    """Http basic authentication helper."""

    def __new__(cls, login: str, password: str = "", encoding: str = "latin1") -> "BasicAuth":
        if login is None:
            raise ValueError("None is not allowed as login value")
        if password is None:
            raise ValueError("None is not allowed as password value")
        if ":" in login:
            raise ValueError('A ":" is not allowed in login (RFC 1945#section-11.1)')
        return super().__new__(cls, login, password, encoding)

    @classmethod
    def decode(cls, auth_header: str, encoding: str = "latin1") -> "BasicAuth":
        """Create a BasicAuth object from an Authorization HTTP header."""
        try:
            auth_type, encoded_credentials = auth_header.split(" ", 1)
        except ValueError:
            raise ValueError("Could not parse authorization header.")
        if auth_type.lower() != "basic":
            raise ValueError("Unknown authorization method %s" % auth_type)
        try:
            decoded = base64.b64decode(encoded_credentials.encode("ascii"), validate=True).decode(encoding)
        except binascii.Error:
            raise ValueError("Invalid base64 encoding.")
        try:
            username, password = decoded.split(":", 1)
        except ValueError:
            raise ValueError("Invalid credentials.")
        return cls(username, password, encoding=encoding)

    @classmethod
    def from_url(cls, url: URL, *, encoding: str = "latin1") -> Optional["BasicAuth"]:
        """Create BasicAuth from url."""
        if not isinstance(url, URL):
            raise TypeError("url should be yarl.URL instance")
        if url.raw_user is None and url.raw_password is None:
            return None
        return cls(url.user or "", url.password or "", encoding=encoding)

    def encode(self) -> str:
        """Encode credentials."""
        creds = f"{self.login}:{self.password}".encode(self.encoding)
        return "Basic %s" % base64.b64encode(creds).decode(self.encoding)


def strip_auth_from_url(url: URL) -> Tuple[URL, Optional[BasicAuth]]:
    """Remove user and password from URL if present and return BasicAuth object."""
    if url.raw_user is None and url.raw_password is None:
        return (url, None)
    return (url.with_user(None), BasicAuth(url.user or "", url.password or ""))


def netrc_from_env() -> Optional[netrc.netrc]:
    """Load netrc from file.

    Attempt to load it from the path specified by the env-var
    NETRC or in the default location in the user's home directory.

    Returns None if it couldn't be found or fails to parse.
    """
    netrc_env: Optional[str] = os.environ.get("NETRC")
    if netrc_env is not None:
        netrc_path = Path(netrc_env)
    else:
        try:
            home_dir: Path = Path.home()
        except RuntimeError as e:
            client_logger.debug("Could not resolve home directory when trying to look for .netrc file: %s", e)
            return None
        netrc_path = home_dir / ("_netrc" if platform.system() == "Windows" else ".netrc")
    try:
        return netrc.netrc(str(netrc_path))
    except netrc.NetrcParseError as e:
        client_logger.warning("Could not parse .netrc file: %s", e)
    except OSError as e:
        netrc_exists: bool = False
        with contextlib.suppress(OSError):
            netrc_exists = netrc_path.is_file()
        if netrc_env or netrc_exists:
            client_logger.warning("Could not read .netrc file: %s", e)
    return None


@functools.partial(dataclasses.dataclass, frozen=True)
class ProxyInfo:
    proxy: URL
    auth: Optional[BasicAuth] = None


def basicauth_from_netrc(netrc_obj: Optional[netrc.netrc], host: str) -> BasicAuth:
    """
    Return :py:class:`~aiohttp.BasicAuth` credentials for ``host`` from ``netrc_obj``.

    :raises LookupError: if ``netrc_obj`` is :py:data:`None` or if no
            entry is found for the ``host``.
    """
    if netrc_obj is None:
        raise LookupError("No .netrc file found")
    auth_from_netrc: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = netrc_obj.authenticators(host)
    if auth_from_netrc is None:
        raise LookupError(f"No entry for {host!s} found in the `.netrc` file.")
    login, account, password = auth_from_netrc
    username: Optional[str] = login if login or account is None else account
    if password is None:
        password = ""
    return BasicAuth(username, password)  # type: ignore


def proxies_from_env() -> Dict[str, ProxyInfo]:
    proxy_urls: Dict[str, URL] = {k: URL(v) for k, v in getproxies().items() if k in ("http", "https", "ws", "wss")}
    netrc_obj: Optional[netrc.netrc] = netrc_from_env()
    stripped: Dict[str, Tuple[URL, Optional[BasicAuth]]] = {k: strip_auth_from_url(v) for k, v in proxy_urls.items()}
    ret: Dict[str, ProxyInfo] = {}
    for proto, val in stripped.items():
        proxy, auth = val
        if proxy.scheme in ("https", "wss"):
            client_logger.warning("%s proxies %s are not supported, ignoring", proxy.scheme.upper(), proxy)
            continue
        if netrc_obj and auth is None:
            if proxy.host is not None:
                try:
                    auth = basicauth_from_netrc(netrc_obj, proxy.host)
                except LookupError:
                    auth = None
        ret[proto] = ProxyInfo(proxy, auth)
    return ret


def get_env_proxy_for_url(url: URL) -> Tuple[URL, Optional[BasicAuth]]:
    """Get a permitted proxy for the given URL from the env."""
    if url.host is not None and proxy_bypass(url.host):
        raise LookupError(f"Proxying is disallowed for `{url.host!r}`")
    proxies_in_env: Dict[str, ProxyInfo] = proxies_from_env()
    try:
        proxy_info: ProxyInfo = proxies_in_env[url.scheme]
    except KeyError:
        raise LookupError(f"No proxies found for `{url!s}` in the env")
    else:
        return (proxy_info.proxy, proxy_info.auth)


@functools.partial(dataclasses.dataclass, frozen=True)
class MimeType:
    type: str
    subtype: str
    suffix: str
    parameters: MultiDictProxy[str]


@functools.lru_cache(maxsize=56)
def parse_mimetype(mimetype: str) -> MimeType:
    """Parses a MIME type into its components.

    mimetype is a MIME type string.

    Returns a MimeType object.

    Example:

    >>> parse_mimetype('text/html; charset=utf-8')
    MimeType(type='text', subtype='html', suffix='',
             parameters=MultiDictProxy(MultiDict({'charset': 'utf-8'})))

    """
    if not mimetype:
        return MimeType(type="", subtype="", suffix="", parameters=MultiDictProxy(MultiDict()))
    parts: List[str] = mimetype.split(";")
    params: MultiDict[str] = MultiDict()
    for item in parts[1:]:
        if not item:
            continue
        key, _, value = item.partition("=")
        params.add(key.lower().strip(), value.strip(' "'))
    fulltype: str = parts[0].strip().lower()
    if fulltype == "*":
        fulltype = "*/*"
    mtype, _, stype = fulltype.partition("/")
    stype, _, suffix = stype.partition("+")
    return MimeType(type=mtype, subtype=stype, suffix=suffix, parameters=MultiDictProxy(params))


def guess_filename(obj: Any, default: Optional[str] = None) -> Optional[str]:
    name: Any = getattr(obj, "name", None)
    if name and isinstance(name, str) and (name[0] != "<") and (name[-1] != ">"):
        return Path(name).name
    return default


not_qtext_re = re.compile(r'[^\041\043-\133\135-\176]')
QCONTENT: set[str] = {chr(i) for i in range(32, 127)} | {"\t"}


def quoted_string(content: str) -> str:
    """Return 7-bit content as quoted-string.

    Format content into a quoted-string as defined in RFC5322 for
    Internet Message Format. Notice that this is not the 8-bit HTTP
    format, but the 7-bit email format. Content must be in usascii or
    a ValueError is raised.
    """
    if not QCONTENT > set(content):
        raise ValueError(f"bad content for quoted-string {content!r}")
    return not_qtext_re.sub(lambda x: "\\" + x.group(0), content)


def content_disposition_header(
    disptype: str,
    quote_fields: bool = True,
    _charset: str = "utf-8",
    params: Optional[Mapping[str, str]] = None,
) -> str:
    """Sets ``Content-Disposition`` header for MIME.

    This is the MIME payload Content-Disposition header from RFC 2183
    and RFC 7579 section 4.2, not the HTTP Content-Disposition from
    RFC 6266.

    disptype is a disposition type: inline, attachment, form-data.
    Should be valid extension token (see RFC 2183)

    quote_fields performs value quoting to 7-bit MIME headers
    according to RFC 7578. Set to quote_fields to False if recipient
    can take 8-bit file names and field values.

    _charset specifies the charset to use when quote_fields is True.

    params is a dict with disposition params.
    """
    if not disptype or not TOKEN > set(disptype):
        raise ValueError(f"bad content disposition type {disptype!r}")
    value: str = disptype
    if params:
        lparams: List[Tuple[str, str]] = []
        for key, val in params.items():
            if not key or not TOKEN > set(key):
                raise ValueError(f"bad content disposition parameter {key!r}={val!r}")
            if quote_fields:
                if key.lower() == "filename":
                    qval: str = quote(val, "", encoding=_charset)
                    lparams.append((key, f'"{qval}"'))
                else:
                    try:
                        qval = quoted_string(val)
                    except ValueError:
                        qval = "".join((_charset, "''", quote(val, "", encoding=_charset)))
                        lparams.append((key + "*", qval))
                    else:
                        lparams.append((key, f'"{qval}"'))
            else:
                qval = val.replace("\\", "\\\\").replace('"', '\\"')
                lparams.append((key, f'"{qval}"'))
        sparams: str = "; ".join(("=".join(pair) for pair in lparams))
        value = "; ".join((value, sparams))
    return value


def is_expected_content_type(response_content_type: str, expected_content_type: str) -> bool:
    """Checks if received content type is processable as an expected one.

    Both arguments should be given without parameters.
    """
    if expected_content_type == "application/json":
        return json_re.match(response_content_type) is not None
    return expected_content_type in response_content_type


def is_ip_address(host: Optional[str]) -> bool:
    """Check if host looks like an IP Address.

    This check is only meant as a heuristic to ensure that
    a host is not a domain name.
    """
    if not host:
        return False
    return ":" in host or host.replace(".", "").isdigit()


_cached_current_datetime: Optional[int] = None
_cached_formatted_datetime: str = ""


def rfc822_formatted_time() -> str:
    global _cached_current_datetime
    global _cached_formatted_datetime
    now: int = int(time.time())
    if now != _cached_current_datetime:
        _weekdayname: Tuple[str, str, str, str, str, str, str] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        _monthname: Tuple[str, ...] = (
            "",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        )
        year, month, day, hh, mm, ss, wd, *tail = time.gmtime(now)
        _cached_formatted_datetime = "%s, %02d %3s %4d %02d:%02d:%02d GMT" % (
            _weekdayname[wd],
            day,
            _monthname[month],
            year,
            hh,
            mm,
            ss,
        )
        _cached_current_datetime = now
    return _cached_formatted_datetime


def _weakref_handle(info: Tuple[weakref.ref[Any], str]) -> None:
    ref, name = info
    ob: Optional[Any] = ref()
    if ob is not None:
        with suppress(Exception):
            getattr(ob, name)()


def weakref_handle(
    ob: Any, name: str, timeout: Optional[float], loop: asyncio.AbstractEventLoop, timeout_ceil_threshold: float = 5
) -> Optional[asyncio.TimerHandle]:
    if timeout is not None and timeout > 0:
        when: float = loop.time() + timeout
        if timeout >= timeout_ceil_threshold:
            when = ceil(when)
        return loop.call_at(when, _weakref_handle, (weakref.ref(ob), name))
    return None


def call_later(
    cb: Callable[..., Any], timeout: Optional[float], loop: asyncio.AbstractEventLoop, timeout_ceil_threshold: float = 5
) -> Optional[asyncio.TimerHandle]:
    if timeout is None or timeout <= 0:
        return None
    now: float = loop.time()
    when: float = calculate_timeout_when(now, timeout, timeout_ceil_threshold)
    return loop.call_at(when, cb)


def calculate_timeout_when(loop_time: float, timeout: float, timeout_ceiling_threshold: float) -> float:
    """Calculate when to execute a timeout."""
    when: float = loop_time + timeout
    if timeout > timeout_ceiling_threshold:
        return ceil(when)
    return when


class TimeoutHandle:
    """Timeout handle"""
    __slots__ = ("_timeout", "_loop", "_ceil_threshold", "_callbacks")

    def __init__(self, loop: asyncio.AbstractEventLoop, timeout: Optional[float], ceil_threshold: float = 5) -> None:
        self._timeout: Optional[float] = timeout
        self._loop: asyncio.AbstractEventLoop = loop
        self._ceil_threshold: float = ceil_threshold
        self._callbacks: List[Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]] = []

    def register(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._callbacks.append((callback, args, kwargs))

    def close(self) -> None:
        self._callbacks.clear()

    def start(self) -> Optional[asyncio.TimerHandle]:
        timeout: Optional[float] = self._timeout
        if timeout is not None and timeout > 0:
            when: float = self._loop.time() + timeout
            if timeout >= self._ceil_threshold:
                when = ceil(when)
            return self._loop.call_at(when, self.__call__)
        else:
            return None

    def timer(self) -> Union["TimerContext", "TimerNoop"]:
        if self._timeout is not None and self._timeout > 0:
            timer = TimerContext(self._loop)
            self.register(timer.timeout)
            return timer
        else:
            return TimerNoop()

    def __call__(self) -> None:
        for cb, args, kwargs in self._callbacks:
            with suppress(Exception):
                cb(*args, **kwargs)
        self._callbacks.clear()


class BaseTimerContext(ContextManager["BaseTimerContext"]):
    __slots__ = ()

    def assert_timeout(self) -> None:
        """Raise TimeoutError if timeout has been exceeded."""
        pass


class TimerNoop(BaseTimerContext):
    __slots__ = ()

    def __enter__(self) -> "TimerNoop":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        return None


class TimerContext(BaseTimerContext):
    """Low resolution timeout context manager"""
    __slots__ = ("_loop", "_tasks", "_cancelled", "_cancelling")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop: asyncio.AbstractEventLoop = loop
        self._tasks: List[asyncio.Task[Any]] = []
        self._cancelled: bool = False
        self._cancelling: int = 0

    def assert_timeout(self) -> None:
        """Raise TimeoutError if timer has already been cancelled."""
        if self._cancelled:
            raise asyncio.TimeoutError from None

    def __enter__(self) -> "TimerContext":
        task: Optional[asyncio.Task[Any]] = asyncio.current_task(loop=self._loop)
        if task is None:
            raise RuntimeError("Timeout context manager should be used inside a task")
        if sys.version_info >= (3, 11):
            self._cancelling = task.cancelling()  # type: ignore
        if self._cancelled:
            raise asyncio.TimeoutError from None
        self._tasks.append(task)
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        enter_task: Optional[asyncio.Task[Any]] = None
        if self._tasks:
            enter_task = self._tasks.pop()
        if exc_type is asyncio.CancelledError and self._cancelled:
            assert enter_task is not None
            if sys.version_info >= (3, 11):
                if enter_task.uncancel() > self._cancelling:  # type: ignore
                    return None
            raise asyncio.TimeoutError from exc_val  # type: ignore
        return None

    def timeout(self) -> None:
        if not self._cancelled:
            for task in set(self._tasks):
                task.cancel()
            self._cancelled = True


def ceil_timeout(delay: Optional[float], ceil_threshold: float = 5) -> Any:
    if delay is None or delay <= 0:
        return async_timeout.timeout(None)
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    now: float = loop.time()
    when: float = now + delay
    if delay > ceil_threshold:
        when = ceil(when)
    return async_timeout.timeout_at(when)


class HeadersMixin:
    """Mixin for handling headers."""
    _content_type: Optional[str] = None
    _content_dict: Optional[Dict[str, str]] = None
    _stored_content_type: Union[str, enum.Enum] = sentinel
    _headers: Mapping[str, str]

    def _parse_content_type(self, raw: Optional[str]) -> None:
        self._stored_content_type = raw
        if raw is None:
            self._content_type = "application/octet-stream"
            self._content_dict = {}
        else:
            msg = HeaderParser().parsestr("Content-Type: " + raw)
            self._content_type = msg.get_content_type()
            params = msg.get_params(())
            self._content_dict = dict(params[1:])

    @property
    def content_type(self) -> str:
        """The value of content part for Content-Type HTTP header."""
        raw: Optional[str] = self._headers.get(hdrs.CONTENT_TYPE)
        if self._stored_content_type != raw:
            self._parse_content_type(raw)
        assert self._content_type is not None
        return self._content_type

    @property
    def charset(self) -> Optional[str]:
        """The value of charset part for Content-Type HTTP header."""
        raw: Optional[str] = self._headers.get(hdrs.CONTENT_TYPE)
        if self._stored_content_type != raw:
            self._parse_content_type(raw)
        assert self._content_dict is not None
        return self._content_dict.get("charset")

    @property
    def content_length(self) -> Optional[int]:
        """The value of Content-Length HTTP header."""
        content_length: Optional[str] = self._headers.get(hdrs.CONTENT_LENGTH)
        return None if content_length is None else int(content_length)


def set_result(fut: asyncio.Future, result: Any) -> None:
    if not fut.done():
        fut.set_result(result)


_EXC_SENTINEL: BaseException = BaseException()


class ErrorableProtocol(Protocol):
    def set_exception(self, exc: Exception, exc_cause: Any = ...) -> None:
        ...


def set_exception(fut: "ErrorableProtocol", exc: Exception, exc_cause: Union[BaseException, Any] = _EXC_SENTINEL) -> None:
    """Set future exception.

    If the future is marked as complete, this function is a no-op.

    :param exc_cause: An exception that is a direct cause of ``exc``.
                      Only set if provided.
    """
    if asyncio.isfuture(fut) and fut.done():
        return
    exc_is_sentinel: bool = exc_cause is _EXC_SENTINEL
    exc_causes_itself: bool = exc is exc_cause
    if not exc_is_sentinel and (not exc_causes_itself):
        exc.__cause__ = exc_cause  # type: ignore
    fut.set_exception(exc)


@functools.total_ordering
class AppKey(Generic[_T]):
    """Keys for static typing support in Application."""
    __slots__ = ("_name", "_t", "__orig_class__")

    def __init__(self, name: str, t: Optional[Any] = None) -> None:
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == "<module>":
                module = frame.f_globals["__name__"]
                break
            frame = frame.f_back
        else:
            raise RuntimeError("Failed to get module name.")
        self._name = module + "." + name
        self._t = t

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, AppKey):
            return self._name < other._name
        return True

    def __repr__(self) -> str:
        t = self._t
        if t is None:
            with suppress(AttributeError):
                t = get_args(self.__orig_class__)[0]
        if t is None:
            t_repr = "<<Unknown>>"
        elif isinstance(t, type):
            if t.__module__ == "builtins":
                t_repr = t.__qualname__
            else:
                t_repr = f"{t.__module__}.{t.__qualname__}"
        else:
            t_repr = repr(t)
        return f"<AppKey({self._name}, type={t_repr})>"


@final
class ChainMapProxy(Mapping[Union[str, AppKey[Any]], Any]):
    __slots__ = ("_maps",)

    def __init__(self, maps: Iterable[Mapping[Union[str, AppKey[Any]], Any]]) -> None:
        self._maps: Tuple[Mapping[Union[str, AppKey[Any]], Any], ...] = tuple(maps)

    def __init_subclass__(cls) -> None:
        raise TypeError("Inheritance class {} from ChainMapProxy is forbidden".format(cls.__name__))

    @overload
    def __getitem__(self, key: Union[str, AppKey[Any]]) -> Any:
        ...
    
    @overload
    def __getitem__(self, key: Union[str, AppKey[Any]]) -> Any:
        ...

    def __getitem__(self, key: Union[str, AppKey[Any]]) -> Any:
        for mapping in self._maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        raise KeyError(key)

    @overload
    def get(self, key: Union[str, AppKey[Any]], default: Any) -> Any:
        ...
    
    @overload
    def get(self, key: Union[str, AppKey[Any]], default: object = ...) -> Any:
        ...
    
    @overload
    def get(self, key: Union[str, AppKey[Any]], default: object = ...) -> Any:
        ...

    def get(self, key: Union[str, AppKey[Any]], default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        return len(set().union(*self._maps))

    def __iter__(self) -> Iterator[Union[str, AppKey[Any]]]:
        d: Dict[Union[str, AppKey[Any]], Any] = {}
        for mapping in reversed(self._maps):
            d.update(mapping)
        return iter(d)

    def __contains__(self, key: object) -> bool:
        return any((key in m for m in self._maps))

    def __bool__(self) -> bool:
        return any(self._maps)

    def __repr__(self) -> str:
        content = ", ".join(map(repr, self._maps))
        return f"ChainMapProxy({content})"


class CookieMixin:
    """Mixin for handling cookies."""
    _cookies: Optional[SimpleCookie] = None

    @property
    def cookies(self) -> SimpleCookie:
        if self._cookies is None:
            self._cookies = SimpleCookie()
        return self._cookies

    def set_cookie(
        self,
        name: str,
        value: str,
        *,
        expires: Optional[str] = None,
        domain: Optional[str] = None,
        max_age: Optional[int] = None,
        path: str = "/",
        secure: Optional[bool] = None,
        httponly: Optional[bool] = None,
        samesite: Optional[str] = None,
        partitioned: Optional[bool] = None,
    ) -> None:
        """Set or update response cookie.

        Sets new cookie or updates existent with new value.
        Also updates only those params which are not None.
        """
        if self._cookies is None:
            self._cookies = SimpleCookie()
        self._cookies[name] = value
        c: Morsel = self._cookies[name]
        if expires is not None:
            c["expires"] = expires
        elif c.get("expires") == "Thu, 01 Jan 1970 00:00:00 GMT":
            del c["expires"]
        if domain is not None:
            c["domain"] = domain
        if max_age is not None:
            c["max-age"] = str(max_age)
        elif "max-age" in c:
            del c["max-age"]
        c["path"] = path
        if secure is not None:
            c["secure"] = secure  # type: ignore
        if httponly is not None:
            c["httponly"] = httponly  # type: ignore
        if samesite is not None:
            c["samesite"] = samesite
        if partitioned is not None:
            c["partitioned"] = partitioned  # type: ignore
        if DEBUG:
            cookie_length: int = len(c.output(header="")[1:])
            if cookie_length > COOKIE_MAX_LENGTH:
                warnings.warn(
                    "The size of is too large, it might get ignored by the client.",
                    UserWarning,
                    stacklevel=2,
                )

    def del_cookie(
        self,
        name: str,
        *,
        domain: Optional[str] = None,
        path: str = "/",
        secure: Optional[bool] = None,
        httponly: Optional[bool] = None,
        samesite: Optional[str] = None,
    ) -> None:
        """Delete cookie.

        Creates new empty expired cookie.
        """
        if self._cookies is not None:
            self._cookies.pop(name, None)
        self.set_cookie(
            name,
            "",
            max_age=0,
            expires="Thu, 01 Jan 1970 00:00:00 GMT",
            domain=domain,
            path=path,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
        )


def populate_with_cookies(headers: CIMultiDict[str], cookies: Mapping[Any, Morsel]) -> None:
    for cookie in cookies.values():
        value: str = cookie.output(header="")[1:]
        headers.add(hdrs.SET_COOKIE, value)


_ETAGC = r'[!\x23-\x7E\x80-\xff]+'
_ETAGC_RE = re.compile(_ETAGC)
_QUOTED_ETAG = f'(W/)?"({_ETAGC})"'
QUOTED_ETAG_RE = re.compile(_QUOTED_ETAG)
LIST_QUOTED_ETAG_RE = re.compile(f'({_QUOTED_ETAG})(?:\\s*,\\s*|$)|(.)')
ETAG_ANY = "*"


@functools.partial(dataclasses.dataclass, frozen=True)
class ETag:
    is_weak: bool = False


def validate_etag_value(value: str) -> None:
    if value != ETAG_ANY and (not _ETAGC_RE.fullmatch(value)):
        raise ValueError(f"Value {value!r} is not a valid etag. Maybe it contains '\"'?")


def parse_http_date(date_str: Optional[str]) -> Optional[datetime.datetime]:
    """Process a date string, return a datetime object"""
    if date_str is not None:
        timetuple = parsedate(date_str)
        if timetuple is not None:
            with suppress(ValueError):
                return datetime.datetime(*timetuple[:6], tzinfo=datetime.timezone.utc)
    return None


@functools.lru_cache
def must_be_empty_body(method: str, code: int) -> bool:
    """Check if a request must return an empty body."""
    return (
        code in EMPTY_BODY_STATUS_CODES
        or method in EMPTY_BODY_METHODS
        or (200 <= code < 300 and method in hdrs.METH_CONNECT_ALL)
    )


def should_remove_content_length(method: str, code: int) -> bool:
    """Check if a Content-Length header should be removed.

    This should always be a subset of must_be_empty_body
    """
    return code in EMPTY_BODY_STATUS_CODES or (200 <= code < 300 and method in hdrs.METH_CONNECT_ALL)
