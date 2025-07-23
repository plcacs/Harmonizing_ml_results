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
from collections import ChainMap, namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from http.cookies import SimpleCookie
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING, Any, Callable, ContextManager, Dict, FrozenSet, Generic, 
    Iterable, Iterator, List, Mapping, Optional, Pattern, Protocol, Set, 
    Tuple, Type, TypeVar, Union, final, get_args, overload
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

if TYPE_CHECKING:
    from dataclasses import dataclass as frozen_dataclass_decorator
elif sys.version_info < (3, 10):
    frozen_dataclass_decorator = functools.partial(dataclasses.dataclass, frozen=True)
else:
    frozen_dataclass_decorator = functools.partial(dataclasses.dataclass, frozen=True, slots=True)

__all__ = ('BasicAuth', 'ChainMapProxy', 'ETag', 'frozen_dataclass_decorator', 'reify')

PY_310: bool = sys.version_info >= (3, 10)
COOKIE_MAX_LENGTH: int = 4096
_T = TypeVar('_T')
_S = TypeVar('_S')
_SENTINEL = enum.Enum('_SENTINEL', 'sentinel')
sentinel: Any = _SENTINEL.sentinel
NO_EXTENSIONS: bool = bool(os.environ.get('AIOHTTP_NO_EXTENSIONS'))
EMPTY_BODY_STATUS_CODES: FrozenSet[int] = frozenset((204, 304, *range(100, 200)))
EMPTY_BODY_METHODS: FrozenSet[str] = hdrs.METH_HEAD_ALL
DEBUG: bool = sys.flags.dev_mode or (not sys.flags.ignore_environment and bool(os.environ.get('PYTHONASYNCIODEBUG')))
CHAR: Set[str] = {chr(i) for i in range(0, 128)}
CTL: Set[str] = {chr(i) for i in range(0, 32)} | {chr(127)}
SEPARATORS: Set[str] = {'(', ')', '<', '>', '@', ',', ';', ':', '\\', '"', '/', '[', ']', '?', '=', '{', '}', ' ', chr(9)}
TOKEN: Set[str] = CHAR ^ CTL ^ SEPARATORS
json_re: Pattern[str] = re.compile('(?:application/|[\\w.-]+/[\\w.+-]+?\\+)json$', re.IGNORECASE)

class BasicAuth(namedtuple('BasicAuth', ['login', 'password', 'encoding'])):
    """Http basic authentication helper."""

    def __new__(cls, login: str, password: str = '', encoding: str = 'latin1') -> 'BasicAuth':
        if login is None:
            raise ValueError('None is not allowed as login value')
        if password is None:
            raise ValueError('None is not allowed as password value')
        if ':' in login:
            raise ValueError('A ":" is not allowed in login (RFC 1945#section-11.1)')
        return super().__new__(cls, login, password, encoding)

    @classmethod
    def decode(cls, auth_header: str, encoding: str = 'latin1') -> 'BasicAuth':
        """Create a BasicAuth object from an Authorization HTTP header."""
        try:
            auth_type, encoded_credentials = auth_header.split(' ', 1)
        except ValueError:
            raise ValueError('Could not parse authorization header.')
        if auth_type.lower() != 'basic':
            raise ValueError('Unknown authorization method %s' % auth_type)
        try:
            decoded = base64.b64decode(encoded_credentials.encode('ascii'), validate=True).decode(encoding)
        except binascii.Error:
            raise ValueError('Invalid base64 encoding.')
        try:
            username, password = decoded.split(':', 1)
        except ValueError:
            raise ValueError('Invalid credentials.')
        return cls(username, password, encoding=encoding)

    @classmethod
    def from_url(cls, url: URL, *, encoding: str = 'latin1') -> Optional['BasicAuth']:
        """Create BasicAuth from url."""
        if not isinstance(url, URL):
            raise TypeError('url should be yarl.URL instance')
        if url.raw_user is None and url.raw_password is None:
            return None
        return cls(url.user or '', url.password or '', encoding=encoding)

    def encode(self) -> str:
        """Encode credentials."""
        creds = f'{self.login}:{self.password}'.encode(self.encoding)
        return 'Basic %s' % base64.b64encode(creds).decode(self.encoding)

def strip_auth_from_url(url: URL) -> Tuple[URL, Optional[BasicAuth]]:
    """Remove user and password from URL if present and return BasicAuth object."""
    if url.raw_user is None and url.raw_password is None:
        return (url, None)
    return (url.with_user(None), BasicAuth(url.user or '', url.password or ''))

def netrc_from_env() -> Optional[netrc.netrc]:
    """Load netrc from file."""
    netrc_env = os.environ.get('NETRC')
    if netrc_env is not None:
        netrc_path = Path(netrc_env)
    else:
        try:
            home_dir = Path.home()
        except RuntimeError as e:
            client_logger.debug('Could not resolve home directory when trying to look for .netrc file: %s', e)
            return None
        netrc_path = home_dir / ('_netrc' if platform.system() == 'Windows' else '.netrc')
    try:
        return netrc.netrc(str(netrc_path))
    except netrc.NetrcParseError as e:
        client_logger.warning('Could not parse .netrc file: %s', e)
    except OSError as e:
        netrc_exists = False
        with contextlib.suppress(OSError):
            netrc_exists = netrc_path.is_file()
        if netrc_env or netrc_exists:
            client_logger.warning('Could not read .netrc file: %s', e)
    return None

@frozen_dataclass_decorator
class ProxyInfo:
    proxy: URL
    proxy_auth: Optional[BasicAuth]

def basicauth_from_netrc(netrc_obj: Optional[netrc.netrc], host: str) -> BasicAuth:
    """Return BasicAuth credentials for host from netrc_obj."""
    if netrc_obj is None:
        raise LookupError('No .netrc file found')
    auth_from_netrc = netrc_obj.authenticators(host)
    if auth_from_netrc is None:
        raise LookupError(f'No entry for {host!s} found in the `.netrc` file.')
    login, account, password = auth_from_netrc
    username = login if login or account is None else account
    if password is None:
        password = ''
    return BasicAuth(username, password)

def proxies_from_env() -> Dict[str, ProxyInfo]:
    proxy_urls = {k: URL(v) for k, v in getproxies().items() if k in ('http', 'https', 'ws', 'wss')}
    netrc_obj = netrc_from_env()
    stripped = {k: strip_auth_from_url(v) for k, v in proxy_urls.items()}
    ret: Dict[str, ProxyInfo] = {}
    for proto, val in stripped.items():
        proxy, auth = val
        if proxy.scheme in ('https', 'wss'):
            client_logger.warning('%s proxies %s are not supported, ignoring', proxy.scheme.upper(), proxy)
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
        raise LookupError(f'Proxying is disallowed for `{url.host!r}`')
    proxies_in_env = proxies_from_env()
    try:
        proxy_info = proxies_in_env[url.scheme]
    except KeyError:
        raise LookupError(f'No proxies found for `{url!s}` in the env')
    else:
        return (proxy_info.proxy, proxy_info.proxy_auth)

@frozen_dataclass_decorator
class MimeType:
    type: str
    subtype: str
    suffix: str
    parameters: MultiDictProxy[str]

@functools.lru_cache(maxsize=56)
def parse_mimetype(mimetype: str) -> MimeType:
    """Parses a MIME type into its components."""
    if not mimetype:
        return MimeType(type='', subtype='', suffix='', parameters=MultiDictProxy(MultiDict()))
    parts = mimetype.split(';')
    params = MultiDict()
    for item in parts[1:]:
        if not item:
            continue
        key, _, value = item.partition('=')
        params.add(key.lower().strip(), value.strip(' "'))
    fulltype = parts[0].strip().lower()
    if fulltype == '*':
        fulltype = '*/*'
    mtype, _, stype = fulltype.partition('/')
    stype, _, suffix = stype.partition('+')
    return MimeType(type=mtype, subtype=stype, suffix=suffix, parameters=MultiDictProxy(params))

def guess_filename(obj: Any, default: Optional[str] = None) -> Optional[str]:
    name = getattr(obj, 'name', None)
    if name and isinstance(name, str) and (name[0] != '<') and (name[-1] != '>'):
        return Path(name).name
    return default

not_qtext_re: Pattern[str] = re.compile('[^\\041\\043-\\133\\135-\\176]')
QCONTENT: Set[str] = {chr(i) for i in range(32, 127)} | {'\t'}

def quoted_string(content: str) -> str:
    """Return 7-bit content as quoted-string."""
    if not QCONTENT > set(content):
        raise ValueError(f'bad content for quoted-string {content!r}')
    return not_qtext_re.sub(lambda x: '\\' + x.group(0), content)

def content_disposition_header(
    disptype: str,
    quote_fields: bool = True,
    _charset: str = 'utf-8',
    params: Optional[Dict[str, str]] = None
) -> str:
    """Sets Content-Disposition header for MIME."""
    if not disptype or not TOKEN > set(disptype):
        raise ValueError(f'bad content disposition type {disptype!r}')
    value = disptype
    if params:
        lparams: List[Tuple[str, str]] = []
        for key, val in params.items():
            if not key or not TOKEN > set(key):
                raise ValueError(f'bad content disposition parameter {key!r}={val!r}')
            if quote_fields:
                if key.lower() == 'filename':
                    qval = quote(val, '', encoding=_charset)
                    lparams.append((key, '"%s"' % qval))
                else:
                    try:
                        qval = quoted_string(val)
                    except ValueError:
                        qval = ''.join((_charset, "''", quote(val, '', encoding=_charset)))
                        lparams.append((key + '*', qval))
                    else:
                        lparams.append((key, '"%s"' % qval))
            else:
                qval = val.replace('\\', '\\\\').replace('"', '\\"')
                lparams.append((key, '"%s"' % qval))
        sparams = '; '.join(('='.join(pair) for pair in lparams))
        value = '; '.join((value, sparams))
    return value

def is_expected_content_type(response_content_type: str, expected_content_type: str) -> bool:
    """Checks if received content type is processable as an expected one."""
    if expected_content_type == 'application/json':
        return json_re.match(response_content_type) is not None
    return expected_content_type in response_content_type

def is_ip_address(host: Optional[str]) -> bool:
    """Check if host looks like an IP Address."""
    if not host:
        return False
    return ':' in host or host.replace('.', '').isdigit()

_cached_current_datetime: Optional[int] = None
_cached_formatted_datetime: str = ''

def rfc822_formatted_time() -> str:
    global _cached_current_datetime
    global _cached_formatted_datetime
    now = int(time.time())
    if now != _cached_current_datetime:
        _weekdayname = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
        _monthname = ('', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
        year, month, day, hh, mm, ss, wd, *tail = time.gmtime(now)
        _cached_formatted_datetime = '%s, %02d %3s %4d %02d:%02d:%02d GMT' % (_weekdayname[wd], day, _monthname[month], year, hh, mm, ss)
        _cached_current_datetime = now
    return _cached_formatted_datetime

def _weakref_handle(info: Tuple[weakref.ref[Any], str]) -> None:
    ref, name = info
    ob = ref()
    if ob is not None:
        with suppress(Exception):
            getattr(ob, name)()

def weakref_handle(
    ob: Any,
    name: str,
    timeout: Optional[float],
    loop: asyncio.AbstractEventLoop,
    timeout_ceil_threshold: float = 5
) -> Optional[asyncio.Handle]:
    if timeout is not None and timeout > 0:
        when = loop.time() + timeout
        if timeout >= timeout_ceil_threshold:
            when = ceil(when)
        return loop.call_at(when, _weakref_handle, (weakref.ref(ob), name))
    return None

def call_later(
    cb: Callable[..., Any],
    timeout: Optional[float],
    loop: asyncio.AbstractEventLoop,
    timeout_ceil_threshold: float = 5
) -> Optional[asyncio.Handle]:
    if timeout is None or timeout <= 0:
        return None
    now = loop.time()
    when = calculate_timeout_when(now, timeout, timeout_ceil_threshold)
    return loop.call_at(when, cb)

def calculate_timeout_when(loop_time: float, timeout: float, timeout_ceiling_threshold: float) -> float:
    """Calculate when to execute a timeout."""
    when = loop_time + timeout
    if timeout > timeout_ceiling_threshold:
        return ceil(when)
    return when

class TimeoutHandle:
    """Timeout handle"""
    __slots__ = ('_timeout', '_loop', '_ceil_threshold', '_callbacks')

    def __init__(self, loop: asyncio.AbstractEventLoop, timeout: Optional[float], ceil_threshold: float = 5):
        self._timeout = timeout
        self._loop = loop
        self._ceil_threshold = ceil_threshold
        self._callbacks: List[Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]] = []

    def register(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._callbacks.append((callback, args, kwargs))

    def close(self) -> None:
        self._callbacks.clear()

    def start(self) -> Optional[asyncio.Handle]:
        timeout = self._timeout
        if timeout is not None and timeout > 0:
            when = self._loop.time() + timeout
            if timeout >= self._ceil_threshold:
                when = ceil(when)
            return self._loop.call_at(when, self.__call__)
        else:
            return None

    def timer(self) -> 'BaseTimerContext':
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

class BaseTimerContext(ContextManager['BaseTimerContext']):
    __slots__ = ()

    def assert_timeout(self) -> None:
        """Raise TimeoutError if timeout has been exceeded."""

class TimerNoop(BaseTimerContext):
    __slots__ = ()

    def __enter__(self) -> 'TimerNoop':
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional