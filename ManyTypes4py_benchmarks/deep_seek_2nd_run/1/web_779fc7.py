from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type, TypeVar, cast, overload
import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import AnyMatches, DefaultHostMatches, HostMatches, ReversibleRouter, Rule, ReversibleRuleRouter, URLSpec, _RuleList
from tornado.util import ObjectDict, unicode_type, _websocket_mask

url = URLSpec

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
    SUPPORTED_METHODS: Tuple[str, ...] = ('GET', 'HEAD', 'POST', 'DELETE', 'PATCH', 'PUT', 'OPTIONS')
    _template_loaders: Dict[str, template.Loader] = {}
    _template_loader_lock: threading.Lock = threading.Lock()
    _remove_control_chars_regex: re.Pattern = re.compile('[\\x00-\\x08\\x0e-\\x1f]')
    _stream_request_body: bool = False
    _transforms: Optional[List[OutputTransform]] = None
    path_args: Optional[List[str]] = None
    path_kwargs: Optional[Dict[str, Any]] = None

    def __init__(self, application: 'Application', request: httputil.HTTPServerRequest, **kwargs: Any) -> None:
        super().__init__()
        self.application = application
        self.request = request
        self._headers_written: bool = False
        self._finished: bool = False
        self._auto_finish: bool = True
        self._prepared_future: Optional[Future[None]] = None
        self.ui = ObjectDict(((n, self._ui_method(m)) for n, m in application.ui_methods.items())
        self.ui['_tt_modules'] = _UIModuleNamespace(self, application.ui_modules)
        self.ui['modules'] = self.ui['_tt_modules']
        self.clear()
        assert self.request.connection is not None
        self.request.connection.set_close_callback(self.on_connection_close)
        self.initialize(**kwargs)

    def initialize(self, **kwargs: Any) -> None:
        pass

    @property
    def settings(self) -> Dict[str, Any]:
        return self.application.settings

    def _unimplemented_method(self, *args: Any, **kwargs: Any) -> None:
        raise HTTPError(405)

    head = _unimplemented_method
    get = _unimplemented_method
    post = _unimplemented_method
    delete = _unimplemented_method
    patch = _unimplemented_method
    put = _unimplemented_method
    options = _unimplemented_method

    def prepare(self) -> Optional[Awaitable[None]]:
        pass

    def on_finish(self) -> None:
        pass

    def on_connection_close(self) -> None:
        pass

    def clear(self) -> None:
        self._headers = httputil.HTTPHeaders({'Server': 'TornadoServer/%s' % tornado.version, 'Content-Type': 'text/html; charset=UTF-8', 'Date': httputil.format_timestamp(time.time())})
        self.set_default_headers()
        self._write_buffer: List[bytes] = []
        self._status_code = 200
        self._reason = httputil.responses[200]

    def set_default_headers(self) -> None:
        pass

    def set_status(self, status_code: int, reason: Optional[str] = None) -> None:
        self._status_code = status_code
        if reason is not None:
            self._reason = escape.native_str(reason)
        else:
            self._reason = httputil.responses.get(status_code, 'Unknown')

    def get_status(self) -> int:
        return self._status_code

    def set_header(self, name: str, value: _HeaderTypes) -> None:
        self._headers[name] = self._convert_header_value(value)

    def add_header(self, name: str, value: _HeaderTypes) -> None:
        self._headers.add(name, self._convert_header_value(value))

    def clear_header(self, name: str) -> None:
        if name in self._headers:
            del self._headers[name]

    _INVALID_HEADER_CHAR_RE = re.compile('[\\x00-\\x1f]')

    def _convert_header_value(self, value: _HeaderTypes) -> str:
        if isinstance(value, str):
            retval = value
        elif isinstance(value, bytes):
            retval = value.decode('latin1')
        elif isinstance(value, numbers.Integral):
            return str(value)
        elif isinstance(value, datetime.datetime):
            return httputil.format_timestamp(value)
        else:
            raise TypeError('Unsupported header value %r' % value)
        if RequestHandler._INVALID_HEADER_CHAR_RE.search(retval):
            raise ValueError('Unsafe header value %r', retval)
        return retval

    @overload
    def get_argument(self, name: str, default: _ArgDefaultMarker, strip: bool = True) -> str: ...

    @overload
    def get_argument(self, name: str, default: str, strip: bool = True) -> str: ...

    def get_argument(self, name: str, default: Union[str, _ArgDefaultMarker] = _ARG_DEFAULT, strip: bool = True) -> str:
        return self._get_argument(name, default, self.request.arguments, strip)

    def get_arguments(self, name: str, strip: bool = True) -> List[str]:
        return self._get_arguments(name, self.request.arguments, strip)

    @overload
    def get_body_argument(self, name: str, default: _ArgDefaultMarker, strip: bool = True) -> str: ...

    @overload
    def get_body_argument(self, name: str, default: str, strip: bool = True) -> str: ...

    def get_body_argument(self, name: str, default: Union[str, _ArgDefaultMarker] = _ARG_DEFAULT, strip: bool = True) -> str:
        return self._get_argument(name, default, self.request.body_arguments, strip)

    def get_body_arguments(self, name: str, strip: bool = True) -> List[str]:
        return self._get_arguments(name, self.request.body_arguments, strip)

    @overload
    def get_query_argument(self, name: str, default: _ArgDefaultMarker, strip: bool = True) -> str: ...

    @overload
    def get_query_argument(self, name: str, default: str, strip: bool = True) -> str: ...

    def get_query_argument(self, name: str, default: Union[str, _ArgDefaultMarker] = _ARG_DEFAULT, strip: bool = True) -> str:
        return self._get_argument(name, default, self.request.query_arguments, strip)

    def get_query_arguments(self, name: str, strip: bool = True) -> List[str]:
        return self._get_arguments(name, self.request.query_arguments, strip)

    def _get_argument(self, name: str, default: Union[str, _ArgDefaultMarker], source: Dict[str, List[bytes]], strip: bool) -> str:
        args = self._get_arguments(name, source, strip=strip)
        if not args:
            if isinstance(default, _ArgDefaultMarker):
                raise MissingArgumentError(name)
            return default
        return args[-1]

    def _get_arguments(self, name: str, source: Dict[str, List[bytes]], strip: bool) -> List[str]:
        values = []
        for v in source.get(name, []):
            s = self.decode_argument(v, name=name)
            if isinstance(s, str):
                s = RequestHandler._remove_control_chars_regex.sub(' ', s)
            if strip:
                s = s.strip()
            values.append(s)
        return values

    def decode_argument(self, value: bytes, name: Optional[str] = None) -> str:
        try:
            return _unicode(value)
        except UnicodeDecodeError:
            raise HTTPError(400, 'Invalid unicode in {}: {!r}'.format(name or 'url', value[:40]))

    @property
    def cookies(self) -> http.cookies.SimpleCookie:
        return self.request.cookies

    @overload
    def get_cookie(self, name: str, default: str) -> str: ...

    @overload
    def get_cookie(self, name: str, default: None = None) -> Optional[str]: ...

    def get_cookie(self, name: str, default: Optional[str] = None) -> Optional[str]:
        if self.request.cookies is not None and name in self.request.cookies:
            return self.request.cookies[name].value
        return default

    def set_cookie(self, name: str, value: str, domain: Optional[str] = None, expires: Optional[Union[float, Tuple, datetime.datetime]] = None, path: str = '/', expires_days: Optional[int] = None, *, max_age: Optional[int] = None, httponly: bool = False, secure: bool = False, samesite: Optional[str] = None, **kwargs: Any) -> None:
        name = escape.native_str(name)
        value = escape.native_str(value)
        if re.search('[\\x00-\\x20]', name + value):
            raise ValueError(f'Invalid cookie {name!r}: {value!r}')
        if not hasattr(self, '_new_cookie'):
            self._new_cookie = http.cookies.SimpleCookie()
        if name in self._new_cookie:
            del self._new_cookie[name]
        self._new_cookie[name] = value
        morsel = self._new_cookie[name]
        if domain:
            morsel['domain'] = domain
        if expires_days is not None and (not expires):
            expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=expires_days)
        if expires:
            morsel['expires'] = httputil.format_timestamp(expires)
        if path:
            morsel['path'] = path
        if max_age:
            morsel['max-age'] = str(max_age)
        if httponly:
            morsel['httponly'] = True
        if secure:
            morsel['secure'] = True
        if samesite:
            morsel['samesite'] = samesite
        if kwargs:
            for k, v in kwargs.items():
                morsel[k] = v
            warnings.warn(f'Deprecated arguments to set_cookie: {set(kwargs.keys())} (should be lowercase)', DeprecationWarning)

    def clear_cookie(self, name: str, **kwargs: Any) -> None:
        for excluded_arg in ['expires', 'max_age']:
            if excluded_arg in kwargs:
                raise TypeError(f"clear_cookie() got an unexpected keyword argument '{excluded_arg}'")
        expires = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=365)
        self.set_cookie(name, value='', expires=expires, **kwargs)

    def clear_all_cookies(self, **kwargs: Any) -> None:
        for name in self.request.cookies:
            self.clear_cookie(name, **kwargs)

    def set_signed_cookie(self, name: str, value: Union[str, bytes], expires_days: int = 30, version: Optional[int] = None, **kwargs: Any) -> None:
        self.set_cookie(name, self.create_signed_value(name, value, version=version), expires_days=expires_days, **kwargs)

    set_secure_cookie = set_signed_cookie

    def create_signed_value(self, name: str, value: Union[str, bytes], version: Optional[int] = None) -> bytes:
        self.require_setting('cookie_secret', 'secure cookies')
        secret = self.application.settings['cookie_secret']
        key_version = None
        if isinstance(secret, dict):
            if self.application.settings.get('key_version') is None:
                raise Exception('key_version setting must be used for secret_key dicts')
            key_version = self.application.settings['key_version']
        return create_signed_value(secret, name, value, version=version, key_version=key_version)

    def get_signed_cookie(self, name: str, value: Optional[str] = None, max_age_days: int = 31, min_version: Optional[int] = None) -> Optional[bytes]:
        self.require_setting('cookie_secret', 'secure cookies')
        if value is None:
            value = self.get_cookie(name)
        return decode_signed_value(self.application.settings['cookie_secret'], name, value, max_age_days=max_age_days, min_version=min_version)

    get_secure_cookie = get_signed_cookie

    def get_signed_cookie_key_version(self, name: str, value: Optional[str] = None) -> Optional[int]:
        self.require_setting('cookie_secret', 'secure cookies')
        if value is None:
            value = self.get_cookie(name)
        if value is None:
            return None
        return get_signature_key_version(value)

    get_secure_cookie_key_version = get_signed_cookie_key_version

    def redirect(self, url: str, permanent: bool = False, status: Optional[int] = None) -> None:
        if self._headers_written:
            raise Exception('Cannot redirect after headers have been written')
        if status is None:
            status = 301 if permanent else 302
        else:
            assert isinstance(status, int) and 300 <= status <= 399
        self.set_status(status)
        self.set_header('Location', utf8(url))
        self.finish()

    def write(self, chunk: Union[str, bytes, Dict[str, Any]]) -> None:
        if self._finished:
            raise RuntimeError('Cannot write() after finish()')
        if not isinstance(chunk, (bytes, str, dict)):
            message = 'write() only accepts bytes, unicode, and dict objects'
            if isinstance(chunk, list):
                message += '. Lists not accepted for security reasons; see ' + 'http://www.tornadoweb.org/en/stable/web.html#tornado.web.RequestHandler.write'
            raise TypeError(message)
        if isinstance(chunk, dict):
            chunk = escape.json_encode(chunk)
            self.set_header('Content-Type', 'application/json; charset=UTF-8')
        chunk = utf8(chunk)
        self._write_buffer.append(chunk)

    def render(self, template_name: str, **kwargs: Any) -> Future[None]:
        if self._finished:
            raise RuntimeError('Cannot render() after finish()')
        html = self.render_string(template_name, **kwargs)
        js_embed = []
        js_files = []
        css_embed = []
        css_files = []
        html_heads = []
        html_bodies = []
        for module in getattr(self, '_active_modules', {}).values():
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
            js = self.render_linked_js(js_files)
            sloc = html.rindex(b'</body>')
            html = html[:sloc] + utf8(js) + b'\n' + html[sloc:]
        if js_embed:
            js_bytes = self.render_embed_js(js_embed)
            sloc = html.rindex(b'</body>')
            html = html[:sloc] + js_bytes + b'\n' + html[sloc:]
        if css_files:
            css = self.render_linked_css(css_files)
            hloc = html.index(b'</head>')
            html = html[:hloc] + utf8(css) + b'\n' + html[hloc:]
        if css_embed:
            css_bytes = self.render_embed_css(css_embed)
            hloc = html.index(b'</head>')
            html = html[:hloc] + css_bytes + b'\n' + html[hloc:]
        if html_heads:
            hloc = html.index(b'</head>')
            html = html[:hloc] + b''.join(html_heads) + b'\n' + html[hloc:]
        if html_bodies:
            hloc = html.index(b'</body>')
            html = html[:hloc] + b''.join(html_bodies) + b'\n' + html[hloc:]
        return self.f