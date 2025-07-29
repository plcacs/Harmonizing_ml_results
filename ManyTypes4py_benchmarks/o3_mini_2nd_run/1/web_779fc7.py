#!/usr/bin/env python3
# type: ignore
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
from tornado import escape, gen, httputil, iostream, locale, template
from tornado.log import access_log, app_log, gen_log
from tornado.routing import AnyMatches, DefaultHostMatches, HostMatches, ReversibleRouter, Rule, ReversibleRuleRouter, URLSpec, _RuleList
from tornado.util import ObjectDict, unicode_type, _websocket_mask
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast
from types import TracebackType

T = TypeVar("T", bound="RequestHandler")
_ArgDefaultMarker = object
_ARG_DEFAULT = _ArgDefaultMarker

class RequestHandler:
    SUPPORTED_METHODS: Tuple[str, ...] = ('GET', 'HEAD', 'POST', 'DELETE', 'PATCH', 'PUT', 'OPTIONS')
    _template_loaders: Dict[str, Any] = {}
    _template_loader_lock = threading.Lock()
    _remove_control_chars_regex = re.compile('[\\x00-\\x08\\x0e-\\x1f]')
    _stream_request_body: bool = False
    _transforms: Optional[List[Any]] = None
    path_args: Optional[List[Any]] = None
    path_kwargs: Optional[Dict[str, Any]] = None

    def __init__(self, application: Any, request: Any, **kwargs: Any) -> None:
        super().__init__()
        self.application = application
        self.request = request
        self._headers_written: bool = False
        self._finished: bool = False
        self._auto_finish: bool = True
        self._prepared_future: Optional[Future] = None
        self.ui = ObjectDict(((n, self._ui_method(m)) for n, m in application.ui_methods.items()))
        self.ui['_tt_modules'] = _UIModuleNamespace(self, application.ui_modules)
        self.ui['modules'] = self.ui['_tt_modules']
        self.clear()
        assert self.request.connection is not None
        self.request.connection.set_close_callback(self.on_connection_close)
        self.initialize(**kwargs)

    def _initialize(self) -> None:
        pass
    initialize = _initialize

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

    def prepare(self) -> None:
        pass

    def on_finish(self) -> None:
        pass

    def on_connection_close(self) -> None:
        if _has_stream_request_body(self.__class__):
            if not self.request._body_future.done():
                self.request._body_future.set_exception(iostream.StreamClosedError())
                self.request._body_future.exception()

    def clear(self) -> None:
        self._headers = httputil.HTTPHeaders({
            'Server': 'TornadoServer/%s' % tornado.version,
            'Content-Type': 'text/html; charset=UTF-8',
            'Date': httputil.format_timestamp(time.time())
        })
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
            self._reason = httputil.responses.get(status_code, 'Unknown')

    def get_status(self) -> int:
        return self._status_code

    def set_header(self, name: str, value: Any) -> None:
        self._headers[name] = self._convert_header_value(value)

    def add_header(self, name: str, value: Any) -> None:
        self._headers.add(name, self._convert_header_value(value))

    def clear_header(self, name: str) -> None:
        if name in self._headers:
            del self._headers[name]

    _INVALID_HEADER_CHAR_RE = re.compile('[\\x00-\\x1f]')

    def _convert_header_value(self, value: Any) -> str:
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

    def get_argument(self, name: str, default: Any = _ARG_DEFAULT, strip: bool = True) -> Any:
        return self._get_argument(name, default, self.request.arguments, strip)

    def get_arguments(self, name: str, strip: bool = True) -> List[Any]:
        assert isinstance(strip, bool)
        return self._get_arguments(name, self.request.arguments, strip)

    def get_body_argument(self, name: str, default: Any = _ARG_DEFAULT, strip: bool = True) -> Any:
        return self._get_argument(name, default, self.request.body_arguments, strip)

    def get_body_arguments(self, name: str, strip: bool = True) -> List[Any]:
        return self._get_arguments(name, self.request.body_arguments, strip)

    def get_query_argument(self, name: str, default: Any = _ARG_DEFAULT, strip: bool = True) -> Any:
        return self._get_argument(name, default, self.request.query_arguments, strip)

    def get_query_arguments(self, name: str, strip: bool = True) -> List[Any]:
        return self._get_arguments(name, self.request.query_arguments, strip)

    def _get_argument(self, name: str, default: Any, source: Dict[str, Any], strip: bool = True) -> Any:
        args: List[Any] = self._get_arguments(name, source, strip=strip)
        if not args:
            if default is _ARG_DEFAULT:
                raise MissingArgumentError(name)
            return default
        return args[-1]

    def _get_arguments(self, name: str, source: Dict[str, Any], strip: bool = True) -> List[Any]:
        values: List[Any] = []
        for v in source.get(name, []):
            s: Any = self.decode_argument(v, name=name)
            if isinstance(s, unicode_type):
                s = RequestHandler._remove_control_chars_regex.sub(' ', s)
            if strip:
                s = s.strip()
            values.append(s)
        return values

    def decode_argument(self, value: bytes, name: Optional[str] = None) -> Any:
        try:
            return _unicode(value)
        except UnicodeDecodeError:
            raise HTTPError(400, 'Invalid unicode in {}: {!r}'.format(name or 'url', value[:40]))

    @property
    def cookies(self) -> Any:
        return self.request.cookies

    def get_cookie(self, name: str, default: Optional[str] = None) -> Optional[str]:
        if self.request.cookies is not None and name in self.request.cookies:
            return self.request.cookies[name].value
        return default

    def set_cookie(self, name: str, value: str, domain: Optional[str] = None, expires: Optional[Any] = None, path: str = '/', expires_days: Optional[int] = None, *, max_age: Optional[int] = None, httponly: bool = False, secure: bool = False, samesite: Optional[str] = None, **kwargs: Any) -> None:
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

    def set_signed_cookie(self, name: str, value: str, expires_days: int = 30, version: Optional[int] = None, **kwargs: Any) -> None:
        self.set_cookie(name, self.create_signed_value(name, value, version=version), expires_days=expires_days, **kwargs)
    set_secure_cookie = set_signed_cookie

    def create_signed_value(self, name: str, value: str, version: Optional[int] = None) -> bytes:
        self.require_setting('cookie_secret', 'secure cookies')
        secret: Any = self.application.settings['cookie_secret']
        key_version: Optional[int] = None
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
        self.set_header('Location', escape.utf8(url))
        self.finish()

    def write(self, chunk: Union[bytes, str, dict]) -> None:
        if self._finished:
            raise RuntimeError('Cannot write() after finish()')
        if not isinstance(chunk, (bytes, unicode_type, dict)):
            message = 'write() only accepts bytes, unicode, and dict objects'
            if isinstance(chunk, list):
                message += '. Lists not accepted for security reasons; see http://www.tornadoweb.org/en/stable/web.html#tornado.web.RequestHandler.write'
            raise TypeError(message)
        if isinstance(chunk, dict):
            chunk = escape.json_encode(chunk)
            self.set_header('Content-Type', 'application/json; charset=UTF-8')
        chunk = escape.utf8(chunk)
        self._write_buffer.append(chunk)

    def render(self, template_name: str, **kwargs: Any) -> Future:
        if self._finished:
            raise RuntimeError('Cannot render() after finish()')
        html: bytes = self.render_string(template_name, **kwargs)
        js_embed: List[bytes] = []
        js_files: List[str] = []
        css_embed: List[bytes] = []
        css_files: List[str] = []
        html_heads: List[bytes] = []
        html_bodies: List[bytes] = []
        for module in getattr(self, '_active_modules', {}).values():
            embed_part = module.embedded_javascript()
            if embed_part:
                js_embed.append(escape.utf8(embed_part))
            file_part = module.javascript_files()
            if file_part:
                if isinstance(file_part, (unicode_type, bytes)):
                    js_files.append(_unicode(file_part))
                else:
                    js_files.extend(file_part)
            embed_part = module.embedded_css()
            if embed_part:
                css_embed.append(escape.utf8(embed_part))
            file_part = module.css_files()
            if file_part:
                if isinstance(file_part, (unicode_type, bytes)):
                    css_files.append(_unicode(file_part))
                else:
                    css_files.extend(file_part)
            head_part = module.html_head()
            if head_part:
                html_heads.append(escape.utf8(head_part))
            body_part = module.html_body()
            if body_part:
                html_bodies.append(escape.utf8(body_part))
        if js_files:
            js = self.render_linked_js(js_files)
            sloc = html.rindex(b'</body>')
            html = html[:sloc] + escape.utf8(js) + b'\n' + html[sloc:]
        if js_embed:
            js_bytes = self.render_embed_js(js_embed)
            sloc = html.rindex(b'</body>')
            html = html[:sloc] + js_bytes + b'\n' + html[sloc:]
        if css_files:
            css = self.render_linked_css(css_files)
            hloc = html.index(b'</head>')
            html = html[:hloc] + escape.utf8(css) + b'\n' + html[hloc:]
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
        return self.finish(html)

    def render_linked_js(self, js_files: List[str]) -> str:
        paths: List[str] = []
        unique_paths: set = set()
        for path in js_files:
            if not is_absolute(path):
                path = self.static_url(path)
            if path not in unique_paths:
                paths.append(path)
                unique_paths.add(path)
        return ''.join(('<script src="' + escape.xhtml_escape(p) + '" type="text/javascript"></script>' for p in paths))

    def render_embed_js(self, js_embed: List[bytes]) -> bytes:
        return b'<script type="text/javascript">\n//<![CDATA[\n' + b'\n'.join(js_embed) + b'\n//]]>\n</script>'

    def render_linked_css(self, css_files: List[str]) -> str:
        paths: List[str] = []
        unique_paths: set = set()
        for path in css_files:
            if not is_absolute(path):
                path = self.static_url(path)
            if path not in unique_paths:
                paths.append(path)
                unique_paths.add(path)
        return ''.join(('<link href="' + escape.xhtml_escape(p) + '" type="text/css" rel="stylesheet"/>' for p in paths))

    def render_embed_css(self, css_embed: List[bytes]) -> bytes:
        return b'<style type="text/css">\n' + b'\n'.join(css_embed) + b'\n</style>'

    def render_string(self, template_name: str, **kwargs: Any) -> bytes:
        template_path: Optional[str] = self.get_template_path()
        if not template_path:
            frame = sys._getframe(0)
            web_file = frame.f_code.co_filename
            while frame.f_code.co_filename == web_file and frame.f_back is not None:
                frame = frame.f_back
            assert frame.f_code.co_filename is not None
            template_path = os.path.dirname(frame.f_code.co_filename)
        with RequestHandler._template_loader_lock:
            if template_path not in RequestHandler._template_loaders:
                loader = self.create_template_loader(template_path)
                RequestHandler._template_loaders[template_path] = loader
            else:
                loader = RequestHandler._template_loaders[template_path]
        t = loader.load(template_name)
        namespace: Dict[str, Any] = self.get_template_namespace()
        namespace.update(kwargs)
        return t.generate(**namespace)

    def get_template_namespace(self) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {
            'handler': self,
            'request': self.request,
            'current_user': self.current_user,
            'locale': self.locale,
            '_': self.locale.translate,
            'pgettext': self.locale.pgettext,
            'static_url': self.static_url,
            'xsrf_form_html': self.xsrf_form_html,
            'reverse_url': self.reverse_url
        }
        namespace.update(self.ui)
        return namespace

    def create_template_loader(self, template_path: str) -> Any:
        settings: Dict[str, Any] = self.application.settings
        if 'template_loader' in settings:
            return settings['template_loader']
        kwargs: Dict[str, Any] = {}
        if 'autoescape' in settings:
            kwargs['autoescape'] = settings['autoescape']
        if 'template_whitespace' in settings:
            kwargs['whitespace'] = settings['template_whitespace']
        return template.Loader(template_path, **kwargs)

    def flush(self, include_footers: bool = False) -> Future:
        assert self.request.connection is not None
        chunk: bytes = b''.join(self._write_buffer)
        self._write_buffer = []
        if not self._headers_written:
            self._headers_written = True
            for transform in self._transforms or []:
                assert chunk is not None
                self._status_code, self._headers, chunk = transform.transform_first_chunk(self._status_code, self._headers, chunk, include_footers)
            if self.request.method == 'HEAD':
                chunk = b''
            if hasattr(self, '_new_cookie'):
                for cookie in self._new_cookie.values():
                    self.add_header('Set-Cookie', cookie.OutputString(None))
            start_line = httputil.ResponseStartLine('', self._status_code, self._reason)
            return self.request.connection.write_headers(start_line, self._headers, chunk)
        else:
            for transform in self._transforms or []:
                chunk = transform.transform_chunk(chunk, include_footers)
            if self.request.method != 'HEAD':
                return self.request.connection.write(chunk)
            else:
                fut: Future = Future()
                fut.set_result(None)
                return fut

    def finish(self, chunk: Optional[Union[bytes, str]] = None) -> Future:
        if self._finished:
            raise RuntimeError('finish() called twice')
        if chunk is not None:
            self.write(chunk)
        if not self._headers_written:
            if self._status_code == 200 and self.request.method in ('GET', 'HEAD') and ('Etag' not in self._headers):
                self.set_etag_header()
                if self.check_etag_header():
                    self._write_buffer = []
                    self.set_status(304)
            if self._status_code in (204, 304) or 100 <= self._status_code < 200:
                assert not self._write_buffer, 'Cannot send body with %s' % self._status_code
                self._clear_representation_headers()
            elif 'Content-Length' not in self._headers:
                content_length = sum((len(part) for part in self._write_buffer))
                self.set_header('Content-Length', content_length)
        assert self.request.connection is not None
        self.request.connection.set_close_callback(None)
        future: Future = self.flush(include_footers=True)
        self.request.connection.finish()
        self._log()
        self._finished = True
        self.on_finish()
        self._break_cycles()
        return future

    def detach(self) -> Any:
        self._finished = True
        return self.request.connection.detach()

    def _break_cycles(self) -> None:
        self.ui = None

    def send_error(self, status_code: int = 500, **kwargs: Any) -> None:
        if self._headers_written:
            gen_log.error('Cannot send error response after headers written')
            if not self._finished:
                try:
                    self.finish()
                except Exception:
                    gen_log.error('Failed to flush partial response', exc_info=True)
            return
        self.clear()
        reason: Optional[str] = kwargs.get('reason')
        if 'exc_info' in kwargs:
            exception: Any = kwargs['exc_info'][1]
            if isinstance(exception, HTTPError) and exception.reason:
                reason = exception.reason
        self.set_status(status_code, reason=reason)
        try:
            self.write_error(status_code, **kwargs)
        except Exception:
            app_log.error('Uncaught exception in write_error', exc_info=True)
        if not self._finished:
            self.finish()

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        if self.settings.get('serve_traceback') and 'exc_info' in kwargs:
            self.set_header('Content-Type', 'text/plain')
            for line in traceback.format_exception(*kwargs['exc_info']):
                self.write(line)
            self.finish()
        else:
            self.finish('<html><title>%(code)d: %(message)s</title><body>%(code)d: %(message)s</body></html>' % {'code': status_code, 'message': self._reason})

    @property
    def locale(self) -> Any:
        if not hasattr(self, '_locale'):
            loc = self.get_user_locale()
            if loc is not None:
                self._locale = loc
            else:
                self._locale = self.get_browser_locale()
                assert self._locale
        return self._locale

    @locale.setter
    def locale(self, value: Any) -> None:
        self._locale = value

    def get_user_locale(self) -> Optional[Any]:
        return None

    def get_browser_locale(self, default: str = 'en_US') -> Any:
        if 'Accept-Language' in self.request.headers:
            languages = self.request.headers['Accept-Language'].split(',')
            locales: List[Tuple[str, float]] = []
            for language in languages:
                parts = language.strip().split(';')
                if len(parts) > 1 and parts[1].strip().startswith('q='):
                    try:
                        score = float(parts[1].strip()[2:])
                        if score < 0:
                            raise ValueError()
                    except (ValueError, TypeError):
                        score = 0.0
                else:
                    score = 1.0
                if score > 0:
                    locales.append((parts[0], score))
            if locales:
                locales.sort(key=lambda pair: pair[1], reverse=True)
                codes = [loc[0] for loc in locales]
                return locale.get(*codes)
        return locale.get(default)

    @property
    def current_user(self) -> Any:
        if not hasattr(self, '_current_user'):
            self._current_user = self.get_current_user()
        return self._current_user

    @current_user.setter
    def current_user(self, value: Any) -> None:
        self._current_user = value

    def get_current_user(self) -> Any:
        return None

    def get_login_url(self) -> str:
        self.require_setting('login_url', '@tornado.web.authenticated')
        return self.application.settings['login_url']

    def get_template_path(self) -> Optional[str]:
        return self.application.settings.get('template_path')

    @property
    def xsrf_token(self) -> bytes:
        if not hasattr(self, '_xsrf_token'):
            version, token, timestamp = self._get_raw_xsrf_token()
            output_version: int = self.settings.get('xsrf_cookie_version', 2)
            cookie_kwargs: Dict[str, Any] = self.settings.get('xsrf_cookie_kwargs', {})
            if output_version == 1:
                self._xsrf_token = binascii.b2a_hex(token)
            elif output_version == 2:
                mask = os.urandom(4)
                self._xsrf_token = b'|'.join([b'2', binascii.b2a_hex(mask), binascii.b2a_hex(_websocket_mask(mask, token)), escape.utf8(str(int(timestamp)))])
            else:
                raise ValueError('unknown xsrf cookie version %d', output_version)
            if version is None:
                if self.current_user and 'expires_days' not in cookie_kwargs:
                    cookie_kwargs['expires_days'] = 30
                cookie_name: str = self.settings.get('xsrf_cookie_name', '_xsrf')
                self.set_cookie(cookie_name, self._xsrf_token, **cookie_kwargs)
        return self._xsrf_token

    def _get_raw_xsrf_token(self) -> Tuple[Optional[int], bytes, float]:
        if not hasattr(self, '_raw_xsrf_token'):
            cookie_name: str = self.settings.get('xsrf_cookie_name', '_xsrf')
            cookie: Optional[str] = self.get_cookie(cookie_name)
            if cookie:
                version, token, timestamp = self._decode_xsrf_token(cookie)
            else:
                version, token, timestamp = (None, None, None)
            if token is None:
                version = None
                token = os.urandom(16)
                timestamp = time.time()
            assert token is not None
            assert timestamp is not None
            self._raw_xsrf_token = (version, token, timestamp)
        return self._raw_xsrf_token

    def _decode_xsrf_token(self, cookie: str) -> Tuple[int, bytes, int]:
        try:
            m = _signed_value_version_re.match(escape.utf8(cookie))
            if m:
                version = int(m.group(1))
                if version == 2:
                    _, mask_str, masked_token, timestamp_str = cookie.split('|')
                    mask = binascii.a2b_hex(escape.utf8(mask_str))
                    token = _websocket_mask(mask, binascii.a2b_hex(escape.utf8(masked_token)))
                    timestamp = int(timestamp_str)
                    return (version, token, timestamp)
                else:
                    raise Exception('Unknown xsrf cookie version')
            else:
                version = 1
                try:
                    token = binascii.a2b_hex(escape.utf8(cookie))
                except (binascii.Error, TypeError):
                    token = escape.utf8(cookie)
                timestamp = int(time.time())
                return (version, token, timestamp)
        except Exception:
            gen_log.debug('Uncaught exception in _decode_xsrf_token', exc_info=True)
            return (0, b'', 0)

    def check_xsrf_cookie(self) -> None:
        input_token: Optional[str] = self.get_argument('_xsrf', None) or self.request.headers.get('X-Xsrftoken') or self.request.headers.get('X-Csrftoken')
        if not input_token:
            raise HTTPError(403, "'_xsrf' argument missing from POST")
        _, token, _ = self._decode_xsrf_token(input_token)
        _, expected_token, _ = self._get_raw_xsrf_token()
        if not token:
            raise HTTPError(403, "'_xsrf' argument has invalid format")
        if not hmac.compare_digest(escape.utf8(token), escape.utf8(expected_token)):
            raise HTTPError(403, 'XSRF cookie does not match POST argument')

    def xsrf_form_html(self) -> str:
        return '<input type="hidden" name="_xsrf" value="' + escape.xhtml_escape(self.xsrf_token) + '"/>'

    def static_url(self, path: str, include_host: Optional[bool] = None, **kwargs: Any) -> str:
        self.require_setting('static_path', 'static_url')
        get_url: Callable[..., str] = self.settings.get('static_handler_class', StaticFileHandler).make_static_url
        if include_host is None:
            include_host = getattr(self, 'include_host', False)
        if include_host:
            base = self.request.protocol + '://' + self.request.host
        else:
            base = ''
        return base + get_url(self.settings, path, **kwargs)

    def require_setting(self, name: str, feature: str = 'this feature') -> None:
        if not self.application.settings.get(name):
            raise Exception("You must define the '%s' setting in your application to use %s" % (name, feature))

    def reverse_url(self, name: str, *args: Any) -> str:
        return self.application.reverse_url(name, *args)

    def compute_etag(self) -> Optional[str]:
        hasher = hashlib.sha1()
        for part in self._write_buffer:
            hasher.update(part)
        return '"%s"' % hasher.hexdigest()

    def set_etag_header(self) -> None:
        etag: Optional[str] = self.compute_etag()
        if etag is not None:
            self.set_header('Etag', etag)

    def check_etag_header(self) -> bool:
        computed_etag: bytes = escape.utf8(self._headers.get('Etag', ''))
        etags = re.findall(b'\\*|(?:W/)?"[^"]*"', escape.utf8(self.request.headers.get('If-None-Match', '')))
        if not computed_etag or not etags:
            return False
        match = False
        if etags[0] == b'*':
            match = True
        else:
            def val(x: bytes) -> bytes:
                return x[2:] if x.startswith(b'W/') else x
            for etag in etags:
                if val(etag) == val(computed_etag):
                    match = True
                    break
        return match

    async def _execute(self, transforms: List[Any], *args: Any, **kwargs: Any) -> None:
        self._transforms = transforms
        try:
            if self.request.method not in self.SUPPORTED_METHODS:
                raise HTTPError(405)
            self.path_args = [self.decode_argument(arg) for arg in args]
            self.path_kwargs = {k: self.decode_argument(v, name=k) for k, v in kwargs.items()}
            if self.request.method not in ('GET', 'HEAD', 'OPTIONS') and self.application.settings.get('xsrf_cookies'):
                self.check_xsrf_cookie()
            result = self.prepare()
            if result is not None:
                result = await result
            if self._prepared_future is not None:
                future_set_result_unless_cancelled(self._prepared_future, None)
            if self._finished:
                return
            if _has_stream_request_body(self.__class__):
                try:
                    await self.request._body_future
                except iostream.StreamClosedError:
                    return
            method = getattr(self, self.request.method.lower())
            result = method(*self.path_args, **self.path_kwargs)
            if result is not None:
                result = await result
            if self._auto_finish and (not self._finished):
                self.finish()
        except Exception as e:
            try:
                self._handle_request_exception(e)
            except Exception:
                app_log.error('Exception in exception handler', exc_info=True)
            finally:
                result = None
            if self._prepared_future is not None and (not self._prepared_future.done()):
                self._prepared_future.set_result(None)

    def data_received(self, chunk: bytes) -> None:
        raise NotImplementedError()

    def _log(self) -> None:
        self.application.log_request(self)

    def _request_summary(self) -> str:
        return '{} {} ({})'.format(self.request.method, self.request.uri, self.request.remote_ip)

    def _handle_request_exception(self, e: Exception) -> None:
        if isinstance(e, Finish):
            if not self._finished:
                self.finish(*e.args)
            return
        try:
            self.log_exception(*sys.exc_info())
        except Exception:
            app_log.error('Error in exception logger', exc_info=True)
        if self._finished:
            return
        if isinstance(e, HTTPError):
            self.send_error(e.status_code, exc_info=sys.exc_info())
        else:
            self.send_error(500, exc_info=sys.exc_info())

    def log_exception(self, typ: Any, value: Any, tb: Optional[TracebackType]) -> None:
        if isinstance(value, HTTPError):
            if value.log_message:
                format_str = '%d %s: ' + value.log_message
                args_list = [value.status_code, self._request_summary()] + list(value.args)
                gen_log.warning(format_str, *args_list)
        else:
            app_log.error('Uncaught exception %s\n%r', self._request_summary(), self.request, exc_info=(typ, value, tb))

    def _ui_module(self, name: str, module: Any) -> Callable[..., str]:
        def render(*args: Any, **kwargs: Any) -> str:
            if not hasattr(self, '_active_modules'):
                self._active_modules = {}
            if name not in self._active_modules:
                self._active_modules[name] = module(self)
            rendered = self._active_modules[name].render(*args, **kwargs)
            return _unicode(rendered)
        return render

    def _ui_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        return lambda *args, **kwargs: method(self, *args, **kwargs)

    def _clear_representation_headers(self) -> None:
        headers = ['Content-Encoding', 'Content-Language', 'Content-Type']
        for h in headers:
            self.clear_header(h)

def stream_request_body(cls: Type[T]) -> Type[T]:
    if not issubclass(cls, RequestHandler):
        raise TypeError('expected subclass of RequestHandler, got %r' % cls)
    cls._stream_request_body = True
    return cls

def _has_stream_request_body(cls: Type[RequestHandler]) -> bool:
    if not issubclass(cls, RequestHandler):
        raise TypeError('expected subclass of RequestHandler, got %r' % cls)
    return cls._stream_request_body

def removeslash(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self: RequestHandler, *args: Any, **kwargs: Any) -> Any:
        if self.request.path.endswith('/'):
            if self.request.method in ('GET', 'HEAD'):
                uri: str = self.request.path.rstrip('/')
                if uri:
                    if self.request.query:
                        uri += '?' + self.request.query
                    self.redirect(uri, permanent=True)
                    return None
            else:
                raise HTTPError(404)
        return method(self, *args, **kwargs)
    return wrapper

def addslash(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self: RequestHandler, *args: Any, **kwargs: Any) -> Any:
        if not self.request.path.endswith('/'):
            if self.request.method in ('GET', 'HEAD'):
                uri: str = self.request.path + '/'
                if self.request.query:
                    uri += '?' + self.request.query
                self.redirect(uri, permanent=True)
                return None
            raise HTTPError(404)
        return method(self, *args, **kwargs)
    return wrapper

class _ApplicationRouter(ReversibleRuleRouter):
    def __init__(self, application: "Application", rules: Optional[Any] = None) -> None:
        assert isinstance(application, Application)
        self.application = application
        super().__init__(rules)

    def process_rule(self, rule: Any) -> Any:
        rule = super().process_rule(rule)
        if isinstance(rule.target, (list, tuple)):
            rule.target = _ApplicationRouter(self.application, rule.target)
        return rule

    def get_target_delegate(self, target: Any, request: Any, **target_params: Any) -> Any:
        if isclass(target) and issubclass(target, RequestHandler):
            return self.application.get_handler_delegate(request, target, **target_params)
        return super().get_target_delegate(target, request, **target_params)

class Application(ReversibleRouter):
    def __init__(self, handlers: Optional[Any] = None, default_host: Optional[str] = None, transforms: Optional[List[Any]] = None, **settings: Any) -> None:
        if transforms is None:
            self.transforms = []
            if settings.get('compress_response') or settings.get('gzip'):
                self.transforms.append(GZipContentEncoding)
        else:
            self.transforms = transforms
        self.default_host = default_host
        self.settings = settings
        self.ui_modules: Dict[str, Any] = {'linkify': _linkify, 'xsrf_form_html': _xsrf_form_html, 'Template': TemplateModule}
        self.ui_methods: Dict[str, Callable[..., Any]] = {}
        self._load_ui_modules(settings.get('ui_modules', {}))
        self._load_ui_methods(settings.get('ui_methods', {}))
        if self.settings.get('static_path'):
            path: str = self.settings['static_path']
            handlers = list(handlers or [])
            static_url_prefix: str = settings.get('static_url_prefix', '/static/')
            static_handler_class: Any = settings.get('static_handler_class', StaticFileHandler)
            static_handler_args: Dict[str, Any] = settings.get('static_handler_args', {})
            static_handler_args['path'] = path
            for pattern in [re.escape(static_url_prefix) + '(.*)', '/(favicon\\.ico)', '/(robots\\.txt)']:
                handlers.insert(0, (pattern, static_handler_class, static_handler_args))
        if self.settings.get('debug'):
            self.settings.setdefault('autoreload', True)
            self.settings.setdefault('compiled_template_cache', False)
            self.settings.setdefault('static_hash_cache', False)
            self.settings.setdefault('serve_traceback', True)
        self.wildcard_router = _ApplicationRouter(self, handlers)
        self.default_router = _ApplicationRouter(self, [Rule(AnyMatches(), self.wildcard_router)])
        if self.settings.get('autoreload'):
            from tornado import autoreload
            autoreload.start()

    def listen(self, port: int, address: Optional[str] = None, *, family: int = socket.AF_UNSPEC, backlog: int = tornado.netutil._DEFAULT_BACKLOG, flags: Optional[int] = None, reuse_port: bool = False, **kwargs: Any) -> Any:
        server = HTTPServer(self, **kwargs)
        server.listen(port, address=address, family=family, backlog=backlog, flags=flags, reuse_port=reuse_port)
        return server

    def add_handlers(self, host_pattern: str, host_handlers: Any) -> None:
        host_matcher = HostMatches(host_pattern)
        rule = Rule(host_matcher, _ApplicationRouter(self, host_handlers))
        self.default_router.rules.insert(-1, rule)
        if self.default_host is not None:
            self.wildcard_router.add_rules([(DefaultHostMatches(self, host_matcher.host_pattern), host_handlers)])

    def add_transform(self, transform_class: Any) -> None:
        self.transforms.append(transform_class)

    def _load_ui_methods(self, methods: Any) -> None:
        if isinstance(methods, types.ModuleType):
            self._load_ui_methods({n: getattr(methods, n) for n in dir(methods)})
        elif isinstance(methods, list):
            for m in methods:
                self._load_ui_methods(m)
        else:
            for name, fn in methods.items():
                if not name.startswith('_') and hasattr(fn, '__call__') and (name[0].lower() == name[0]):
                    self.ui_methods[name] = fn

    def _load_ui_modules(self, modules: Any) -> None:
        if isinstance(modules, types.ModuleType):
            self._load_ui_modules({n: getattr(modules, n) for n in dir(modules)})
        elif isinstance(modules, list):
            for m in modules:
                self._load_ui_modules(m)
        else:
            assert isinstance(modules, dict)
            for name, cls in modules.items():
                try:
                    if issubclass(cls, UIModule):
                        self.ui_modules[name] = cls
                except TypeError:
                    pass

    def __call__(self, request: Any) -> Any:
        dispatcher = self.find_handler(request)
        return dispatcher.execute()

    def find_handler(self, request: Any, **kwargs: Any) -> Any:
        route = self.default_router.find_handler(request)
        if route is not None:
            return cast('_HandlerDelegate', route)
        if self.settings.get('default_handler_class'):
            return self.get_handler_delegate(request, self.settings['default_handler_class'], self.settings.get('default_handler_args', {}))
        return self.get_handler_delegate(request, ErrorHandler, {'status_code': 404})

    def get_handler_delegate(self, request: Any, target_class: Type[RequestHandler], target_kwargs: Optional[Dict[str, Any]] = None, path_args: Optional[List[Any]] = None, path_kwargs: Optional[Dict[str, Any]] = None) -> '_HandlerDelegate':
        return _HandlerDelegate(self, request, target_class, target_kwargs, path_args, path_kwargs)

    def reverse_url(self, name: str, *args: Any) -> str:
        reversed_url = self.default_router.reverse_url(name, *args)
        if reversed_url is not None:
            return reversed_url
        raise KeyError('%s not found in named urls' % name)

    def log_request(self, handler: RequestHandler) -> None:
        if 'log_function' in self.settings:
            self.settings['log_function'](handler)
            return
        if handler.get_status() < 400:
            log_method = access_log.info
        elif handler.get_status() < 500:
            log_method = access_log.warning
        else:
            log_method = access_log.error
        request_time = 1000.0 * handler.request.request_time()
        log_method('%d %s %.2fms', handler.get_status(), handler._request_summary(), request_time)

class _HandlerDelegate(httputil.HTTPMessageDelegate):
    def __init__(self, application: Application, request: Any, handler_class: Type[RequestHandler], handler_kwargs: Optional[Dict[str, Any]], path_args: Optional[List[Any]], path_kwargs: Optional[Dict[str, Any]]) -> None:
        self.application = application
        self.connection = request.connection
        self.request = request
        self.handler_class = handler_class
        self.handler_kwargs = handler_kwargs or {}
        self.path_args = path_args or []
        self.path_kwargs = path_kwargs or {}
        self.chunks: List[bytes] = []
        self.stream_request_body: bool = _has_stream_request_body(self.handler_class)

    def headers_received(self, start_line: Any, headers: Any) -> Any:
        if self.stream_request_body:
            self.request._body_future = Future()
            return self.execute()
        return None

    def data_received(self, data: bytes) -> Any:
        if self.stream_request_body:
            return self.handler.data_received(data)
        else:
            self.chunks.append(data)
            return None

    def finish(self) -> None:
        if self.stream_request_body:
            future_set_result_unless_cancelled(self.request._body_future, None)
        else:
            self.request.body = b''.join(self.chunks)
            self.request._parse_body()
            self.execute()

    def on_connection_close(self) -> None:
        if self.stream_request_body:
            self.handler.on_connection_close()
        else:
            self.chunks = None

    def execute(self) -> Future:
        if not self.application.settings.get('compiled_template_cache', True):
            with RequestHandler._template_loader_lock:
                for loader in RequestHandler._template_loaders.values():
                    loader.reset()
        if not self.application.settings.get('static_hash_cache', True):
            static_handler_class = self.application.settings.get('static_handler_class', StaticFileHandler)
            static_handler_class.reset()
        self.handler = self.handler_class(self.application, self.request, **self.handler_kwargs)
        transforms = [t(self.request) for t in self.application.transforms]
        if self.stream_request_body:
            self.handler._prepared_future = Future()
        fut: Future = gen.convert_yielded(self.handler._execute(transforms, *self.path_args, **self.path_kwargs))
        fut.add_done_callback(lambda f: f.result())
        return self.handler._prepared_future

class HTTPError(Exception):
    def __init__(self, status_code: int = 500, log_message: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        self.status_code = status_code
        self.log_message = log_message
        self.args = args
        self.reason = kwargs.get('reason', None)
        if log_message and (not args):
            self.log_message = log_message.replace('%', '%%')

    def __str__(self) -> str:
        message = 'HTTP %d: %s' % (self.status_code, self.reason or httputil.responses.get(self.status_code, 'Unknown'))
        if self.log_message:
            return message + ' (' + self.log_message % self.args + ')'
        else:
            return message

class Finish(Exception):
    pass

class MissingArgumentError(HTTPError):
    def __init__(self, arg_name: str) -> None:
        super().__init__(400, 'Missing argument %s' % arg_name)
        self.arg_name = arg_name

class ErrorHandler(RequestHandler):
    def initialize(self, status_code: int) -> None:
        self.set_status(status_code)

    def prepare(self) -> None:
        raise HTTPError(self._status_code)

    def check_xsrf_cookie(self) -> None:
        pass

class RedirectHandler(RequestHandler):
    def initialize(self, url: str, permanent: bool = True) -> None:
        self._url = url
        self._permanent = permanent

    def get(self, *args: Any, **kwargs: Any) -> None:
        to_url = self._url.format(*args, **kwargs)
        if self.request.query_arguments:
            to_url = httputil.url_concat(to_url, list(httputil.qs_to_qsl(self.request.query_arguments)))
        self.redirect(to_url, permanent=self._permanent)

class StaticFileHandler(RequestHandler):
    CACHE_MAX_AGE: int = 86400 * 365 * 10
    _static_hashes: Dict[str, Optional[str]] = {}
    _lock = threading.Lock()

    def initialize(self, path: str, default_filename: Optional[str] = None) -> None:
        self.root = path
        self.default_filename = default_filename

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._static_hashes = {}

    def head(self, path: str) -> Any:
        return self.get(path, include_body=False)

    async def get(self, path: str, include_body: bool = True) -> None:
        self.path = self.parse_url_path(path)
        del path
        absolute_path = self.get_absolute_path(self.root, self.path)
        self.absolute_path = self.validate_absolute_path(self.root, absolute_path)
        if self.absolute_path is None:
            return
        self.modified = self.get_modified_time()
        self.set_headers()
        if self.should_return_304():
            self.set_status(304)
            return
        request_range = None
        range_header = self.request.headers.get('Range')
        if range_header:
            request_range = httputil._parse_request_range(range_header)
        size = self.get_content_size()
        if request_range:
            start, end = request_range
            if start is not None and start < 0:
                start += size
                if start < 0:
                    start = 0
            if start is not None and (start >= size or (end is not None and start >= end)) or end == 0:
                self.set_status(416)
                self.set_header('Content-Type', 'text/plain')
                self.set_header('Content-Range', f'bytes */{size}')
                return
            if end is not None and end > size:
                end = size
            if size != (end or size) - (start or 0):
                self.set_status(206)
                self.set_header('Content-Range', httputil._get_content_range(start, end, size))
        else:
            start = end = None
        if start is not None and end is not None:
            content_length = end - start
        elif end is not None:
            content_length = end
        elif start is not None:
            content_length = size - start
        else:
            content_length = size
        self.set_header('Content-Length', content_length)
        if include_body:
            content = self.get_content(self.absolute_path, start, end)
            if isinstance(content, bytes):
                content = [content]
            for chunk in content:
                try:
                    self.write(chunk)
                    await self.flush()
                except iostream.StreamClosedError:
                    return
        else:
            assert self.request.method == 'HEAD'

    def compute_etag(self) -> Optional[str]:
        assert self.absolute_path is not None
        version_hash = self._get_cached_version(self.absolute_path)
        if not version_hash:
            return None
        return f'"{version_hash}"'

    def set_headers(self) -> None:
        self.set_header('Accept-Ranges', 'bytes')
        self.set_etag_header()
        if self.modified is not None:
            self.set_header('Last-Modified', self.modified)
        content_type = self.get_content_type()
        if content_type:
            self.set_header('Content-Type', content_type)
        cache_time = self.get_cache_time(self.path, self.modified, content_type)
        if cache_time > 0:
            self.set_header('Expires', datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=cache_time))
            self.set_header('Cache-Control', 'max-age=' + str(cache_time))
        self.set_extra_headers(self.path)

    def should_return_304(self) -> bool:
        if self.request.headers.get('If-None-Match'):
            return self.check_etag_header()
        ims_value = self.request.headers.get('If-Modified-Since')
        if ims_value is not None:
            try:
                if_since = email.utils.parsedate_to_datetime(ims_value)
            except Exception:
                return False
            if if_since.tzinfo is None:
                if_since = if_since.replace(tzinfo=datetime.timezone.utc)
            assert self.modified is not None
            if if_since >= self.modified:
                return True
        return False

    @classmethod
    def get_absolute_path(cls, root: str, path: str) -> str:
        abspath = os.path.abspath(os.path.join(root, path))
        return abspath

    def validate_absolute_path(self, root: str, absolute_path: str) -> Optional[str]:
        root = os.path.abspath(root)
        if not root.endswith(os.path.sep):
            root += os.path.sep
        if not (absolute_path + os.path.sep).startswith(root):
            raise HTTPError(403, '%s is not in root static directory', self.path)
        if os.path.isdir(absolute_path) and self.default_filename is not None:
            if not self.request.path.endswith('/'):
                if self.request.path.startswith('//'):
                    raise HTTPError(403, 'cannot redirect path with two initial slashes')
                self.redirect(self.request.path + '/', permanent=True)
                return None
            absolute_path = os.path.join(absolute_path, self.default_filename)
        if not os.path.exists(absolute_path):
            raise HTTPError(404)
        if not os.path.isfile(absolute_path):
            raise HTTPError(403, '%s is not a file', self.path)
        return absolute_path

    @classmethod
    def get_content(cls, abspath: str, start: Optional[int] = None, end: Optional[int] = None) -> Iterable[bytes]:
        with open(abspath, 'rb') as file:
            if start is not None:
                file.seek(start)
            if end is not None:
                remaining = end - (start or 0)
            else:
                remaining = None
            while True:
                chunk_size = 64 * 1024
                if remaining is not None and remaining < chunk_size:
                    chunk_size = remaining
                chunk = file.read(chunk_size)
                if chunk:
                    if remaining is not None:
                        remaining -= len(chunk)
                    yield chunk
                else:
                    if remaining is not None:
                        assert remaining == 0
                    return

    @classmethod
    def get_content_version(cls, abspath: str) -> str:
        data = cls.get_content(abspath)
        hasher = hashlib.sha512()
        if isinstance(data, bytes):
            hasher.update(data)
        else:
            for chunk in data:
                hasher.update(chunk)
        return hasher.hexdigest()

    def _stat(self) -> os.stat_result:
        assert self.absolute_path is not None
        if not hasattr(self, '_stat_result'):
            self._stat_result = os.stat(self.absolute_path)
        return self._stat_result

    def get_content_size(self) -> int:
        stat_result = self._stat()
        return stat_result.st_size

    def get_modified_time(self) -> Optional[datetime.datetime]:
        stat_result = self._stat()
        modified = datetime.datetime.fromtimestamp(int(stat_result.st_mtime), datetime.timezone.utc)
        return modified

    def get_content_type(self) -> str:
        assert self.absolute_path is not None
        mime_type, encoding = mimetypes.guess_type(self.absolute_path)
        if encoding == 'gzip':
            return 'application/gzip'
        elif encoding is not None:
            return 'application/octet-stream'
        elif mime_type is not None:
            return mime_type
        else:
            return 'application/octet-stream'

    def set_extra_headers(self, path: str) -> None:
        pass

    def get_cache_time(self, path: str, modified: Optional[datetime.datetime], mime_type: Optional[str]) -> int:
        return self.CACHE_MAX_AGE if 'v' in self.request.arguments else 0

    @classmethod
    def make_static_url(cls, settings: Dict[str, Any], path: str, include_version: bool = True) -> str:
        url: str = settings.get('static_url_prefix', '/static/') + path
        if not include_version:
            return url
        version_hash = cls.get_version(settings, path)
        if not version_hash:
            return url
        return f'{url}?v={version_hash}'

    def parse_url_path(self, url_path: str) -> str:
        if os.path.sep != '/':
            url_path = url_path.replace('/', os.path.sep)
        return url_path

    @classmethod
    def get_version(cls, settings: Dict[str, Any], path: str) -> Optional[str]:
        abs_path = cls.get_absolute_path(settings['static_path'], path)
        return cls._get_cached_version(abs_path)

    @classmethod
    def _get_cached_version(cls, abs_path: str) -> Optional[str]:
        with cls._lock:
            hashes = cls._static_hashes
            if abs_path not in hashes:
                try:
                    hashes[abs_path] = cls.get_content_version(abs_path)
                except Exception:
                    gen_log.error('Could not open static file %r', abs_path)
                    hashes[abs_path] = None
            hsh = hashes.get(abs_path)
            if hsh:
                return hsh
        return None

class FallbackHandler(RequestHandler):
    def initialize(self, fallback: Callable[[Any], Any]) -> None:
        self.fallback = fallback

    def prepare(self) -> None:
        self.fallback(self.request)
        self._finished = True
        self.on_finish()

class OutputTransform:
    def __init__(self, request: Any) -> None:
        pass

    def transform_first_chunk(self, status_code: int, headers: Any, chunk: bytes, finishing: bool) -> Tuple[int, Any, bytes]:
        return (status_code, headers, chunk)

    def transform_chunk(self, chunk: bytes, finishing: bool) -> bytes:
        return chunk

class GZipContentEncoding(OutputTransform):
    CONTENT_TYPES = {'application/javascript', 'application/x-javascript', 'application/xml', 'application/atom+xml', 'application/json', 'application/xhtml+xml', 'image/svg+xml'}
    GZIP_LEVEL: int = 6
    MIN_LENGTH: int = 1024

    def __init__(self, request: Any) -> None:
        self._gzipping: bool = 'gzip' in request.headers.get('Accept-Encoding', '')

    def _compressible_type(self, ctype: str) -> bool:
        return ctype.startswith('text/') or ctype in self.CONTENT_TYPES

    def transform_first_chunk(self, status_code: int, headers: Any, chunk: bytes, finishing: bool) -> Tuple[int, Any, bytes]:
        if 'Vary' in headers:
            headers['Vary'] += ', Accept-Encoding'
        else:
            headers['Vary'] = 'Accept-Encoding'
        if self._gzipping:
            ctype = _unicode(headers.get('Content-Type', '')).split(';')[0]
            self._gzipping = self._compressible_type(ctype) and (not finishing or len(chunk) >= self.MIN_LENGTH) and ('Content-Encoding' not in headers)
        if self._gzipping:
            headers['Content-Encoding'] = 'gzip'
            self._gzip_value = BytesIO()
            self._gzip_file = gzip.GzipFile(mode='w', fileobj=self._gzip_value, compresslevel=self.GZIP_LEVEL)
            chunk = self.transform_chunk(chunk, finishing)
            if 'Content-Length' in headers:
                if finishing:
                    headers['Content-Length'] = str(len(chunk))
                else:
                    del headers['Content-Length']
        return (status_code, headers, chunk)

    def transform_chunk(self, chunk: bytes, finishing: bool) -> bytes:
        if self._gzipping:
            self._gzip_file.write(chunk)
            if finishing:
                self._gzip_file.close()
            else:
                self._gzip_file.flush()
            chunk = self._gzip_value.getvalue()
            self._gzip_value.truncate(0)
            self._gzip_value.seek(0)
        return chunk

def authenticated(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self: RequestHandler, *args: Any, **kwargs: Any) -> Any:
        if not self.current_user:
            if self.request.method in ('GET', 'HEAD'):
                url = self.get_login_url()
                if '?' not in url:
                    if urllib.parse.urlsplit(url).scheme:
                        next_url = self.request.full_url()
                    else:
                        assert self.request.uri is not None
                        next_url = self.request.uri
                    url += '?' + urlencode(dict(next=next_url))
                self.redirect(url)
                return None
            raise HTTPError(403)
        return method(self, *args, **kwargs)
    return wrapper

class UIModule:
    def __init__(self, handler: RequestHandler) -> None:
        self.handler = handler
        self.request = handler.request
        self.ui = handler.ui
        self.locale = handler.locale

    @property
    def current_user(self) -> Any:
        return self.handler.current_user

    def render(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def embedded_javascript(self) -> Optional[str]:
        return None

    def javascript_files(self) -> Optional[Any]:
        return None

    def embedded_css(self) -> Optional[str]:
        return None

    def css_files(self) -> Optional[Any]:
        return None

    def html_head(self) -> Optional[str]:
        return None

    def html_body(self) -> Optional[str]:
        return None

    def render_string(self, path: str, **kwargs: Any) -> str:
        return self.handler.render_string(path, **kwargs)

class _linkify(UIModule):
    def render(self, text: str, **kwargs: Any) -> str:
        return escape.linkify(text, **kwargs)

class _xsrf_form_html(UIModule):
    def render(self) -> str:
        return self.handler.xsrf_form_html()

class TemplateModule(UIModule):
    def __init__(self, handler: RequestHandler) -> None:
        super().__init__(handler)
        self._resource_list: List[Dict[str, Any]] = []
        self._resource_dict: Dict[str, Dict[str, Any]] = {}

    def render(self, path: str, **kwargs: Any) -> str:
        def set_resources(**kwargs_inner: Any) -> str:
            if path not in self._resource_dict:
                self._resource_list.append(kwargs_inner)
                self._resource_dict[path] = kwargs_inner
            elif self._resource_dict[path] != kwargs_inner:
                raise ValueError('set_resources called with different resources for the same template')
            return ''
        return self.render_string(path, set_resources=set_resources, **kwargs)

    def _get_resources(self, key: str) -> Iterable[Any]:
        return (r[key] for r in self._resource_list if key in r)

    def embedded_javascript(self) -> str:
        return '\n'.join(self._get_resources('embedded_javascript'))

    def javascript_files(self) -> List[str]:
        result: List[str] = []
        for f in self._get_resources('javascript_files'):
            if isinstance(f, (unicode_type, bytes)):
                result.append(_unicode(f))
            else:
                result.extend(f)
        return result

    def embedded_css(self) -> str:
        return '\n'.join(self._get_resources('embedded_css'))

    def css_files(self) -> List[str]:
        result: List[str] = []
        for f in self._get_resources('css_files'):
            if isinstance(f, (unicode_type, bytes)):
                result.append(_unicode(f))
            else:
                result.extend(f)
        return result

    def html_head(self) -> str:
        return ''.join(self._get_resources('html_head'))

    def html_body(self) -> str:
        return ''.join(self._get_resources('html_body'))

class _UIModuleNamespace:
    def __init__(self, handler: RequestHandler, ui_modules: Dict[str, Any]) -> None:
        self.handler = handler
        self.ui_modules = ui_modules

    def __getitem__(self, key: str) -> Any:
        return self.handler._ui_module(key, self.ui_modules[key])

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(str(e))

def create_signed_value(secret: Any, name: str, value: str, version: Optional[int] = None, clock: Optional[Callable[[], float]] = None, key_version: Optional[int] = None) -> bytes:
    if version is None:
        version = DEFAULT_SIGNED_VALUE_VERSION
    if clock is None:
        clock = time.time
    timestamp = escape.utf8(str(int(clock())))
    value = base64.b64encode(escape.utf8(value))
    if version == 1:
        assert not isinstance(secret, dict)
        signature = _create_signature_v1(secret, name, value, timestamp)
        value = b'|'.join([value, timestamp, signature])
        return value
    elif version == 2:
        def format_field(s: str) -> bytes:
            return escape.utf8('%d:' % len(s)) + escape.utf8(s)
        to_sign = b'|'.join([b'2', format_field(str(key_version or 0)), format_field(timestamp), format_field(name), format_field(value), b''])
        if isinstance(secret, dict):
            assert key_version is not None, 'Key version must be set when sign key dict is used'
            assert version >= 2, 'Version must be at least 2 for key version support'
            secret = secret[key_version]
        signature = _create_signature_v2(secret, to_sign)
        return to_sign + signature
    else:
        raise ValueError('Unsupported version %d' % version)

_signed_value_version_re = re.compile(b'^([1-9][0-9]*)\\|(.*)$')

def _get_version(value: bytes) -> int:
    m = _signed_value_version_re.match(value)
    if m is None:
        version = 1
    else:
        try:
            version = int(m.group(1))
            if version > 999:
                version = 1
        except ValueError:
            version = 1
    return version

def decode_signed_value(secret: Any, name: str, value: Optional[str], max_age_days: int = 31, clock: Optional[Callable[[], float]] = None, min_version: Optional[int] = None) -> Optional[bytes]:
    if clock is None:
        clock = time.time
    if min_version is None:
        min_version = DEFAULT_SIGNED_VALUE_MIN_VERSION
    if min_version > 2:
        raise ValueError('Unsupported min_version %d' % min_version)
    if not value:
        return None
    value = escape.utf8(value)
    version = _get_version(value)
    if version < min_version:
        return None
    if version == 1:
        assert not isinstance(secret, dict)
        return _decode_signed_value_v1(secret, name, value, max_age_days, clock)
    elif version == 2:
        return _decode_signed_value_v2(secret, name, value, max_age_days, clock)
    else:
        return None

def _decode_signed_value_v1(secret: Any, name: str, value: bytes, max_age_days: int, clock: Callable[[], float]) -> Optional[bytes]:
    parts = escape.utf8(value).split(b'|')
    if len(parts) != 3:
        return None
    signature = _create_signature_v1(secret, name, parts[0], parts[1])
    if not hmac.compare_digest(parts[2], signature):
        gen_log.warning('Invalid cookie signature %r', value)
        return None
    timestamp = int(parts[1])
    if timestamp < clock() - max_age_days * 86400:
        gen_log.warning('Expired cookie %r', value)
        return None
    if timestamp > clock() + 31 * 86400:
        gen_log.warning('Cookie timestamp in future; possible tampering %r', value)
        return None
    if parts[1].startswith(b'0'):
        gen_log.warning('Tampered cookie %r', value)
        return None
    try:
        return base64.b64decode(parts[0])
    except Exception:
        return None

def _decode_fields_v2(value: bytes) -> Tuple[int, bytes, bytes, bytes, bytes]:
    def _consume_field(s: bytes) -> Tuple[bytes, bytes]:
        length, _, rest = s.partition(b':')
        n = int(length)
        field_value = rest[:n]
        if rest[n:n + 1] != b'|':
            raise ValueError('malformed v2 signed value field')
        rest = rest[n + 1:]
        return (field_value, rest)
    rest = value[2:]
    key_version_bytes, rest = _consume_field(rest)
    timestamp, rest = _consume_field(rest)
    name_field, rest = _consume_field(rest)
    value_field, passed_sig = _consume_field(rest)
    return (int(key_version_bytes), timestamp, name_field, value_field, passed_sig)

def _decode_signed_value_v2(secret: Any, name: str, value: bytes, max_age_days: int, clock: Callable[[], float]) -> Optional[bytes]:
    try:
        key_version, timestamp_bytes, name_field, value_field, passed_sig = _decode_fields_v2(value)
    except ValueError:
        return None
    signed_string = value[:-len(passed_sig)]
    if isinstance(secret, dict):
        try:
            secret = secret[key_version]
        except KeyError:
            return None
    expected_sig = _create_signature_v2(secret, signed_string)
    if not hmac.compare_digest(passed_sig, expected_sig):
        return None
    if name_field != escape.utf8(name):
        return None
    timestamp = int(timestamp_bytes)
    if timestamp < clock() - max_age_days * 86400:
        return None
    try:
        return base64.b64decode(value_field)
    except Exception:
        return None

def get_signature_key_version(value: str) -> Optional[int]:
    value_bytes = escape.utf8(value)
    version = _get_version(value_bytes)
    if version < 2:
        return None
    try:
        key_version, _, _, _, _ = _decode_fields_v2(value_bytes)
    except ValueError:
        return None
    return key_version

def _create_signature_v1(secret: Any, *parts: Any) -> bytes:
    hash_obj = hmac.new(escape.utf8(secret), digestmod=hashlib.sha1)
    for part in parts:
        hash_obj.update(escape.utf8(part))
    return escape.utf8(hash_obj.hexdigest())

def _create_signature_v2(secret: Any, s: bytes) -> bytes:
    hash_obj = hmac.new(escape.utf8(secret), digestmod=hashlib.sha256)
    hash_obj.update(escape.utf8(s))
    return escape.utf8(hash_obj.hexdigest())

def is_absolute(path: str) -> bool:
    return any((path.startswith(x) for x in ['/', 'http:', 'https:']))

# Constants for signed value versions
MIN_SUPPORTED_SIGNED_VALUE_VERSION = 1
MAX_SUPPORTED_SIGNED_VALUE_VERSION = 2
DEFAULT_SIGNED_VALUE_VERSION = 2
DEFAULT_SIGNED_VALUE_MIN_VERSION = 1
