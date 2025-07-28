#!/usr/bin/env python3
"""
Tornado web module with type annotations.
"""
import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import types
import traceback
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, Awaitable

import tornado
from tornado import escape, gen, iostream, locale, template
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado.log import access_log, app_log, gen_log

# Type variables and alias definitions
_RequestHandlerType = TypeVar('_RequestHandlerType', bound='RequestHandler')
_ArgDefault = object()  # marker for default argument

# Utility function for type hints
def is_absolute(path: str) -> bool:
    return any((path.startswith(x) for x in ['/', 'http:', 'https:']))

class RequestHandler:
    SUPPORTED_METHODS: Tuple[str, ...] = ('GET', 'HEAD', 'POST', 'DELETE', 'PATCH', 'PUT', 'OPTIONS')
    
    def __init__(self, application: "Application", request: httputil.HTTPServerRequest, **kwargs: Any) -> None:
        self.application = application
        self.request = request
        self._finished: bool = False
        self._auto_finish: bool = True
        self._prepared_future: Optional[Future] = None
        self.ui: Dict[str, Any] = {}  # UI namespace
        self.clear()
        self.initialize(**kwargs)

    def initialize(self, **kwargs: Any) -> None:
        pass

    def clear(self) -> None:
        self._finished = False
        self._auto_finish = True

    def prepare(self) -> Optional[Awaitable[Any]]:
        # Called before the http method is called.
        pass

    def on_finish(self) -> None:
        pass

    def check_xsrf_cookie(self) -> None:
        pass

    @property
    def current_user(self) -> Any:
        return None

    async def _execute(self, transforms: List[Any], *args: Any, **kwargs: Any) -> None:
        self._transforms = transforms
        try:
            if self.request.method not in self.SUPPORTED_METHODS:
                raise HTTPError(405)
            # Decode path arguments (simplified)
            self.path_args: List[Any] = [self.decode_argument(arg) for arg in args]
            self.path_kwargs: Dict[str, Any] = {k: self.decode_argument(v, name=k) for k, v in kwargs.items()}
            if self.request.method not in ('GET', 'HEAD', 'OPTIONS') and self.application.settings.get('xsrf_cookies'):
                self.check_xsrf_cookie()
            result = self.prepare()
            if result is not None:
                await result
            if self._prepared_future is not None:
                future_set_result_unless_cancelled(self._prepared_future, None)
            if self._finished:
                return
            if _has_stream_request_body(self.__class__):
                try:
                    await self.request._body_future  # type: ignore
                except iostream.StreamClosedError:
                    return
            method: Callable[..., Any] = getattr(self, self.request.method.lower())
            res = method(*self.path_args, **self.path_kwargs)
            if res is not None:
                await res
            if self._auto_finish and (not self._finished):
                self.finish()
        except Exception as e:
            try:
                self._handle_request_exception(e)
            except Exception:
                app_log.error('Exception in exception handler', exc_info=True)
            finally:
                if self._prepared_future is not None and (not self._prepared_future.done()):
                    self._prepared_future.set_result(None)

    def decode_argument(self, value: Any, name: Optional[str] = None) -> Any:
        try:
            return escape.utf8(value)
        except Exception:
            raise HTTPError(400, f"Invalid argument {name}: {value}")

    def finish(self, chunk: Optional[Union[bytes, str]] = None) -> Future:
        if self._finished:
            raise RuntimeError('finish() called twice')
        if chunk is not None:
            self.write(chunk)
        future = self.flush(include_footers=True)
        self._finished = True
        self.on_finish()
        return future

    def write(self, chunk: Union[bytes, str, Dict[Any, Any]]) -> None:
        if self._finished:
            raise RuntimeError('Cannot write() after finish()')
        # Handle writing chunk (simplified)
        pass

    def flush(self, include_footers: bool = False) -> Future:
        # Flush output buffer (simplified)
        fut: Future = Future()
        fut.set_result(None)
        return fut

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

    def log_exception(self, typ: Any, value: Any, tb: Any) -> None:
        if isinstance(value, HTTPError):
            if value.log_message:
                format_str = '%d %s: ' + value.log_message
                args = [value.status_code, self._request_summary()] + list(value.args)
                gen_log.warning(format_str, *args)
        else:
            app_log.error('Uncaught exception %s\n%r', self._request_summary(), self.request, exc_info=(typ, value, tb))

    def _request_summary(self) -> str:
        return '{} {} ({})'.format(self.request.method, self.request.uri, self.request.remote_ip)

class HTTPError(Exception):
    def __init__(self, status_code: int = 500, log_message: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        self.status_code: int = status_code
        self.log_message: Optional[str] = log_message
        self.args = args
        self.reason: Optional[str] = kwargs.get('reason', None)
        if log_message and (not args):
            self.log_message = log_message.replace('%', '%%')

    def __str__(self) -> str:
        message = f'HTTP {self.status_code}: {self.reason or "Unknown"}'
        if self.log_message:
            return message + ' (' + (self.log_message % self.args) + ')'
        else:
            return message

class Finish(Exception):
    pass

class MissingArgumentError(HTTPError):
    def __init__(self, arg_name: str) -> None:
        super().__init__(400, f'Missing argument {arg_name}')
        self.arg_name: str = arg_name

class ErrorHandler(RequestHandler):
    def initialize(self, status_code: int) -> None:
        self.set_status(status_code)

    def prepare(self) -> None:
        raise HTTPError(self._status_code)

    def check_xsrf_cookie(self) -> None:
        pass

    def set_status(self, status_code: int, reason: Optional[str] = None) -> None:
        self._status_code = status_code
        self._reason = reason

class RedirectHandler(RequestHandler):
    def initialize(self, url: str, permanent: bool = True) -> None:
        self._url: str = url
        self._permanent: bool = permanent

    def get(self, *args: Any, **kwargs: Any) -> None:
        to_url = self._url.format(*args, **kwargs)
        if self.request.query_arguments:
            from urllib.parse import urlencode
            to_url += '?' + urlencode(dict(next=self.request.uri))
        self.redirect(to_url, permanent=self._permanent)

    def redirect(self, url: str, permanent: bool = False, status: Optional[int] = None) -> None:
        # Simplified redirect implementation.
        self.set_status(301 if permanent else 302)
        self.finish()

class StaticFileHandler(RequestHandler):
    CACHE_MAX_AGE: int = 86400 * 365 * 10
    _static_hashes: Dict[str, Optional[str]] = {}
    _lock = threading.Lock()

    def initialize(self, path: str, default_filename: Optional[str] = None) -> None:
        self.root: str = path
        self.default_filename: Optional[str] = default_filename

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._static_hashes = {}

    async def get(self, path: str, include_body: bool = True) -> None:
        self.path = self.parse_url_path(path)
        absolute_path = self.get_absolute_path(self.root, self.path)
        self.absolute_path = self.validate_absolute_path(self.root, absolute_path)
        if self.absolute_path is None:
            return
        self.modified = self.get_modified_time()
        self.set_headers()
        if self.should_return_304():
            self.set_status(304)
            return
        # Range and content handling omitted for brevity.
        if include_body:
            content = self.get_content(self.absolute_path)
            if isinstance(content, bytes):
                content = [content]
            for chunk in content:
                try:
                    self.write(chunk)
                    await self.flush()
                except iostream.StreamClosedError:
                    return

    def head(self, path: str) -> None:
        return self.get(path, include_body=False)

    def compute_etag(self) -> Optional[str]:
        version_hash = self._get_cached_version(self.absolute_path)  # type: ignore
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
            expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=cache_time)
            self.set_header('Expires', expires)
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
            if self.modified is not None and if_since >= self.modified:
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
            raise HTTPError(403, '%s is not in root static directory' % self.path)
        if os.path.isdir(absolute_path) and self.default_filename is not None:
            if not self.request.path.endswith('/'):
                self.redirect(self.request.path + '/', permanent=True)
                return None
            absolute_path = os.path.join(absolute_path, self.default_filename)
        if not os.path.exists(absolute_path):
            raise HTTPError(404)
        if not os.path.isfile(absolute_path):
            raise HTTPError(403, '%s is not a file' % self.path)
        return absolute_path

    @classmethod
    def get_content(cls, abspath: str, start: Optional[int] = None, end: Optional[int] = None) -> Union[bytes, List[bytes]]:
        with open(abspath, 'rb') as file:
            if start is not None:
                file.seek(start)
            remaining = None
            if end is not None:
                remaining = end - (start or 0)
            chunks: List[bytes] = []
            while True:
                chunk_size = 64 * 1024
                if remaining is not None and remaining < chunk_size:
                    chunk_size = remaining
                chunk = file.read(chunk_size)
                if chunk:
                    if remaining is not None:
                        remaining -= len(chunk)
                    chunks.append(chunk)
                else:
                    break
            if len(chunks) == 1:
                return chunks[0]
            return chunks

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
        if not hasattr(self, '_stat_result'):
            self._stat_result = os.stat(self.absolute_path)  # type: ignore
        return self._stat_result

    def get_content_size(self) -> int:
        stat_result = self._stat()
        return stat_result.st_size

    def get_modified_time(self) -> Optional[datetime.datetime]:
        stat_result = self._stat()
        modified = datetime.datetime.fromtimestamp(int(stat_result.st_mtime), datetime.timezone.utc)
        return modified

    def get_content_type(self) -> str:
        mime_type, encoding = template.guess_type(self.absolute_path)  # type: ignore
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

    def get_cache_time(self, path: str, modified: Optional[datetime.datetime], mime_type: str) -> int:
        return self.CACHE_MAX_AGE if 'v' in self.request.arguments else 0

    @classmethod
    def make_static_url(cls, settings: Dict[str, Any], path: str, include_version: bool = True) -> str:
        url = settings.get('static_url_prefix', '/static/') + path
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
    def initialize(self, fallback: Callable[[httputil.HTTPServerRequest], Any]) -> None:
        self.fallback = fallback

    def prepare(self) -> None:
        self.fallback(self.request)
        self._finished = True
        self.on_finish()

class OutputTransform:
    def __init__(self, request: httputil.HTTPServerRequest) -> None:
        pass

    def transform_first_chunk(self, status_code: int, headers: Dict[str, Any], chunk: bytes, finishing: bool) -> Tuple[int, Dict[str, Any], bytes]:
        return (status_code, headers, chunk)

    def transform_chunk(self, chunk: bytes, finishing: bool) -> bytes:
        return chunk

class GZipContentEncoding(OutputTransform):
    CONTENT_TYPES = {'application/javascript', 'application/x-javascript', 'application/xml',
                     'application/atom+xml', 'application/json', 'application/xhtml+xml', 'image/svg+xml'}
    GZIP_LEVEL: int = 6
    MIN_LENGTH: int = 1024

    def __init__(self, request: httputil.HTTPServerRequest) -> None:
        super().__init__(request)
        self._gzipping: bool = 'gzip' in request.headers.get('Accept-Encoding', '')
        self._gzip_value: Optional[BytesIO] = None
        self._gzip_file: Any = None

    def _compressible_type(self, ctype: str) -> bool:
        return ctype.startswith('text/') or ctype in self.CONTENT_TYPES

    def transform_first_chunk(self, status_code: int, headers: Dict[str, Any], chunk: bytes, finishing: bool) -> Tuple[int, Dict[str, Any], bytes]:
        if 'Vary' in headers:
            headers['Vary'] += ', Accept-Encoding'
        else:
            headers['Vary'] = 'Accept-Encoding'
        if self._gzipping:
            ctype = str(headers.get('Content-Type', '')).split(';')[0]
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
            chunk = self._gzip_value.getvalue()  # type: ignore
            self._gzip_value.truncate(0)  # type: ignore
            self._gzip_value.seek(0)  # type: ignore
        return chunk

def stream_request_body(cls: Type[_RequestHandlerType]) -> Type[_RequestHandlerType]:
    if not issubclass(cls, RequestHandler):
        raise TypeError('expected subclass of RequestHandler, got %r' % cls)
    cls._stream_request_body = True
    return cls

def _has_stream_request_body(cls: Type[RequestHandler]) -> bool:
    if not issubclass(cls, RequestHandler):
        raise TypeError('expected subclass of RequestHandler, got %r' % cls)
    return getattr(cls, '_stream_request_body', False)

def removeslash(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self: RequestHandler, *args: Any, **kwargs: Any) -> Any:
        if self.request.path.endswith('/'):
            if self.request.method in ('GET', 'HEAD'):
                uri = self.request.path.rstrip('/')
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
                uri = self.request.path + '/'
                if self.request.query:
                    uri += '?' + self.request.query
                self.redirect(uri, permanent=True)
                return None
            raise HTTPError(404)
        return method(self, *args, **kwargs)
    return wrapper

class _ApplicationRouter:
    def __init__(self, application: "Application", rules: Optional[List[Any]] = None) -> None:
        self.application = application
        self.rules = rules or []

    def process_rule(self, rule: Any) -> Any:
        rule = rule  # simplified processing
        if isinstance(rule.target, (list, tuple)):
            rule.target = _ApplicationRouter(self.application, rule.target)
        return rule

    def get_target_delegate(self, target: Any, request: httputil.HTTPServerRequest, **target_params: Any) -> Any:
        from inspect import isclass
        if isclass(target) and issubclass(target, RequestHandler):
            return self.application.get_handler_delegate(request, target, **target_params)
        return target

class Application:
    def __init__(self, handlers: Optional[List[Any]] = None, default_host: Optional[str] = None, transforms: Optional[List[Any]] = None, **settings: Any) -> None:
        if transforms is None:
            self.transforms: List[Any] = []
            if settings.get('compress_response') or settings.get('gzip'):
                self.transforms.append(GZipContentEncoding)
        else:
            self.transforms = transforms
        self.default_host: Optional[str] = default_host
        self.settings: Dict[str, Any] = settings
        self.ui_modules: Dict[str, Any] = {'linkify': _linkify, 'xsrf_form_html': _xsrf_form_html, 'Template': TemplateModule}
        self.ui_methods: Dict[str, Callable[..., Any]] = {}
        self._load_ui_modules(settings.get('ui_modules', {}))
        self._load_ui_methods(settings.get('ui_methods', {}))
        if self.settings.get('static_path'):
            path = self.settings['static_path']
            handlers = list(handlers or [])
            static_url_prefix = settings.get('static_url_prefix', '/static/')
            static_handler_class = settings.get('static_handler_class', StaticFileHandler)
            static_handler_args = settings.get('static_handler_args', {})
            static_handler_args['path'] = path
            for pattern in [re.escape(static_url_prefix) + '(.*)', '/(favicon\\.ico)', '/(robots\\.txt)']:
                handlers.insert(0, (pattern, static_handler_class, static_handler_args))
        if self.settings.get('debug'):
            self.settings.setdefault('autoreload', True)
            self.settings.setdefault('compiled_template_cache', False)
            self.settings.setdefault('static_hash_cache', False)
            self.settings.setdefault('serve_traceback', True)
        self.wildcard_router = _ApplicationRouter(self, handlers)
        self.default_router = _ApplicationRouter(self, [{'matcher': 'AnyMatches', 'target': self.wildcard_router}])
        if self.settings.get('autoreload'):
            from tornado import autoreload
            autoreload.start()

    def listen(self, port: int, address: Optional[str] = None, *, family: int = socket.AF_UNSPEC, backlog: int = 128, flags: Optional[int] = None, reuse_port: bool = False, **kwargs: Any) -> HTTPServer:
        server = HTTPServer(self, **kwargs)
        server.listen(port, address=address, family=family, backlog=backlog, flags=flags, reuse_port=reuse_port)
        return server

    def add_handlers(self, host_pattern: str, host_handlers: List[Any]) -> None:
        host_matcher = host_pattern
        rule = {'matcher': host_matcher, 'target': _ApplicationRouter(self, host_handlers)}
        self.default_router.rules.insert(-1, rule)

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

    def __call__(self, request: httputil.HTTPServerRequest) -> Any:
        dispatcher = self.find_handler(request)
        return dispatcher.execute()

    def find_handler(self, request: httputil.HTTPServerRequest, **kwargs: Any) -> Any:
        route = self.default_router.rules[0] if self.default_router.rules else None
        if route is not None:
            return route['target']
        if self.settings.get('default_handler_class'):
            return self.get_handler_delegate(request, self.settings['default_handler_class'], self.settings.get('default_handler_args', {}))
        return self.get_handler_delegate(request, ErrorHandler, {'status_code': 404})

    def get_handler_delegate(self, request: httputil.HTTPServerRequest, target_class: Type[RequestHandler], target_kwargs: Optional[Dict[str, Any]] = None, path_args: Optional[List[Any]] = None, path_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return _HandlerDelegate(self, request, target_class, target_kwargs, path_args, path_kwargs)

    def reverse_url(self, name: str, *args: Any) -> str:
        reversed_url = "/dummy_url"  # simplified reverse_url implementation
        if reversed_url is not None:
            return reversed_url
        raise KeyError(f'{name} not found in named urls')

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
        request_time = 1000.0 * handler.request.request_time()  # type: ignore
        log_method('%d %s %.2fms', handler.get_status(), handler._request_summary(), request_time)

class _HandlerDelegate(httputil.HTTPMessageDelegate):
    def __init__(self, application: Application, request: httputil.HTTPServerRequest, handler_class: Type[RequestHandler], handler_kwargs: Optional[Dict[str, Any]] = None, path_args: Optional[List[Any]] = None, path_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.application = application
        self.connection = request.connection
        self.request = request
        self.handler_class = handler_class
        self.handler_kwargs = handler_kwargs or {}
        self.path_args = path_args or []
        self.path_kwargs = path_kwargs or {}
        self.chunks: List[bytes] = []
        self.stream_request_body: bool = _has_stream_request_body(self.handler_class)

    def headers_received(self, start_line: Any, headers: Dict[str, Any]) -> Optional[Any]:
        if self.stream_request_body:
            self.request._body_future = Future()  # type: ignore
            return self.execute()
        return None

    def data_received(self, data: bytes) -> Optional[Any]:
        if self.stream_request_body:
            return self.handler.data_received(data)  # type: ignore
        else:
            self.chunks.append(data)
            return None

    def finish(self) -> None:
        if self.stream_request_body:
            future_set_result_unless_cancelled(self.request._body_future, None)  # type: ignore
        else:
            self.request.body = b''.join(self.chunks)  # type: ignore
            self.request._parse_body()  # type: ignore
            self.execute()

    def on_connection_close(self) -> None:
        if self.stream_request_body:
            self.handler.on_connection_close()  # type: ignore
        else:
            self.chunks = []

    def execute(self) -> Future:
        self.handler = self.handler_class(self.application, self.request, **self.handler_kwargs)
        transforms = [t(self.request) for t in self.application.transforms]
        if self.stream_request_body:
            self.handler._prepared_future = Future()
        fut = gen.convert_yielded(self.handler._execute(transforms, *self.path_args, **self.path_kwargs))
        fut.add_done_callback(lambda f: f.result())
        return self.handler._prepared_future  # type: ignore

class HTTPError(Exception):
    pass

def authenticated(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self: RequestHandler, *args: Any, **kwargs: Any) -> Any:
        if not self.current_user:
            if self.request.method in ('GET', 'HEAD'):
                url = self.get_login_url()  # type: ignore
                if '?' not in url:
                    import urllib.parse
                    next_url = self.request.full_url() if urllib.parse.urlsplit(url).scheme else self.request.uri  # type: ignore
                    url += '?' + urllib.parse.urlencode(dict(next=next_url))
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

    def javascript_files(self) -> Optional[List[Union[str, bytes]]]:
        return None

    def embedded_css(self) -> Optional[str]:
        return None

    def css_files(self) -> Optional[List[Union[str, bytes]]]:
        return None

    def html_head(self) -> Optional[str]:
        return None

    def html_body(self) -> Optional[str]:
        return None

    def render_string(self, path: str, **kwargs: Any) -> str:
        return self.handler.render_string(path, **kwargs)  # type: ignore

class _linkify(UIModule):
    def render(self, text: str, **kwargs: Any) -> str:
        return escape.linkify(text, **kwargs)

class _xsrf_form_html(UIModule):
    def render(self) -> str:
        return self.handler.xsrf_form_html()  # type: ignore

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

    def _get_resources(self, key: str) -> List[Any]:
        return [r[key] for r in self._resource_list if key in r]

    def embedded_javascript(self) -> str:
        return '\n'.join(self._get_resources('embedded_javascript'))

    def javascript_files(self) -> List[Any]:
        result: List[Any] = []
        for f in self._get_resources('javascript_files'):
            if isinstance(f, (str, bytes)):
                result.append(f)
            else:
                result.extend(f)
        return result

    def embedded_css(self) -> str:
        return '\n'.join(self._get_resources('embedded_css'))

    def css_files(self) -> List[Any]:
        result: List[Any] = []
        for f in self._get_resources('css_files'):
            if isinstance(f, (str, bytes)):
                result.append(f)
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
        return self.handler._ui_module(key, self.ui_modules[key])  # type: ignore

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(str(e))

def create_signed_value(secret: Union[str, Dict[int, str]], name: str, value: str, version: Optional[int] = None, clock: Optional[Callable[[], float]] = None, key_version: Optional[int] = None) -> bytes:
    DEFAULT_SIGNED_VALUE_VERSION = 2
    if version is None:
        version = DEFAULT_SIGNED_VALUE_VERSION
    if clock is None:
        clock = time.time
    timestamp: bytes = escape.utf8(str(int(clock())))
    value_b: bytes = base64.b64encode(escape.utf8(value))
    if version == 1:
        assert not isinstance(secret, dict)
        signature: bytes = _create_signature_v1(secret, name, value_b, timestamp)
        result: bytes = b'|'.join([value_b, timestamp, signature])
        return result
    elif version == 2:
        def format_field(s: str) -> bytes:
            return escape.utf8(f'{len(s)}:') + escape.utf8(s)
        to_sign: bytes = b'|'.join([b'2', format_field(str(key_version or 0)), format_field(timestamp.decode("utf8")), format_field(name), format_field(value), b''])
        if isinstance(secret, dict):
            assert key_version is not None, 'Key version must be set when sign key dict is used'
            secret = secret[key_version]
        signature = _create_signature_v2(secret, to_sign)
        return to_sign + signature
    else:
        raise ValueError('Unsupported version %d' % version)

_signed_value_version_re = re.compile(b'^([1-9][0-9]*)\\|(.*)$')

def _get_version(value: bytes) -> int:
    m = _signed_value_version_re.match(value)
    if m is None:
        return 1
    else:
        try:
            version = int(m.group(1))
            if version > 999:
                return 1
            return version
        except ValueError:
            return 1

def decode_signed_value(secret: Union[str, Dict[int, str]], name: str, value: str, max_age_days: int = 31, clock: Optional[Callable[[], float]] = None, min_version: Optional[int] = None) -> Optional[bytes]:
    DEFAULT_SIGNED_VALUE_MIN_VERSION = 1
    if clock is None:
        clock = time.time
    if min_version is None:
        min_version = DEFAULT_SIGNED_VALUE_MIN_VERSION
    if min_version > 2:
        raise ValueError('Unsupported min_version %d' % min_version)
    if not value:
        return None
    value_b: bytes = escape.utf8(value)
    version: int = _get_version(value_b)
    if version < min_version:
        return None
    if version == 1:
        assert not isinstance(secret, dict)
        return _decode_signed_value_v1(secret, name, value_b, max_age_days, clock)
    elif version == 2:
        return _decode_signed_value_v2(secret, name, value_b, max_age_days, clock)
    else:
        return None

def _decode_signed_value_v1(secret: str, name: str, value: bytes, max_age_days: int, clock: Callable[[], float]) -> Optional[bytes]:
    parts = value.split(b'|')
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
        length, sep, rest = s.partition(b':')
        n = int(length)
        field_value = rest[:n]
        if rest[n:n + 1] != b'|':
            raise ValueError('malformed v2 signed value field')
        rest = rest[n + 1:]
        return (field_value, rest)
    rest = value[2:]
    key_version_field, rest = _consume_field(rest)
    timestamp_field, rest = _consume_field(rest)
    name_field, rest = _consume_field(rest)
    value_field, passed_sig = _consume_field(rest)
    return (int(key_version_field), timestamp_field, name_field, value_field, passed_sig)

def _decode_signed_value_v2(secret: Union[str, Dict[int, str]], name: str, value: bytes, max_age_days: int, clock: Callable[[], float]) -> Optional[bytes]:
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
    value_b: bytes = escape.utf8(value)
    version: int = _get_version(value_b)
    if version < 2:
        return None
    try:
        key_version, _, _, _, _ = _decode_fields_v2(value_b)
        return key_version
    except ValueError:
        return None

def _create_signature_v1(secret: str, *parts: Any) -> bytes:
    hash_obj = hmac.new(escape.utf8(secret), digestmod=hashlib.sha1)
    for part in parts:
        hash_obj.update(escape.utf8(part))
    return escape.utf8(hash_obj.hexdigest())

def _create_signature_v2(secret: str, s: bytes) -> bytes:
    hash_obj = hmac.new(escape.utf8(secret), digestmod=hashlib.sha256)
    hash_obj.update(escape.utf8(s))
    return escape.utf8(hash_obj.hexdigest())

# End of annotated code.
