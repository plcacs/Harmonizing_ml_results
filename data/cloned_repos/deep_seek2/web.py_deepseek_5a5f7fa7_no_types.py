from typing import Dict, Any, Union, Optional, Awaitable, Tuple, List, Callable, Iterable, Generator, Type, TypeVar, cast, overload
from types import TracebackType
import typing
if typing.TYPE_CHECKING:
    from typing import Set
_HeaderTypes = Union[bytes, str, int, numbers.Integral, datetime.datetime]
_CookieSecretTypes = Union[str, bytes, Dict[int, str], Dict[int, bytes]]
MIN_SUPPORTED_SIGNED_VALUE_VERSION = 1
'The oldest signed value version supported by this version of Tornado.\n\nSigned values older than this version cannot be decoded.\n\n.. versionadded:: 3.2.1\n'
MAX_SUPPORTED_SIGNED_VALUE_VERSION = 2
'The newest signed value version supported by this version of Tornado.\n\nSigned values newer than this version cannot be decoded.\n\n.. versionadded:: 3.2.1\n'
DEFAULT_SIGNED_VALUE_VERSION = 2
'The signed value version produced by `.RequestHandler.create_signed_value`.\n\nMay be overridden by passing a ``version`` keyword argument.\n\n.. versionadded:: 3.2.1\n'
DEFAULT_SIGNED_VALUE_MIN_VERSION = 1
'The oldest signed value accepted by `.RequestHandler.get_signed_cookie`.\n\nMay be overridden by passing a ``min_version`` keyword argument.\n\n.. versionadded:: 3.2.1\n'

class _ArgDefaultMarker:
    pass
_ARG_DEFAULT = _ArgDefaultMarker()

class RequestHandler:
    """Base class for HTTP request handlers.

    Subclasses must define at least one of the methods defined in the
    "Entry points" section below.

    Applications should not construct `RequestHandler` objects
    directly and subclasses should not override ``__init__`` (override
    `~RequestHandler.initialize` instead).

    """
    SUPPORTED_METHODS: Tuple[str, ...] = ('GET', 'HEAD', 'POST', 'DELETE', 'PATCH', 'PUT', 'OPTIONS')
    _template_loaders: Dict[str, template.BaseLoader] = {}
    _template_loader_lock = threading.Lock()
    _remove_control_chars_regex = re.compile('[\\x00-\\x08\\x0e-\\x1f]')
    _stream_request_body: bool = False
    _transforms: Optional[List[OutputTransform]] = None
    path_args: Optional[List[str]] = None
    path_kwargs: Optional[Dict[str, str]] = None

    def __init__(self, application, request, **kwargs: Any):
        super().__init__()
        self.application: Application = application
        self.request: httputil.HTTPServerRequest = request
        self._headers_written: bool = False
        self._finished: bool = False
        self._auto_finish: bool = True
        self._prepared_future: Optional[Future[None]] = None
        self.ui: ObjectDict = ObjectDict(((n, self._ui_method(m)) for n, m in application.ui_methods.items()))
        self.ui['_tt_modules'] = _UIModuleNamespace(self, application.ui_modules)
        self.ui['modules'] = self.ui['_tt_modules']
        self.clear()
        assert self.request.connection is not None
        self.request.connection.set_close_callback(self.on_connection_close)
        self.initialize(**kwargs)

    def _initialize(self):
        pass
    initialize: Callable[..., None] = _initialize
    "Hook for subclass initialization. Called for each request.\n\n    A dictionary passed as the third argument of a ``URLSpec`` will be\n    supplied as keyword arguments to ``initialize()``.\n\n    Example::\n\n        class ProfileHandler(RequestHandler):\n            def initialize(self, database):\n                self.database = database\n\n            def get(self, username):\n                ...\n\n        app = Application([\n            (r'/user/(.*)', ProfileHandler, dict(database=database)),\n            ])\n    "

    @property
    def settings(self):
        """An alias for `self.application.settings <Application.settings>`."""
        return self.application.settings

    def _unimplemented_method(self, *args: str, **kwargs: str):
        raise HTTPError(405)
    head: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    get: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    post: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    delete: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    patch: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    put: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method
    options: Callable[..., Optional[Awaitable[None]]] = _unimplemented_method

    def prepare(self):
        """Called at the beginning of a request before  `get`/`post`/etc.

        Override this method to perform common initialization regardless
        of the request method.

        Asynchronous support: Use ``async def`` or decorate this method with
        `.gen.coroutine` to make it asynchronous.
        If this method returns an  ``Awaitable`` execution will not proceed
        until the ``Awaitable`` is done.

        .. versionadded:: 3.1
           Asynchronous support.
        """
        pass

    def on_finish(self):
        """Called after the end of a request.

        Override this method to perform cleanup, logging, etc.
        This method is a counterpart to `prepare`.  ``on_finish`` may
        not produce any output, as it is called after the response
        has been sent to the client.
        """
        pass

    def on_connection_close(self):
        """Called in async handlers if the client closed the connection.

        Override this to clean up resources associated with
        long-lived connections.  Note that this method is called only if
        the connection was closed during asynchronous processing; if you
        need to do cleanup after every request override `on_finish`
        instead.

        Proxies may keep a connection open for a time (perhaps
        indefinitely) after the client has gone away, so this method
        may not be called promptly after the end user closes their
        connection.
        """
        if _has_stream_request_body(self.__class__):
            if not self.request._body_future.done():
                self.request._body_future.set_exception(iostream.StreamClosedError())
                self.request._body_future.exception()

    def clear(self):
        """Resets all headers and content for this response."""
        self._headers = httputil.HTTPHeaders({'Server': 'TornadoServer/%s' % tornado.version, 'Content-Type': 'text/html; charset=UTF-8', 'Date': httputil.format_timestamp(time.time())})
        self.set_default_headers()
        self._write_buffer: List[bytes] = []
        self._status_code = 200
        self._reason = httputil.responses[200]

    def set_default_headers(self):
        """Override this to set HTTP headers at the beginning of the request.

        For example, this is the place to set a custom ``Server`` header.
        Note that setting such headers in the normal flow of request
        processing may not do what you want, since headers may be reset
        during error handling.
        """
        pass

    def set_status(self, status_code, reason=None):
        """Sets the status code for our response.

        :arg int status_code: Response status code.
        :arg str reason: Human-readable reason phrase describing the status
            code. If ``None``, it will be filled in from
            `http.client.responses` or "Unknown".

        .. versionchanged:: 5.0

           No longer validates that the response code is in
           `http.client.responses`.
        """
        self._status_code = status_code
        if reason is not None:
            self._reason = escape.native_str(reason)
        else:
            self._reason = httputil.responses.get(status_code, 'Unknown')

    def get_status(self):
        """Returns the status code for our response."""
        return self._status_code

    def set_header(self, name, value):
        """Sets the given response header name and value.

        All header values are converted to strings (`datetime` objects
        are formatted according to the HTTP specification for the
        ``Date`` header).

        """
        self._headers[name] = self._convert_header_value(value)

    def add_header(self, name, value):
        """Adds the given response header and value.

        Unlike `set_header`, `add_header` may be called multiple times
        to return multiple values for the same header.
        """
        self._headers.add(name, self._convert_header_value(value))

    def clear_header(self, name):
        """Clears an outgoing header, undoing a previous `set_header` call.

        Note that this method does not apply to multi-valued headers
        set by `add_header`.
        """
        if name in self._headers:
            del self._headers[name]
    _INVALID_HEADER_CHAR_RE = re.compile('[\\x00-\\x1f]')

    def _convert_header_value(self, value):
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
    def get_argument(self, name, default, strip=True):
        pass

    @overload
    def get_argument(self, name, default=_ARG_DEFAULT, strip=True):
        pass

    @overload
    def get_argument(self, name, default, strip=True):
        pass

    def get_argument(self, name, default=_ARG_DEFAULT, strip=True):
        """Returns the value of the argument with the given name.

        If default is not provided, the argument is considered to be
        required, and we raise a `MissingArgumentError` if it is missing.

        If the argument appears in the request more than once, we return the
        last value.

        This method searches both the query and body arguments.
        """
        return self._get_argument(name, default, self.request.arguments, strip)

    def get_arguments(self, name, strip=True):
        """Returns a list of the arguments with the given name.

        If the argument is not present, returns an empty list.

        This method searches both the query and body arguments.
        """
        assert isinstance(strip, bool)
        return self._get_arguments(name, self.request.arguments, strip)

    @overload
    def get_body_argument(self, name, default, strip=True):
        pass

    @overload
    def get_body_argument(self, name, default=_ARG_DEFAULT, strip=True):
        pass

    @overload
    def get_body_argument(self, name, default, strip=True):
        pass

    def get_body_argument(self, name, default=_ARG_DEFAULT, strip=True):
        """Returns the value of the argument with the given name
        from the request body.

        If default is not provided, the argument is considered to be
        required, and we raise a `MissingArgumentError` if it is missing.

        If the argument appears in the url more than once, we return the
        last value.

        .. versionadded:: 3.2
        """
        return self._get_argument(name, default, self.request.body_arguments, strip)

    def get_body_arguments(self, name, strip=True):
        """Returns a list of the body arguments with the given name.

        If the argument is not present, returns an empty list.

        .. versionadded:: 3.2
        """
        return self._get_arguments(name, self.request.body_arguments, strip)

    @overload
    def get_query_argument(self, name, default, strip=True):
        pass

    @overload
    def get_query_argument(self, name, default=_ARG_DEFAULT, strip=True):
        pass

    @overload
    def get_query_argument(self, name, default, strip=True):
        pass

    def get_query_argument(self, name, default=_ARG_DEFAULT, strip=True):
        """Returns the value of the argument with the given name
        from the request query string.

        If default is not provided, the argument is considered to be
        required, and we raise a `MissingArgumentError` if it is missing.

        If the argument appears in the url more than once, we return the
        last value.

        .. versionadded:: 3.2
        """
        return self._get_argument(name, default, self.request.query_arguments, strip)

    def get_query_arguments(self, name, strip=True):
        """Returns a list of the query arguments with the given name.

        If the argument is not present, returns an empty list.

        .. versionadded:: 3.2
        """
        return self._get_arguments(name, self.request.query_arguments, strip)

    def _get_argument(self, name, default, source, strip=True):
        args = self._get_arguments(name, source, strip=strip)
        if not args:
            if isinstance(default, _ArgDefaultMarker):
                raise MissingArgumentError(name)
            return default
        return args[-1]

    def _get_arguments(self, name, source, strip=True):
        values = []
        for v in source.get(name, []):
            s = self.decode_argument(v, name=name)
            if isinstance(s, str):
                s = RequestHandler._remove_control_chars_regex.sub(' ', s)
            if strip:
                s = s.strip()
            values.append(s)
        return values

    def decode_argument(self, value, name=None):
        """Decodes an argument from the request.

        The argument has been percent-decoded and is now a byte string.
        By default, this method decodes the argument as utf-8 and returns
        a unicode string, but this may be overridden in subclasses.

        This method is used as a filter for both `get_argument()` and for
        values extracted from the url and passed to `get()`/`post()`/etc.

        The name of the argument is provided if known, but may be None
        (e.g. for unnamed groups in the url regex).
        """
        try:
            return _unicode(value)
        except UnicodeDecodeError:
            raise HTTPError(400, 'Invalid unicode in {}: {!r}'.format(name or 'url', value[:40]))

    @property
    def cookies(self):
        """An alias for
        `self.request.cookies <.httputil.HTTPServerRequest.cookies>`."""
        return self.request.cookies

    @overload
    def get_cookie(self, name, default):
        pass

    @overload
    def get_cookie(self, name, default=None):
        pass

    def get_cookie(self, name, default=None):
        """Returns the value of the request cookie with the given name.

        If the named cookie is not present, returns ``default``.

        This method only returns cookies that were present in the request.
        It does not see the outgoing cookies set by `set_cookie` in this
        handler.
        """
        if self.request.cookies is not None and name in self.request.cookies:
            return self.request.cookies[name].value
        return default

    def set_cookie(self, name, value, domain=None, expires=None, path='/', expires_days=None, *, max_age: Optional[int]=None, httponly: bool=False, secure: bool=False, samesite: Optional[str]=None, **kwargs: Any):
        """Sets an outgoing cookie name/value with the given options.

        Newly-set cookies are not immediately visible via `get_cookie`;
        they are not present until the next request.

        Most arguments are passed directly to `http.cookies.Morsel` directly.
        See https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie
        for more information.

        ``expires`` may be a numeric timestamp as returned by `time.time`,
        a time tuple as returned by `time.gmtime`, or a
        `datetime.datetime` object. ``expires_days`` is provided as a convenience
        to set an expiration time in days from today (if both are set, ``expires``
        is used).

        .. deprecated:: 6.3
           Keyword arguments are currently accepted case-insensitively.
           In Tornado 7.0 this will be changed to only accept lowercase
           arguments.
        """
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

    def clear_cookie(self, name, **kwargs: Any):
        """Deletes the cookie with the given name.

        This method accepts the same arguments as `set_cookie`, except for
        ``expires`` and ``max_age``. Clearing a cookie requires the same
        ``domain`` and ``path`` arguments as when it was set. In some cases the
        ``samesite`` and ``secure`` arguments are also required to match. Other
        arguments are ignored.

        Similar to `set_cookie`, the effect of this method will not be
        seen until the following request.

        .. versionchanged:: 6.3

           Now accepts all keyword arguments that ``set_cookie`` does.
           The ``samesite`` and ``secure`` flags have recently become
           required for clearing ``samesite="none"`` cookies.
        """
        for excluded_arg in ['expires', 'max_age']:
            if excluded_arg in kwargs:
                raise TypeError(f"clear_cookie() got an unexpected keyword argument '{excluded_arg}'")
        expires = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=365)
        self.set_cookie(name, value='', expires=expires, **kwargs)

    def clear_all_cookies(self, **kwargs: Any):
        """Attempt to delete all the cookies the user sent with this request.

        See `clear_cookie` for more information on keyword arguments. Due to
        limitations of the cookie protocol, it is impossible to determine on the
        server side which values are necessary for the ``domain``, ``path``,
        ``samesite``, or ``secure`` arguments, this method can only be
        successful if you consistently use the same values for these arguments
        when setting cookies.

        Similar to `set_cookie`, the effect of this method will not be seen
        until the following request.

        .. versionchanged:: 3.2

           Added the ``path`` and ``domain`` parameters.

        .. versionchanged:: 6.3

           Now accepts all keyword arguments that ``set_cookie`` does.

        .. deprecated:: 6.3

           The increasingly complex rules governing cookies have made it
           impossible for a ``clear_all_cookies`` method to work reliably
           since all we know about cookies are their names. Applications
           should generally use ``clear_cookie`` one at a time instead.
        """
        for name in self.request.cookies:
            self.clear_cookie(name, **kwargs)

    def set_signed_cookie(self, name, value, expires_days=30, version=None, **kwargs: Any):
        """Signs and timestamps a cookie so it cannot be forged.

        You must specify the ``cookie_secret`` setting in your Application
        to use this method. It should be a long, random sequence of bytes
        to be used as the HMAC secret for the signature.

        To read a cookie set with this method, use `get_signed_cookie()`.

        Note that the ``expires_days`` parameter sets the lifetime of the
        cookie in the browser, but is independent of the ``max_age_days``
        parameter to `get_signed_cookie`.
        A value of None limits the lifetime to the current browser session.

        Secure cookies may contain arbitrary byte values, not just unicode
        strings (unlike regular cookies)

        Similar to `set_cookie`, the effect of this method will not be
        seen until the following request.

        .. versionchanged:: 3.2.1

           Added the ``version`` argument.  Introduced cookie version 2
           and made it the default.

        .. versionchanged:: 6.3

           Renamed from ``set_secure_cookie`` to ``set_signed_cookie`` to
           avoid confusion with other uses of "secure" in cookie attributes
           and prefixes. The old name remains as an alias.
        """
        self.set_cookie(name, self.create_signed_value(name, value, version=version), expires_days=expires_days, **kwargs)
    set_secure_cookie = set_signed_cookie

    def create_signed_value(self, name, value, version=None):
        """Signs and timestamps a string so it cannot be forged.

        Normally used via set_signed_cookie, but provided as a separate
        method for non-cookie uses.  To decode a value not stored
        as a cookie use the optional value argument to get_signed_cookie.

        .. versionchanged:: 3.2.1

           Added the ``version`` argument.  Introduced cookie version 2
           and made it the default.
        """
        self.require_setting('cookie_secret', 'secure cookies')
        secret = self.application.settings['cookie_secret']
        key_version = None
        if isinstance(secret, dict):
            if self.application.settings.get('key_version') is None:
                raise Exception('key_version setting must be used for secret_key dicts')
            key_version = self.application.settings['key_version']
        return create_signed_value(secret, name, value, version=version, key_version=key_version)

    def get_signed_cookie(self, name, value=None, max_age_days=31, min_version=None):
        """Returns the given signed cookie if it validates, or None.

        The decoded cookie value is returned as a byte string (unlike
        `get_cookie`).

        Similar to `get_cookie`, this method only returns cookies that
        were present in the request. It does not see outgoing cookies set by
        `set_signed_cookie` in this handler.

        .. versionchanged:: 3.2.1

           Added the ``min_version`` argument.  Introduced cookie version 2;
           both versions 1 and 2 are accepted by default.

         .. versionchanged:: 6.3

           Renamed from ``get_secure_cookie`` to ``get_signed_cookie`` to
           avoid confusion with other uses of "secure" in cookie attributes
           and prefixes. The old name remains as an alias.

        """
        self.require_setting('cookie_secret', 'secure cookies')
        if value is None:
            value = self.get_cookie(name)
        return decode_signed_value(self.application.settings['cookie_secret'], name, value, max_age_days=max_age_days, min_version=min_version)
    get_secure_cookie = get_signed_cookie

    def get_signed_cookie_key_version(self, name, value=None):
        """Returns the signing key version of the secure cookie.

        The version is returned as int.

        .. versionchanged:: 6.3

           Renamed from ``get_secure_cookie_key_version`` to
           ``set_signed_cookie_key_version`` to avoid confusion with other
           uses of "secure" in cookie attributes and prefixes. The old name
           remains as an alias.

        """
        self.require_setting('cookie_secret', 'secure cookies')
        if value is None:
            value = self.get_cookie(name)
        if value is None:
            return None
        return get_signature_key_version(value)
    get_secure_cookie_key_version = get_signed_cookie_key_version

    def redirect(self, url, permanent=False, status=None):
        """Sends a redirect to the given (optionally relative) URL.

        If the ``status`` argument is specified, that value is used as the
        HTTP status code; otherwise either 301 (permanent) or 302
        (temporary) is chosen based on the ``permanent`` argument.
        The default is 302 (temporary).
        """
        if self._headers_written:
            raise Exception('Cannot redirect after headers have been written')
        if status is None:
            status = 301 if permanent else 302
        else:
            assert isinstance(status, int) and 300 <= status <= 399
        self.set_status(status)
        self.set_header('Location', utf8(url))
        self.finish()

    def write(self, chunk):
        """Writes the given chunk to the output buffer.

        To write the output to the network, use the `flush()` method below.

        If the given chunk is a dictionary, we write it as JSON and set
        the Content-Type of the response to be ``application/json``.
        (if you want to send JSON as a different ``Content-Type``, call
        ``set_header`` *after* calling ``write()``).

        Note that lists are not converted to JSON because of a potential
        cross-site security vulnerability.  All JSON output should be
        wrapped in a dictionary.  More details at
        http://haacked.com/archive/2009/06/25/json-hijacking.aspx/ and
        https://github.com/facebook/tornado/issues/1009
        """
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

    def render(self, template_name, **kwargs: Any):
        """Renders the template with the given arguments as the response.

        ``render()`` calls ``finish()``, so no other output methods can be called
        after it.

        Returns a `.Future` with the same semantics as the one returned by `finish`.
        Awaiting this `.Future` is optional.

        .. versionchanged:: 5.1

           Now returns a `.Future` instead of ``None``.
        """
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