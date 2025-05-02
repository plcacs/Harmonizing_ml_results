"""Class-based views."""
from functools import wraps
from typing import Any, Awaitable, Callable, ClassVar, Mapping, MutableMapping, Optional, Type, Union, cast, no_type_check
from faust.types import AppT, ModelT
from faust.types.web import ViewDecorator, ViewHandlerFun
from yarl import URL
from . import exceptions
from .base import Request, Response, Web
from .exceptions import WebError
__all__ = ['View', 'gives_model', 'takes_model']
_bytes = bytes

class View:
    """Web view (HTTP endpoint)."""
    ServerError = exceptions.ServerError
    ValidationError = exceptions.ValidationError
    ParseError = exceptions.ParseError
    NotAuthenticated = exceptions.NotAuthenticated
    PermissionDenied = exceptions.PermissionDenied
    NotFound = exceptions.NotFound

    @classmethod
    def from_handler(cls, fun):
        """Decorate ``async def`` handler function to create view."""
        if not callable(fun):
            raise TypeError(f'View handler must be callable, not {fun!r}')
        return type(fun.__name__, (cls,), {'get': fun, '__doc__': fun.__doc__, '__module__': fun.__module__})

    def __init__(self, app, web):
        self.app = app
        self.web = web
        self.methods = {'head': self.head, 'get': self.get, 'post': self.post, 'patch': self.patch, 'delete': self.delete, 'put': self.put, 'options': self.options, 'search': self.search}
        self.__post_init__()

    def __post_init__(self):
        """Override this to add custom initialization to your view."""
        ...

    async def __call__(self, request):
        """Perform HTTP request."""
        return await self.dispatch(request)

    async def dispatch(self, request):
        """Dispatch the request and perform any callbacks/cleanup."""
        app = self.app
        sensors = app.sensors
        method = request.method.lower()
        kwargs = request.match_info or {}
        method = cast(Callable[..., Awaitable[Response]], self.methods[method])
        sensor_state = sensors.on_web_request_start(app, request, view=self)
        response = None
        try:
            response = await method(cast(Request, request), **kwargs)
        except WebError as exc:
            response = await self.on_request_error(request, exc)
        finally:
            sensors.on_web_request_end(app, request, response, sensor_state, view=self)
        return response

    async def on_request_error(self, request, exc):
        """Call when a request raises an exception."""
        return self.error(exc.code, exc.detail, **exc.extra_context)

    def path_for(self, view_name, **kwargs):
        """Return the URL path for view by name.

        Supports match keyword arguments.
        """
        return self.web.url_for(view_name, **kwargs)

    def url_for(self, view_name, _base_url=None, **kwargs):
        """Return the canonical URL for view by name.

        Supports match keyword arguments.
        Can take optional base name, which if not set will
        be the canonical URL of the app.
        """
        if _base_url is None:
            _base_url = self.app.conf.canonical_url
        return URL('/'.join([str(_base_url).rstrip('/'), str(self.path_for(view_name, **kwargs)).lstrip('/')]))

    @no_type_check
    async def head(self, request, **kwargs):
        """Override ``head`` to define the HTTP HEAD handler."""
        return await self.get(request, **kwargs)

    @no_type_check
    async def get(self, request, **kwargs):
        """Override ``get`` to define the HTTP GET handler."""
        raise exceptions.MethodNotAllowed('Method GET not allowed.')

    @no_type_check
    async def post(self, request, **kwargs):
        """Override ``post`` to define the HTTP POST handler."""
        raise exceptions.MethodNotAllowed('Method POST not allowed.')

    @no_type_check
    async def put(self, request, **kwargs):
        """Override ``put`` to define the HTTP PUT handler."""
        raise exceptions.MethodNotAllowed('Method PUT not allowed.')

    @no_type_check
    async def patch(self, request, **kwargs):
        """Override ``patch`` to define the HTTP PATCH handler."""
        raise exceptions.MethodNotAllowed('Method PATCH not allowed.')

    @no_type_check
    async def delete(self, request, **kwargs):
        """Override ``delete`` to define the HTTP DELETE handler."""
        raise exceptions.MethodNotAllowed('Method DELETE not allowed.')

    @no_type_check
    async def options(self, request, **kwargs):
        """Override ``options`` to define the HTTP OPTIONS handler."""
        raise exceptions.MethodNotAllowed('Method OPTIONS not allowed.')

    @no_type_check
    async def search(self, request, **kwargs):
        """Override ``search`` to define the HTTP SEARCH handler."""
        raise exceptions.MethodNotAllowed('Method SEARCH not allowed.')

    def text(self, value, *, content_type=None, status=200, reason=None, headers=None):
        """Create text response, using "text/plain" content-type."""
        return self.web.text(value, content_type=content_type, status=status, reason=reason, headers=headers)

    def html(self, value, *, content_type=None, status=200, reason=None, headers=None):
        """Create HTML response from string, ``text/html`` content-type."""
        return self.web.html(value, content_type=content_type, status=status, reason=reason, headers=headers)

    def json(self, value, *, content_type=None, status=200, reason=None, headers=None):
        """Create new JSON response.

        Accepts any JSON-serializable value and will automatically
        serialize it for you.

        The content-type is set to "application/json".
        """
        return self.web.json(value, content_type=content_type, status=status, reason=reason, headers=headers)

    def bytes(self, value, *, content_type=None, status=200, reason=None, headers=None):
        """Create new ``bytes`` response - for binary data."""
        return self.web.bytes(value, content_type=content_type, status=status, reason=reason, headers=headers)

    async def read_request_content(self, request):
        """Return the request body as bytes."""
        return await self.web.read_request_content(request)

    def bytes_to_response(self, s):
        """Deserialize byte string back into a response object."""
        return self.web.bytes_to_response(s)

    def response_to_bytes(self, response):
        """Convert response to serializable byte string.

        The result is a byte string that can be deserialized
        using :meth:`bytes_to_response`.
        """
        return self.web.response_to_bytes(response)

    def route(self, pattern, handler):
        """Create new route from pattern and handler."""
        self.web.route(pattern, handler)
        return handler

    def notfound(self, reason='Not Found', **kwargs):
        """Create not found error response.

        Deprecated: Use ``raise self.NotFound()`` instead.
        """
        return self.error(404, reason, **kwargs)

    def error(self, status, reason, **kwargs):
        """Create error JSON response."""
        return self.json({'error': reason, **kwargs}, status=status)

def takes_model(Model):
    """Decorate view function to return model data."""

    def _decorate_view(fun):

        @wraps(fun)
        async def _inner(view, request, *args, **kwargs):
            data = await view.read_request_content(request)
            obj = Model.loads(data, serializer='json')
            return await fun(view, request, obj, *args, **kwargs)
        return _inner
    return _decorate_view

def gives_model(Model):
    """Decorate view function to automatically decode POST data.

    The POST data is decoded using the model you specify.
    """

    def _decorate_view(fun):

        @wraps(fun)
        async def _inner(view, request, *args, **kwargs):
            response = await fun(view, request, *args, **kwargs)
            return view.json(response)
        return _inner
    return _decorate_view