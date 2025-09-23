from collections.abc import Iterator, Mapping
from typing import Any
import orjson
from django.http import HttpRequest, HttpResponse, HttpResponseNotAllowed
from typing_extensions import override
from zerver.lib.exceptions import JsonableError, UnauthorizedError

class MutableJsonResponse(HttpResponse):

    def __init__(self, data: dict[str, Any], *, content_type: str, status: int, exception: Exception | None=None) -> None:
        super().__init__('', content_type=content_type, status=status)
        self._data = data
        self._needs_serialization = True
        self.exception = exception

    def get_data(self) -> dict[str, Any]:
        """Get data for this MutableJsonResponse. Calling this method
        after the response's content has already been serialized
        will mean the next time the response's content is accessed
        it will be reserialized because the caller may have mutated
        the data."""
        self._needs_serialization = True
        return self._data

    @override
    @property
    def content(self) -> Any:
        """Get content for the response. If the content hasn't been
        overridden by the property setter, it will be the response data
        serialized lazily to JSON."""
        if self._needs_serialization:
            self.content = orjson.dumps(self._data, option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_PASSTHROUGH_DATETIME)
        return super().content

    @content.setter
    def content(self, value: Any) -> None:
        """Set the content for the response."""
        assert isinstance(HttpResponse.content, property)
        assert HttpResponse.content.fset is not None
        HttpResponse.content.fset(self, value)
        self._needs_serialization = False

    @override
    def __iter__(self) -> Iterator[bytes]:
        return iter([self.content])

def json_unauthorized(message=None, www_authenticate=None):
    return json_response_from_error(UnauthorizedError(msg=message, www_authenticate=www_authenticate))

def json_method_not_allowed(methods):
    resp = HttpResponseNotAllowed(methods)
    resp.content = orjson.dumps({'result': 'error', 'msg': 'Method Not Allowed', 'allowed_methods': methods})
    return resp

def json_response(res_type='success', msg: str='', data: Mapping[str, Any]={}, status: int=200, exception: Exception | None=None) -> MutableJsonResponse:
    content = {'result': res_type, 'msg': msg}
    content.update(data)
    return MutableJsonResponse(data=content, content_type='application/json', status=status, exception=exception)

def json_success(request: HttpRequest, data: Mapping[str, Any]={}) -> MutableJsonResponse:
    return json_response(data=data)

def json_response_from_error(exception: JsonableError) -> MutableJsonResponse:
    """
    This should only be needed in middleware; in app code, just raise.

    When app code raises a JsonableError, the JsonErrorHandler
    middleware takes care of transforming it into a response by
    calling this function.
    """
    response_type = 'error'
    if 200 <= exception.http_status_code < 300:
        response_type = 'success'
    response = json_response(response_type, msg=exception.msg, data=exception.data, status=exception.http_status_code, exception=exception)
    for (header, value) in exception.extra_headers.items():
        response[header] = value
    return response

class AsynchronousResponse(HttpResponse):
    """
    This response is just a sentinel to be discarded by Tornado and replaced
    with a real response later; see zulip_finish.
    """
    status_code = 399