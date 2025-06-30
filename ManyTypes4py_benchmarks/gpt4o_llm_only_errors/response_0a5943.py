from collections.abc import Iterator, Mapping
from typing import Any, Optional, Dict, List, Union
import orjson
from django.http import HttpRequest, HttpResponse, HttpResponseNotAllowed
from typing_extensions import override
from zerver.lib.exceptions import JsonableError, UnauthorizedError

class MutableJsonResponse(HttpResponse):

    def __init__(self, data: Dict[str, Any], *, content_type: str, status: int, exception: Optional[Exception] = None) -> None:
        super().__init__('', content_type=content_type, status=status)
        self._data: Dict[str, Any] = data
        self._needs_serialization: bool = True
        self.exception: Optional[Exception] = exception

    def get_data(self) -> Dict[str, Any]:
        self._needs_serialization = True
        return self._data

    @override
    @property
    def content(self) -> bytes:
        if self._needs_serialization:
            self.content = orjson.dumps(self._data, option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_PASSTHROUGH_DATETIME)
        return super().content

    @content.setter
    def content(self, value: bytes) -> None:
        assert isinstance(HttpResponse.content, property)
        assert HttpResponse.content.fset is not None
        HttpResponse.content.fset(self, value)
        self._needs_serialization = False

    @override
    def __iter__(self) -> Iterator[bytes]:
        return iter([self.content])

def json_unauthorized(message: Optional[str] = None, www_authenticate: Optional[str] = None) -> MutableJsonResponse:
    return json_response_from_error(UnauthorizedError(msg=message, www_authenticate=www_authenticate))

def json_method_not_allowed(methods: List[str]) -> HttpResponseNotAllowed:
    resp = HttpResponseNotAllowed(methods)
    resp.content = orjson.dumps({'result': 'error', 'msg': 'Method Not Allowed', 'allowed_methods': methods})
    return resp

def json_response(res_type: str = 'success', msg: str = '', data: Dict[str, Any] = {}, status: int = 200, exception: Optional[Exception] = None) -> MutableJsonResponse:
    content = {'result': res_type, 'msg': msg}
    content.update(data)
    return MutableJsonResponse(data=content, content_type='application/json', status=status, exception=exception)

def json_success(request: HttpRequest, data: Dict[str, Any] = {}) -> MutableJsonResponse:
    return json_response(data=data)

def json_response_from_error(exception: JsonableError) -> MutableJsonResponse:
    response_type = 'error'
    if 200 <= exception.http_status_code < 300:
        response_type = 'success'
    response = json_response(response_type, msg=exception.msg, data=exception.data, status=exception.http_status_code, exception=exception)
    for header, value in exception.extra_headers.items():
        response[header] = value
    return response

class AsynchronousResponse(HttpResponse):
    status_code: int = 399
