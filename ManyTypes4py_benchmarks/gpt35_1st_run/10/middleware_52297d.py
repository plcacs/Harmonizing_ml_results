from typing import Annotated, Any, Concatenate

def record_request_stop_data(log_data: MutableMapping[str, Any]):
    ...

def async_request_timer_stop(request: HttpRequest):
    ...

def record_request_restart_data(log_data: MutableMapping[str, Any]):
    ...

def async_request_timer_restart(request: HttpRequest):
    ...

def record_request_start_data(log_data: MutableMapping[str, Any]):
    ...

def timedelta_ms(timedelta: float) -> float:
    ...

def format_timedelta(timedelta: float) -> str:
    ...

def is_slow_query(time_delta: float, path: str) -> bool:
    ...

def write_log_line(log_data: MutableMapping[str, Any], path: str, method: str, remote_ip: str, requester_for_logs: str, client_name: str, client_version: str = None, status_code: int = 200, error_content: Any = None):
    ...

def parse_client(request: HttpRequest, *, req_client: str = None) -> Tuple[str, str]:
    ...

class LogRequests(MiddlewareMixin):

    def process_request(self, request: HttpRequest):
        ...

    def process_view(self, request: HttpRequest, view_func: Callable, args: Any, kwargs: Any):
        ...

    def process_response(self, request: HttpRequest, response: HttpResponseBase) -> HttpResponseBase:
        ...

class JsonErrorHandler(MiddlewareMixin):

    def process_exception(self, request: HttpRequest, exception: Exception) -> Optional[HttpResponseBase]:
        ...

class TagRequests(MiddlewareMixin):

    def process_view(self, request: HttpRequest, view_func: Callable, args: Any, kwargs: Any):
        ...

    def process_request(self, request: HttpRequest):
        ...

class CsrfFailureError(JsonableError):

    def __init__(self, reason: str):
        ...

    @staticmethod
    @override
    def msg_format() -> str:
        ...

def csrf_failure(request: HttpRequest, reason: str = '') -> HttpResponse:
    ...

class LocaleMiddleware(DjangoLocaleMiddleware):

    @override
    def process_response(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        ...

class RateLimitMiddleware(MiddlewareMixin):

    def set_response_headers(self, response: HttpResponse, rate_limit_results: List[RateLimitResult]):
        ...

    def process_response(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        ...

class FlushDisplayRecipientCache(MiddlewareMixin):

    def process_response(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        ...

class HostDomainMiddleware(MiddlewareMixin):

    def process_request(self, request: HttpRequest):
        ...

class SetRemoteAddrFromRealIpHeader(MiddlewareMixin):

    def process_request(self, request: HttpRequest):
        ...

class ProxyMisconfigurationError(JsonableError):

    def __init__(self, proxy_reason: str):
        ...

    @staticmethod
    @override
    def msg_format() -> str:
        ...

class DetectProxyMisconfiguration(MiddlewareMixin):

    def process_view(self, request: HttpRequest, view_func: Callable, args: Any, kwargs: Any):
        ...

def validate_scim_bearer_token(request: HttpRequest) -> bool:
    ...

class ZulipSCIMAuthCheckMiddleware(SCIMAuthCheckMiddleware):

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        ...
