from typing import Any

def record_request_stop_data(log_data: dict[str, Any]) -> None:
    ...

def async_request_timer_stop(request: Any) -> None:
    ...

def record_request_restart_data(log_data: dict[str, Any]) -> None:
    ...

def async_request_timer_restart(request: Any) -> None:
    ...

def record_request_start_data(log_data: dict[str, Any]) -> None:
    ...

def timedelta_ms(timedelta: float) -> float:
    ...

def format_timedelta(timedelta: float) -> str:
    ...

def is_slow_query(time_delta: float, path: str) -> bool:
    ...

def write_log_line(log_data: dict[str, Any], path: str, method: str, remote_ip: str, requester_for_logs: str, client_name: str, client_version: str = None, status_code: int = 200, error_content: Any = None) -> None:
    ...

def parse_client(request: Any, *, req_client: Any = None) -> tuple[str, str]:
    ...

class LogRequests(MiddlewareMixin):

    def process_request(self, request: Any) -> None:
        ...

    def process_view(self, request: Any, view_func: Any, args: Any, kwargs: Any) -> None:
        ...

    def process_response(self, request: Any, response: Any) -> Any:
        ...

class JsonErrorHandler(MiddlewareMixin):

    def process_exception(self, request: Any, exception: Exception) -> Any:
        ...

class TagRequests(MiddlewareMixin):

    def process_view(self, request: Any, view_func: Any, args: Any, kwargs: Any) -> None:
        ...

    def process_request(self, request: Any) -> None:
        ...

class CsrfFailureError(JsonableError):

    def __init__(self, reason: str) -> None:
        ...

    @staticmethod
    def msg_format() -> str:
        ...

def csrf_failure(request: Any, reason: str = '') -> Any:
    ...

class LocaleMiddleware(DjangoLocaleMiddleware):

    def process_response(self, request: Any, response: Any) -> Any:
        ...

class RateLimitMiddleware(MiddlewareMixin):

    def set_response_headers(self, response: Any, rate_limit_results: list[RateLimitResult]) -> None:
        ...

    def process_response(self, request: Any, response: Any) -> Any:
        ...

class FlushDisplayRecipientCache(MiddlewareMixin):

    def process_response(self, request: Any, response: Any) -> Any:
        ...

class HostDomainMiddleware(MiddlewareMixin):

    def process_request(self, request: Any) -> Any:
        ...

class SetRemoteAddrFromRealIpHeader(MiddlewareMixin):

    def process_request(self, request: Any) -> None:
        ...

class ProxyMisconfigurationError(JsonableError):

    def __init__(self, proxy_reason: str) -> None:
        ...

    @staticmethod
    def msg_format() -> str:
        ...

class DetectProxyMisconfiguration(MiddlewareMixin):

    def process_view(self, request: Any, view_func: Any, args: Any, kwargs: Any) -> None:
        ...

def validate_scim_bearer_token(request: Any) -> bool:
    ...

class ZulipSCIMAuthCheckMiddleware(SCIMAuthCheckMiddleware):

    def process_request(self, request: Any) -> Any:
        ...
