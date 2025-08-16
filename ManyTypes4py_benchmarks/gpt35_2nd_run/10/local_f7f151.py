from typing import List, Any, Dict, Tuple, Callable, Optional, Union

MatchResult = namedtuple('MatchResult', ['route', 'captured', 'query_params'])
EventType = Dict[str, Any]
ContextType = Dict[str, Any]
HeaderType = Dict[str, Any]
ResponseType = Dict[str, Any]
HandlerCls = Callable[..., 'ChaliceRequestHandler']
ServerCls = Callable[..., 'HTTPServer']

class Clock(object):

    def time(self) -> float:
        return time.time()

def create_local_server(app_obj: Chalice, config: Config, host: str, port: int) -> LocalDevServer:

class LocalARNBuilder(object):

    def build_arn(self, method: str, path: str) -> str:

class ARNMatcher(object):

    def __init__(self, target_arn: str):

    def does_any_resource_match(self, resources: List[str]) -> bool:

class RouteMatcher(object):

    def __init__(self, route_urls: List[str]):

    def match_route(self, url: str) -> MatchResult:

class LambdaEventConverter(object):

    def __init__(self, route_matcher: RouteMatcher, binary_types: Optional[List[str]] = None):

    def create_lambda_event(self, method: str, path: str, headers: HeaderType, body: Optional[bytes] = None) -> EventType:

class LocalGatewayAuthorizer(object):

    def __init__(self, app_object: Chalice):

    def authorize(self, raw_path: str, lambda_event: EventType, lambda_context: LambdaContext) -> LocalAuthPair:

class LocalGateway(object):

    def __init__(self, app_object: Chalice, config: Config):

    def handle_request(self, method: str, path: str, headers: HeaderType, body: Optional[bytes]) -> ResponseType:

class ChaliceRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, request, client_address, server, app_object: Chalice, config: Config):

    def _parse_payload(self) -> Tuple[HeaderType, Optional[bytes]]:

    def _generic_handle(self):

    def _handle_binary(self, response: ResponseType) -> ResponseType:

    def _send_error_response(self, error: LocalGatewayException):

    def _send_http_response(self, code: int, headers: HeaderType, body: Optional[bytes]):

    def _send_http_response_with_body(self, code: int, headers: HeaderType, body: bytes):

    def _send_http_response_no_body(self, code: int, headers: HeaderType):

    def _send_headers(self, headers: HeaderType):

class LocalDevServer(object):

    def __init__(self, app_object: Chalice, config: Config, host: str, port: int, handler_cls: HandlerCls = ChaliceRequestHandler, server_cls: ServerCls = ThreadedHTTPServer):

    def handle_single_request(self):

    def serve_forever(self):

    def shutdown(self):

class HTTPServerThread(threading.Thread):

    def __init__(self, server_factory: Callable[[], LocalDevServer]):

    def run(self):

    def shutdown(self):

class LocalChalice(Chalice):

    @property
    def current_request(self) -> Any:

    @current_request.setter
    def current_request(self, value: Any):

class CustomLocalChalice(LocalChalice):
