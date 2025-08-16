    def request(self, method: str, url: str, *, auth: Any = None, status: HTTPStatus = HTTPStatus.OK, text: str = None, data: Any = None, content: Any = None, json: Any = None, params: Any = None, headers: Any = None, exc: Exception = None, cookies: Any = None, side_effect: Any = None, closing: Any = None) -> None:

    def match_request(self, method: str, url: str, *, data: Any = None, auth: Any = None, params: Any = None, headers: Any = None, allow_redirects: Any = None, timeout: Any = None, json: Any = None, cookies: Any = None, **kwargs: Any) -> AiohttpClientMockResponse:

    def match_request(self, method: str, url: URL, params: Any = None) -> bool:

    def __init__(self, method: str, url: URL, status: HTTPStatus = HTTPStatus.OK, response: Any = None, json: Any = None, text: str = None, cookies: Any = None, exc: Exception = None, headers: Any = None, side_effect: Any = None, closing: Any = None) -> None:

    def text(self, encoding: str = 'utf-8', errors: str = 'strict') -> str:

    def json(self, encoding: str = 'utf-8', content_type: Any = None, loads: Any = json_loads) -> Any:

    def __aenter__(self) -> 'AiohttpClientMockResponse':

    def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:

    def create_session(self, loop: Any) -> ClientSession:

    def __init__(self) -> None:

    def __call__(self, method: str, url: str, data: Any) -> AiohttpClientMockResponse:

    def queue_response(self, **kwargs: Any) -> None:
