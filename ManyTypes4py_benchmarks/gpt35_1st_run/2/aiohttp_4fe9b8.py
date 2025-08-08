    def __init__(self) -> None:
    def request(self, method: str, url: str, *, auth: Any = None, status: HTTPStatus = HTTPStatus.OK, text: str = None, data: Any = None, content: Any = None, json: Any = None, params: Any = None, headers: Any = None, exc: Exception = None, cookies: Any = None, side_effect: Any = None, closing: Any = None) -> None:
    def get(self, *args: Any, **kwargs: Any) -> None:
    def put(self, *args: Any, **kwargs: Any) -> None:
    def post(self, *args: Any, **kwargs: Any) -> None:
    def delete(self, *args: Any, **kwargs: Any) -> None:
    def options(self, *args: Any, **kwargs: Any) -> None:
    def patch(self, *args: Any, **kwargs: Any) -> None:
    @property
    def call_count(self) -> int:
    def clear_requests(self) -> None:
    def create_session(self, loop: Any) -> ClientSession:
    async def match_request(self, method: str, url: str, *, data: Any = None, auth: Any = None, params: Any = None, headers: Any = None, allow_redirects: Any = None, timeout: Any = None, json: Any = None, cookies: Any = None, **kwargs: Any) -> AiohttpClientMockResponse:
    def __init__(self, method: str, url: str, status: HTTPStatus = HTTPStatus.OK, response: Any = None, json: Any = None, text: str = None, cookies: Any = None, exc: Exception = None, headers: Any = None, side_effect: Any = None, closing: Any = None) -> None:
    def match_request(self, method: str, url: str, params: Any = None) -> bool:
    @property
    def headers(self) -> CIMultiDict:
    @property
    def cookies(self) -> dict:
    @property
    def url(self) -> URL:
    @property
    def content_type(self) -> str:
    @property
    def content(self) -> StreamReader:
    async def read(self) -> Any:
    async def text(self, encoding: str = 'utf-8', errors: str = 'strict') -> str:
    async def json(self, encoding: str = 'utf-8', content_type: Any = None, loads: Any = json_loads) -> Any:
    def release(self) -> None:
    def raise_for_status(self) -> None:
    def close(self) -> None:
    async def wait_for_close(self) -> None:
    @property
    def response(self) -> bytes:
    async def __aenter__(self) -> AiohttpClientMockResponse:
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
def mock_aiohttp_client() -> Iterator[AiohttpClientMocker]:
    def create_session(hass: HomeAssistant, *args: Any, **kwargs: Any) -> ClientSession:
    def __init__(self) -> None:
    async def __call__(self, method: str, url: str, data: Any) -> AiohttpClientMockResponse:
    def queue_response(self, **kwargs: Any) -> None:
    def stop(self) -> None:
