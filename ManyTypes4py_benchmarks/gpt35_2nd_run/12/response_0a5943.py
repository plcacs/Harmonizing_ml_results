    def __init__(self, data: Any, *, content_type: str, status: int, exception: JsonableError = None) -> None:
    def get_data(self) -> Any:
    def content(self) -> bytes:
    def content(self, value: bytes) -> None:
    def __iter__(self) -> Iterator[bytes]:
def json_unauthorized(message: str = None, www_authenticate: str = None) -> MutableJsonResponse:
def json_method_not_allowed(methods: list[str]) -> HttpResponseNotAllowed:
def json_response(res_type: str = 'success', msg: str = '', data: dict[str, Any] = {}, status: int = 200, exception: JsonableError = None) -> MutableJsonResponse:
def json_success(request: HttpRequest, data: dict[str, Any] = {}) -> MutableJsonResponse:
def json_response_from_error(exception: JsonableError) -> MutableJsonResponse:
