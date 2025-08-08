    def __init__(self, timeout: Optional[Seconds] = None, include_headers: bool = False, key_prefix: Optional[str] = None, backend: Optional[CacheBackendT] = None, **kwargs: Any) -> None:
    def view(self, timeout: Optional[Seconds] = None, include_headers: bool = False, key_prefix: Optional[str] = None, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    async def get_view(self, key: str, view: View) -> Optional[Response]:
    def _view_backend(self, view: View) -> CacheBackendT:
    async def set_view(self, key: str, view: View, response: Response, timeout: Optional[Seconds] = None) -> None:
    def can_cache_request(self, request: Request) -> bool:
    def can_cache_response(self, request: Request, response: Response) -> bool:
    def key_for_request(self, request: Request, prefix: Optional[str] = None, method: Optional[str] = None, include_headers: bool = False) -> str:
    def build_key(self, request: Request, method: str, prefix: str, headers: Mapping[str, str]) -> str:
