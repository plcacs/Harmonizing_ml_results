from aredis import StrictRedis
from faust.types import AppT
from . import base

class CacheBackend(base.CacheBackend):
    _client: Optional[Union[StrictRedis, None]] = None
    operational_errors: Tuple[Type[Exception], ...]
    invalidating_errors: Tuple[Type[Exception], ...]
    irrecoverable_errors: Tuple[Type[Exception], ...]

    def __init__(self, app: AppT, url: URL, *, connect_timeout: Optional[float] = None, stream_timeout: Optional[float] = None, max_connections: Optional[int] = None, max_connections_per_node: Optional[int] = None, **kwargs: Any) -> None:
        self.connect_timeout: Optional[float] = connect_timeout
        self.stream_timeout: Optional[float] = stream_timeout
        self.max_connections: Optional[int] = max_connections
        self.max_connections_per_node: Optional[int] = max_connections_per_node
        self._client_by_scheme: Mapping[str, Type[StrictRedis]] = self._init_schemes()

    def _init_schemes(self) -> Mapping[str, Type[StrictRedis]]:
        ...

    async def _get(self, key: str) -> Optional[bytes]:
        ...

    async def _set(self, key: str, value: bytes, timeout: Optional[int] = None) -> None:
        ...

    async def _delete(self, key: str) -> None:
        ...

    async def on_start(self) -> None:
        ...

    async def connect(self) -> None:
        ...

    def _new_client(self) -> StrictRedis:
        ...

    def _client_from_url_and_query(self, url: URL, *, connect_timeout: Optional[float] = None, stream_timeout: Optional[float] = None, max_connections: Optional[int] = None, max_connections_per_node: Optional[int] = None, **kwargs: Any) -> StrictRedis:
        ...

    def _prepare_client_kwargs(self, url: URL, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def _as_cluster_kwargs(self, db: Optional[int] = None, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def _int_from_str(self, val: Optional[str] = None, default: Optional[int] = None) -> int:
        ...

    def _float_from_str(self, val: Optional[str] = None, default: Optional[float] = None) -> float:
        ...

    def _db_from_path(self, path: str) -> int:
        ...

    @cached_property
    def client(self) -> StrictRedis:
        ...
