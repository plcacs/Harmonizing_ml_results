"""Redis cache backend."""
import socket
import typing
from enum import Enum
from typing import Any, Mapping, Optional, Type, Union
from mode.utils.compat import want_bytes
from mode.utils.objects import cached_property
from yarl import URL
from faust.exceptions import ImproperlyConfigured
from faust.types import AppT
from . import base
try:
    import aredis
    import aredis.exceptions
except ImportError:
    aredis = None
if typing.TYPE_CHECKING:
    from aredis import StrictRedis as _RedisClientT
else:

    class _RedisClientT:
        ...

class RedisScheme(Enum):
    """Types of Redis configurations."""
    SINGLE_NODE = 'redis'
    CLUSTER = 'rediscluster'

class CacheBackend(base.CacheBackend):
    """Backend for cache operations using Redis."""
    _client = None
    if aredis is None:
        ...
    else:
        operational_errors = (socket.error, IOError, OSError, aredis.exceptions.ConnectionError, aredis.exceptions.TimeoutError)
        invalidating_errors = (aredis.exceptions.DataError, aredis.exceptions.InvalidResponse, aredis.exceptions.ResponseError)
        irrecoverable_errors = (aredis.exceptions.AuthenticationError,)

    def __init__(self, app: Union[faustypes.AppT, yarl.URL, str], url: Union[faustypes.AppT, yarl.URL, str], *, connect_timeout: Union[None, int, float]=None, stream_timeout: Union[None, int]=None, max_connections: Union[None, int, str]=None, max_connections_per_node: Union[None, int, float]=None, **kwargs) -> None:
        super().__init__(app, url, **kwargs)
        self.connect_timeout = connect_timeout
        self.stream_timeout = stream_timeout
        self.max_connections = max_connections
        self.max_connections_per_node = max_connections_per_node
        self._client_by_scheme = self._init_schemes()

    def _init_schemes(self) -> dict:
        if aredis is None:
            return {}
        else:
            return {RedisScheme.SINGLE_NODE.value: aredis.StrictRedis, RedisScheme.CLUSTER.value: aredis.StrictRedisCluster}

    async def _get(self, key):
        value = await self.client.get(key)
        if value is not None:
            return want_bytes(value)
        return None

    async def _set(self, key, value, timeout=None):
        if timeout is not None:
            await self.client.setex(key, int(timeout), value)
        else:
            await self.client.set(key, value)

    async def _delete(self, key):
        await self.client.delete(key)

    async def on_start(self):
        """Call when Redis backend starts."""
        if aredis is None:
            raise ImproperlyConfigured('Redis cache backend requires `pip install aredis`')
        await self.connect()

    async def connect(self):
        """Connect to Redis/Redis Cluster server."""
        if self._client is None:
            self._client = self._new_client()
        await self.client.ping()

    def _new_client(self):
        return self._client_from_url_and_query(self.url, **self.url.query)

    def _client_from_url_and_query(self, url: Union[str, yarl.URL], *, connect_timeout: Union[None, str, yarl.URL, int]=None, stream_timeout: Union[None, str, yarl.URL, int]=None, max_connections: Union[None, str, yarl.URL, int]=None, max_connections_per_node: Union[None, str, yarl.URL, int]=None, **kwargs):
        Client = self._client_by_scheme[url.scheme]
        return Client(**self._prepare_client_kwargs(url, host=url.host, port=url.port, db=self._db_from_path(url.path), password=url.password, connect_timeout=self._float_from_str(connect_timeout, self.connect_timeout), stream_timeout=self._float_from_str(stream_timeout, self.stream_timeout), max_connections=self._int_from_str(max_connections, self.max_connections), max_connections_per_node=self._int_from_str(max_connections_per_node, self.max_connections_per_node), skip_full_coverage_check=True))

    def _prepare_client_kwargs(self, url: str, **kwargs):
        if url.scheme == RedisScheme.CLUSTER.value:
            return self._as_cluster_kwargs(**kwargs)
        return kwargs

    def _as_cluster_kwargs(self, db: None=None, **kwargs):
        return kwargs

    def _int_from_str(self, val: Any=None, default: Union[None, str, int, float]=None) -> Union[int, None, str, float]:
        return int(val) if val else default

    def _float_from_str(self, val: Union[None, str, float, int]=None, default: Union[None, str, float, int]=None) -> Union[float, None, str, int]:
        return float(val) if val else default

    def _db_from_path(self, path: norfs.fs.base.Path) -> int:
        if not path or path == '/':
            return 0
        try:
            return int(path.strip('/'))
        except ValueError:
            raise ValueError(f'Database is int between 0 and limit - 1, not {path!r}')

    @cached_property
    def client(self) -> None:
        """Return Redis client instance."""
        if self._client is None:
            raise RuntimeError('Cache backend not started')
        return self._client