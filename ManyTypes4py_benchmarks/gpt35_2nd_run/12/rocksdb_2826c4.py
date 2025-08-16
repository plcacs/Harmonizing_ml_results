from typing import Any, Callable, DefaultDict, Dict, Iterable, Iterator, Mapping, MutableMapping, NamedTuple, Optional, Set, Tuple, Union
from mode.utils.collections import LRUCache
from yarl import URL
from faust.exceptions import ImproperlyConfigured
from faust.streams import current_event
from faust.types import AppT, CollectionT, EventT, TP
from faust.utils import platforms

class PartitionDB(NamedTuple):
    partition: int
    db: rocksdb.DB

class _DBValueTuple(NamedTuple):
    db: rocksdb.DB
    value: Any

class RocksDBOptions:
    max_open_files: int
    write_buffer_size: int
    max_write_buffer_number: int
    target_file_size_base: int
    block_cache_size: int
    block_cache_compressed_size: int
    bloom_filter_size: int

    def __init__(self, max_open_files=None, write_buffer_size=None, max_write_buffer_number=None, target_file_size_base=None, block_cache_size=None, block_cache_compressed_size=None, bloom_filter_size=None, **kwargs):
        ...

    def open(self, path, *, read_only=False) -> rocksdb.DB:
        ...

    def as_options(self) -> rocksdb.Options:
        ...

class Store(base.SerializedStore):
    offset_key: bytes

    def __init__(self, url: URL, app: AppT, table: CollectionT, *, key_index_size: Optional[int] = None, options: Optional[Dict[str, Any]] = None, **kwargs):
        ...

    def persisted_offset(self, tp: TP) -> Optional[int]:
        ...

    def set_persisted_offset(self, tp: TP, offset: int):
        ...

    async def need_active_standby_for(self, tp: TP) -> bool:
        ...

    def apply_changelog_batch(self, batch: Iterable[EventT], to_key: Callable, to_value: Callable):
        ...

    def _set(self, key: bytes, value: Any):
        ...

    def _db_for_partition(self, partition: int) -> rocksdb.DB:
        ...

    def _open_for_partition(self, partition: int) -> rocksdb.DB:
        ...

    def _get(self, key: bytes) -> Any:
        ...

    def _get_bucket_for_key(self, key: bytes) -> Union[_DBValueTuple, None]:
        ...

    def _del(self, key: bytes):
        ...

    async def on_rebalance(self, table: CollectionT, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]):
        ...

    def revoke_partitions(self, table: CollectionT, tps: Set[TP]):
        ...

    async def assign_partitions(self, table: CollectionT, tps: Set[TP]):
        ...

    async def _try_open_db_for_partition(self, partition: int, max_retries: int = 5, retry_delay: float = 1.0):
        ...

    def _contains(self, key: bytes) -> bool:
        ...

    def _dbs_for_key(self, key: bytes) -> Iterable[rocksdb.DB]:
        ...

    def _dbs_for_actives(self) -> Iterable[rocksdb.DB]:
        ...

    def _size(self) -> int:
        ...

    def _visible_keys(self, db: rocksdb.DB) -> Iterable[bytes]:
        ...

    def _visible_items(self, db: rocksdb.DB) -> Iterable[Tuple[bytes, Any]]:
        ...

    def _visible_values(self, db: rocksdb.DB) -> Iterable[Any]:
        ...

    def _size1(self, db: rocksdb.DB) -> int:
        ...

    def _iterkeys(self) -> Iterable[bytes]:
        ...

    def _itervalues(self) -> Iterable[Any]:
        ...

    def _iteritems(self) -> Iterable[Tuple[bytes, Any]]:
        ...

    def _clear(self):
        ...

    def reset_state(self):
        ...

    def partition_path(self, partition: int) -> Path:
        ...

    def _path_with_suffix(self, path: Path, *, suffix: str = '.db') -> Path:
        ...

    @property
    def path(self) -> Path:
        ...

    @property
    def basename(self) -> Path:
        ...
