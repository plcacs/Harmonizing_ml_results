from typing import Any, Callable, DefaultDict, Dict, Iterable, Iterator, Mapping, MutableMapping, NamedTuple, Optional, Set, Tuple, Union

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

    def __init__(self, max_open_files: Optional[int] = None, write_buffer_size: Optional[int] = None, max_write_buffer_number: Optional[int] = None, target_file_size_base: Optional[int] = None, block_cache_size: Optional[int] = None, block_cache_compressed_size: Optional[int] = None, bloom_filter_size: Optional[int] = None, **kwargs: Any) -> None:
        ...

    def open(self, path: Path, *, read_only: bool = False) -> rocksdb.DB:
        ...

    def as_options(self) -> rocksdb.Options:
        ...

class Store(base.SerializedStore):
    offset_key: bytes

    def __init__(self, url: URL, app: AppT, table: CollectionT, *, key_index_size: Optional[int] = None, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        ...

    def persisted_offset(self, tp: TP) -> Optional[int]:
        ...

    def set_persisted_offset(self, tp: TP, offset: int) -> None:
        ...

    async def need_active_standby_for(self, tp: TP) -> bool:
        ...

    def apply_changelog_batch(self, batch: Iterable[EventT], to_key: Callable, to_value: Callable) -> None:
        ...

    def _set(self, key: bytes, value: Any) -> None:
        ...

    def _db_for_partition(self, partition: int) -> rocksdb.DB:
        ...

    def _open_for_partition(self, partition: int) -> rocksdb.DB:
        ...

    def _get(self, key: bytes) -> Any:
        ...

    def _get_bucket_for_key(self, key: bytes) -> Union[_DBValueTuple, None]:
        ...

    def _del(self, key: bytes) -> None:
        ...

    async def on_rebalance(self, table: Any, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
        ...

    def revoke_partitions(self, table: Any, tps: Set[TP]) -> None:
        ...

    async def assign_partitions(self, table: Any, tps: Set[TP]) -> None:
        ...

    async def _try_open_db_for_partition(self, partition: int, max_retries: int = 5, retry_delay: float = 1.0) -> rocksdb.DB:
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

    def _clear(self) -> None:
        ...

    def reset_state(self) -> None:
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
