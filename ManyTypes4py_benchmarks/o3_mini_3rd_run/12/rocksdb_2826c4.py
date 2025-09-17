#!/usr/bin/env python3
"""RocksDB storage."""
import asyncio
import gc
import math
import shutil
import typing
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, Iterator, Mapping, MutableMapping, NamedTuple, Optional, Set, Tuple, Union, cast

from mode.utils.collections import LRUCache
from yarl import URL
from faust.exceptions import ImproperlyConfigured
from faust.streams import current_event
from faust.types import AppT, CollectionT, EventT, TP
from faust.utils import platforms
from . import base

_max_open_files = platforms.max_open_files()
if _max_open_files is not None:
    _max_open_files = math.ceil(_max_open_files * 0.9)
DEFAULT_MAX_OPEN_FILES: Optional[int] = _max_open_files
DEFAULT_WRITE_BUFFER_SIZE: int = 67108864
DEFAULT_MAX_WRITE_BUFFER_NUMBER: int = 3
DEFAULT_TARGET_FILE_SIZE_BASE: int = 67108864
DEFAULT_BLOCK_CACHE_SIZE: int = 2 * 1024 ** 3
DEFAULT_BLOCK_CACHE_COMPRESSED_SIZE: int = 500 * 1024 ** 2
DEFAULT_BLOOM_FILTER_SIZE: int = 3

try:
    import rocksdb  # type: ignore
except ImportError:
    rocksdb = None

if typing.TYPE_CHECKING:
    from rocksdb import DB, Options  # type: ignore
else:

    class DB:
        """Dummy DB."""
        def get(self, key: bytes) -> Optional[bytes]:
            ...
        def put(self, key: bytes, value: bytes) -> None:
            ...
        def delete(self, key: bytes) -> None:
            ...
        def write(self, batch: Any) -> None:
            ...
        def iterkeys(self) -> Any:
            ...
        def iteritems(self) -> Any:
            ...
        def key_may_exist(self, key: bytes) -> Tuple[bool, Optional[bytes]]:
            return (False, None)

    class Options:
        """Dummy Options."""
        ...

class PartitionDB(NamedTuple):
    """Tuple of ``(partition, rocksdb.DB)``."""
    partition: int
    db: DB

class _DBValueTuple(NamedTuple):
    db: DB
    value: bytes

class RocksDBOptions:
    """Options required to open a RocksDB database."""
    max_open_files: Optional[int] = DEFAULT_MAX_OPEN_FILES
    write_buffer_size: int = DEFAULT_WRITE_BUFFER_SIZE
    max_write_buffer_number: int = DEFAULT_MAX_WRITE_BUFFER_NUMBER
    target_file_size_base: int = DEFAULT_TARGET_FILE_SIZE_BASE
    block_cache_size: int = DEFAULT_BLOCK_CACHE_SIZE
    block_cache_compressed_size: int = DEFAULT_BLOCK_CACHE_COMPRESSED_SIZE
    bloom_filter_size: int = DEFAULT_BLOOM_FILTER_SIZE

    def __init__(
        self,
        max_open_files: Optional[int] = None,
        write_buffer_size: Optional[int] = None,
        max_write_buffer_number: Optional[int] = None,
        target_file_size_base: Optional[int] = None,
        block_cache_size: Optional[int] = None,
        block_cache_compressed_size: Optional[int] = None,
        bloom_filter_size: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        if max_open_files is not None:
            self.max_open_files = max_open_files
        if write_buffer_size is not None:
            self.write_buffer_size = write_buffer_size
        if max_write_buffer_number is not None:
            self.max_write_buffer_number = max_write_buffer_number
        if target_file_size_base is not None:
            self.target_file_size_base = target_file_size_base
        if block_cache_size is not None:
            self.block_cache_size = block_cache_size
        if block_cache_compressed_size is not None:
            self.block_cache_compressed_size = block_cache_compressed_size
        if bloom_filter_size is not None:
            self.bloom_filter_size = bloom_filter_size
        self.extra_options: Dict[str, Any] = kwargs

    def open(self, path: Path, *, read_only: bool = False) -> DB:
        """Open RocksDB database using this configuration."""
        assert rocksdb is not None, "rocksdb library not available"
        return rocksdb.DB(str(path), self.as_options(), read_only=read_only)

    def as_options(self) -> Any:
        """Return :class:`rocksdb.Options` object using this configuration."""
        assert rocksdb is not None, "rocksdb library not available"
        return rocksdb.Options(
            create_if_missing=True,
            max_open_files=self.max_open_files,
            write_buffer_size=self.write_buffer_size,
            max_write_buffer_number=self.max_write_buffer_number,
            target_file_size_base=self.target_file_size_base,
            table_factory=rocksdb.BlockBasedTableFactory(
                filter_policy=rocksdb.BloomFilterPolicy(self.bloom_filter_size),
                block_cache=rocksdb.LRUCache(self.block_cache_size),
                block_cache_compressed=rocksdb.LRUCache(self.block_cache_compressed_size),
            ),
            **self.extra_options
        )

class Store(base.SerializedStore):
    """RocksDB table storage."""
    offset_key: bytes = b'__faust\x00offset__'

    def __init__(
        self,
        url: URL,
        app: AppT,
        table: Any,
        *,
        key_index_size: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        if rocksdb is None:
            error = ImproperlyConfigured('RocksDB bindings not installed? pip install python-rocksdb')
            try:
                import rocksdb  # type: ignore
            except Exception as exc:
                raise error from exc
            else:
                raise error
        super().__init__(url, app, table, **kwargs)
        if not self.url.path:
            self.url /= self.table_name  # type: ignore
        self.options: Mapping[str, Any] = options or {}
        self.rocksdb_options: RocksDBOptions = RocksDBOptions(**self.options)
        if key_index_size is None:
            key_index_size = app.conf.table_key_index_size  # type: ignore
        self.key_index_size: int = key_index_size
        self._dbs: Dict[int, DB] = {}
        self._key_index: LRUCache = LRUCache(limit=self.key_index_size)

    def persisted_offset(self, tp: TP) -> Optional[int]:
        """Return the last persisted offset.

        See :meth:`set_persisted_offset`.
        """
        db: DB = self._db_for_partition(tp.partition)
        offset: Optional[bytes] = db.get(self.offset_key)
        if offset is not None:
            return int(offset)
        return None

    def set_persisted_offset(self, tp: TP, offset: int) -> None:
        """Set the last persisted offset for this table.

        This will remember the last offset that we wrote to RocksDB,
        so that on rebalance/recovery we can seek past this point
        to only read the events that occurred recently while
        we were not an active replica.
        """
        self._db_for_partition(tp.partition).put(self.offset_key, str(offset).encode())

    async def need_active_standby_for(self, tp: TP) -> bool:
        """Decide if an active standby is needed for this topic partition.

        Since other workers may be running on the same local machine,
        we can decide to not actively read standby messages, since
        that database file is already being populated.
        """
        try:
            self._db_for_partition(tp.partition)
        except rocksdb.errors.RocksIOError as exc:  # type: ignore
            if 'lock' not in repr(exc):
                raise
            return False
        else:
            return True

    def apply_changelog_batch(
        self,
        batch: Iterable[EventT],
        to_key: Callable[[Any], Any],
        to_value: Callable[[Any], Any]
    ) -> None:
        """Write batch of changelog events to local RocksDB storage.

        Arguments:
            batch: Iterable of changelog events (:class:`faust.Event`)
            to_key: A callable you can use to deserialize the key
                of a changelog event.
            to_value: A callable you can use to deserialize the value
                of a changelog event.
        """
        batches: DefaultDict[int, Any] = defaultdict(rocksdb.WriteBatch)  # type: ignore
        tp_offsets: Dict[TP, int] = {}
        for event in batch:
            msg = event.message
            tp: TP = msg.tp
            offset: int = msg.offset
            tp_offsets[tp] = offset if tp not in tp_offsets else max(offset, tp_offsets[tp])
            if msg.value is None:
                batches[msg.partition].delete(msg.key)
            else:
                batches[msg.partition].put(msg.key, msg.value)
        for partition, batch_ in batches.items():
            self._db_for_partition(partition).write(batch_)
        for tp, offset in tp_offsets.items():
            self.set_persisted_offset(tp, offset)

    def _set(self, key: bytes, value: bytes) -> None:
        event: Optional[EventT] = current_event()
        assert event is not None
        partition: int = event.message.partition
        db: DB = self._db_for_partition(partition)
        self._key_index[key] = partition
        db.put(key, value)

    def _db_for_partition(self, partition: int) -> DB:
        try:
            return self._dbs[partition]
        except KeyError:
            db: DB = self._dbs[partition] = self._open_for_partition(partition)
            return db

    def _open_for_partition(self, partition: int) -> DB:
        return self.rocksdb_options.open(self.partition_path(partition))

    def _get(self, key: bytes) -> Optional[bytes]:
        dbvalue: Optional[_DBValueTuple] = self._get_bucket_for_key(key)
        if dbvalue is None:
            return None
        db, value = dbvalue
        if value is None:
            if db.key_may_exist(key)[0]:
                return db.get(key)
        return value

    def _get_bucket_for_key(self, key: bytes) -> Optional[_DBValueTuple]:
        try:
            partition: int = self._key_index[key]
            dbs: Iterable[PartitionDB] = [PartitionDB(partition, self._dbs[partition])]
        except KeyError:
            dbs = cast(Iterable[PartitionDB], self._dbs.items())
        for partition, db in dbs:
            if db.key_may_exist(key)[0]:
                value: Optional[bytes] = db.get(key)
                if value is not None:
                    self._key_index[key] = partition
                    return _DBValueTuple(db, value)
        return None

    def _del(self, key: bytes) -> None:
        for db in self._dbs_for_key(key):
            db.delete(key)

    async def on_rebalance(
        self,
        table: Any,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP]
    ) -> None:
        """Rebalance occurred.

        Arguments:
            table: The table that we store data for.
            assigned: Set of all assigned topic partitions.
            revoked: Set of newly revoked topic partitions.
            newly_assigned: Set of newly assigned topic partitions,
                for which we were not assigned the last time.
        """
        self.revoke_partitions(table, revoked)
        await self.assign_partitions(table, newly_assigned)

    def revoke_partitions(self, table: Any, tps: Set[TP]) -> None:
        """De-assign partitions used on this worker instance.

        Arguments:
            table: The table that we store data for.
            tps: Set of topic partitions that we should no longer
                be serving data for.
        """
        dbs_closed: int = 0
        for tp in tps:
            if tp.topic in table.changelog_topic.topics:
                db = self._dbs.pop(tp.partition, None)
                if db is not None:
                    del db
                    dbs_closed += 1
        if dbs_closed:
            gc.collect()

    async def assign_partitions(self, table: Any, tps: Set[TP]) -> None:
        """Assign partitions to this worker instance.

        Arguments:
            table: The table that we store data for.
            tps: Set of topic partitions we have been assigned.
        """
        standby_tps: Set[TP] = self.app.assignor.assigned_standbys()  # type: ignore
        my_topics: Set[str] = table.changelog_topic.topics
        for tp in tps:
            if tp.topic in my_topics and tp not in standby_tps:
                await self._try_open_db_for_partition(tp.partition)
                await asyncio.sleep(0)

    async def _try_open_db_for_partition(
        self,
        partition: int,
        max_retries: int = 5,
        retry_delay: float = 1.0
    ) -> DB:
        for i in range(max_retries):
            try:
                return self._db_for_partition(partition)
            except rocksdb.errors.RocksIOError as exc:  # type: ignore
                if i == max_retries - 1 or 'lock' not in repr(exc):
                    raise
                self.log.info('DB for partition %r is locked! Retry in 1s...', partition)
                await asyncio.sleep(retry_delay)
        # Will never reach here since loop either returns or raises
        raise RuntimeError("Unable to open DB for partition")

    def _contains(self, key: bytes) -> bool:
        for db in self._dbs_for_key(key):
            if db.key_may_exist(key)[0] and db.get(key) is not None:
                return True
        return False

    def _dbs_for_key(self, key: bytes) -> Iterable[DB]:
        try:
            return [self._dbs[self._key_index[key]]]
        except KeyError:
            return self._dbs.values()

    def _dbs_for_actives(self) -> Iterator[DB]:
        actives: Set[TP] = self.app.assignor.assigned_actives()  # type: ignore
        topic: str = self.table._changelog_topic_name()  # type: ignore
        for partition, db in self._dbs.items():
            tp = TP(topic=topic, partition=partition)  # type: ignore
            if tp in actives or self.table.is_global:
                yield db

    def _size(self) -> int:
        return sum((self._size1(db) for db in self._dbs_for_actives()))

    def _visible_keys(self, db: DB) -> Iterator[bytes]:
        it = db.iterkeys()
        it.seek_to_first()
        for key in it:
            if key != self.offset_key:
                yield key

    def _visible_items(self, db: DB) -> Iterator[Tuple[bytes, bytes]]:
        it = db.iteritems()
        it.seek_to_first()
        for key, value in it:
            if key != self.offset_key:
                yield (key, value)

    def _visible_values(self, db: DB) -> Iterator[bytes]:
        for _, value in self._visible_items(db):
            yield value

    def _size1(self, db: DB) -> int:
        return sum((1 for _ in self._visible_keys(db)))

    def _iterkeys(self) -> Iterator[bytes]:
        for db in self._dbs_for_actives():
            yield from self._visible_keys(db)

    def _itervalues(self) -> Iterator[bytes]:
        for db in self._dbs_for_actives():
            yield from self._visible_values(db)

    def _iteritems(self) -> Iterator[Tuple[bytes, bytes]]:
        for db in self._dbs_for_actives():
            yield from self._visible_items(db)

    def _clear(self) -> None:
        raise NotImplementedError('TODO')

    def reset_state(self) -> None:
        """Remove all data stored in this table.

        Notes:
            Only local data will be removed, table changelog partitions
            in Kafka will not be affected.
        """
        self._dbs.clear()
        self._key_index.clear()
        with suppress(FileNotFoundError):
            shutil.rmtree(self.path.absolute())

    def partition_path(self, partition: int) -> Path:
        """Return :class:`pathlib.Path` to db file of specific partition."""
        p: Path = self.path / self.basename
        return self._path_with_suffix(p.with_name(f'{p.name}-{partition}'))

    def _path_with_suffix(self, path: Path, *, suffix: str = '.db') -> Path:
        return path.with_name(f'{path.name}{suffix}')

    @property
    def path(self) -> Path:
        """Path to directory where tables are stored.

        See Also:
            :setting:`tabledir` (default value for this path).

        Returns:
            :class:`pathlib.Path`.
        """
        return self.app.conf.tabledir  # type: ignore

    @property
    def basename(self) -> Path:
        """Return the name of this table, used as filename prefix."""
        return Path(self.url.path)
