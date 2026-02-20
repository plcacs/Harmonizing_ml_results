"""Base class Collection for Table and future data structures."""
import abc
import time

from contextlib import suppress
from collections import defaultdict
from functools import lru_cache
from datetime import datetime
from heapq import heappop, heappush
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    MutableSet,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
    no_type_check,
)

from mode.utils.futures import maybe_async
from mode import Seconds, Service
from yarl import URL

from faust import stores
from faust import joins
from faust.exceptions import PartitionsMismatch
from faust.streams import current_event
from faust.types import (
    AppT,
    CodecArg,
    EventT,
    FieldDescriptorT,
    FutureMessage,
    JoinT,
    RecordMetadata,
    SchemaT,
    TP,
    TopicT,
)
from faust.types.models import ModelArg, ModelT
from faust.types.stores import StoreT
from faust.types.streams import JoinableT, StreamT
from faust.types.tables import (
    ChangelogEventCallback,
    CollectionT,
    RecoverCallback,
    RelativeHandler,
    WindowCloseCallback,
)
from faust.types.windows import WindowRange, WindowT

__all__ = ['Collection']

TABLE_CLEANING = 'CLEANING'

E_SOURCE_PARTITIONS_MISMATCH = '''\
The source topic {source_topic!r} for table {table_name!r}
has {source_n} partitions, but the changelog
topic {change_topic!r} has {change_n} partitions.

Please make sure the topics have the same number of partitions
by configuring Kafka correctly.
'''


class Collection(Service, CollectionT):
    """Base class for changelog-backed data structures stored in Kafka."""

    _store: Optional[URL]
    _changelog_topic: Optional[TopicT]
    _partition_timestamp_keys: MutableMapping[
        Tuple[int, float], MutableSet[Tuple[Any, WindowRange]]]
    _partition_timestamps: MutableMapping[int, List[float]]
    _partition_latest_timestamp: MutableMapping[int, float]
    _recover_callbacks: MutableSet[RecoverCallback]
    _data: Optional[StoreT] = None
    _changelog_compacting: Optional[bool] = True
    _changelog_deleting: Optional[bool] = None

    @abc.abstractmethod
    def _has_key(self, key: Any) -> bool:  # pragma: no cover
        ...

    @abc.abstractmethod
    def _get_key(self, key: Any) -> Any:  # pragma: no cover
        ...

    @abc.abstractmethod
    def _set_key(self, key: Any, value: Any) -> None:  # pragma: no cover
        ...

    @abc.abstractmethod
    def _del_key(self, key: Any) -> None:  # pragma: no cover
        ...

    def __init__(self,
                 app: AppT,
                 *,
                 name: Optional[str] = None,
                 default: Any = None,
                 store: Optional[Union[str, URL]] = None,
                 schema: Optional[SchemaT] = None,
                 key_type: Optional[ModelArg] = None,
                 value_type: Optional[ModelArg] = None,
                 partitions: Optional[int] = None,
                 window: Optional[WindowT] = None,
                 changelog_topic: Optional[TopicT] = None,
                 help: Optional[str] = None,
                 on_recover: Optional[RecoverCallback] = None,
                 on_changelog_event: Optional[ChangelogEventCallback] = None,
                 recovery_buffer_size: int = 1000,
                 standby_buffer_size: Optional[int] = None,
                 extra_topic_configs: Optional[Mapping[str, Any]] = None,
                 recover_callbacks: Optional[Set[RecoverCallback]] = None,
                 options: Optional[Mapping[str, Any]] = None,
                 use_partitioner: bool = False,
                 on_window_close: Optional[WindowCloseCallback] = None,
                 is_global: bool = False,
                 **kwargs: Any) -> None:
        Service.__init__(self, **kwargs)
        self.app: AppT = app
        self.name: str = cast(str, name)  # set lazily so CAN BE NONE!
        self.default: Any = default
        self._store: Optional[URL] = URL(store) if store else None
        self.schema: Optional[SchemaT] = schema
        self.key_type: Optional[ModelArg] = key_type
        self.value_type: Optional[ModelArg] = value_type
        self.partitions: Optional[int] = partitions
        self.window: Optional[WindowT] = window
        self._changelog_topic: Optional[TopicT] = changelog_topic
        self.extra_topic_configs: Mapping[str, Any] = extra_topic_configs or {}
        self.help: str = help or ''
        self._on_changelog_event: Optional[ChangelogEventCallback] = on_changelog_event
        self.recovery_buffer_size: int = recovery_buffer_size
        self.standby_buffer_size: int = standby_buffer_size or recovery_buffer_size
        self.use_partitioner: bool = use_partitioner
        self._on_window_close: Optional[WindowCloseCallback] = on_window_close
        self.last_closed_window: float = 0.0
        self.is_global: bool = is_global
        assert self.recovery_buffer_size > 0 and self.standby_buffer_size > 0

        self.options: Optional[Mapping[str, Any]] = options

        # Setting Serializers from key_type and value_type
        # Possible values json and raw
        # Fallback to json
        self.key_serializer: Optional[CodecArg] = self._serializer_from_type(self.key_type)
        self.value_serializer: Optional[CodecArg] = self._serializer_from_type(self.value_type)

        # Table key expiration
        self._partition_timestamp_keys: MutableMapping[Tuple[int, float], MutableSet[Tuple[Any, WindowRange]]] = defaultdict(set)
        self._partition_timestamps: MutableMapping[int, List[float]] = defaultdict(list)
        self._partition_latest_timestamp: MutableMapping[int, float] = defaultdict(int)

        self._recover_callbacks: MutableSet[RecoverCallback] = set(recover_callbacks or [])
        if on_recover:
            self.on_recover(on_recover)

        # Aliases
        self._sensor_on_get: Callable[[CollectionT, Any], None] = self.app.sensors.on_table_get
        self._sensor_on_set: Callable[[CollectionT, Any, Any], None] = self.app.sensors.on_table_set
        self._sensor_on_del: Callable[[CollectionT, Any], None] = self.app.sensors.on_table_del

    def _serializer_from_type(
            self, typ: Optional[ModelArg]) -> Optional[CodecArg]:
        if typ is bytes:
            return 'raw'
        serializer: Optional[CodecArg] = None
        with suppress(AttributeError):
            serializer = typ._options.serializer  # type: ignore
        return serializer or 'json'

    def __hash__(self) -> int:
        # We have to override MutableMapping __hash__, so that this table
        # can be registered in the app.tables mapping.
        return object.__hash__(self)

    def _new_store(self) -> StoreT:
        return self._new_store_by_url(self._store or self.app.conf.store)

    def _new_store_by_url(self, url: Union[str, URL]) -> StoreT:
        return stores.by_url(url)(
            url, self.app, self,
            table_name=self.name,
            key_type=self.key_type,
            key_serializer=self.key_serializer,
            value_serializer=self.value_serializer,
            value_type=self.value_type,
            loop=self.loop,
            options=self.options,
        )

    @property  # type: ignore
    @no_type_check  # XXX https://github.com/python/mypy/issues/4125
    def data(self) -> StoreT:
        """Underlying table storage."""
        if self._data is None:
            self._data = self._new_store()
        return self._data

    async def on_start(self) -> None:
        """Call when table starts."""
        await self.add_runtime_dependency(self.data)
        await self.changelog_topic.maybe_declare()

    def on_recover(self, fun: RecoverCallback) -> RecoverCallback:
        """Add function as callback to be called on table recovery."""
        assert fun not in self._recover_callbacks
        self._recover_callbacks.add(fun)
        return fun

    def info(self) -> Mapping[str, Any]:
        """Return table attributes as dictionary."""
        # Used to recreate object in .clone()
        return {
            'app': self.app,
            'name': self.name,
            'default': self.default,
            'store': self._store,
            'schema': self.schema,
            'key_type': self.key_type,
            'value_type': self.value_type,
            'partitions': self.partitions,
            'window': self.window,
            'changelog_topic': self._changelog_topic,
            'recover_callbacks': self._recover_callbacks,
            'on_changelog_event': self._on_changelog_event,
            'recovery_buffer_size': self.recovery_buffer_size,
            'standby_buffer_size': self.standby_buffer_size,
            'extra_topic_configs': self.extra_topic_configs,
            'use_partitioner': self.use_partitioner,
        }

    def persisted_offset(self, tp: TP) -> Optional[int]:
        """Return the last persisted offset for topic partition."""
        return self.data.persisted_offset(tp)

    async def need_active_standby_for(self, tp: TP) -> bool:
        """Return :const:`False` if we have access to partition data."""
        return await self.data.need_active_standby_for(tp)

    def reset_state(self) -> None:
        """Reset local state."""
        self.data.reset_state()

    def send_changelog(self,
                       partition: Optional[int],
                       key: Any,
                       value: Any,
                       key_serializer: Optional[CodecArg] = None,
                       value_serializer: Optional[CodecArg] = None) -> FutureMessage:
        """Send modification event to changelog topic."""
        if key_serializer is None:
            key_serializer = self.key_serializer
        if value_serializer is None:
            value_serializer = self.value_serializer
        return self.changelog_topic.send_soon(
            key=key,
            value=value,
            partition=partition,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            callback=self._on_changelog_sent,
            # Ensures final partition number is ready in ret.message.partition
            eager_partitioning=True,
        )

    def _send_changelog(self,
                        event: Optional[EventT],
                        key: Any,
                        value: Any,
                        key_serializer: Optional[CodecArg] = None,
                        value_serializer: Optional[CodecArg] = None) -> None:
        # XXX compat version of send_changelog that needs event argument.
        if event is None:
            raise RuntimeError('Cannot modify table outside of agent/stream.')
        self.send_changelog(
            event.message.partition,
            key, value, key_serializer, value_serializer)

    def partition_for_key(self, key: Any) -> Optional[int]:
        """Return partition number for table key.

        Always returns :const:`None` when :attr:`use_partitioner`
        is enabled.

        Returns:
            Optional[int]: specific partition or :const:`None` if
                the producer should select partition using its partitioner.
        """
        if self.use_partitioner:
            return None
        else:
            event = current_event()
            if event is None:
                raise TypeError(
                    'Cannot modify table key from outside of stream iteration')
            self._verify_source_topic_partitions(event.message.topic)
            return event.message.partition

    @lru_cache()
    def _verify_source_topic_partitions(self, source_topic: str) -> None:
        change_topic = self.changelog_topic_name
        source_n = self.app.consumer.topic_partitions(source_topic)
        if source_n is not None:
            change_n = self.app.consumer.topic_partitions(change_topic)
            if change_n is not None:
                if source_n != change_n:
                    raise PartitionsMismatch(
                        E_SOURCE_PARTITIONS_MISMATCH.format(
                            source_topic=source_topic,
                            table_name=self.name,
                            source_n=source_n,
                            change_topic=change_topic,
                            change_n=change_n,
                        ),
                    )

    def _on_changelog_sent(self, fut: FutureMessage) -> None:
        # This is what keeps the offset in RocksDB so that at startup
        # we know what offsets we already have data for in the database.
        #
        # Kafka Streams has a global ".checkpoint" file, but we store
        # it in each individual RocksDB database file.
        # Every partition in the table will have its own database file,
        #  this is required as partitions can easily move from and to
        #  machine as nodes die and recover.
        res: RecordMetadata = fut.result()
        if self.app.in_transaction:
            # for exactly-once semantics we only write the
            # persisted offset to RocksDB on disk when that partition
            # is committed.
            self.app.tables.persist_offset_on_commit(
                self.data, res.topic_partition, res.offset)
        else:
            # for normal processing (at-least-once) we just write
            # the persisted offset immediately.
            self.data.set_persisted_offset(res.topic_partition, res.offset)

    @Service.task
    @Service.transitions_to(TABLE_CLEANING)
    async def _clean_data(self) -> None:
        interval: Seconds = self.app.conf.table_cleanup_interval
        if self._should_expire_keys():
            await self.sleep(interval)
            async for sleep_time in self.itertimer(
                    interval, name='table_cleanup'):
                await self._del_old_keys()

    async def _del_old_keys(self) -> None:
        window = cast(WindowT, self.window)
        assert window
        for partition, timestamps in self._partition_timestamps.items():
            while timestamps and window.stale(
                    timestamps[0],
                    self._partition_latest_timestamp[partition]):
                timestamp = heappop(timestamps)
                keys_to_remove = self._partition_timestamp_keys.pop(
                    (partition, timestamp), None)
                if keys_to_remove:
                    for key in keys_to_remove:
                        value = self.data.pop(key, None)
                        if key[1][0] > self.last_closed_window:
                            await self.on_window_close(key, value)
                    self.last_closed_window = max(
                        self.last_closed_window,
                        max(key[1][0] for key in keys_to_remove),
                    )

    async def on_window_close(self, key: Tuple[Any, WindowRange], value: Any) -> None:
        if self._on_window_close:
            await maybe_async(self._on_window_close(key, value))

    def _should_expire_keys(self) -> bool:
        window = self.window
        return not (window is None or window.expires is None)

    def _maybe_set_key_ttl(self, key: Tuple[Any, WindowRange], partition: int) -> None:
        if not self._should_expire_keys():
            return
        _, window_range = key
        _, range_end = window_range
        heappush(self._partition_timestamps[partition], range_end)
        self._partition_latest_timestamp[partition] = max(
            self._partition_latest_timestamp[partition], range_end)
        self._partition_timestamp_keys[(partition, range_end)].add(key)

    def _maybe_del_key_ttl(self, key: Tuple[Any, WindowRange], partition: int) -> None:
        if not self._should_expire_keys():
            return
        _, window_range = key
        ts_keys = self._partition_timestamp_keys.get(
            (partition, window_range[1]))
        if ts_keys is not None:
            ts_keys.discard(key)

    def _changelog_topic_name(self) -> str:
        return f'{self.app.conf.id}-{self.name}-changelog'

    def join(self, *fields: FieldDescriptorT) -> StreamT:
        """Right join of this table and another stream/table."""
        return self._join(joins.RightJoin(stream=self, fields=fields))

    def left_join(self, *fields: FieldDescriptorT) -> StreamT:
        """Left join of this table and another stream/table."""
        return self._join(joins.LeftJoin(stream=self, fields=fields))

    def inner_join(self, *fields: FieldDescriptorT) -> StreamT:
        """Inner join of this table and another stream/table."""
        return self._join(joins.InnerJoin(stream=self, fields=fields))

    def outer_join(self, *fields: FieldDescriptorT) -> StreamT:
        """Outer join of this table and another stream/table."""
        return self._join(joins.OuterJoin(stream=self, fields=fields))

    def _join(self, join_strategy: JoinT) -> StreamT:
        # TODO
        raise NotImplementedError('TODO')

    def clone(self, **kwargs: Any) -> Any:
        """Clone table instance."""
        return self.__class__(**{**self.info(), **kwargs})

    def combine(self, *nodes: JoinableT, **kwargs: Any) -> StreamT:
        """Combine tables and streams."""
        # TODO
        raise NotImplementedError('TODO')

    def contribute_to_stream(self, active: StreamT) -> None:
        """Contribute table to stream join."""
        # TODO  See Stream.contribute_to_stream()
        # Should probably connect to Table changelog.
