#!/usr/bin/env python3
"""Base class Collection for Table and future data structures."""
import abc
import time
from contextlib import suppress
from collections import defaultdict
from functools import lru_cache
from datetime import datetime
from heapq import heappop, heappush
from typing import Any, Callable, Iterable, Iterator, List, Mapping, MutableMapping, MutableSet, Optional, Set, Tuple, Union, cast, no_type_check
from mode.utils.futures import maybe_async
from mode import Seconds, Service
from yarl import URL
from faust import stores
from faust import joins
from faust.exceptions import PartitionsMismatch
from faust.streams import current_event
from faust.types import AppT, CodecArg, EventT, FieldDescriptorT, FutureMessage, JoinT, RecordMetadata, SchemaT, TP, TopicT
from faust.types.models import ModelArg, ModelT
from faust.types.stores import StoreT
from faust.types.streams import JoinableT, StreamT
from faust.types.tables import ChangelogEventCallback, CollectionT, RecoverCallback, RelativeHandler, WindowCloseCallback
from faust.types.windows import WindowRange, WindowT

__all__ = ['Collection']
TABLE_CLEANING = 'CLEANING'
E_SOURCE_PARTITIONS_MISMATCH = (
    'The source topic {source_topic!r} for table {table_name!r}\n'
    'has {source_n} partitions, but the changelog\n'
    'topic {change_topic!r} has {change_n} partitions.\n\n'
    'Please make sure the topics have the same number of partitions\n'
    'by configuring Kafka correctly.\n'
)

class Collection(Service, CollectionT):
    """Base class for changelog-backed data structures stored in Kafka."""
    _data: Optional[Any] = None
    _changelog_compacting: bool = True
    _changelog_deleting: Optional[Any] = None

    @abc.abstractmethod
    def _has_key(self, key: Any) -> bool:
        ...

    @abc.abstractmethod
    def _get_key(self, key: Any) -> Any:
        ...

    @abc.abstractmethod
    def _set_key(self, key: Any, value: Any) -> None:
        ...

    @abc.abstractmethod
    def _del_key(self, key: Any) -> None:
        ...

    def __init__(
        self,
        app: AppT,
        *,
        name: Optional[str] = None,
        default: Any = None,
        store: Optional[str] = None,
        schema: Optional[SchemaT] = None,
        key_type: Optional[type] = None,
        value_type: Optional[type] = None,
        partitions: Optional[int] = None,
        window: Optional[WindowT] = None,
        changelog_topic: Optional[Any] = None,
        help: Optional[str] = None,
        on_recover: Optional[Callable[..., Any]] = None,
        on_changelog_event: Optional[Callable[[Any], Any]] = None,
        recovery_buffer_size: int = 1000,
        standby_buffer_size: Optional[int] = None,
        extra_topic_configs: Optional[Mapping[str, Any]] = None,
        recover_callbacks: Optional[Iterable[Callable[..., Any]]] = None,
        options: Optional[Any] = None,
        use_partitioner: bool = False,
        on_window_close: Optional[Callable[[Any, Any], Any]] = None,
        is_global: bool = False,
        **kwargs: Any,
    ) -> None:
        Service.__init__(self, **kwargs)
        self.app: AppT = app
        self.name: str = cast(str, name)
        self.default: Any = default
        self._store: Optional[URL] = URL(store) if store else None
        self.schema: Optional[SchemaT] = schema
        self.key_type: Optional[type] = key_type
        self.value_type: Optional[type] = value_type
        self.partitions: Optional[int] = partitions
        self.window: Optional[WindowT] = window
        self._changelog_topic: Optional[Any] = changelog_topic
        self.extra_topic_configs: Mapping[str, Any] = extra_topic_configs or {}
        self.help: str = help or ''
        self._on_changelog_event: Optional[Callable[[Any], Any]] = on_changelog_event
        self.recovery_buffer_size: int = recovery_buffer_size
        self.standby_buffer_size: int = standby_buffer_size or recovery_buffer_size
        self.use_partitioner: bool = use_partitioner
        self._on_window_close: Optional[Callable[[Any, Any], Any]] = on_window_close
        self.last_closed_window: float = 0.0
        self.is_global: bool = is_global
        assert self.recovery_buffer_size > 0 and self.standby_buffer_size > 0
        self.options: Optional[Any] = options
        self.key_serializer: Any = self._serializer_from_type(self.key_type)
        self.value_serializer: Any = self._serializer_from_type(self.value_type)
        self._partition_timestamp_keys: MutableMapping[Tuple[int, float], Set[Any]] = defaultdict(set)
        self._partition_timestamps: MutableMapping[int, List[float]] = defaultdict(list)
        self._partition_latest_timestamp: MutableMapping[int, float] = defaultdict(int)
        self._recover_callbacks: Set[Callable[..., Any]] = set(recover_callbacks or [])
        if on_recover:
            self.on_recover(on_recover)
        self._sensor_on_get: Callable[..., Any] = self.app.sensors.on_table_get
        self._sensor_on_set: Callable[..., Any] = self.app.sensors.on_table_set
        self._sensor_on_del: Callable[..., Any] = self.app.sensors.on_table_del

    def _serializer_from_type(self, typ: Optional[type]) -> Any:
        if typ is bytes:
            return 'raw'
        serializer: Optional[Any] = None
        with suppress(AttributeError):
            serializer = typ._options.serializer  # type: ignore
        return serializer or 'json'

    def __hash__(self) -> int:
        return object.__hash__(self)

    def _new_store(self) -> Any:
        return self._new_store_by_url(self._store or self.app.conf.store)

    def _new_store_by_url(self, url: Any) -> Any:
        return stores.by_url(url)(
            url,
            self.app,
            self,
            table_name=self.name,
            key_type=self.key_type,
            key_serializer=self.key_serializer,
            value_serializer=self.value_serializer,
            value_type=self.value_type,
            loop=self.loop,
            options=self.options,
        )

    @property
    @no_type_check
    def data(self) -> Any:
        """Underlying table storage."""
        if self._data is None:
            self._data = self._new_store()
        return self._data

    async def on_start(self) -> None:
        """Call when table starts."""
        await self.add_runtime_dependency(self.data)
        await self.changelog_topic.maybe_declare()

    def on_recover(self, fun: Callable[..., Any]) -> Callable[..., Any]:
        """Add function as callback to be called on table recovery."""
        assert fun not in self._recover_callbacks
        self._recover_callbacks.add(fun)
        return fun

    def info(self) -> Mapping[str, Any]:
        """Return table attributes as dictionary."""
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

    def persisted_offset(self, tp: TP) -> Any:
        """Return the last persisted offset for topic partition."""
        return self.data.persisted_offset(tp)

    async def need_active_standby_for(self, tp: TP) -> Any:
        """Return :const:`False` if we have access to partition data."""
        return await self.data.need_active_standby_for(tp)

    def reset_state(self) -> None:
        """Reset local state."""
        self.data.reset_state()

    def send_changelog(
        self,
        partition: int,
        key: Any,
        value: Any,
        key_serializer: Optional[Any] = None,
        value_serializer: Optional[Any] = None,
    ) -> Any:
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
            eager_partitioning=True,
        )

    def _send_changelog(self, event: Any, key: Any, value: Any, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = None) -> None:
        if event is None:
            raise RuntimeError('Cannot modify table outside of agent/stream.')
        self.send_changelog(event.message.partition, key, value, key_serializer, value_serializer)

    def partition_for_key(self, key: Any) -> Optional[int]:
        """Return partition number for table key.

        Always returns :const:`None` when :attr:`use_partitioner`
        is enabled.
        """
        if self.use_partitioner:
            return None
        else:
            event = current_event()
            if event is None:
                raise TypeError('Cannot modify table key from outside of stream iteration')
            self._verify_source_topic_partitions(event.message.topic)
            return event.message.partition

    @lru_cache()
    def _verify_source_topic_partitions(self, source_topic: str) -> None:
        change_topic: str = self.changelog_topic_name
        source_n: Optional[int] = self.app.consumer.topic_partitions(source_topic)
        if source_n is not None:
            change_n: Optional[int] = self.app.consumer.topic_partitions(change_topic)
            if change_n is not None:
                if source_n != change_n:
                    raise PartitionsMismatch(
                        E_SOURCE_PARTITIONS_MISMATCH.format(
                            source_topic=source_topic,
                            table_name=self.name,
                            source_n=source_n,
                            change_topic=change_topic,
                            change_n=change_n,
                        )
                    )

    def _on_changelog_sent(self, fut: Any) -> None:
        res = fut.result()
        if self.app.in_transaction:
            self.app.tables.persist_offset_on_commit(self.data, res.topic_partition, res.offset)
        else:
            self.data.set_persisted_offset(res.topic_partition, res.offset)

    @Service.task
    @Service.transitions_to(TABLE_CLEANING)
    async def _clean_data(self) -> None:
        interval: Seconds = self.app.conf.table_cleanup_interval
        if self._should_expire_keys():
            await self.sleep(interval)
            async for sleep_time in self.itertimer(interval, name='table_cleanup'):
                await self._del_old_keys()

    async def _del_old_keys(self) -> None:
        assert self.window is not None
        window: WindowT = cast(WindowT, self.window)
        for partition, timestamps in self._partition_timestamps.items():
            while timestamps and window.stale(timestamps[0], self._partition_latest_timestamp[partition]):
                timestamp: float = heappop(timestamps)
                keys_to_remove: Optional[Set[Any]] = self._partition_timestamp_keys.pop((partition, timestamp), None)
                if keys_to_remove:
                    for key in keys_to_remove:
                        value: Any = self.data.pop(key, None)
                        if key[1][0] > self.last_closed_window:
                            await self.on_window_close(key, value)
                    self.last_closed_window = max(self.last_closed_window, max((key[1][0] for key in keys_to_remove)))
    
    async def on_window_close(self, key: Any, value: Any) -> None:
        if self._on_window_close:
            await maybe_async(self._on_window_close(key, value))

    def _should_expire_keys(self) -> bool:
        return not (self.window is None or getattr(self.window, 'expires', None) is None)

    def _maybe_set_key_ttl(self, key: Any, partition: int) -> None:
        if not self._should_expire_keys():
            return
        _, window_range = key
        # Assuming window_range is a tuple with second element as range end (float)
        range_end: float = window_range[1]
        heappush(self._partition_timestamps[partition], range_end)
        self._partition_latest_timestamp[partition] = max(self._partition_latest_timestamp[partition], range_end)
        self._partition_timestamp_keys[(partition, range_end)].add(key)

    def _maybe_del_key_ttl(self, key: Any, partition: int) -> None:
        if not self._should_expire_keys():
            return
        _, window_range = key
        ts_keys: Optional[Set[Any]] = self._partition_timestamp_keys.get((partition, window_range[1]))
        if ts_keys is not None:
            ts_keys.discard(key)

    def _changelog_topic_name(self) -> str:
        return f'{self.app.conf.id}-{self.name}-changelog'

    def join(self, *fields: FieldDescriptorT) -> Any:
        """Right join of this table and another stream/table."""
        return self._join(joins.RightJoin(stream=self, fields=fields))

    def left_join(self, *fields: FieldDescriptorT) -> Any:
        """Left join of this table and another stream/table."""
        return self._join(joins.LeftJoin(stream=self, fields=fields))

    def inner_join(self, *fields: FieldDescriptorT) -> Any:
        """Inner join of this table and another stream/table."""
        return self._join(joins.InnerJoin(stream=self, fields=fields))

    def outer_join(self, *fields: FieldDescriptorT) -> Any:
        """Outer join of this table and another stream/table."""
        return self._join(joins.OuterJoin(stream=self, fields=fields))

    def _join(self, join_strategy: JoinT) -> Any:
        raise NotImplementedError('TODO')

    def clone(self, **kwargs: Any) -> "Collection":
        """Clone table instance."""
        return self.__class__(**{**self.info(), **kwargs})

    def combine(self, *nodes: Any, **kwargs: Any) -> Any:
        """Combine tables and streams."""
        raise NotImplementedError('TODO')

    def contribute_to_stream(self, active: Any) -> Any:
        """Contribute table to stream join."""
        ...

    async def remove_from_stream(self, stream: StreamT) -> None:
        """Remove table from stream join after stream stopped."""
        ...

    def _new_changelog_topic(self, *, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[Any] = None) -> TopicT:
        if compacting is None:
            compacting = self._changelog_compacting
        if deleting is None:
            deleting = self._changelog_deleting
        if retention is None and self.window:
            retention = getattr(self.window, 'expires', None)
        return self.app.topic(
            self._changelog_topic_name(),
            schema=self.schema,
            key_type=self.key_type,
            value_type=self.value_type,
            key_serializer=self.key_serializer,
            value_serializer=self.value_serializer,
            partitions=self.partitions,
            retention=retention,
            compacting=compacting,
            deleting=deleting,
            acks=False,
            internal=True,
            config=self.extra_topic_configs,
            maxsize=131072,
            allow_empty=True,
        )

    def __copy__(self) -> "Collection":
        return self.clone()

    def __and__(self, other: Any) -> Any:
        return self.combine(self, other)

    def _apply_window_op(self, op: Callable[[Any, Any], Any], key: Any, value: Any, timestamp: float) -> None:
        get_ = self._get_key
        set_ = self._set_key
        for window_range in self._window_ranges(timestamp):
            set_((key, window_range), op(get_((key, window_range)), value))

    def _set_windowed(self, key: Any, value: Any, timestamp: float) -> None:
        for window_range in self._window_ranges(timestamp):
            self._set_key((key, window_range), value)

    def _del_windowed(self, key: Any, timestamp: float) -> None:
        for window_range in self._window_ranges(timestamp):
            self._del_key((key, window_range))

    def _window_ranges(self, timestamp: float) -> Iterator[WindowRange]:
        assert self.window is not None
        window: WindowT = cast(WindowT, self.window)
        for window_range in window.ranges(timestamp):
            yield window_range

    def _relative_now(self, event: Optional[EventT] = None) -> float:
        event = event if event is not None else current_event()
        if event is None:
            return time.time()
        return self._partition_latest_timestamp[event.message.partition]

    def _relative_event(self, event: Optional[EventT] = None) -> float:
        event = event if event is not None else current_event()
        if event is None:
            raise RuntimeError('Operation outside of stream iteration')
        return event.message.timestamp

    def _relative_field(self, field: FieldDescriptorT) -> Callable[[Optional[EventT]], Any]:
        def to_value(event: Optional[EventT] = None) -> Any:
            if event is None:
                raise RuntimeError('Operation outside of stream iteration')
            return field.getattr(cast(ModelT, event.value))
        return to_value

    def _relative_timestamp(self, timestamp: float) -> Callable[[Optional[EventT]], float]:
        def handler(event: Optional[EventT] = None) -> float:
            return timestamp
        return handler

    def _windowed_now(self, key: Any) -> Any:
        assert self.window is not None
        window: WindowT = cast(WindowT, self.window)
        return self._get_key((key, window.earliest(self._relative_now())))

    def _windowed_timestamp(self, key: Any, timestamp: float) -> Any:
        assert self.window is not None
        window: WindowT = cast(WindowT, self.window)
        return self._get_key((key, window.current(timestamp)))

    def _windowed_contains(self, key: Any, timestamp: float) -> bool:
        assert self.window is not None
        window: WindowT = cast(WindowT, self.window)
        return self._has_key((key, window.current(timestamp)))

    def _windowed_delta(self, key: Any, d: float, event: Optional[EventT] = None) -> Any:
        assert self.window is not None
        window: WindowT = cast(WindowT, self.window)
        return self._get_key((key, window.delta(self._relative_event(event), d)))

    async def on_rebalance(self, assigned: Iterable[TP], revoked: Iterable[TP], newly_assigned: Iterable[TP]) -> None:
        """Call when cluster is rebalancing."""
        await self.data.on_rebalance(self, assigned, revoked, newly_assigned)

    async def on_recovery_completed(self, active_tps: Iterable[TP], standby_tps: Iterable[TP]) -> None:
        """Call when recovery has completed after rebalancing."""
        await self.data.on_recovery_completed(active_tps, standby_tps)
        await self.call_recover_callbacks()

    async def call_recover_callbacks(self) -> None:
        """Call any configured recovery callbacks after rebalancing."""
        for fun in self._recover_callbacks:
            await fun()

    async def on_changelog_event(self, event: Any) -> None:
        """Call when a new changelog event is received."""
        if self._on_changelog_event:
            await self._on_changelog_event(event)

    @property
    def label(self) -> str:
        """Return human-readable label used to represent this table."""
        return f'{self.shortlabel}@{self._store}'

    @property
    def shortlabel(self) -> str:
        """Return short label used to represent this table in logs."""
        return f'{type(self).__name__}: {self.name}'

    @property
    def changelog_topic(self) -> TopicT:
        """Return the changelog topic used by this table."""
        if self._changelog_topic is None:
            self._changelog_topic = self._new_changelog_topic()
        return self._changelog_topic

    @changelog_topic.setter
    def changelog_topic(self, topic: TopicT) -> None:
        self._changelog_topic = topic

    @property
    def changelog_topic_name(self) -> str:
        return self.changelog_topic.get_topic_name()

    def apply_changelog_batch(self, batch: Any) -> None:
        """Apply batch of events from changelog topic local table storage."""
        self.data.apply_changelog_batch(batch, to_key=self._to_key, to_value=self._to_value)

    def _to_key(self, k: Any) -> Any:
        if isinstance(k, list):
            return tuple((tuple(v) if isinstance(v, list) else v for v in k))
        return k

    def _to_value(self, v: Any) -> Any:
        return v

    def _human_channel(self) -> str:
        return f'{type(self).__name__}: {self.name}'

    def _repr_info(self) -> Any:
        return self.name
