import asyncio
import typing
from collections import deque
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
)

import aiokafka
import aiokafka.abc
import opentracing
from aiokafka.structs import TopicPartition as _TopicPartition
from mode.utils.futures import StampedeWrapper
from mode.utils.objects import cached_property
from mode.utils.times import Seconds
from mode.utils.typing import Deque
from yarl import URL

from faust.auth import GSSAPICredentials, SASLCredentials, SSLCredentials
from faust.transport import base
from faust.transport.consumer import ConsumerThread, RecordMap, ThreadDelegateConsumer
from faust.types import ConsumerMessage, HeadersArg, RecordMetadata, TP
from faust.types.auth import CredentialsT
from faust.types.transports import ConsumerT, PartitionerT, ProducerT

__all__: List[str]

logger: Any
DEFAULT_GENERATION_ID: int
TOPIC_LENGTH_MAX: int
SLOW_PROCESSING_CAUSE_AGENT: str
SLOW_PROCESSING_CAUSE_STREAM: str
SLOW_PROCESSING_CAUSE_COMMIT: str
SLOW_PROCESSING_EXPLAINED: str
SLOW_PROCESSING_NO_FETCH_SINCE_START: str
SLOW_PROCESSING_NO_RESPONSE_SINCE_START: str
SLOW_PROCESSING_NO_RECENT_FETCH: str
SLOW_PROCESSING_NO_RECENT_RESPONSE: str
SLOW_PROCESSING_NO_HIGHWATER_SINCE_START: str
SLOW_PROCESSING_STREAM_IDLE_SINCE_START: str
SLOW_PROCESSING_STREAM_IDLE: str
SLOW_PROCESSING_NO_COMMIT_SINCE_START: str
SLOW_PROCESSING_NO_RECENT_COMMIT: str

def server_list(urls: Iterable[URL], default_port: int) -> List[str]: ...

class ConsumerRebalanceListener(aiokafka.abc.ConsumerRebalanceListener):
    _thread: AIOKafkaConsumerThread
    def __init__(self, thread: AIOKafkaConsumerThread) -> None: ...
    def on_partitions_revoked(self, revoked: Iterable[Any]) -> Any: ...
    async def on_partitions_assigned(self, assigned: Iterable[Any]) -> None: ...

class Consumer(ThreadDelegateConsumer):
    logger: ClassVar[Any]
    RebalanceListener: ClassVar[Type[ConsumerRebalanceListener]]
    consumer_stopped_errors: ClassVar[Tuple[Type[BaseException], ...]]

    def _new_consumer_thread(self) -> AIOKafkaConsumerThread: ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = ...,
        timeout: Seconds = ...,
        retention: Optional[Seconds] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        ensure_created: bool = ...,
    ) -> None: ...
    def _new_topicpartition(self, topic: str, partition: int) -> TP: ...
    def _to_message(self, tp: TP, record: Any) -> ConsumerMessage: ...
    async def on_stop(self) -> None: ...

class AIOKafkaConsumerThread(ConsumerThread):
    _consumer: Optional[aiokafka.AIOKafkaConsumer]
    _partitioner: Any
    _rebalance_listener: ConsumerRebalanceListener
    _pending_rebalancing_spans: Deque[Any]
    tp_last_committed_at: MutableMapping[TP, float]
    tp_fetch_request_timeout_secs: float
    tp_fetch_response_timeout_secs: float
    tp_stream_timeout_secs: float
    tp_commit_timeout_secs: float
    time_started: float
    _assignor: Any

    def __post_init__(self) -> None: ...
    async def on_start(self) -> None: ...
    async def on_thread_stop(self) -> None: ...
    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> aiokafka.AIOKafkaConsumer: ...
    def _create_worker_consumer(
        self, transport: Transport, loop: asyncio.AbstractEventLoop
    ) -> aiokafka.AIOKafkaConsumer: ...
    def _create_client_consumer(
        self, transport: Transport, loop: asyncio.AbstractEventLoop
    ) -> aiokafka.AIOKafkaConsumer: ...
    @cached_property
    def trace_category(self) -> str: ...
    def start_rebalancing_span(self) -> opentracing.Span: ...
    def start_coordinator_span(self) -> opentracing.Span: ...
    def _start_span(self, name: str, *, lazy: bool = ...) -> opentracing.Span: ...
    def _transform_span_lazy(self, span: Any) -> None: ...
    def _span_finish(self, span: Any) -> None: ...
    def _on_span_generation_pending(self, span: Any) -> None: ...
    def _on_span_generation_known(self, span: Any) -> None: ...
    def _on_span_cancelled_early(self, span: Any) -> None: ...
    def traced_from_parent_span(
        self, parent_span: Optional[opentracing.Span], lazy: bool = ..., **extra_context: Any
    ) -> Any: ...
    def flush_spans(self) -> None: ...
    def on_generation_id_known(self) -> None: ...
    def close(self) -> None: ...
    async def subscribe(self, topics: Iterable[str]) -> None: ...
    async def seek_to_committed(self) -> Optional[Mapping[TP, int]]: ...
    async def commit(self, offsets: Mapping[TP, int]) -> bool: ...
    async def _commit(self, offsets: Mapping[TP, int]) -> bool: ...
    def verify_event_path(self, now: float, tp: TP) -> Optional[None]: ...
    def verify_recovery_event_path(self, now: float, tp: TP) -> None: ...
    def _verify_aiokafka_event_path(self, now: float, tp: TP) -> bool: ...
    def _log_slow_processing_stream(self, msg: str, *args: Any) -> None: ...
    def _log_slow_processing_commit(self, msg: str, *args: Any) -> None: ...
    def _make_slow_processing_error(self, msg: str, causes: List[str]) -> str: ...
    def _log_slow_processing(
        self,
        msg: str,
        *args: Any,
        causes: List[str],
        setting: str,
        current_value: Any,
    ) -> Any: ...
    async def position(self, tp: TP) -> Optional[int]: ...
    async def seek_to_beginning(self, *partitions: TP) -> None: ...
    async def seek_wait(self, partitions: Mapping[TP, int]) -> None: ...
    async def _seek_wait(
        self, consumer: aiokafka.AIOKafkaConsumer, partitions: Mapping[TP, int]
    ) -> None: ...
    def seek(self, partition: TP, offset: int) -> None: ...
    def assignment(self) -> Set[TP]: ...
    def highwater(self, tp: TP) -> Optional[int]: ...
    def topic_partitions(self, topic: str) -> Optional[int]: ...
    async def earliest_offsets(self, *partitions: TP) -> Mapping[TP, int]: ...
    async def highwaters(self, *partitions: TP) -> Mapping[TP, int]: ...
    async def _highwaters(self, partitions: Tuple[TP, ...]) -> Mapping[TP, int]: ...
    def _ensure_consumer(self) -> aiokafka.AIOKafkaConsumer: ...
    async def getmany(
        self, active_partitions: Optional[Set[TP]], timeout: float
    ) -> RecordMap: ...
    async def _fetch_records(
        self,
        consumer: aiokafka.AIOKafkaConsumer,
        active_partitions: Optional[Set[TP]],
        timeout: Optional[float] = ...,
        max_records: Optional[int] = ...,
    ) -> RecordMap: ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = ...,
        timeout: Seconds = ...,
        retention: Optional[Seconds] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        ensure_created: bool = ...,
    ) -> None: ...
    def key_partition(
        self, topic: str, key: Optional[bytes], partition: Optional[int] = ...
    ) -> Optional[int]: ...

class Producer(base.Producer):
    logger: ClassVar[Any]
    allow_headers: bool
    _producer: Optional[Any]
    _send_on_produce_message: Any

    def __post_init__(self) -> None: ...
    def _settings_default(self) -> MutableMapping[str, Any]: ...
    def _settings_auth(self) -> MutableMapping[str, Any]: ...
    async def begin_transaction(self, transactional_id: str) -> None: ...
    async def commit_transaction(self, transactional_id: str) -> None: ...
    async def abort_transaction(self, transactional_id: str) -> None: ...
    async def stop_transaction(self, transactional_id: str) -> None: ...
    async def maybe_begin_transaction(self, transactional_id: str) -> None: ...
    async def commit_transactions(
        self,
        tid_to_offset_map: Mapping[str, Mapping[TP, int]],
        group_id: str,
        start_new_transaction: bool = ...,
    ) -> None: ...
    def _settings_extra(self) -> MutableMapping[str, Any]: ...
    def _new_producer(self) -> Any: ...
    @property
    def _producer_type(self) -> Type[Any]: ...
    async def _on_irrecoverable_error(self, exc: BaseException) -> None: ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = ...,
        timeout: Seconds = ...,
        retention: Optional[Seconds] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        ensure_created: bool = ...,
    ) -> None: ...
    def _ensure_producer(self) -> Any: ...
    async def on_start(self) -> None: ...
    async def on_stop(self) -> None: ...
    async def send(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: Optional[HeadersArg],
        *,
        transactional_id: Optional[str] = ...,
    ) -> Awaitable[RecordMetadata]: ...
    async def send_and_wait(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: Optional[HeadersArg],
        *,
        transactional_id: Optional[str] = ...,
    ) -> RecordMetadata: ...
    async def flush(self) -> None: ...
    def key_partition(self, topic: str, key: Optional[bytes]) -> TP: ...
    def supports_headers(self) -> bool: ...

class Transport(base.Transport):
    Consumer: ClassVar[Type[Consumer]]
    Producer: ClassVar[Type[Producer]]
    default_port: ClassVar[int]
    driver_version: ClassVar[str]
    _topic_waiters: MutableMapping[str, StampedeWrapper]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def _topic_config(
        self,
        retention: Optional[int] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
    ) -> MutableMapping[str, Any]: ...
    async def _create_topic(
        self,
        owner: Any,
        client: Any,
        topic: str,
        partitions: int,
        replication: int,
        **kwargs: Any,
    ) -> None: ...
    async def _get_controller_node(
        self, owner: Any, client: Any, timeout: int = ...
    ) -> Optional[int]: ...
    async def _really_create_topic(
        self,
        owner: Any,
        client: Any,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = ...,
        timeout: int = ...,
        retention: Optional[int] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        ensure_created: bool = ...,
    ) -> None: ...

def credentials_to_aiokafka_auth(
    credentials: Optional[CredentialsT] = ...,
    ssl_context: Optional[Any] = ...,
) -> MutableMapping[str, Any]: ...