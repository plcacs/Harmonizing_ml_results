"""Message transport using :pypi:`aiokafka`."""
import asyncio
import typing
from collections import deque
from time import monotonic
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Deque,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)
import aiokafka
import aiokafka.abc
import opentracing
from aiokafka.consumer.group_coordinator import OffsetCommitRequest
from aiokafka.errors import (
    CommitFailedError,
    ConsumerStoppedError,
    IllegalStateError,
    KafkaError,
)
from aiokafka.structs import OffsetAndMetadata, TopicPartition as _TopicPartition
from aiokafka.util import parse_kafka_version
from kafka.errors import NotControllerError, TopicAlreadyExistsError as TopicExistsError
from kafka.partitioner.default import DefaultPartitioner
from mode import Service
from mode.utils.objects import cached_property
from mode.utils.times import Seconds
from faust.auth import (
    GSSAPICredentials,
    SASLCredentials,
    SSLCredentials,
)
from faust.exceptions import (
    ConsumerNotStarted,
    ImproperlyConfigured,
    NotReady,
    ProducerSendError,
)
from faust.transport import base
from faust.types import (
    ConsumerMessage,
    HeadersArg,
    RecordMetadata,
    TP,
)
from faust.types.transports import (
    ConsumerT,
    PartitionerT,
    ProducerT,
)
from yarl import URL

__all__: List[str] = ['Consumer', 'Producer', 'Transport']
logger: Any = ...

DEFAULT_GENERATION_ID: int = OffsetCommitRequest.DEFAULT_GENERATION_ID
TOPIC_LENGTH_MAX: int = 249

class ConsumerRebalanceListener(aiokafka.abc.ConsumerRebalanceListener):
    def __init__(self, thread: 'AIOKafkaConsumerThread') -> None: ...
    def on_partitions_revoked(self, revoked: Set[TP]) -> None: ...
    async def on_partitions_assigned(self, assigned: Set[TP]) -> None: ...

class Consumer(ThreadDelegateConsumer):
    logger: Any = ...
    RebalanceListener: ClassVar[Type[ConsumerRebalanceListener]] = ConsumerRebalanceListener
    consumer_stopped_errors: ClassVar[Tuple[Type[Exception], ...]] = (ConsumerStoppedError,)
    
    def _new_consumer_thread(self) -> 'AIOKafkaConsumerThread': ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: float = 30.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None: ...
    def _new_topicpartition(self, topic: str, partition: int) -> TP: ...
    def _to_message(self, tp: TP, record: Any) -> ConsumerMessage: ...
    async def on_stop(self) -> None: ...

class AIOKafkaConsumerThread(ConsumerThread):
    _consumer: Optional[aiokafka.AIOKafkaConsumer] = None
    
    def __post_init__(self) -> None: ...
    async def on_start(self) -> None: ...
    async def on_thread_stop(self) -> None: ...
    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> aiokafka.AIOKafkaConsumer: ...
    def _create_worker_consumer(self, transport: 'Transport', loop: asyncio.AbstractEventLoop) -> aiokafka.AIOKafkaConsumer: ...
    def _create_client_consumer(self, transport: 'Transport', loop: asyncio.AbstractEventLoop) -> aiokafka.AIOKafkaConsumer: ...
    @cached_property
    def trace_category(self) -> str: ...
    def start_rebalancing_span(self) -> Any: ...
    def start_coordinator_span(self) -> Any: ...
    def _start_span(self, name: str, *, lazy: bool = False) -> Any: ...
    @no_type_check
    def _transform_span_lazy(self, span: Any) -> None: ...
    def _span_finish(self, span: Any) -> None: ...
    def _on_span_generation_pending(self, span: Any) -> None: ...
    def _on_span_generation_known(self, span: Any) -> None: ...
    def _on_span_cancelled_early(self, span: Any) -> None: ...
    def traced_from_parent_span(
        self,
        parent_span: Any,
        lazy: bool = False,
        **extra_context: Any,
    ) -> Any: ...
    def flush_spans(self) -> None: ...
    def on_generation_id_known(self) -> None: ...
    def close(self) -> None: ...
    async def subscribe(self, topics: Iterable[str]) -> None: ...
    async def seek_to_committed(self) -> None: ...
    async def commit(self, offsets: Mapping[TP, int]) -> bool: ...
    async def _commit(self, offsets: Mapping[TP, int]) -> bool: ...
    def verify_event_path(self, now: float, tp: TP) -> Optional[str]: ...
    def verify_recovery_event_path(self, now: float, tp: TP) -> None: ...
    def _verify_aiokafka_event_path(self, now: float, tp: TP) -> bool: ...
    def _log_slow_processing_stream(self, msg: str, *args: Any) -> None: ...
    def _log_slow_processing_commit(self, msg: str, *args: Any) -> None: ...
    def _make_slow_processing_error(self, msg: str, causes: Iterable[str]) -> str: ...
    def _log_slow_processing(
        self,
        msg: str,
        *args: Any,
        causes: Iterable[str],
        setting: str,
        current_value: Seconds,
    ) -> None: ...
    async def position(self, tp: TP) -> int: ...
    async def seek_to_beginning(self, *partitions: TP) -> None: ...
    async def seek_wait(self, partitions: Mapping[TP, int]) -> None: ...
    async def _seek_wait(self, consumer: aiokafka.AIOKafkaConsumer, partitions: Mapping[TP, int]) -> None: ...
    def seek(self, partition: TP, offset: int) -> None: ...
    def assignment(self) -> Set[TP]: ...
    def highwater(self, tp: TP) -> Optional[int]: ...
    def topic_partitions(self, topic: str) -> Optional[int]: ...
    async def earliest_offsets(self, *partitions: TP) -> Mapping[TP, int]: ...
    async def highwaters(self, *partitions: TP) -> Mapping[TP, int]: ...
    async def _highwaters(self, partitions: Iterable[TP]) -> Mapping[TP, int]: ...
    def _ensure_consumer(self) -> aiokafka.AIOKafkaConsumer: ...
    async def getmany(
        self,
        active_partitions: Iterable[TP],
        timeout: Seconds,
    ) -> Mapping[TP, List[Any]]: ...
    async def _fetch_records(
        self,
        consumer: aiokafka.AIOKafkaConsumer,
        active_partitions: Iterable[TP],
        timeout: Optional[Seconds] = None,
        max_records: Optional[int] = None,
    ) -> Mapping[TP, List[Any]]: ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: float = 30.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None: ...
    def key_partition(self, topic: str, key: Optional[bytes], partition: Optional[int] = None) -> Optional[int]: ...

class Producer(base.Producer):
    logger: Any = ...
    allow_headers: bool = ...
    _producer: Optional[Union[aiokafka.AIOKafkaProducer, aiokafka.MultiTXNProducer]] = None
    
    def __post_init__(self) -> None: ...
    def _settings_default(self) -> Mapping[str, Any]: ...
    def _settings_auth(self) -> Mapping[str, Any]: ...
    async def begin_transaction(self, transactional_id: str) -> None: ...
    async def commit_transaction(self, transactional_id: str) -> None: ...
    async def abort_transaction(self, transactional_id: str) -> None: ...
    async def stop_transaction(self, transactional_id: str) -> None: ...
    async def maybe_begin_transaction(self, transactional_id: str) -> None: ...
    async def commit_transactions(
        self,
        tid_to_offset_map: Mapping[str, Mapping[TP, int]],
        group_id: str,
        start_new_transaction: bool = True,
    ) -> None: ...
    def _settings_extra(self) -> Mapping[str, Any]: ...
    def _new_producer(self) -> Union[aiokafka.AIOKafkaProducer, aiokafka.MultiTXNProducer]: ...
    @property
    def _producer_type(self) -> Type[Union[aiokafka.AIOKafkaProducer, aiokafka.MultiTXNProducer]]: ...
    async def _on_irrecoverable_error(self, exc: Exception) -> None: ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: float = 20.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None: ...
    def _ensure_producer(self) -> Union[aiokafka.AIOKafkaProducer, aiokafka.MultiTXNProducer]: ...
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
        transactional_id: Optional[str] = None,
    ) -> RecordMetadata: ...
    async def send_and_wait(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: Optional[HeadersArg],
        *,
        transactional_id: Optional[str] = None,
    ) -> RecordMetadata: ...
    async def flush(self) -> None: ...
    def key_partition(self, topic: str, key: Optional[bytes]) -> TP: ...
    def supports_headers(self) -> bool: ...

class Transport(base.Transport):
    Consumer: ClassVar[Type[Consumer]] = Consumer
    Producer: ClassVar[Type[Producer]] = Producer
    default_port: int = 9092
    driver_version: str = ...
    
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def _topic_config(
        self,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
    ) -> Mapping[str, Any]: ...
    async def _create_topic(
        self,
        owner: Union[Consumer, Producer],
        client: aiokafka.abc.KafkaClient,
        topic: str,
        partitions: int,
        replication: int,
        **kwargs: Any,
    ) -> None: ...
    async def _get_controller_node(
        self,
        owner: Union[Consumer, Producer],
        client: aiokafka.abc.KafkaClient,
        timeout: int = 30000,
    ) -> Optional[int]: ...
    async def _really_create_topic(
        self,
        owner: Union[Consumer, Producer],
        client: aiokafka.abc.KafkaClient,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: int = 30000,
        retention: Optional[int] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None: ...

def credentials_to_aiokafka_auth(
    credentials: Optional[CredentialsT] = None,
    ssl_context: Optional[Any] = None,
) -> Mapping[str, Any]: ...