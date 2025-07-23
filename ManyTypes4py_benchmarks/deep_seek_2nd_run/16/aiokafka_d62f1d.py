"""Message transport using :pypi:`aiokafka`."""
import asyncio
import typing
from collections import deque
from time import monotonic
from typing import (
    Any, Awaitable, Callable, ClassVar, Dict, Iterable, List, Mapping,
    MutableMapping, Optional, Set, Tuple, Type, Union, cast, no_type_check
)
import aiokafka
import aiokafka.abc
import opentracing
from aiokafka.consumer.group_coordinator import OffsetCommitRequest
from aiokafka.errors import (
    CommitFailedError, ConsumerStoppedError, IllegalStateError, KafkaError
)
from aiokafka.structs import OffsetAndMetadata, TopicPartition as _TopicPartition
from aiokafka.util import parse_kafka_version
from kafka.errors import (
    NotControllerError, TopicAlreadyExistsError as TopicExistsError, for_code
)
from kafka.partitioner.default import DefaultPartitioner
from kafka.partitioner.hashed import murmur2
from kafka.protocol.metadata import MetadataRequest_v1
from mode import Service, get_logger
from mode.utils.futures import StampedeWrapper
from mode.utils.objects import cached_property
from mode.utils import text
from mode.utils.times import Seconds, humanize_seconds_ago, want_seconds
from mode.utils.typing import Deque
from opentracing.ext import tags
from yarl import URL
from faust.auth import GSSAPICredentials, SASLCredentials, SSLCredentials
from faust.exceptions import (
    ConsumerNotStarted, ImproperlyConfigured, NotReady, ProducerSendError
)
from faust.transport import base
from faust.transport.consumer import (
    ConsumerThread, RecordMap, ThreadDelegateConsumer, ensure_TPset
)
from faust.types import ConsumerMessage, HeadersArg, RecordMetadata, TP
from faust.types.auth import CredentialsT
from faust.types.transports import ConsumerT, PartitionerT, ProducerT
from faust.utils.kafka.protocol.admin import CreateTopicsRequest
from faust.utils.tracing import noop_span, set_current_span, traced_from_parent_span

__all__ = ['Consumer', 'Producer', 'Transport']

if not hasattr(aiokafka, '__robinhood__'):
    raise RuntimeError('Please install robinhood-aiokafka, not aiokafka')

logger = get_logger(__name__)
DEFAULT_GENERATION_ID = OffsetCommitRequest.DEFAULT_GENERATION_ID
TOPIC_LENGTH_MAX = 249
SLOW_PROCESSING_CAUSE_AGENT = '\nThe agent processing the stream is hanging (waiting for network, I/O or infinite loop).\n'.strip()
SLOW_PROCESSING_CAUSE_STREAM = '\nThe stream has stopped processing events for some reason.\n'.strip()
SLOW_PROCESSING_CAUSE_COMMIT = '\nThe commit handler background thread has stopped working (report as bug).\n'.strip()
SLOW_PROCESSING_EXPLAINED = '\n\nThere are multiple possible explanations for this:\n\n1) The processing of a single event in the stream\n   is taking too long.\n\n    The timeout for this is defined by the %(setting)s setting,\n    currently set to %(current_value)r.  If you expect the time\n    required to process an event, to be greater than this then please\n    increase the timeout.\n\n'
SLOW_PROCESSING_NO_FETCH_SINCE_START = '\nAiokafka has not sent fetch request for %r since start (started %s)\n'.strip()
SLOW_PROCESSING_NO_RESPONSE_SINCE_START = '\nAiokafka has not received fetch response for %r since start (started %s)\n'.strip()
SLOW_PROCESSING_NO_RECENT_FETCH = '\nAiokafka stopped fetching from %r (last done %s)\n'.strip()
SLOW_PROCESSING_NO_RECENT_RESPONSE = '\nBroker stopped responding to fetch requests for %r (last responded %s)\n'.strip()
SLOW_PROCESSING_NO_HIGHWATER_SINCE_START = '\nHighwater not yet available for %r (started %s).\n'.strip()
SLOW_PROCESSING_STREAM_IDLE_SINCE_START = '\nStream has not started processing %r (started %s).\n'.strip()
SLOW_PROCESSING_STREAM_IDLE = '\nStream stopped processing, or is slow for %r (last inbound %s).\n'.strip()
SLOW_PROCESSING_NO_COMMIT_SINCE_START = '\nHas not committed %r at all since worker start (started %s).\n'.strip()
SLOW_PROCESSING_NO_RECENT_COMMIT = '\nHas not committed %r (last commit %s).\n'.strip()

def server_list(urls: List[URL], default_port: int) -> List[str]:
    """Convert list of urls to list of servers accepted by :pypi:`aiokafka`."""
    default_host = '127.0.0.1'
    return [f'{u.host or default_host}:{u.port or default_port}' for u in urls]

class ConsumerRebalanceListener(aiokafka.abc.ConsumerRebalanceListener):

    def __init__(self, thread: Any) -> None:
        self._thread = thread

    def on_partitions_revoked(self, revoked: Iterable[_TopicPartition]) -> Awaitable[None]:
        """Call when partitions are being revoked."""
        thread = self._thread
        thread.app.on_rebalance_start()
        return thread.on_partitions_revoked(ensure_TPset(revoked))

    async def on_partitions_assigned(self, assigned: Iterable[_TopicPartition]) -> None:
        """Call when partitions are being assigned."""
        await self._thread.on_partitions_assigned(ensure_TPset(assigned))

class Consumer(ThreadDelegateConsumer):
    """Kafka consumer using :pypi:`aiokafka`."""
    logger = logger
    RebalanceListener: ClassVar[Type[ConsumerRebalanceListener]] = ConsumerRebalanceListener
    consumer_stopped_errors: ClassVar[Tuple[Type[Exception], ...]] = (ConsumerStoppedError,)

    def _new_consumer_thread(self) -> 'AIOKafkaConsumerThread':
        return AIOKafkaConsumerThread(self, loop=self.loop, beacon=self.beacon)

    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: float = 30.0,
        retention: Optional[float] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False
    ) -> None:
        """Create/declare topic on server."""
        await self._thread.create_topic(
            topic, partitions, replication,
            config=config, timeout=timeout,
            retention=retention, compacting=compacting,
            deleting=deleting, ensure_created=ensure_created
        )

    def _new_topicpartition(self, topic: str, partition: int) -> TP:
        return cast(TP, _TopicPartition(topic, partition))

    def _to_message(self, tp: TP, record: Any) -> ConsumerMessage:
        timestamp = record.timestamp
        timestamp_s = cast(float, None)
        if timestamp is not None:
            timestamp_s = timestamp / 1000.0
        return ConsumerMessage(
            record.topic, record.partition, record.offset,
            timestamp_s, record.timestamp_type, record.headers,
            record.key, record.value, record.checksum,
            record.serialized_key_size, record.serialized_value_size,
            tp
        )

    async def on_stop(self) -> None:
        """Call when consumer is stopping."""
        await super().on_stop()
        transport = cast(Transport, self.transport)
        transport._topic_waiters.clear()

class AIOKafkaConsumerThread(ConsumerThread):
    _consumer: Optional[aiokafka.AIOKafkaConsumer] = None

    def __post_init__(self) -> None:
        consumer = cast(Consumer, self.consumer)
        self._partitioner = self.app.conf.producer_partitioner or DefaultPartitioner()
        self._rebalance_listener = consumer.RebalanceListener(self)
        self._pending_rebalancing_spans: Deque[Any] = deque()
        self.tp_last_committed_at: Dict[TP, float] = {}
        app = self.consumer.app
        stream_processing_timeout = app.conf.stream_processing_timeout
        self.tp_fetch_request_timeout_secs = stream_processing_timeout
        self.tp_fetch_response_timeout_secs = stream_processing_timeout
        self.tp_stream_timeout_secs = stream_processing_timeout
        commit_livelock_timeout = app.conf.broker_commit_livelock_soft_timeout
        self.tp_commit_timeout_secs = commit_livelock_timeout

    async def on_start(self) -> None:
        """Call when consumer starts."""
        self._consumer = self._create_consumer(loop=self.thread_loop)
        self.time_started = monotonic()
        await self._consumer.start()

    async def on_thread_stop(self) -> None:
        """Call when consumer thread is stopping."""
        await super().on_thread_stop()
        if self._consumer is not None:
            await self._consumer.stop()

    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> aiokafka.AIOKafkaConsumer:
        transport = cast(Transport, self.transport)
        if self.app.client_only:
            return self._create_client_consumer(transport, loop=loop)
        else:
            return self._create_worker_consumer(transport, loop=loop)

    def _create_worker_consumer(
        self,
        transport: 'Transport',
        loop: asyncio.AbstractEventLoop
    ) -> aiokafka.AIOKafkaConsumer:
        isolation_level = 'read_uncommitted'
        conf = self.app.conf
        if self.consumer.in_transaction:
            isolation_level = 'read_committed'
        self._assignor = self.app.assignor
        auth_settings = credentials_to_aiokafka_auth(conf.broker_credentials, conf.ssl_context)
        max_poll_interval = conf.broker_max_poll_interval or 0
        request_timeout = conf.broker_request_timeout
        session_timeout = conf.broker_session_timeout
        rebalance_timeout = conf.broker_rebalance_timeout
        if session_timeout > request_timeout:
            raise ImproperlyConfigured(
                f'Setting broker_session_timeout={session_timeout} cannot be '
                f'greater than broker_request_timeout={request_timeout}'
            )
        return aiokafka.AIOKafkaConsumer(
            loop=loop,
            api_version=conf.consumer_api_version,
            client_id=conf.broker_client_id,
            group_id=conf.id,
            group_instance_id=conf.consumer_group_instance_id,
            bootstrap_servers=server_list(transport.url, transport.default_port),
            partition_assignment_strategy=[self._assignor],
            enable_auto_commit=False,
            auto_offset_reset=conf.consumer_auto_offset_reset,
            max_poll_records=conf.broker_max_poll_records,
            max_poll_interval_ms=int(max_poll_interval * 1000.0),
            max_partition_fetch_bytes=conf.consumer_max_fetch_size,
            fetch_max_wait_ms=1500,
            request_timeout_ms=int(request_timeout * 1000.0),
            check_crcs=conf.broker_check_crcs,
            session_timeout_ms=int(session_timeout * 1000.0),
            rebalance_timeout_ms=int(rebalance_timeout * 1000.0),
            heartbeat_interval_ms=int(conf.broker_heartbeat_interval * 1000.0),
            isolation_level=isolation_level,
            traced_from_parent_span=self.traced_from_parent_span,
            start_rebalancing_span=self.start_rebalancing_span,
            start_coordinator_span=self.start_coordinator_span,
            on_generation_id_known=self.on_generation_id_known,
            flush_spans=self.flush_spans,
            **auth_settings
        )

    def _create_client_consumer(
        self,
        transport: 'Transport',
        loop: asyncio.AbstractEventLoop
    ) -> aiokafka.AIOKafkaConsumer:
        conf = self.app.conf
        auth_settings = credentials_to_aiokafka_auth(conf.broker_credentials, conf.ssl_context)
        max_poll_interval = conf.broker_max_poll_interval or 0
        return aiokafka.AIOKafkaConsumer(
            loop=loop,
            client_id=conf.broker_client_id,
            bootstrap_servers=server_list(transport.url, transport.default_port),
            request_timeout_ms=int(conf.broker_request_timeout * 1000.0),
            enable_auto_commit=True,
            max_poll_records=conf.broker_max_poll_records,
            max_poll_interval_ms=int(max_poll_interval * 1000.0),
            auto_offset_reset=conf.consumer_auto_offset_reset,
            check_crcs=conf.broker_check_crcs,
            **auth_settings
        )

    @cached_property
    def trace_category(self) -> str:
        return f'{self.app.conf.name}-_aiokafka'

    def start_rebalancing_span(self) -> Any:
        return self._start_span('rebalancing', lazy=True)

    def start_coordinator_span(self) -> Any:
        return self._start_span('coordinator')

    def _start_span(self, name: str, *, lazy: bool = False) -> Any:
        tracer = self.app.tracer
        if tracer is not None:
            span = tracer.get_tracer(self.trace_category).start_span(operation_name=name)
            span.set_tag(tags.SAMPLING_PRIORITY, 1)
            self.app._span_add_default_tags(span)
            set_current_span(span)
            if lazy:
                self._transform_span_lazy(span)
            return span
        else:
            return noop_span()

    @no_type_check
    def _transform_span_lazy(self, span: Any) -> None:
        consumer = self
        if typing.TYPE_CHECKING:
            pass
        else:
            cls = span.__class__

            class LazySpan(cls):

                def finish():
                    consumer._span_finish(span)
            span._real_finish, span.finish = (span.finish, LazySpan.finish)

    def _span_finish(self, span: Any) -> None:
        assert self._consumer is not None
        if self._consumer._coordinator.generation == DEFAULT_GENERATION_ID:
            self._on_span_generation_pending(span)
        else:
            self._on_span_generation_known(span)

    def _on_span_generation_pending(self, span: Any) -> None:
        self._pending_rebalancing_spans.append(span)

    def _on_span_generation_known(self, span: Any) -> None:
        if self._consumer:
            coordinator = self._consumer._coordinator
            coordinator_id = coordinator.coordinator_id
            app_id = self.app.conf.id
            generation = coordinator.generation
            member_id = coordinator.member_id
            try:
                op_name = span.operation_name
                set_tag = span.set_tag
            except AttributeError:
                pass
            else:
                trace_id_str = f'reb-{app_id}-{generation}'
                trace_id = murmur2(trace_id_str.encode())
                span.context.trace_id = trace_id
                if op_name.endswith('.REPLACE_WITH_MEMBER_ID'):
                    span.set_operation_name(f'rebalancing node {member_id}')
                set_tag('kafka_generation', generation)
                set_tag('kafka_member_id', member_id)
                set_tag('kafka_coordinator_id', coordinator_id)
                self.app._span_add_default_tags(span)
                span._real_finish()

    def _on_span_cancelled_early(self, span: Any) -> None:
        try:
            op_name = span.operation_name
        except AttributeError:
            return
        else:
            span.set_operation_name(f'{op_name} (CANCELLED)')
            span._real_finish()

    def traced_from_parent_span(
        self,
        parent_span: Any,
        lazy: bool = False,
        **extra_context: Any
    ) -> Any:
        return traced_from_parent_span(
            parent_span,
            callback=self._transform_span_lazy if lazy else None,
            **extra_context
        )

    def flush_spans(self) -> None:
        while self._pending_rebalancing_spans:
            span = self._pending_rebalancing_spans.popleft()
            self._on_span_cancelled_early(span)

    def on_generation_id_known(self) -> None:
        while self._pending_rebalancing_spans:
            span = self._pending_rebalancing_spans.popleft()
            self._on_span_generation_known(span)

    def close(self) -> None:
        """Close consumer for graceful shutdown."""
        if self._consumer is not None:
            self._consumer.set_close()
            self._consumer._coordinator.set_close()

    async def subscribe(self, topics: Set[str]) -> None:
        """Reset subscription (requires rebalance)."""
        await self.call_thread(
            self._ensure_consumer().subscribe,
            topics=set(topics),
            listener=self._rebalance_listener
        )

    async def seek_to_committed(self) -> None:
        """Seek partitions to the last committed offset."""
        return await self.call_thread(self._ensure_consumer().seek_to_committed)

    async def commit(self, offsets: Mapping[TP, int]) -> bool:
        """Commit topic offsets."""
        return await self.call_thread(self._commit, offsets)

