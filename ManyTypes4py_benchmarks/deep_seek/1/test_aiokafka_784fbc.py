import aiokafka
import faust
import opentracing
import pytest
import random
import string
from contextlib import contextmanager
from typing import Optional, Dict, List, Set, Tuple, Any, Callable, Deque, Union, cast
from aiokafka.errors import CommitFailedError, IllegalStateError, KafkaError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from opentracing.ext import tags
from faust import auth
from faust.exceptions import ImproperlyConfigured, NotReady
from faust.sensors.monitor import Monitor
from faust.transport.drivers import aiokafka as mod
from faust.transport.drivers.aiokafka import (
    AIOKafkaConsumerThread, Consumer, ConsumerNotStarted, ConsumerRebalanceListener,
    ConsumerStoppedError, Producer, ProducerSendError, TOPIC_LENGTH_MAX, Transport,
    credentials_to_aiokafka_auth, server_list
)
from faust.types import TP
from mode.utils.futures import done_future
from mode.utils.mocks import ANY, AsyncMock, MagicMock, Mock, call, patch
from types import TracebackType
from typing_extensions import Literal

TP1: TP = TP('topic', 23)
TP2: TP = TP('topix', 23)
TESTED_MODULE: str = 'faust.transport.drivers.aiokafka'

@pytest.fixture()
def thread() -> Mock:
    return Mock(name='thread', create_topic=AsyncMock())

@pytest.fixture()
def consumer(*, thread: Mock, app: Any, callback: Mock,
             on_partitions_revoked: Mock, on_partitions_assigned: Mock) -> Consumer:
    consumer = Consumer(
        app.transport,
        callback=callback,
        on_partitions_revoked=on_partitions_revoked,
        on_partitions_assigned=on_partitions_assigned
    )
    consumer._thread = thread
    return consumer

@pytest.fixture()
def callback() -> Mock:
    return Mock(name='callback')

@pytest.fixture()
def on_partitions_revoked() -> Mock:
    return Mock(name='on_partitions_revoked')

@pytest.fixture()
def on_partitions_assigned() -> Mock:
    return Mock(name='on_partitions_assigned')

class test_ConsumerRebalanceListener:

    @pytest.fixture()
    def handler(self, *, thread: Mock) -> ConsumerRebalanceListener:
        return ConsumerRebalanceListener(thread)

    @pytest.fixture()
    def thread(self) -> Mock:
        return Mock(
            name='thread',
            on_partitions_assigned=AsyncMock(),
            on_partitions_revoked=AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_on_partitions_revoked(self, *, handler: ConsumerRebalanceListener,
                                        thread: Mock) -> None:
        await handler.on_partitions_revoked([
            TopicPartition('A', 0),
            TopicPartition('B', 3)
        ])
        thread.on_partitions_revoked.assert_called_once_with({TP('A', 0), TP('B', 3)})

    @pytest.mark.asyncio
    async def test_on_partitions_assigned(self, *, handler: ConsumerRebalanceListener,
                                         thread: Mock) -> None:
        await handler.on_partitions_assigned([
            TopicPartition('A', 0),
            TopicPartition('B', 3)
        ])
        thread.on_partitions_assigned.assert_called_once_with({TP('A', 0), TP('B', 3)})

class test_Consumer:

    @pytest.fixture()
    def thread(self) -> Mock:
        return Mock(name='thread', create_topic=AsyncMock())

    @pytest.fixture()
    def consumer(self, *, thread: Mock, app: Any, callback: Mock,
                 on_partitions_revoked: Mock, on_partitions_assigned: Mock) -> Consumer:
        consumer = Consumer(
            app.transport,
            callback=callback,
            on_partitions_revoked=on_partitions_revoked,
            on_partitions_assigned=on_partitions_assigned
        )
        consumer._thread = thread
        return consumer

    @pytest.fixture()
    def callback(self) -> Mock:
        return Mock(name='callback')

    @pytest.fixture()
    def on_partitions_revoked(self) -> Mock:
        return Mock(name='on_partitions_revoked')

    @pytest.fixture()
    def on_partitions_assigned(self) -> Mock:
        return Mock(name='on_partitions_assigned')

    @pytest.mark.asyncio
    async def test_create_topic(self, *, consumer: Consumer, thread: Mock) -> None:
        await consumer.create_topic(
            'topic', 30, 3, timeout=40.0, retention=50.0,
            compacting=True, deleting=True, ensure_created=True
        )
        thread.create_topic.assert_called_once_with(
            'topic', 30, 3, config=None, timeout=40.0,
            retention=50.0, compacting=True, deleting=True, ensure_created=True
        )

    def test__new_topicpartition(self, *, consumer: Consumer) -> None:
        tp = consumer._new_topicpartition('t', 3)
        assert isinstance(tp, TopicPartition)
        assert tp.topic == 't'
        assert tp.partition == 3

    def test__to_message(self, *, consumer: Consumer) -> None:
        record = self.mock_record(timestamp=3000, headers=[('a', b'b')])
        m = consumer._to_message(TopicPartition('t', 3), record)
        assert m.topic == record.topic
        assert m.partition == record.partition
        assert m.offset == record.offset
        assert m.timestamp == 3.0
        assert m.headers == record.headers
        assert m.key == record.key
        assert m.value == record.value
        assert m.checksum == record.checksum
        assert m.serialized_key_size == record.serialized_key_size
        assert m.serialized_value_size == record.serialized_value_size

    def test__to_message__no_timestamp(self, *, consumer: Consumer) -> None:
        record = self.mock_record(timestamp=None)
        m = consumer._to_message(TopicPartition('t', 3), record)
        assert m.timestamp is None

    def mock_record(self, topic: str = 't', partition: int = 3, offset: int = 1001,
                    timestamp: Optional[int] = None, timestamp_type: int = 1,
                    headers: Optional[List[Tuple[str, bytes]]] = None,
                    key: bytes = b'key', value: bytes = b'value',
                    checksum: int = 312, serialized_key_size: int = 12,
                    serialized_value_size: int = 40, **kwargs: Any) -> Mock:
        return Mock(
            name='record',
            topic=topic,
            partition=partition,
            offset=offset,
            timestamp=timestamp,
            timestamp_type=timestamp_type,
            headers=headers,
            key=key,
            value=value,
            checksum=checksum,
            serialized_key_size=serialized_key_size,
            serialized_value_size=serialized_value_size
        )

    @pytest.mark.asyncio
    async def test_on_stop(self, *, consumer: Consumer) -> None:
        consumer.transport._topic_waiters = {'topic': Mock()}
        await consumer.on_stop()
        assert not consumer.transport._topic_waiters

class AIOKafkaConsumerThreadFixtures:

    @pytest.fixture()
    def cthread(self, *, consumer: Consumer) -> AIOKafkaConsumerThread:
        return AIOKafkaConsumerThread(consumer)

    @pytest.fixture()
    def tracer(self, *, app: Any) -> Mock:
        tracer = app.tracer = Mock(name='tracer')
        tobj = tracer.get_tracer.return_value

        def start_span(operation_name: Optional[str] = None, **kwargs: Any) -> opentracing.Span:
            span = opentracing.Span(
                tracer=tobj,
                context=opentracing.SpanContext()
            )
            if operation_name is not None:
                span.operation_name = operation_name
                assert span.operation_name == operation_name
            return span
        tobj.start_span = start_span
        return tracer

    @pytest.fixture()
    def _consumer(self) -> Mock:
        return Mock(
            name='AIOKafkaConsumer',
            autospec=aiokafka.AIOKafkaConsumer,
            start=AsyncMock(),
            stop=AsyncMock(),
            commit=AsyncMock(),
            position=AsyncMock(),
            end_offsets=AsyncMock()
        )

    @pytest.fixture()
    def now(self) -> int:
        return 1201230410

    @pytest.fixture()
    def tp(self) -> TP:
        return TP('foo', 30)

    @pytest.fixture()
    def aiotp(self, *, tp: TP) -> TopicPartition:
        return TopicPartition(tp.topic, tp.partition)

    @pytest.fixture()
    def logger(self, *, cthread: AIOKafkaConsumerThread) -> Mock:
        cthread.log = Mock(name='cthread.log')
        return cthread.log

class test_verify_event_path_base(AIOKafkaConsumerThreadFixtures):
    last_request: Optional[float] = None
    last_response: Optional[float] = None
    highwater: Optional[int] = 1
    committed_offset: int = 1
    acks_enabled: bool = False
    stream_inbound: Optional[float] = None
    last_commit: Optional[float] = None
    expected_message: Optional[str] = None
    has_monitor: bool = True

    def _set_started(self, t: float) -> None:
        self._cthread.time_started = t

    def _set_last_request(self, last_request: float) -> None:
        self.__consumer.records_last_request[self._aiotp] = last_request

    def _set_last_response(self, last_response: float) -> None:
        self.__consumer.records_last_response[self._aiotp] = last_response

    def _set_stream_inbound(self, inbound_time: float) -> None:
        self._app.monitor.stream_inbound_time[self._tp] = inbound_time

    def _set_last_commit(self, commit_time: float) -> None:
        self._cthread.tp_last_committed_at[self._tp] = commit_time

    @pytest.fixture(autouse=True)
    def aaaa_setup_attributes(self, *, app: Any, cthread: AIOKafkaConsumerThread,
                             _consumer: Mock, now: int, tp: TP, aiotp: TopicPartition) -> None:
        self._app = app
        self._tp = tp
        self._aiotp = aiotp
        self._now = now
        self._cthread = cthread
        self.__consumer = _consumer

    @pytest.fixture(autouse=True)
    def setup_consumer(self, *, app: Any, cthread: AIOKafkaConsumerThread,
                      _consumer: Mock, now: int, tp: TP, aiotp: TopicPartition) -> None:
        assert self._tp is tp
        assert self._aiotp is aiotp
        app.topics.acks_enabled_for = Mock(name='acks_enabled_for')
        app.topics.acks_enabled_for.return_value = self.acks_enabled
        self._set_started(now)
        cthread._consumer = _consumer
        _consumer.records_last_request = {}
        if self.last_request is not None:
            self._set_last_request(self.last_request)
        _consumer.records_last_response = {}
        if self.last_response is not None:
            self._set_last_response(self.last_response)
        if self.has_monitor:
            cthread.consumer.app.monitor = Mock(name='monitor', spec=Monitor)
            app.monitor = cthread.consumer.app.monitor
            app.monitor = Mock(name='monitor', spec=Monitor)
            app.monitor.stream_inbound_time = {}
            self._set_stream_inbound(self.stream_inbound)
        else:
            app.monitor = None
        cthread.highwater = Mock(name='highwater')
        cthread.highwater.return_value = self.highwater
        cthread.consumer._committed_offset = {tp: self.committed_offset}
        cthread.tp_last_committed_at = {}
        self._set_last_commit(self.last_commit)

    def test_state(self, *, cthread: AIOKafkaConsumerThread, now: int) -> None:
        assert cthread.time_started == now

class test_VEP_no_fetch_since_start(test_verify_event_path_base):

    def test_just_started(self, *, cthread: AIOKafkaConsumerThread,
                         now: int, tp: TP, logger: Mock) -> None:
        self._set_started(now - 2.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread,
                      now: int, tp: TP, logger: Mock) -> None:
        self._set_started(now - cthread.tp_fetch_request_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_FETCH_SINCE_START, ANY, ANY
        )

class test_VEP_no_response_since_start(test_verify_event_path_base):

    def test_just_started(self, *, cthread: AIOKafkaConsumerThread,
                         _consumer: Mock, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 5.0)
        self._set_started(now - 2.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread,
                      _consumer: Mock, now: int, tp: TP, logger: Mock) -> None:
        assert cthread.verify_event_path(now, tp) is None
        self._set_last_request(now - 5.0)
        self._set_started(now - cthread.tp_fetch_response_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_RESPONSE_SINCE_START, ANY, ANY
        )

class test_VEP_no_recent_fetch(test_verify_event_path_base):

    def test_recent_fetch(self, *, cthread: AIOKafkaConsumerThread,
                         now: int, tp: TP, logger: Mock) -> None:
        self._set_last_response(now - 30.0)
        self._set_last_request(now - 2.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread,
                      now: int, tp: TP, logger: Mock) -> None:
        self._set_last_response(now - 30.0)
        self._set_last_request(now - cthread.tp_fetch_request_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_RECENT_FETCH, ANY, ANY
        )

class test_VEP_no_recent_response(test_verify_event_path_base):

    def test_recent_response(self, *, cthread: AIOKafkaConsumerThread,
                            now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 2.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread,
                      now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - cthread.tp_fetch_response_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_RECENT_RESPONSE, ANY, ANY
        )

class test_VEP_no_highwater_since_start(test_verify_event_path_base):
    highwater: Optional[int] = None

    def test_no_monitor(self, *, app: Any, cthread: AIOKafkaConsumerThread,
                       now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now)
        app.monitor = None
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_just_started(self, *, cthread: AIOKafkaConsumerThread,
                         now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread,
                      now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now - cthread.tp_stream_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.S