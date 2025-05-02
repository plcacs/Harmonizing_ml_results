import aiokafka
import faust
import opentracing
import pytest
import random
import string
from contextlib import contextmanager
from typing import Optional, Dict, List, Set, Tuple, Any, Callable, AsyncIterator, Union, Deque

from aiokafka.errors import CommitFailedError, IllegalStateError, KafkaError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from opentracing.ext import tags
from faust import auth
from faust.exceptions import ImproperlyConfigured, NotReady
from faust.sensors.monitor import Monitor
from faust.transport.drivers import aiokafka as mod
from faust.transport.drivers.aiokafka import (
    AIOKafkaConsumerThread,
    Consumer,
    ConsumerNotStarted,
    ConsumerRebalanceListener,
    ConsumerStoppedError,
    Producer,
    ProducerSendError,
    TOPIC_LENGTH_MAX,
    Transport,
    credentials_to_aiokafka_auth,
    server_list,
)
from faust.types import TP
from mode.utils.futures import done_future
from mode.utils.mocks import ANY, AsyncMock, MagicMock, Mock, call, patch

TP1: TP = TP('topic', 23)
TP2: TP = TP('topix', 23)

TESTED_MODULE: str = 'faust.transport.drivers.aiokafka'


@pytest.fixture()
def thread() -> Mock:
    return Mock(
        name='thread',
        create_topic=AsyncMock(),
    )


@pytest.fixture()
def consumer(*,
             thread: Mock,
             app: Any,
             callback: Mock,
             on_partitions_revoked: Mock,
             on_partitions_assigned: Mock) -> Consumer:
    consumer: Consumer = Consumer(
        app.transport,
        callback=callback,
        on_partitions_revoked=on_partitions_revoked,
        on_partitions_assigned=on_partitions_assigned,
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
            on_partitions_revoked=AsyncMock(),
        )

    @pytest.mark.asyncio
    async def test_on_partitions_revoked(self, *, handler: ConsumerRebalanceListener, thread: Mock) -> None:
        await handler.on_partitions_revoked([
            TopicPartition('A', 0),
            TopicPartition('B', 3),
        ])
        thread.on_partitions_revoked.assert_called_once_with({
            TP('A', 0), TP('B', 3),
        })

    @pytest.mark.asyncio
    async def test_on_partitions_assigned(self, *, handler: ConsumerRebalanceListener, thread: Mock) -> None:
        await handler.on_partitions_assigned([
            TopicPartition('A', 0),
            TopicPartition('B', 3),
        ])
        thread.on_partitions_assigned.assert_called_once_with({
            TP('A', 0), TP('B', 3),
        })


class test_Consumer:

    @pytest.fixture()
    def thread(self) -> Mock:
        return Mock(
            name='thread',
            create_topic=AsyncMock(),
        )

    @pytest.fixture()
    def consumer(self, *,
                 thread: Mock,
                 app: Any,
                 callback: Mock,
                 on_partitions_revoked: Mock,
                 on_partitions_assigned: Mock) -> Consumer:
        consumer: Consumer = Consumer(
            app.transport,
            callback=callback,
            on_partitions_revoked=on_partitions_revoked,
            on_partitions_assigned=on_partitions_assigned,
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
            'topic', 30, 3,
            timeout=40.0,
            retention=50.0,
            compacting=True,
            deleting=True,
            ensure_created=True,
        )
        thread.create_topic.assert_called_once_with(
            'topic',
            30,
            3,
            config=None,
            timeout=40.0,
            retention=50.0,
            compacting=True,
            deleting=True,
            ensure_created=True,
        )

    def test__new_topicpartition(self, *, consumer: Consumer) -> None:
        tp: TopicPartition = consumer._new_topicpartition('t', 3)
        assert isinstance(tp, TopicPartition)
        assert tp.topic == 't'
        assert tp.partition == 3

    def test__to_message(self, *, consumer: Consumer) -> None:
        record: Mock = self.mock_record(
            timestamp=3000,
            headers=[('a', b'b')],
        )
        m: Any = consumer._to_message(TopicPartition('t', 3), record)
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
        record: Mock = self.mock_record(timestamp=None)
        m: Any = consumer._to_message(TopicPartition('t', 3), record)
        assert m.timestamp is None

    def mock_record(self,
                    topic: str = 't',
                    partition: int = 3,
                    offset: int = 1001,
                    timestamp: Optional[int] = None,
                    timestamp_type: int = 1,
                    headers: Optional[List[Tuple[str, bytes]]] = None,
                    key: bytes = b'key',
                    value: bytes = b'value',
                    checksum: int = 312,
                    serialized_key_size: int = 12,
                    serialized_value_size: int = 40,
                    **kwargs: Any) -> Mock:
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
            serialized_value_size=serialized_value_size,
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
        tracer: Mock = app.tracer = Mock(name='tracer')
        tobj: Mock = tracer.get_tracer.return_value

        def start_span(operation_name: Optional[str] = None, **kwargs: Any) -> opentracing.Span:
            span: opentracing.Span = opentracing.Span(
                tracer=tobj,
                context=opentracing.SpanContext(),
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
            end_offsets=AsyncMock(),
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
    highwater: int = 1
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
    def aaaa_setup_attributes(self, *,
                              app: Any,
                              cthread: AIOKafkaConsumerThread,
                              _consumer: Mock,
                              now: int,
                              tp: TP,
                              aiotp: TopicPartition) -> None:
        self._app = app
        self._tp = tp
        self._aiotp = aiotp
        self._now = now
        self._cthread = cthread
        self.__consumer = _consumer

    @pytest.fixture(autouse=True)
    def setup_consumer(self, *, app: Any, cthread: AIOKafkaConsumerThread, _consumer: Mock, now: int, tp: TP, aiotp: TopicPartition) -> None:
        assert self._tp is tp
        assert self._aiotp is aiotp
        # patch self.acks_enabledc
        app.topics.acks_enabled_for = Mock(name='acks_enabled_for')
        app.topics.acks_enabled_for.return_value = self.acks_enabled

        # patch consumer.time_started
        self._set_started(now)

        # connect underlying AIOKafkaConsumer object.
        cthread._consumer = _consumer

        # patch AIOKafkaConsumer.records_last_request to self.last_request
        _consumer.records_last_request = {}
        if self.last_request is not None:
            self._set_last_request(self.last_request)

        # patch AIOKafkaConsumer.records_last_response to self.last_response
        _consumer.records_last_response = {}
        if self.last_response is not None:
            self._set_last_response(self.last_response)

        # patch app.monitor
        if self.has_monitor:
            cthread.consumer.app.monitor = Mock(name='monitor', spec=Monitor)
            app.monitor = cthread.consumer.app.monitor
            app.monitor = Mock(name='monitor', spec=Monitor)
            # patch monitor.stream_inbound_time
            # this is the time when a stream last processed a record
            # for tp
            app.monitor.stream_inbound_time = {}
            self._set_stream_inbound(self.stream_inbound)
        else:
            app.monitor = None

        # patch highwater
        cthread.highwater = Mock(name='highwater')
        cthread.highwater.return_value = self.highwater

        # patch committed offset
        cthread.consumer._committed_offset = {
            tp: self.committed_offset,
        }

        cthread.tp_last_committed_at = {}
        self._set_last_commit(self.last_commit)

    def test_state(self, *, cthread: AIOKafkaConsumerThread, now: int) -> None:
        # verify that setup_consumer fixture was applied
        assert cthread.time_started == now


class test_VEP_no_fetch_since_start(test_verify_event_path_base):

    def test_just_started(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_started(now - 2.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_started(
            now - cthread.tp_fetch_request_timeout_secs * 2,
        )
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_FETCH_SINCE_START,
            ANY, ANY,
        )


class test_VEP_no_response_since_start(test_verify_event_path_base):

    def test_just_started(self, *, cthread: AIOKafkaConsumerThread, _consumer: Mock, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 5.0)
        self._set_started(now - 2.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread, _consumer: Mock, now: int, tp: TP, logger: Mock) -> None:
        assert cthread.verify_event_path(now, tp) is None
        self._set_last_request(now - 5.0)
        self._set_started(
            now - cthread.tp_fetch_response_timeout_secs * 2,
        )
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_RESPONSE_SINCE_START,
            ANY, ANY,
        )


class test_VEP_no_recent_fetch(test_verify_event_path_base):

    def test_recent_fetch(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_response(now - 30.0)
        self._set_last_request(now - 2.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_response(now - 30.0)
        self._set_last_request(now - cthread.tp_fetch_request_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_RECENT_FETCH,
            ANY, ANY,
        )


class test_VEP_no_recent_response(test_verify_event_path_base):

    def test_recent_response(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 2.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(
            now - cthread.tp_fetch_response_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_RECENT_RESPONSE,
            ANY, ANY,
        )


class test_VEP_no_highwater_since_start(test_verify_event_path_base):
    highwater: Optional[int] = None

    def test_no_monitor(self, *, app: Any, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now)
        app.monitor = None
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_just_started(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now - cthread.tp_stream_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_called_with(
            mod.SLOW_PROCESSING_NO_HIGHWATER_SINCE_START,
            ANY, ANY,
        )


class test_VEP_stream_idle_no_highwater(test_verify_event_path_base):

    highwater: int = 10
    committed_offset: int = 10

    def test_highwater_same_as_offset(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now - 300.0)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()


class test_VEP_stream_idle_highwater_no_acks(
        test_verify_event_path_base):
    acks_enabled: bool = False

    def test_no_acks(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()


class test_VEP_stream_idle_highwater_same_has_acks_everything_OK(
        test_verify_event_path_base):
    highwater: int = 10
    committed_offset: int = 10
    inbound_time: Optional[float] = None
    acks_enabled: bool = True

    def test_main(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()


class test_VEP_stream_idle_highwater_no_inbound(
        test_verify_event_path_base):
    highwater: int = 20
    committed_offset: int = 10
    inbound_time: Optional[float] = None
    acks_enabled: bool = True

    def test_just_started(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out_since_start(self, *, app: Any, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now - cthread.tp_stream_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        expected_message: str = cthread._make_slow_processing_error(
            mod.SLOW_PROCESSING_STREAM_IDLE_SINCE_START,
            [mod.SLOW_PROCESSING_CAUSE_STREAM,
             mod.SLOW_PROCESSING_CAUSE_AGENT])
        logger.error.assert_called_once_with(
            expected_message,
            tp, ANY,
            setting='stream_processing_timeout',
            current_value=app.conf.stream_processing_timeout,
        )

    def test_has_inbound(self, *, app: Any, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now - cthread.tp_stream_timeout_secs * 2)
        self._set_stream_inbound(now)
        self._set_last_commit(now)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_inbound_timed_out(self, *, app: Any, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now - cthread.tp_stream_timeout_secs * 4)
        self._set_stream_inbound(now - cthread.tp_stream_timeout_secs * 2)
        self._set_last_commit(now)
        assert cthread.verify_event_path(now, tp) is None
        expected_message: str = cthread._make_slow_processing_error(
            mod.SLOW_PROCESSING_STREAM_IDLE,
            [mod.SLOW_PROCESSING_CAUSE_STREAM,
             mod.SLOW_PROCESSING_CAUSE_AGENT])
        logger.error.assert_called_once_with(
            expected_message,
            tp, ANY,
            setting='stream_processing_timeout',
            current_value=app.conf.stream_processing_timeout,
        )


class test_VEP_no_commit(test_verify_event_path_base):
    highwater: int = 20
    committed_offset: int = 10
    inbound_time: Optional[float] = None
    acks_enabled: bool = True

    def _configure(self, now: int, cthread: AIOKafkaConsumerThread) -> None:
        self._set_last_request(now - 10.0)
        self._set_last_response(now - 5.0)
        self._set_started(now - cthread.tp_stream_timeout_secs * 4)
        self._set_stream_inbound(now - 0.01)

    def test_just_started(self, *, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._configure(now, cthread)
        self._set_last_commit(None)
        self._set_started(now)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()

    def test_timed_out_since_start(self, *, app: Any, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._configure(now, cthread)
        self._set_last_commit(None)
        self._set_started(now - cthread.tp_commit_timeout_secs * 2)
        assert cthread.verify_event_path(now, tp) is None
        expected_message: str = cthread._make_slow_processing_error(
            mod.SLOW_PROCESSING_NO_COMMIT_SINCE_START,
            [mod.SLOW_PROCESSING_CAUSE_COMMIT],
        )
        logger.error.assert_called_once_with(
            expected_message,
            tp, ANY,
            setting='broker_commit_livelock_soft_timeout',
            current_value=app.conf.broker_commit_livelock_soft_timeout,
        )

    def test_timed_out_since_last(self, *, app: Any, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._configure(now, cthread)
        self._set_last_commit(cthread.tp_commit_timeout_secs * 2)
        self._set_started(now - cthread.tp_commit_timeout_secs * 4)
        assert cthread.verify_event_path(now, tp) is None
        expected_message: str = cthread._make_slow_processing_error(
            mod.SLOW_PROCESSING_NO_RECENT_COMMIT,
            [mod.SLOW_PROCESSING_CAUSE_COMMIT],
        )
        logger.error.assert_called_once_with(
            expected_message,
            tp, ANY,
            setting='broker_commit_livelock_soft_timeout',
            current_value=app.conf.broker_commit_livelock_soft_timeout,
        )

    def test_committing_fine(self, *, app: Any, cthread: AIOKafkaConsumerThread, now: int, tp: TP, logger: Mock) -> None:
        self._configure(now, cthread)
        self._set_last_commit(now - 2.0)
        self._set_started(now - cthread.tp_commit_timeout_secs * 4)
        assert cthread.verify_event_path(now, tp) is None
        logger.error.assert_not_called()


class test_AIOKafkaConsumerThread(AIOKafkaConsumerThreadFixtures):

    def test_constructor(self, *, cthread: AIOKafkaConsumerThread) -> None:
        assert cthread._partitioner
        assert cthread._rebalance_listener

    @pytest.mark.asyncio
    async def test_on_start(self, *, cthread: AIOKafkaConsumerThread, _consumer: Mock) -> None:
        cthread._create_consumer = Mock(
            name='_create_consumer',
            return_value=_consumer,
        )
        await cthread.on_start()

        assert cthread._consumer is cthread._create_consumer.return_value
        cthread._create_consumer.assert_called_once_with(
            loop=cthread.thread_loop)
        cthread._consumer.start.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_thread_stop(self, *, cthread: AIOKafkaConsumerThread, _consumer: Mock) -> None:
        cthread._consumer = _consumer
        await cthread.on_thread_stop()
        cthread._consumer.stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_thread_stop__consumer_not_started(self, *, cthread: AIOKafkaConsumerThread) -> None:
        cthread._consumer = None
        await cthread.on_thread_stop()

    def test__create_consumer__client(self, *, cthread: AIOKafkaConsumerThread, app: Any) -> None:
        app.client_only = True
        loop: Mock = Mock(name='loop')
        cthread._create_client_consumer = Mock(name='_create_client_consumer')
        c: Any = cthread._create_consumer(loop=loop)
        assert c is cthread._create_client_consumer.return_value
        cthread._create_client_consumer.assert_called_once_with(
            cthread.transport, loop=loop)

    def test__create_consumer__worker(self, *, cthread: AIOKafkaConsumerThread, app: Any) -> None:
        app.client_only = False
        loop: Mock = Mock(name='loop')
        cthread._create_worker_consumer = Mock(name='_create_worker_consumer')
        c: Any = cthread._create_consumer(loop=loop)
        assert c is cthread._create_worker_consumer.return_value
        cthread._create_worker_consumer.assert_called_once_with(
            cthread.transport, loop=loop)

    def test_session_gt_request_timeout(self, *, cthread: AIOKafkaConsumerThread, app: Any) -> None:
        app.conf.broker_session_timeout = 90
        app.conf.broker_request_timeout = 10

        with pytest.raises(ImproperlyConfigured):
            self.assert_create_worker_consumer(
                cthread, app,
                in_transaction=False)

    def test__create_worker_consumer(self, *, cthread: AIOKafkaConsumerThread, app: Any) -> None:
        self.assert_create_worker_consumer(
            cthread, app,
            in_transaction=False,
            isolation_level='read_uncommitted',
        )

    def test__create_worker_consumer__transaction(self, *, cthread: AIOKafkaConsumerThread, app: Any) -> None:
        self.assert_create_worker_consumer(
            cthread, app,
            in_transaction=True,
            isolation_level='read_committed',
        )

    def assert_create_worker_consumer(self, cthread: AIOKafkaConsumerThread, app: Any,
                                      in_transaction: bool = False,
                                      isolation_level: str = 'read_uncommitted',
                                      api_version: Optional[str] = None) -> None:
        loop: Mock = Mock(name='loop')
        transport: Any = cthread.transport
        conf: Any = app.conf
        cthread.consumer.in_transaction = in_transaction
        auth_settings: Dict[str, Any] = credentials_to_aiokafka_auth(
            conf.broker_credentials, conf.ssl_context)
        with patch('aiokafka.AIOKafkaConsumer') as AIOKafkaConsumer:
            c: Any = cthread._create_worker_consumer(transport, loop)
            assert c is AIOKafkaConsumer.return_value
            max_poll_interval: float = conf.broker_max_poll_interval
            AIOKafkaConsumer.assert_called_once_with(
                loop=loop,
                api_version=app.conf.consumer_api_version,
                client_id=conf.broker_client_id,
                group_id=conf.id,
                group_instance_id=conf.consumer_group_instance_id,
                bootstrap_servers=server_list(
                    transport.url, transport.default_port),
                partition_assignment_strategy=[cthread._assignor],
                enable_auto_commit=False,
                auto_offset_reset=conf.consumer_auto_offset_reset,
                max_poll_records=conf.broker_max_poll_records,
                max_poll_interval_ms=int(max_poll_interval * 1000.0),
                max_partition_fetch_bytes=conf.consumer_max_fetch_size,
                fetch_max_wait_ms=1500,
                request_timeout_ms=int(conf.broker_request_timeout * 1000.0),
                rebalance_timeout_ms=int(
                    conf.broker_rebalance_timeout * 1000.0),
                check_crcs=conf.broker_check_crcs,
                session_timeout_ms=int(conf.broker_session_timeout * 1000.0),
                heartbeat_interval_ms=int(
                    conf.broker_heartbeat_interval * 1000.0),
                isolation_level=isolation_level,
                traced_from_parent_span=cthread.traced_from_parent_span,
                start_rebalancing_span=cthread.start_rebalancing_span,
                start_coordinator_span=cthread.start_coordinator_span,
                on_generation_id_known=cthread.on_generation_id_known,
                flush_spans=cthread.flush_spans,
                **auth_settings,
            )

    def test__create_client_consumer(self, *, cthread: AIOKafkaConsumerThread, app: Any) -> None:
        loop: Mock = Mock(name='loop')
        transport: Any = cthread.transport
        conf: Any = app.conf
        auth_settings: Dict[str, Any] = credentials_to_aiokafka_auth(
            conf.broker_credentials, conf.ssl_context)
        with patch('aiokafka.AIOKafkaConsumer') as AIOKafkaConsumer:
            c: Any = cthread._create_client_consumer(transport, loop)
            max_poll_interval: float = conf.broker_max_poll_interval
            assert c is AIOKafkaConsumer.return_value
            AIOKafkaConsumer.assert_called_once_with(
                loop=loop,
                client_id=conf.broker_client_id,
                bootstrap_servers=server_list(
                    transport.url, transport.default_port),
                request_timeout_ms=int(conf.broker_request_timeout * 1000.0),
                max_poll_interval_ms=int(max_poll_interval * 1000.0),
                enable_auto_commit=True,
                max_poll_records=conf.broker_max_poll_records,
                auto_offset_reset=conf.consumer_auto_offset_reset,
                check_crcs=conf.broker_check_crcs,
                **auth_settings,
            )

    def test__start_span(self, *, cthread: AIOKafkaConsumerThread, app: Any) -> None:
        with patch(TESTED_MODULE + '.set_current_span') as s:
            app.tracer = Mock(name='tracer')
            span: opentracing.Span = cthread._start_span('test')
            app.tracer.get_tracer.assert_called_once_with(
                f'{app.conf.name}-_aiokafka')
            tracer: Mock = app.tracer.get_tracer.return_value
            tracer.start_span.assert_called_once_with(
                operation_name='test')
           