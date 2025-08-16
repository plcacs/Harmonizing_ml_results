import re
import collections
import faust
from faust.agents import Agent
from faust.app.base import SCAN_AGENT, SCAN_PAGE, SCAN_TASK
from faust.assignor.leader_assignor import LeaderAssignor, LeaderAssignorT
from faust.channels import Channel, ChannelT
from faust.cli.base import AppCommand
from faust.exceptions import (
    AlreadyConfiguredWarning,
    ConsumerNotStarted,
    ImproperlyConfigured,
    SameNode,
)
from faust.fixups.base import Fixup
from faust.sensors.monitor import Monitor
from faust.serializers import codecs
from faust.transport.base import Transport
from faust.transport.conductor import Conductor
from faust.transport.consumer import Consumer, Fetcher
from faust.types import TP
from faust.types.models import ModelT
from faust.types.settings import Settings
from faust.types.tables import GlobalTableT
from faust.types.web import ResourceOptions
from mode import Service
from mode.utils.compat import want_bytes
from mode.utils.mocks import ANY, AsyncMock, Mock, call, patch
from yarl import URL
import pytest
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast

TEST_TOPIC = 'test'
CONFIG_DICT: Dict[str, Any] = {
    'broker': 'kafka://foo',
    'stream_buffer_maxsize': 1,
}
CONFIG_PATH = 't.unit.app.test_base.ConfigClass'

TP1 = TP('foo', 0)
TP2 = TP('bar', 1)
TP3 = TP('baz', 2)
TP4 = TP('xuz', 3)


class ConfigClass:
    broker: str = 'kafka://foo'
    stream_buffer_maxsize: int = 1


class Key(faust.Record):
    value: int


class Value(faust.Record, serializer='json'):
    amount: float


@pytest.mark.asyncio
@pytest.mark.parametrize('key,topic_name,expected_topic,key_serializer', [
    ('key', TEST_TOPIC, TEST_TOPIC, None),
    (Key(value=10), TEST_TOPIC, TEST_TOPIC, None),
    ({'key': 'k'}, TEST_TOPIC, TEST_TOPIC, 'json'),
    (None, 'topic', 'topic', None),
    (b'key', TEST_TOPIC, TEST_TOPIC, None),
    ('key', 'topic', 'topic', None),
])
async def test_send(
        key: Union[str, bytes, Dict[str, str], Key, None],
        topic_name: str,
        expected_topic: str,
        key_serializer: Optional[str],
        app: Any) -> None:
    topic = app.topic(topic_name)
    event = Value(amount=0.0)
    await app.send(topic, key, event, key_serializer=key_serializer)
    await app.send(topic, key, event, key_serializer=key_serializer)
    expected_sender = app.producer.send
    expected_key: Optional[bytes] = None
    if key is not None:
        if isinstance(key, str):
            key_serializer = 'raw'
        if isinstance(key, ModelT):
            expected_key = key.dumps(serializer='raw')
        elif key_serializer:
            expected_key = codecs.dumps(key_serializer, key)
        else:
            expected_key = want_bytes(key)
    else:
        expected_key = None
    expected_sender.assert_called_with(
        expected_topic, expected_key, event.dumps(),
        partition=None,
        timestamp=None,
        headers={},
    )


@pytest.mark.asyncio
async def test_send_str(app: Any) -> None:
    await app.send('foo', Value(amount=0.0))


class test_App:

    def test_stream(self, *, app: Any) -> None:
        s = app.topic(TEST_TOPIC).stream()
        assert s.channel.topics == (TEST_TOPIC,)
        assert s.channel in app.topics
        assert s.channel.app == app

    def test_new_producer(self, *, app: Any) -> None:
        app._producer = None
        transport = app._transport = Mock(
            name='transport',
            autospec=Transport,
        )
        assert app._new_producer() is transport.create_producer.return_value
        transport.create_producer.assert_called_with(beacon=ANY)
        assert app.producer is transport.create_producer.return_value

    @pytest.mark.parametrize('broker_url,broker_consumer_url', [
        ('moo://', None),
        ('moo://', 'zoo://'),
    ])
    def test_new_transport(self, broker_url: str, broker_consumer_url: Optional[str], *,
                           app: Any, patching: Any) -> None:
        app.conf.broker = broker_url
        if broker_consumer_url:
            app.conf.broker_consumer = broker_consumer_url
        by_url = patching('faust.transport.by_url')
        assert app._new_transport() is by_url.return_value.return_value
        assert app.transport is by_url.return_value.return_value
        by_url.assert_called_with(app.conf.broker_consumer[0])
        by_url.return_value.assert_called_with(
            app.conf.broker_consumer, app, loop=app.loop)
        app.transport = 10
        assert app.transport == 10

    @pytest.mark.parametrize('broker_url,broker_producer_url', [
        ('moo://', None),
        ('moo://', 'zoo://'),
    ])
    def test_new_producer_transport(self, broker_url: str, broker_producer_url: Optional[str], *,
                                    app: Any, patching: Any) -> None:
        app.conf.broker = broker_url
        if broker_producer_url:
            app.conf.broker_producer = broker_producer_url
        by_url = patching('faust.transport.by_url')
        transport = app._new_producer_transport()
        assert transport is by_url.return_value.return_value
        assert app.producer_transport is by_url.return_value.return_value
        by_url.assert_called_with(app.conf.broker_producer[0])
        by_url.return_value.assert_called_with(
            app.conf.broker_producer, app, loop=app.loop)
        app.producer_transport = 10
        assert app.producer_transport == 10

    @pytest.mark.asyncio
    async def test_on_stop(self, *, app: Any) -> None:
        app._http_client = Mock(name='http_client', close=AsyncMock())
        app._producer = Mock(name='producer', flush=AsyncMock())
        await app.on_stop()
        app._http_client.close.assert_called_once_with()
        app._http_client = None
        await app.on_stop()
        app._producer = None
        await app.on_stop()

    @pytest.mark.asyncio
    async def test_stop_consumer__wait_empty_enabled(self, *, app: Any) -> None:
        app.conf.stream_wait_empty = True
        await self.assert_stop_consumer(app)
        app._consumer.wait_empty.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_stop_consumer__wait_empty_disabled(self, *, app: Any) -> None:
        app.conf.stream_wait_empty = False
        await self.assert_stop_consumer(app)
        app.consumer.wait_empty.assert_not_called()

    async def assert_stop_consumer(self, app: Any) -> None:
        consumer = app._consumer = Mock(
            wait_empty=AsyncMock(),
        )
        consumer.assignment.return_value = set()
        app.tables = Mock()
        app.flow_control = Mock()
        app._stop_fetcher = AsyncMock()
        await app._stop_consumer()

        consumer.assignment.side_effect = ConsumerNotStarted()
        await app._stop_consumer()

        consumer.assignment.side_effect = None
        assigned = {TP('foo', 0), TP('bar', 1)}
        consumer.assignment.return_value = assigned

        await app._stop_consumer()
        app.tables.on_partitions_revoked.assert_called_once_with(assigned)
        consumer.stop_flow.assert_called_once_with()
        app.flow_control.suspend.assert_called_once_with()
        app._stop_fetcher.assert_called_once_with()

    def test_on_rebalance_start__existing_state(self, *, app: Any) -> None:
        app._rebalancing_sensor_state = {'foo': 'bar'}
        app.log.warning = Mock()
        app.on_rebalance_start()

    def test_on_rebalance_return__no_state(self, *, app: Any) -> None:
        app._rebalancing_sensor_state = None
        app.log.warning = Mock()
        app.on_rebalance_return()
        app.log.warning.assert_called_once()

    def test_on_rebalance_return__has_state(self, *, app: Any) -> None:
        app._rebalancing_sensor_state = {'time_start': 100.0}
        app.sensors.on_rebalance_return = Mock()
        app.on_rebalance_return()
        app.sensors.on_rebalance_return.assert_called_once_with(
            app, app._rebalancing_sensor_state,
        )

    def test_on_rebalance_start_end(self, *, app: Any) -> None:
        app.tables = Mock()
        app.sensors = Mock()
        app.sensors.on_rebalance_start.return_value = {
            'time_start': 100.,
        }
        assert not app.rebalancing

        app.on_rebalance_start()
        assert app._rebalancing_sensor_state
        assert app.rebalancing
        app.tables.on_rebalance_start.assert_called_once_with()

        app.on_rebalance_return()

        app.on_rebalance_end()
        assert not app.rebalancing
        assert not app._rebalancing_sensor_state

        app.tracer = Mock(name='tracer')
        app.on_rebalance_start()
        span = app._rebalancing_span
        assert span is not None
        app._rebalancing_sensor_state = None
        app.log.warning = Mock()
        app.on_rebalance_end()
        span.finish.assert_called_once_with()
        app.log.warning.assert_called_once()

        assert not app.rebalancing

    def test_trace(self, *, app: Any) -> None:
        app.tracer = None
        with app.trace('foo'):
            pass
        app.tracer = Mock()
        assert app.trace('foo') is app.tracer.trace.return_value

    def test_traced(self, *, app: Any) -> None:
        @app.traced
        def foo(val: int) -> int:
            return val
        assert foo(42) == 42

    def test__start_span_from_rebalancing(self, *, app: Any) -> None:
        app.tracer = None
        app._rebalancing_span = None
        assert app._start_span_from_rebalancing('foo')
        app.tracer = Mock(name='tracer')
        try:
            app._rebalancing_span = Mock(name='span')
            assert app._start_span_from_rebalancing('foo')
        finally:
            app.tracer = None
            app._rebalancing_span = None

    @pytest.mark.asyncio
    async def test_on_partitions_revoked(self, *, app: Any) -> None:
        app.on_partitions_revoked = Mock(send=AsyncMock())
        consumer = app.consumer = Mock(
            wait_empty=AsyncMock(),
            transactions=Mock(
                on_partitions_revoked=AsyncMock(),
            ),
        )
        app.tables = Mock()
        app.flow_control = Mock()
        app._producer = Mock(flush=AsyncMock())
        revoked = {TP('foo', 0), TP('bar', 1)}
        ass = app.consumer.assignment.return_value = {TP('foo', 0)}

        app.in_transaction = False
        await app._on_partitions_revoked(revoked)

        app.on_partitions_revoked.send.assert_called_once_with(revoked)
        consumer.stop_flow.assert_called_once_with()
        app.flow_control.suspend.assert_called_once_with()
        consumer.pause_partitions.assert_called_once_with(ass)
        app.flow_control.clear.assert_called_once_with()
        consumer.wait_empty.assert_called_once_with()
        app._producer.flush.assert_called_once_with()
        consumer.transactions.on_partitions_revoked.assert_not_called()

        app.in_transaction = True
        await app._on_partitions_revoked(revoked)
        consumer.transactions.on_partitions_revoked.assert_called_once_with(
            revoked)

    @pytest.mark.asyncio
    async def test_on_partitions_revoked__no_assignment(self, *, app: Any) -> None:
        app.on_partitions_revoked = Mock(send=AsyncMock())
        app.consumer = Mock()
        app.tables = Mock()
        revoked = {TP('foo', 0), TP('bar', 1)}
        app.consumer.assignment.return_value = set()
        await app._on_partitions_revoked(revoked)

        app.on_partitions_revoked.send.assert_called_once_with(revoked)

    @pytest.mark.asyncio
    async def test_on_partitions_revoked__crashes(self, *, app: Any) -> None:
        app.on_partitions_revoked = Mock(send=AsyncMock())
        app.crash = AsyncMock()
        app.consumer = Mock()
        app.tables = Mock()
        revoked = {TP('foo', 0), TP('bar', 1)}
        app.consumer.assignment.side_effect = RuntimeError()
        await app._on_partitions_revoked(revoked)

        app.crash.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_partitions_revoked__when_stopped(self, *, app: Any) -> None:
        app._stopped.set()
        app._on_rebalance_when_stopped = AsyncMock()
        await app._on_partitions_revoked(set())
        app._on_rebalance_when_stopped.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_stop_fetcher(self, *, app: Any) -> None:
        app._fetcher = Mock(stop=AsyncMock())
        await app._stop_fetcher()
        app._fetcher.stop.assert_called_once_with()
        app._fetcher.service_reset.assert_called_once_with()

    def test_on_rebalance_when_stopped(self, *, app: Any) -> None:
        app.consumer = Mock()
        app._on_rebalance_when_stopped()
        app.consumer.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_partitions_assigned__when_stopped(self, *, app: Any) -> None:
        app._stopped.set()
        app._on_rebalance_when_stopped = AsyncMock()
        await app._on_partitions_assigned(set())
        app._on_rebalance_when_stopped.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_partitions_assigned(self, *, app: Any) -> None:
        app._assignment = {TP('foo', 1), TP('bar', 2)}
        app.on_partitions_assigned = Mock(send=AsyncMock())
        app.consumer = Mock(
            transactions=Mock(
                on_rebalance=AsyncMock(),
            ),
        )
        app.agents = Mock(
            on_rebalance=AsyncMock(),
        )
        app.tables = Mock(
            on_rebalance=AsyncMock(),
        )
        app.topics = Mock(
            maybe_wait_for_subscriptions=AsyncMock(),
            on_partitions_assigned=AsyncMock(),
        )

        assigned = {TP('foo', 1), TP('baz', 3)}
        revoked = {TP('bar', 2)}
        newly_assigned = {TP('baz', 3)}

        app.in_transaction = False
        await app._on_partitions_assigned(assigned)

        app.agents.on_rebalance.assert_called_once_with(
            revoked, newly_assigned)
        app.topics.maybe_wait_for_subscriptions.assert_called_once_with()
        app.consumer.pause_partitions.assert_called_once_with(assigned)
        app.topics.on_partitions_assigned.assert_called_once_with(assigned)
        app.consumer.transactions.on_rebalance.assert_not_called()
        app.tables.on_rebalance.assert_called_once_with(
            assigned, revoked, newly_assigned)
        app.on_partitions_assigned.send.assert_called_once_with(assigned)

        app.in_transaction = True
        app._assignment = {TP('foo', 1), TP('bar', 2)}
        await app._on_partitions_assigned(assigned)
        app.consumer.transactions.on_rebalance.assert_called_once_with(
            assigned, revoked, newly_assigned)

    @pytest.mark.asyncio
    async def test_on_partitions_assigned__crashes(self, *, app: Any) -> None:
        app._assignment = {TP('foo', 1), TP('bar', 2)}
        app.on_partitions_assigned = Mock(send=AsyncMock())
        app.consumer = Mock()
        app.agents = Mock(
            on_rebalance=AsyncMock(),
        )
        app.agents.on_rebalance.coro.side_effect = RuntimeError()
        app.crash = AsyncMock()

        await app._on_partitions_assigned(set())
        app.crash.assert_called_once()

    @pytest.mark.parametrize('prev,new,expected_revoked,expected_assigned', [
        (None, {TP1, TP2}, set(), {TP1, TP2}),
        (set(), set(), set(), set()),
        (set(), {TP1, TP2}, set(), {TP1, TP2}),
        ({TP1, TP2}, {TP1, TP2}, set(),