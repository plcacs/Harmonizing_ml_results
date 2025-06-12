import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union, cast
import pytest
from faust import App
from faust.app._attached import Attachments
from faust.exceptions import AlreadyConfiguredWarning
from faust.tables.manager import TableManager
from faust.transport.base import Producer, Transport
from faust.transport.consumer import Consumer, ConsumerThread, Fetcher, ProducerSendError, ThreadDelegateConsumer, TransactionManager
from faust.transport.conductor import Conductor
from faust.types import Message, TP
from mode import Service
from mode.threads import MethodQueue
from mode.utils.futures import done_future
from mode.utils.mocks import ANY, AsyncMock, Mock, call, patch

TP1: TP = TP('foo', 0)
TP2: TP = TP('foo', 1)
TP3: TP = TP('bar', 3)

class test_Fetcher:

    @pytest.fixture
    def consumer(self) -> Mock:
        return Mock(name='consumer', autospec=Consumer, _drain_messages=AsyncMock())

    @pytest.fixture
    def fetcher(self, *, app: App, consumer: Mock) -> Fetcher:
        fetcher = Fetcher(app)
        app.consumer = consumer
        fetcher.loop = asyncio.get_event_loop()
        return fetcher

    @pytest.mark.asyncio
    async def test_fetcher(self, *, fetcher: Fetcher, app: App) -> None:
        await fetcher._fetcher(fetcher)
        app.consumer._drain_messages.assert_called_once_with(fetcher)

    @pytest.mark.asyncio
    async def test_fetcher__raises_CancelledError(self, *, fetcher: Fetcher, app: App) -> None:
        app.consumer._drain_messages.side_effect = asyncio.CancelledError
        await fetcher._fetcher(fetcher)
        app.consumer._drain_messages.assert_called_once_with(fetcher)

    @pytest.mark.asyncio
    async def test_on_stop__no_drainer(self, *, fetcher: Fetcher) -> None:
        fetcher._drainer = None
        await fetcher.on_stop()

    @pytest.mark.asyncio
    async def test_on_stop__drainer_done(self, *, fetcher: Fetcher) -> None:
        fetcher._drainer = Mock(done=Mock(return_value=True))
        await fetcher.on_stop()

    @pytest.mark.asyncio
    async def test_on_stop_drainer__drainer_done2(self, *, fetcher: Fetcher) -> None:
        fetcher._drainer = Mock(done=Mock(return_value=False))
        with patch('asyncio.wait_for', AsyncMock()) as wait_for:
            wait_for.coro.return_value = None
            await fetcher.on_stop()
        fetcher._drainer.cancel.assert_called_once_with()
        assert wait_for.call_count

    @pytest.mark.asyncio
    async def test_on_stop__drainer_pending(self, *, fetcher: Fetcher) -> None:
        fetcher._drainer = Mock(done=Mock(return_value=False))
        with patch('asyncio.wait_for', AsyncMock()) as wait_for:
            await fetcher.on_stop()
            wait_for.assert_called_once_with(fetcher._drainer, timeout=1.0)

    @pytest.mark.asyncio
    async def test_on_stop__drainer_raises_StopIteration(self, *, fetcher: Fetcher) -> None:
        fetcher._drainer = Mock(done=Mock(return_value=False))
        with patch('asyncio.wait_for', AsyncMock()) as wait_for:
            wait_for.side_effect = StopIteration()
            await fetcher.on_stop()
            wait_for.assert_called_once_with(fetcher._drainer, timeout=1.0)

    @pytest.mark.asyncio
    async def test_on_stop__drainer_raises_CancelledError(self, *, fetcher: Fetcher) -> None:
        fetcher._drainer = Mock(done=Mock(return_value=False))
        with patch('asyncio.wait_for', AsyncMock()) as wait_for:
            wait_for.coro.side_effect = asyncio.CancelledError()
            await fetcher.on_stop()
            wait_for.assert_called_once_with(fetcher._drainer, timeout=1.0)

    @pytest.mark.asyncio
    async def test_on_stop__drainer_raises_TimeoutError(self, *, fetcher: Fetcher) -> None:
        fetcher._drainer = Mock(done=Mock(return_value=False))
        with patch('asyncio.wait_for', AsyncMock()) as wait_for:
            wait_for.coro.side_effect = [asyncio.TimeoutError(), asyncio.TimeoutError(), None]
            await fetcher.on_stop()
            wait_for.assert_called_with(fetcher._drainer, timeout=1.0)
            assert wait_for.call_count == 3

class test_TransactionManager:

    @pytest.fixture()
    def consumer(self) -> Mock:
        return Mock(name='consumer', spec=Consumer)

    @pytest.fixture()
    def producer(self) -> Mock:
        return Mock(name='producer', spec=Producer, create_topic=AsyncMock(), stop_transaction=AsyncMock(), maybe_begin_transaction=AsyncMock(), commit_transactions=AsyncMock(), send=AsyncMock(), flush=AsyncMock())

    @pytest.fixture()
    def transport(self, *, app: App) -> Mock:
        return Mock(name='transport', spec=Transport, app=app)

    @pytest.fixture()
    def manager(self, *, consumer: Mock, producer: Mock, transport: Mock) -> TransactionManager:
        return TransactionManager(transport, consumer=consumer, producer=producer)

    @pytest.mark.asyncio
    async def test_on_partitions_revoked(self, *, manager: TransactionManager) -> None:
        manager.flush = AsyncMock()
        await manager.on_partitions_revoked({TP1})
        manager.flush.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_rebalance(self, *, manager: TransactionManager) -> None:
        TP3_group: int = 0
        TP2_group: int = 2
        manager.app.assignor._topic_groups = {TP3.topic: TP3_group, TP2.topic: TP2_group}
        assert TP3.topic != TP2.topic
        manager._stop_transactions = AsyncMock()
        manager._start_transactions = AsyncMock()
        assigned: Set[TP] = {TP2}
        revoked: Set[TP] = {TP3}
        newly_assigned: Set[TP] = {TP2}
        await manager.on_rebalance(assigned, revoked, newly_assigned)
        manager._stop_transactions.assert_called_once_with([f'{manager.app.conf.id}-{TP3_group}-{TP3.partition}'])
        manager._start_transactions.assert_called_once_with([f'{manager.app.conf.id}-{TP2_group}-{TP2.partition}'])
        await manager.on_rebalance(set(), set(), set())

    @pytest.mark.asyncio
    async def test__stop_transactions(self, *, manager: TransactionManager, producer: Mock) -> None:
        await manager._stop_transactions(['0-0', '1-0'])
        producer.stop_transaction.assert_has_calls([call('0-0'), call.coro('0-0'), call('1-0'), call.coro('1-0')])

    @pytest.mark.asyncio
    async def test_start_transactions(self, *, manager: TransactionManager, producer: Mock) -> None:
        manager._start_new_producer = AsyncMock()
        await manager._start_transactions(['0-0', '1-0'])
        producer.maybe_begin_transaction.assert_has_calls([call('0-0'), call.coro('0-0'), call('1-0'), call.coro('1-0')])

    @pytest.mark.asyncio
    async def test_send(self, *, manager: TransactionManager, producer: Mock) -> None:
        manager.app.assignor._topic_groups = {'t': 3}
        manager.consumer.key_partition.return_value = 1
        await manager.send('t', 'k', 'v', partition=None, headers=None, timestamp=None)
        manager.consumer.key_partition.assert_called_once_with('t', 'k', None)
        producer.send.assert_called_once_with('t', 'k', 'v', 1, None, None, transactional_id='testid-3-1')

    @pytest.mark.asyncio
    async def test_send__topic_not_transactive(self, *, manager: TransactionManager, producer: Mock) -> None:
        manager.app.assignor._topic_groups = {'t': 3}
        manager.consumer.key_partition.return_value = None
        await manager.send('t', 'k', 'v', partition=None, headers=None, timestamp=None)
        manager.consumer.key_partition.assert_called_once_with('t', 'k', None)
        producer.send.assert_called_once_with('t', 'k', 'v', None, None, None, transactional_id=None)

    def test_send_soon(self, *, manager: TransactionManager) -> None:
        with pytest.raises(NotImplementedError):
            manager.send_soon(Mock(name='FutureMessage'))

    @pytest.mark.asyncio
    async def test_send_and_wait(self, *, manager: TransactionManager) -> None:
        on_send = Mock()

        async def send(*args: Any, **kwargs: Any) -> Any:
            on_send(*args, **kwargs)
            return done_future()
        manager.send = send
        await manager.send_and_wait('t', 'k', 'v', 3, 43.2, {})
        on_send.assert_called_once_with('t', 'k', 'v', 3, 43.2, {})

    @pytest.mark.asyncio
    async def test_commit(self, *, manager: TransactionManager, producer: Mock) -> None:
        manager.app.assignor._topic_groups = {'foo': 1, 'bar': 2}
        await manager.commit({TP('foo', 0): 3003, TP('bar', 0): 3004, TP('foo', 3): 4004, TP('foo', 1): 4005}, start_new_transaction=False)
        producer.commit_transactions.assert_called_once_with({'testid-1-0': {TP('foo', 0): 3003}, 'testid-1-3': {TP('foo', 3): 4004}, 'testid-1-1': {TP('foo', 1): 4005}, 'testid-2-0': {TP('bar', 0): 3004}}, 'testid', start_new_transaction=False)

    @pytest.mark.asyncio
    async def test_commit__empty(self, *, manager: TransactionManager) -> None:
        await manager.commit({}, start_new_transaction=False)

    def test_key_partition(self, *, manager: TransactionManager) -> None:
        with pytest.raises(NotImplementedError):
            manager.key_partition('topic', 'key')

    @pytest.mark.asyncio
    async def test_flush(self, *, manager: TransactionManager, producer: Mock) -> None:
        await manager.flush()
        producer.flush.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_create_topic(self, *, manager: TransactionManager) -> None:
        await manager.create_topic(topic='topic', partitions=100, replication=3, config={'C': 1}, timeout=30.0, retention=40.0, compacting=True, deleting=True, ensure_created=True)
        manager.producer.create_topic.assert_called_once_with('topic', 100, 3, config={'C': 1}, timeout=30.0, retention=40.0, compacting=True, deleting=True, ensure_created=True)

    def test_supports_headers(self, *, manager: TransactionManager) -> bool:
        ret = manager.supports_headers()
        assert ret is manager.producer.supports_headers.return_value
        return ret

class MockedConsumerAbstractMethods:

    def assignment(self) -> Set[TP]:
        return self.current_assignment

    def position(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def create_topic(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def earliest_offsets(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def highwater(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def highwaters(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def _getmany(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def _seek(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def _to_message(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def seek_to_committed(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def seek_wait(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def subscribe(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def seek_to_beginning(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def _commit(self, offsets: Dict[TP, int]) -> Any:
        ...

    def topic_partitions(self, topic: str) -> Any:
        ...

    def _new_topicpartition(self, topic: str, partition: int) -> TP:
        return TP(topic, partition)

    def key_partition(self, *args: Any, **kwargs: Any) -> Any:
        ...

class MyConsumer(MockedConsumerAbstractMethods, Consumer):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.current_assignment: Set[TP] = set()
        super().__init__(*args, **kwargs)

class test_Consumer:

    @pytest.fixture
    def callback(self) -> Mock:
        return Mock(name='callback')

    @pytest.fixture
    def on_P_revoked(self) -> Mock:
        return Mock(name='on_partitions_revoked')

    @pytest.fixture
    def on_P_assigned(self) -> Mock:
        return Mock(name='on_partitions_assigned')

    @pytest.fixture
    def consumer(self, *, app: App, callback: Mock, on_P_revoked: Mock, on_P_assigned: Mock) -> MyConsumer:
        return MyConsumer(app.transport, callback=callback, on_partitions_revoked=on_P_revoked, on_partitions_assigned=on_P_assigned)

    @pytest.fixture
    def message(self) -> Mock:
        return Mock(name='message', autospec=Message)

    def test_on_init_dependencies__default(self, *, consumer: MyConsumer) -> None:
        consumer.in_transaction = False
        assert consumer.on_init_dependencies() == []

    def test_on_init_dependencies__exactly_once(self, *, consumer: MyConsumer) -> None:
        consumer.in_transaction = True
        assert consumer.on_init_dependencies() == [consumer.transactions]

    @pytest.mark.asyncio
    async def test_getmany__stopped_after_wait(self, *, consumer: MyConsumer) -> None:
        consumer._wait_next_records = AsyncMock()

        async def on_wait(timeout: float) -> Tuple[Optional[Dict[TP, List[Any]]], Optional[Set[TP]]]:
            consumer._stopped.set()
            return (None, None)
        consumer._wait_next_records.side_effect = on_wait
        assert [a async for a in consumer.getmany(1.0)] == []

    @pytest.mark.asyncio
    async def test_getmany__flow_inactive(self, *, consumer: MyConsumer) -> None:
        consumer._wait_next_records = AsyncMock(return_value=({TP1: ['A', 'B', 'C']}, {TP1}))
        consumer.flow_active = False
        assert [a async for a in consumer.getmany(1.0)] == []

    @pytest.mark.asyncio
    async def test_getmany__flow_inactive2(self, *, consumer: MyConsumer) -> None:
        consumer._wait_next_records = AsyncMock(return_value=({TP1: ['A', 'B', 'C'], TP2: ['D']}, {TP1}))
        consumer.scheduler = Mock()

        def se(records: Dict[TP, List[Any]]) -> AsyncGenerator[Tuple[TP, Any], None]:
            for value in records.items():
                yield value
                consumer.flow_active = False
        consumer.scheduler.iterate.side_effect = se
        consumer.flow_active = True
        res = [a async for a in consumer.getmany(1.0)]
        assert res
        assert len(res) == 1

    @pytest.mark.asyncio
    async def test_getmany(self, *, consumer: MyConsumer) -> None:

        def to_message(tp: TP, record: Any) -> Any:
            return record
        consumer._to_message = to_message
        self._setup_records(consumer, active_partitions={TP1, TP2}, records={TP1: ['A', 'B', 'C'], TP2: ['D', 'E', 'F', 'G'], TP3: ['H', 'I', 'J']})
        assert not consumer.should_stop
        consumer.flow_active = False
        consumer.can_resume_flow.set()
        assert [a async for a in consumer.getmany(1.0)] == []
        assert not consumer.should_stop
        consumer.flow_active = True
        assert [a async for a in consumer.get