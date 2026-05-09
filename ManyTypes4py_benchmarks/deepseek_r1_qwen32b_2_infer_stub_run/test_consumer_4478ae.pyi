import asyncio
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

TP1 = TP('foo', 0)
TP2 = TP('foo', 1)
TP3 = TP('bar', 3)

class test_Fetcher:
    @pytest.fixture
    def consumer(self) -> Mock:
        ...

    @pytest.fixture
    def fetcher(self, app: App, consumer: Mock) -> Fetcher:
        ...

    @pytest.mark.asyncio
    async def test_fetcher(self, fetcher: Fetcher, app: App) -> None:
        ...

    @pytest.mark.asyncio
    async def test_fetcher__raises_CancelledError(self, fetcher: Fetcher, app: App) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_stop__no_drainer(self, fetcher: Fetcher) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_stop__drainer_done(self, fetcher: Fetcher) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_stop_drainer__drainer_done2(self, fetcher: Fetcher) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_stop__drainer_pending(self, fetcher: Fetcher) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_stop__drainer_raises_StopIteration(self, fetcher: Fetcher) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_stop__drainer_raises_CancelledError(self, fetcher: Fetcher) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_stop__drainer_raises_TimeoutError(self, fetcher: Fetcher) -> None:
        ...

class test_TransactionManager:
    @pytest.fixture
    def consumer(self) -> Mock:
        ...

    @pytest.fixture
    def producer(self) -> Mock:
        ...

    @pytest.fixture
    def transport(self, app: App) -> Mock:
        ...

    @pytest.fixture
    def manager(self, transport: Mock, consumer: Mock, producer: Mock) -> TransactionManager:
        ...

    @pytest.mark.asyncio
    async def test_on_partitions_revoked(self, manager: TransactionManager) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_rebalance(self, manager: TransactionManager) -> None:
        ...

    @pytest.mark.asyncio
    async def test__stop_transactions(self, manager: TransactionManager, producer: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_start_transactions(self, manager: TransactionManager, producer: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_send(self, manager: TransactionManager, producer: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_send__topic_not_transactive(self, manager: TransactionManager, producer: Mock) -> None:
        ...

    def test_send_soon(self, manager: TransactionManager) -> None:
        ...

    @pytest.mark.asyncio
    async def test_send_and_wait(self, manager: TransactionManager) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit(self, manager: TransactionManager, producer: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit__empty(self, manager: TransactionManager) -> None:
        ...

    def test_key_partition(self, manager: TransactionManager) -> None:
        ...

    @pytest.mark.asyncio
    async def test_flush(self, manager: TransactionManager, producer: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_create_topic(self, manager: TransactionManager) -> None:
        ...

    def test_supports_headers(self, manager: TransactionManager) -> None:
        ...

class MockedConsumerAbstractMethods:
    def assignment(self) -> set:
        ...

    def position(self, *args, **kwargs) -> None:
        ...

    async def create_topic(self, *args, **kwargs) -> None:
        ...

    def earliest_offsets(self, *args, **kwargs) -> None:
        ...

    def highwater(self, *args, **kwargs) -> None:
        ...

    def highwaters(self, *args, **kwargs) -> None:
        ...

    async def _getmany(self, *args, **kwargs) -> None:
        ...

    async def _seek(self, *args, **kwargs) -> None:
        ...

    def _to_message(self, *args, **kwargs) -> None:
        ...

    async def seek_to_committed(self, *args, **kwargs) -> None:
        ...

    async def seek_wait(self, *args, **kwargs) -> None:
        ...

    async def subscribe(self, *args, **kwargs) -> None:
        ...

    async def seek_to_beginning(self, *args, **kwargs) -> None:
        ...

    async def _commit(self, offsets: dict) -> None:
        ...

    def topic_partitions(self, topic: str) -> None:
        ...

    def _new_topicpartition(self, topic: str, partition: int) -> TP:
        ...

    def key_partition(self, *args, **kwargs) -> None:
        ...

class MyConsumer(MockedConsumerAbstractMethods, Consumer):
    def __init__(self, *args, **kwargs) -> None:
        ...

class test_Consumer:
    @pytest.fixture
    def callback(self) -> Mock:
        ...

    @pytest.fixture
    def on_P_revoked(self) -> Mock:
        ...

    @pytest.fixture
    def on_P_assigned(self) -> Mock:
        ...

    @pytest.fixture
    def consumer(self, app: App, callback: Mock, on_P_revoked: Mock, on_P_assigned: Mock) -> MyConsumer:
        ...

    @pytest.fixture
    def message(self) -> Mock:
        ...

    def test_on_init_dependencies__default(self, consumer: MyConsumer) -> None:
        ...

    def test_on_init_dependencies__exactly_once(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_getmany__stopped_after_wait(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_getmany__flow_inactive(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_getmany__flow_inactive2(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_getmany(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    @pytest.mark.parametrize('client_only', [False, True])
    async def test__wait_next_records(self, client_only: bool, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test__wait_next_records__flow_inactive(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test__wait_next_records__no_active_tps(self, consumer: MyConsumer) -> None:
        ...

    def _setup_records(self, consumer: MyConsumer, active_partitions: set, records: dict | None = None, flow_active: bool = True) -> None:
        ...

    @pytest.mark.asyncio
    async def test__wait_for_ack(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_restart(self, consumer: MyConsumer) -> None:
        ...

    def test__get_active_partitions__when_empty(self, consumer: MyConsumer) -> None:
        ...

    def test__get_active_partitions__when_set(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_perform_seek(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit__client_only(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_seek(self, consumer: MyConsumer) -> None:
        ...

    def test_stop_flow(self, consumer: MyConsumer) -> None:
        ...

    def test_resume_flow(self, consumer: MyConsumer) -> None:
        ...

    def test_pause_partitions(self, consumer: MyConsumer) -> None:
        ...

    def test_resume_partitions(self, consumer: MyConsumer) -> None:
        ...

    def test_read_offset_default(self, consumer: MyConsumer) -> None:
        ...

    def test_committed_offset_default(self, consumer: MyConsumer) -> None:
        ...

    def test_is_changelog_tp(self, app: App, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_partitions_revoked(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_partitions_revoked__updates_active(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_partitions_assigned(self, consumer: MyConsumer) -> None:
        ...

    def test_track_message(self, consumer: MyConsumer, message: Mock) -> None:
        ...

    @pytest.mark.parametrize('offset', [int, int])
    def test_ack(self, offset: int, consumer: MyConsumer, message: Mock) -> None:
        ...

    def test_ack__already_acked(self, consumer: MyConsumer, message: Mock) -> None:
        ...

    def test_ack__disabled(self, consumer: MyConsumer, message: Mock, app: App) -> None:
        ...

    @pytest.mark.asyncio
    async def test_wait_empty(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_wait_empty__when_stopped(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_stop(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_force_commit(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit_tps(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit_tps__ProducerSendError(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit_tps__no_committable(self, consumer: MyConsumer) -> None:
        ...

    def test_filter_committable_offsets(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_handle_attached(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit_offsets(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit_offsets__did_not_commit(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit_offsets__in_transaction(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit_offsets__no_committable_offsets(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit__already_committing(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit(self, consumer: MyConsumer) -> None:
        ...

    def test_filter_tps_with_pending_acks(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.parametrize('tp,offset,committed,should', [(TP1, 0, 0, False), (TP1, 1, 0, True), (TP1, 6, 8, False), (TP1, 100, 8, True)])
    def test_should_commit(self, tp: TP, offset: int, committed: int, should: bool, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.parametrize('tp,acked,expected_offset', [(TP1, [], None), (TP1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10), (TP1, [1, 2, 3, 4, 5, 6, 7, 8, 10], 8), (TP1, [1, 2, 3, 4, 6, 7, 8, 10], 4), (TP1, [1, 3, 4, 6, 7, 8, 10], 1)])
    def test_new_offset(self, tp: TP, acked: list[int], expected_offset: int | None, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.parametrize('tp,acked,gaps,expected_offset', [(TP1, [], [], None), (TP1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [], 10), (TP1, [1, 2, 3, 4, 5, 6, 7, 8, 10], [9], 10), (TP1, [1, 2, 3, 4, 6, 7, 8, 10], [5], 8), (TP1, [1, 3, 4, 6, 7, 8, 10], [2, 5, 9], 10)])
    def test_new_offset_with_gaps(self, tp: TP, acked: list[int], gaps: list[int], expected_offset: int | None, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_task_error(self, consumer: MyConsumer) -> None:
        ...

    def test__add_gap(self, consumer: MyConsumer) -> None:
        ...

    def test__add_gap__previous_to_committed(self, consumer: MyConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit_handler(self, consumer: MyConsumer) -> None:
        ...

    def test_close(self, consumer: MyConsumer) -> None:
        ...

class test_ConsumerThread:
    class MyConsumerThread(MockedConsumerAbstractMethods, ConsumerThread):
        def close(self) -> None:
            ...

        async def getmany(self, *args, **kwargs) -> None:
            ...

        def pause_partitions(self, *args, **kwargs) -> None:
            ...

        def resume_partitions(self, *args, **kwargs) -> None:
            ...

        def stop_flow(self, *args, **kwargs) -> None:
            ...

        def resume_flow(self, *args, **kwargs) -> None:
            ...

        async def commit(self, *args, **kwargs) -> None:
            ...

        async def perform_seek(self, *args, **kwargs) -> None:
            ...

        async def seek(self, *args, **kwargs) -> None:
            ...

    @pytest.fixture
    def consumer(self) -> Mock:
        ...

    @pytest.fixture
    def thread(self, consumer: Mock) -> MyConsumerThread:
        ...

    @pytest.mark.asyncio
    async def test_on_partitions_revoked(self, thread: MyConsumerThread, consumer: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_partitions_assigned(self, thread: MyConsumerThread, consumer: Mock) -> None:
        ...

class test_ThreadDelegateConsumer:
    class TestThreadDelegateConsumer(ThreadDelegateConsumer):
        def _new_consumer_thread(self) -> Mock:
            ...

        def _new_topicpartition(self, *args, **kwargs) -> TP:
            ...

        def _to_message(self, *args, **kwargs) -> tuple:
            ...

        def create_topic(self, *args, **kwargs) -> None:
            ...

    @pytest.fixture
    def message_callback(self) -> Mock:
        ...

    @pytest.fixture
    def partitions_revoked_callback(self) -> Mock:
        ...

    @pytest.fixture
    def partitions_assigned_callback(self) -> Mock:
        ...

    @pytest.fixture
    def consumer(self, app: App, message_callback: Mock, partitions_revoked_callback: Mock, partitions_assigned_callback: Mock) -> TestThreadDelegateConsumer:
        ...

    @pytest.mark.asyncio
    async def test_threadsafe_partitions_revoked(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_threadsafe_partitions_assigned(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test__getmany(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_subscribe(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_seek_to_committed(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_position(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_seek_wait(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test__seek(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    def test_assignment(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    def test_highwater(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    def test_topic_partitions(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_earliest_offsets(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_highwaters(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_commit(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_maybe_wait_for_commit_to_finish(self, loop: asyncio.AbstractEventLoop, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_close(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    def test_key_partition(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    def test_verify_event_path(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_verify_all_partitions_active(self, consumer: TestThreadDelegateConsumer) -> None:
        ...

    @pytest.mark.asyncio
    async def test_verify_all_partitions_active__bail_on_sleep(self, consumer: TestThreadDelegateConsumer) -> None:
        ...