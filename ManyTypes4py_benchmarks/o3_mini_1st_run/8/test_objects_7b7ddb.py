import pytest
from typing import Any, Callable, Generator, Iterator, List, Tuple
from faust.tables.objects import ChangeloggedObjectManager
from faust.types import TP
from mode.utils.mocks import AsyncMock, Mock, patch

TP1: TP = TP('foo', 3)


@pytest.fixture()
def key() -> Mock:
    return Mock(name='key')


@pytest.fixture()
def table() -> Mock:
    return Mock(name='table', _new_store_by_url=Mock(return_value=Mock(on_rebalance=AsyncMock())))


@pytest.yield_fixture()
def current_event() -> Iterator[Any]:
    with patch('faust.tables.objects.current_event') as current_event:
        yield current_event


class ValueType:
    def __init__(self, man: ChangeloggedObjectManager, key: Any) -> None:
        self.man: ChangeloggedObjectManager = man
        self.key: Any = key
        self.synced: set = set()
        self.changes: List[Tuple[Any, Any]] = []

    def sync_from_storage(self, value: Any) -> None:
        self.synced.add(value)

    def as_stored_value(self) -> str:
        return f'{self.key}-stored'

    def apply_changelog_event(self, operation: Any, value: Any) -> None:
        self.changes.append((operation, value))


class test_ChangeloggedObjectManager:
    @pytest.fixture()
    def man(self, *, table: Mock) -> ChangeloggedObjectManager:
        man: ChangeloggedObjectManager = ChangeloggedObjectManager(table)
        man.ValueType = ValueType  # type: ignore
        return man

    @pytest.fixture()
    def storage(self, *, table: Mock) -> Mock:
        return table._new_store_by_url.return_value

    def test_send_changelog_event(self, *, man: ChangeloggedObjectManager, table: Mock, key: Mock, current_event: Any) -> None:
        man.send_changelog_event(key, 3, 'value')
        assert key in man._dirty
        table._send_changelog.assert_called_once_with(current_event(), (3, key), 'value')

    def test__getitem__(self, *, man: ChangeloggedObjectManager) -> None:
        v1 = man['k']
        v2 = man['k']
        assert v1 is v2
        v3 = man['j']
        assert v3 is not v1
        assert man.data['k'].man is man
        assert man.data['k'].key == 'k'
        assert man.data['j'].man is man
        assert man.data['j'].key == 'j'

    def test__setitem__(self, *, man: ChangeloggedObjectManager) -> None:
        with pytest.raises(NotImplementedError):
            man['k'] = 3

    def test__delitem__(self, *, man: ChangeloggedObjectManager) -> None:
        with pytest.raises(NotImplementedError):
            del man['k']

    def test_table_type_name(self, *, man: ChangeloggedObjectManager) -> None:
        assert man._table_type_name

    @pytest.mark.asyncio
    async def test_on_start(self, *, man: ChangeloggedObjectManager) -> None:
        man.add_runtime_dependency = AsyncMock()
        await man.on_start()
        man.add_runtime_dependency.assert_called_once_with(man.storage)

    @pytest.mark.asyncio
    async def test_on_stop(self, *, man: ChangeloggedObjectManager) -> None:
        man.flush_to_storage = Mock()
        await man.on_stop()
        man.flush_to_storage.assert_called_once_with()

    def test_persisted_offset(self, *, man: ChangeloggedObjectManager, storage: Mock) -> None:
        ret = man.persisted_offset(TP1)
        storage.persisted_offset.assert_called_once_with(TP1)
        assert ret is storage.persisted_offset()

    def test_set_persisted_offset(self, *, man: ChangeloggedObjectManager, storage: Mock) -> None:
        man.set_persisted_offset(TP1, 3003)
        storage.set_persisted_offset.assert_called_once_with(TP1, 3003)

    @pytest.mark.asyncio
    async def test_on_rebalance(self, *, man: ChangeloggedObjectManager, storage: Mock, table: Mock) -> None:
        await man.on_rebalance(table, {TP1}, {TP1}, {TP1})
        man.storage.on_rebalance.assert_called_once_with(table, {TP1}, {TP1}, {TP1})

    @pytest.mark.asyncio
    async def test_on_recovery_completed(self, *, man: ChangeloggedObjectManager) -> None:
        man.sync_from_storage = Mock()
        await man.on_recovery_completed({TP1}, {TP1})
        man.sync_from_storage.assert_called_once_with()

    def test_sync_from_storage(self, *, man: ChangeloggedObjectManager, storage: Mock) -> None:
        storage.items.return_value = [('foo', 1), ('bar', 2)]
        man.sync_from_storage()
        assert 1 in man['foo'].synced
        assert 2 in man['bar'].synced

    def test_flush_to_storage(self, *, man: ChangeloggedObjectManager) -> None:
        man._storage = {}  # type: ignore
        man._dirty = {'foo', 'bar'}  # type: ignore
        assert man['foo']
        assert man['bar']
        man.flush_to_storage()
        assert man._storage['foo'] == 'foo-stored'  # type: ignore

    def test_reset_state(self, *, man: ChangeloggedObjectManager, storage: Mock) -> None:
        man.reset_state()
        storage.reset_state.assert_called_once_with()

    def test_apply_changelog_batch__key_is_None(self, *, man: ChangeloggedObjectManager) -> None:
        event1: Mock = Mock(name='event1')
        event1.key = None
        with pytest.raises(RuntimeError):
            man.apply_changelog_batch([event1], lambda k: k, lambda v: v)

    def test_apply_changelog_batch__empty(self, *, man: ChangeloggedObjectManager) -> None:
        man.apply_changelog_batch([], lambda k: k, lambda v: v)

    def test_apply_changelog_batch(self, *, man: ChangeloggedObjectManager) -> None:
        event1: Mock = Mock(name='event1')
        event1.key = (3, 'k')
        event1.value = 'foo'
        man.apply_changelog_batch([event1], lambda k: k, lambda v: v)
        assert (3, 'foo') in man['k'].changes
