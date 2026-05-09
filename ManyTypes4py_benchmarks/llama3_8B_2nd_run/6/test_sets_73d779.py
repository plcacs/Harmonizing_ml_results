import faust
import pytest
from faust.tables.sets import ChangeloggedSet, ChangeloggedSetManager, OPERATION_ADD, OPERATION_DISCARD, OPERATION_UPDATE, SetAction, SetManagerOperation, SetTableManager, SetWindowSet
from mode.utils.mocks import AsyncMock, Mock, call

@pytest.fixture()
def key() -> Mock:
    return Mock(name='key')

@pytest.fixture()
def table() -> Mock:
    return Mock(name='table')

class test_SetWindowSet:
    @pytest.fixture()
    def wrapper(self) -> Mock:
        return Mock(name='wrapper')

    @pytest.fixture()
    def wset(self, *, key: Mock, table: Mock, wrapper: Mock) -> SetWindowSet:
        return SetWindowSet(key, table, wrapper)

    def test_add(self, *, wset: SetWindowSet) -> None:
        event = Mock(name='event')
        wset._apply_set_operation = Mock()
        wset.add('value', event=event)
        wset._apply_set_operation.assert_called_once_with('add', 'value', event)

    # ... rest of the test cases ...

class test_ChangeloggedSet:
    @pytest.fixture()
    def manager(self) -> Mock:
        return Mock(name='manager')

    @pytest.fixture()
    def cset(self, *, manager: Mock, key: Mock) -> ChangeloggedSet:
        return ChangeloggedSet(manager, key)

    # ... rest of the test cases ...

class test_SetTableManager:
    @pytest.fixture()
    def stable(self, *, app: faust.App) -> faust.SetTable:
        return app.SetTable('name', start_manager=True)

    @pytest.fixture()
    def man(self, *, stable: faust.SetTable) -> SetTableManager:
        return SetTableManager(stable)

    # ... rest of the test cases ...

class test_SetTable:
    @pytest.fixture()
    def stable(self, *, app: faust.App) -> faust.SetTable:
        return app.SetTable('name')

    # ... rest of the test cases ...
