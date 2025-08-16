from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Optional, Set, Type, cast, Union
from yarl import URL
from faust.models import Record, maybe_model
from faust.types import AgentT, AppT, EventT, StreamT, TopicT
from faust.types.tables import GlobalTableT, KT, VT
from faust.types.stores import StoreT
from . import wrappers
from .objects import ChangeloggedObject, ChangeloggedObjectManager
from .table import Table

OPERATION_ADD: int = 1
OPERATION_DISCARD: int = 2
OPERATION_UPDATE: int = 15

class SetWindowSet(wrappers.WindowSet):
    def add(self, element: Any, *, event: Optional[Any] = None) -> None:
        self._apply_set_operation('add', element, event)

    def discard(self, element: Any, *, event: Optional[Any] = None) -> None:
        self._apply_set_operation('discard', element, event)

    def _apply_set_operation(self, op: str, element: Any, event: Optional[Any] = None) -> None:
        ...

class SetWindowWrapper(wrappers.WindowWrapper):
    ValueType: Type[SetWindowSet] = SetWindowSet

class ChangeloggedSet(ChangeloggedObject, ManagedUserSet[VT]):
    def __post_init__(self) -> None:
        ...

    def on_add(self, value: VT) -> None:
        ...

    def on_discard(self, value: VT) -> None:
        ...

    def on_change(self, added: List[VT], removed: List[VT]) -> None:
        ...

    def sync_from_storage(self, value: Any) -> None:
        ...

    def as_stored_value(self) -> Any:
        ...

    def apply_changelog_event(self, operation: int, value: Any) -> None:
        ...

class ChangeloggedSetManager(ChangeloggedObjectManager):
    url: Optional[URL] = None
    ValueType: Type[ChangeloggedSet] = ChangeloggedSet

class SetAction(Enum):
    ADD: str = 'ADD'
    DISCARD: str = 'DISCARD'
    CLEAR: str = 'CLEAR'
    INTERSECTION: str = 'INTERSECTION'
    SYMDIFF: str = 'SYMDIFF'

class SetManagerOperation(Record, namespace='@SetManagerOperation'):
    pass

class SetTableManager(Service, Generic[KT, VT]):
    def __init__(self, set_table: SetTable, **kwargs: Any) -> None:
        ...

    async def add(self, key: KT, member: VT) -> None:
        ...

    async def discard(self, key: KT, member: VT) -> None:
        ...

    async def clear(self, key: KT) -> None:
        ...

    async def difference_update(self, key: KT, members: List[VT]) -> None:
        ...

    async def intersection_update(self, key: KT, members: List[VT]) -> None:
        ...

    async def symmetric_difference_update(self, key: KT, members: List[VT]) -> None:
        ...

    def _update(self, key: KT, members: List[VT]) -> None:
        ...

    def _difference_update(self, key: KT, members: List[VT]) -> None:
        ...

    def _clear(self, key: KT, members: List[VT]) -> None:
        ...

    def _intersection_update(self, key: KT, members: List[VT]) -> None:
        ...

    def _symmetric_difference_update(self, key: KT, members: List[VT]) -> None:
        ...

    async def _send_operation(self, action: SetAction, key: KT, members: List[VT]) -> None:
        ...

    def _enable(self) -> None:
        ...

    async def _modify_set(self, stream: StreamT) -> None:
        ...

    @cached_property
    def topic(self) -> TopicT:
        ...

class SetTable(Table[KT, ChangeloggedSet[VT]]):
    Manager: ClassVar[Type[SetTableManager]] = SetTableManager
    manager_topic_suffix: str = '-setmanager'
    WindowWrapper: Type[SetWindowWrapper] = SetWindowWrapper
    _changelog_compacting: bool = False

    def __init__(self, app: AppT, *, start_manager: bool = False, manager_topic_name: Optional[str] = None, manager_topic_suffix: Optional[str] = None, **kwargs: Any) -> None:
        ...

    async def on_start(self) -> None:
        ...

    def _new_store(self) -> ChangeloggedSetManager:
        ...

    def __getitem__(self, key: KT) -> ChangeloggedSet[VT]:
        ...

class SetGlobalTable(SetTable, GlobalTableT):
    pass
