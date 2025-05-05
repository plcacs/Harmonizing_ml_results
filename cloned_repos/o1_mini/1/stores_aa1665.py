import abc
import typing
from typing import Any, Callable, Iterable, Mapping, Optional, Set, TypeVar, Union
from mode import ServiceT
from mode.utils.collections import FastUserDict
from yarl import URL
from .codecs import CodecArg
from .events import EventT
from .tuples import TP

if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .models import ModelArg as _ModelArg
    from .tables import CollectionT as _CollectionT
else:

    class _AppT:
        ...

    class _ModelArg:
        ...

    class _CollectionT:
        ...

__all__ = ['StoreT']
KT = TypeVar('KT')
VT = TypeVar('VT')


class StoreT(ServiceT, FastUserDict[KT, VT]):

    @abc.abstractmethod
    def __init__(
        self,
        url: URL,
        app: "_AppT",
        table: "_CollectionT",
        *,
        table_name: str = '',
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        key_serializer: str = '',
        value_serializer: str = '',
        options: Optional[Mapping[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        ...

    @abc.abstractmethod
    def persisted_offset(self, tp: TP) -> int:
        ...

    @abc.abstractmethod
    def set_persisted_offset(self, tp: TP, offset: int) -> None:
        ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp: TP) -> bool:
        ...

    @abc.abstractmethod
    def apply_changelog_batch(
        self,
        batch: Iterable[EventT],
        to_key: Callable[[Any], KT],
        to_value: Callable[[Any], VT]
    ) -> None:
        ...

    @abc.abstractmethod
    def reset_state(self) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(
        self,
        table: "_CollectionT",
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP]
    ) -> None:
        ...

    @abc.abstractmethod
    async def on_recovery_completed(
        self,
        active_tps: Set[TP],
        standby_tps: Set[TP]
    ) -> None:
        ...
