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
    def __init__(self, url: Union[models.ModelArg, codecs.CodecArg, str], app: Union[models.ModelArg, codecs.CodecArg, str], table: Union[models.ModelArg, codecs.CodecArg, str], *, table_name: typing.Text='', key_type: Union[None, models.ModelArg, codecs.CodecArg, str]=None, value_type: Union[None, models.ModelArg, codecs.CodecArg, str]=None, key_serializer: typing.Text='', value_serializer: typing.Text='', options: Union[None, models.ModelArg, codecs.CodecArg, str]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def persisted_offset(self, tp: Union[tuples.TP, typing.Type, None]) -> None:
        ...

    @abc.abstractmethod
    def set_persisted_offset(self, tp: Union[int, tuples.TP], offset: Union[int, tuples.TP]) -> None:
        ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp):
        ...

    @abc.abstractmethod
    def apply_changelog_batch(self, batch: Union[typing.Callable, typing.Iterable[faustypes.EventT], bool], to_key: Union[typing.Callable, typing.Iterable[faustypes.EventT], bool], to_value: Union[typing.Callable, typing.Iterable[faustypes.EventT], bool]) -> None:
        ...

    @abc.abstractmethod
    def reset_state(self) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self, table, assigned, revoked, newly_assigned):
        ...

    @abc.abstractmethod
    async def on_recovery_completed(self, active_tps, standby_tps):
        ...