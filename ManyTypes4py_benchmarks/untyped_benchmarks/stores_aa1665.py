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
    def __init__(self, url, app, table, *, table_name='', key_type=None, value_type=None, key_serializer='', value_serializer='', options=None, **kwargs):
        ...

    @abc.abstractmethod
    def persisted_offset(self, tp):
        ...

    @abc.abstractmethod
    def set_persisted_offset(self, tp, offset):
        ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp):
        ...

    @abc.abstractmethod
    def apply_changelog_batch(self, batch, to_key, to_value):
        ...

    @abc.abstractmethod
    def reset_state(self):
        ...

    @abc.abstractmethod
    async def on_rebalance(self, table, assigned, revoked, newly_assigned):
        ...

    @abc.abstractmethod
    async def on_recovery_completed(self, active_tps, standby_tps):
        ...