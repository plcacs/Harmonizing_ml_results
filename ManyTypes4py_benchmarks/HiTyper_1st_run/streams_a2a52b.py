import abc
import asyncio
import typing
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union, no_type_check
from mode import Seconds, ServiceT
from mode.utils.trees import NodeT
from .channels import ChannelT
from .core import K
from .events import EventT
from .models import FieldDescriptorT, ModelArg
from .topics import TopicT
from .tuples import TP
if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .join import JoinT as _JoinT
    from .serializers import SchemaT as _SchemaT
else:

    class _AppT:
        ...

    class _JoinT:
        ...

    class _SchemaT:
        ...
__all__ = ['Processor', 'GroupByKeyArg', 'StreamT', 'T', 'T_co', 'T_contra']
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)
Processor = Callable[[T], Union[T, Awaitable[T]]]
GroupByKeyArg = Union[FieldDescriptorT, Callable[[T], K]]

class JoinableT(abc.ABC):

    @abc.abstractmethod
    def combine(self, *nodes, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def join(self, *fields) -> None:
        ...

    @abc.abstractmethod
    def left_join(self, *fields) -> None:
        ...

    @abc.abstractmethod
    def inner_join(self, *fields) -> None:
        ...

    @abc.abstractmethod
    def outer_join(self, *fields) -> None:
        ...

    @abc.abstractmethod
    def __and__(self, other: Union[T, list[str], list, ListProxy]) -> None:
        ...

    @abc.abstractmethod
    def contribute_to_stream(self, active: Union[bool, list[str]]) -> None:
        ...

    @abc.abstractmethod
    async def remove_from_stream(self, stream):
        ...

    @abc.abstractmethod
    def _human_channel(self) -> None:
        ...

class StreamT(AsyncIterable[T_co], JoinableT, ServiceT):
    outbox = None
    join_strategy = None
    task_owner = None
    current_event = None
    active_partitions = None
    concurrency_index = None
    enable_acks = True
    prefix = ''
    _next = None
    _prev = None

    @abc.abstractmethod
    def __init__(self, channel: Union[None, str, typing.Callable, int]=None, *, app: Union[None, str, typing.Callable, int]=None, processors: Union[None, str, typing.Callable, int]=None, combined: Union[None, str, typing.Callable, int]=None, on_start: Union[None, str, typing.Callable, int]=None, join_strategy: Union[None, str, typing.Callable, int]=None, beacon: Union[None, str, typing.Callable, int]=None, concurrency_index: Union[None, str, typing.Callable, int]=None, prev: Union[None, str, typing.Callable, int]=None, active_partitions: Union[None, str, typing.Callable, int]=None, enable_acks: bool=True, prefix: typing.Text='', loop: Union[None, str, typing.Callable, int]=None) -> None:
        ...

    @abc.abstractmethod
    def get_active_stream(self) -> None:
        ...

    @abc.abstractmethod
    def add_processor(self, processor: deeplearning.ml4pl.models.epoch.Type) -> None:
        ...

    @abc.abstractmethod
    def info(self) -> None:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs) -> None:
        ...

    @abc.abstractmethod
    @no_type_check
    async def items(self):
        ...

    @abc.abstractmethod
    @no_type_check
    async def events(self):
        ...

    @abc.abstractmethod
    @no_type_check
    async def take(self, max_, within):
        ...

    @abc.abstractmethod
    def enumerate(self, start: int=0) -> None:
        ...

    @abc.abstractmethod
    def through(self, channel: Union[str, topics.ChannelT, int]) -> None:
        ...

    @abc.abstractmethod
    def echo(self, *channels) -> None:
        ...

    @abc.abstractmethod
    def group_by(self, key: Union[str, bool, set[str]], *, name: Union[None, str, bool, set[str]]=None, topic: Union[None, str, bool, set[str]]=None) -> None:
        ...

    @abc.abstractmethod
    def derive_topic(self, name: Union[str, models.ModelArg, list[str], None], *, schema: Union[None, str, models.ModelArg, list[str]]=None, key_type: Union[None, str, models.ModelArg, list[str]]=None, value_type: Union[None, str, models.ModelArg, list[str]]=None, prefix: typing.Text='', suffix: typing.Text='') -> None:
        ...

    @abc.abstractmethod
    async def throw(self, exc):
        ...

    @abc.abstractmethod
    def __copy__(self) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> None:
        ...

    @abc.abstractmethod
    def __next__(self) -> None:
        ...

    @abc.abstractmethod
    def __aiter__(self) -> None:
        ...

    @abc.abstractmethod
    async def ack(self, event):
        ...