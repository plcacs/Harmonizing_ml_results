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
    def combine(self, *nodes, **kwargs):
        ...

    @abc.abstractmethod
    def join(self, *fields):
        ...

    @abc.abstractmethod
    def left_join(self, *fields):
        ...

    @abc.abstractmethod
    def inner_join(self, *fields):
        ...

    @abc.abstractmethod
    def outer_join(self, *fields):
        ...

    @abc.abstractmethod
    def __and__(self, other):
        ...

    @abc.abstractmethod
    def contribute_to_stream(self, active):
        ...

    @abc.abstractmethod
    async def remove_from_stream(self, stream):
        ...

    @abc.abstractmethod
    def _human_channel(self):
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
    def __init__(self, channel=None, *, app=None, processors=None, combined=None, on_start=None, join_strategy=None, beacon=None, concurrency_index=None, prev=None, active_partitions=None, enable_acks=True, prefix='', loop=None):
        ...

    @abc.abstractmethod
    def get_active_stream(self):
        ...

    @abc.abstractmethod
    def add_processor(self, processor):
        ...

    @abc.abstractmethod
    def info(self):
        ...

    @abc.abstractmethod
    def clone(self, **kwargs):
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
    def enumerate(self, start=0):
        ...

    @abc.abstractmethod
    def through(self, channel):
        ...

    @abc.abstractmethod
    def echo(self, *channels):
        ...

    @abc.abstractmethod
    def group_by(self, key, *, name=None, topic=None):
        ...

    @abc.abstractmethod
    def derive_topic(self, name, *, schema=None, key_type=None, value_type=None, prefix='', suffix=''):
        ...

    @abc.abstractmethod
    async def throw(self, exc):
        ...

    @abc.abstractmethod
    def __copy__(self):
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @abc.abstractmethod
    def __next__(self):
        ...

    @abc.abstractmethod
    def __aiter__(self):
        ...

    @abc.abstractmethod
    async def ack(self, event):
        ...