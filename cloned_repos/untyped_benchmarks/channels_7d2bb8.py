import abc
import asyncio
import typing
from typing import Any, AsyncIterator, Awaitable, Generic, Optional, Set, TypeVar
from mode import Seconds
from mode.utils.futures import stampede
from mode.utils.queues import ThrowableQueue
from .codecs import CodecArg
from .core import HeadersArg, K, V
from .tuples import FutureMessage, Message, MessageSentCallback, RecordMetadata, TP
_T = TypeVar('_T')
_T_contra = TypeVar('_T_contra', contravariant=True)
if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .events import EventT as _EventT
    from .models import ModelArg as _ModelArg
    from .serializers import SchemaT as _SchemaT
    from .streams import StreamT as _StreamT
else:

    class _AppT:
        ...

    class _EventT(Generic[_T]):
        ...

    class _ModelArg:
        ...

    class _SchemaT:
        ...

    class _StreamT:
        ...

class ChannelT(AsyncIterator[_EventT[_T]]):

    @abc.abstractmethod
    def __init__(self, app, *, schema=None, key_type=None, value_type=None, is_iterator=False, queue=None, maxsize=None, root=None, active_partitions=None, loop=None):
        ...

    @abc.abstractmethod
    def clone(self, *, is_iterator=None, **kwargs):
        ...

    @abc.abstractmethod
    def clone_using_queue(self, queue):
        ...

    @abc.abstractmethod
    def stream(self, **kwargs):
        ...

    @abc.abstractmethod
    def get_topic_name(self):
        ...

    @abc.abstractmethod
    async def send(self, *, key=None, value=None, partition=None, timestamp=None, headers=None, schema=None, key_serializer=None, value_serializer=None, callback=None, force=False):
        ...

    @abc.abstractmethod
    def send_soon(self, *, key=None, value=None, partition=None, timestamp=None, headers=None, schema=None, key_serializer=None, value_serializer=None, callback=None, force=False, eager_partitioning=False):
        ...

    @abc.abstractmethod
    def as_future_message(self, key=None, value=None, partition=None, timestamp=None, headers=None, schema=None, key_serializer=None, value_serializer=None, callback=None, eager_partitioning=False):
        ...

    @abc.abstractmethod
    async def publish_message(self, fut, wait=True):
        ...

    @stampede
    @abc.abstractmethod
    async def maybe_declare(self):
        ...

    @abc.abstractmethod
    async def declare(self):
        ...

    @abc.abstractmethod
    def prepare_key(self, key, key_serializer, schema=None):
        ...

    @abc.abstractmethod
    def prepare_value(self, value, value_serializer, schema=None):
        ...

    @abc.abstractmethod
    async def decode(self, message, *, propagate=False):
        ...

    @abc.abstractmethod
    async def deliver(self, message):
        ...

    @abc.abstractmethod
    async def put(self, value):
        ...

    @abc.abstractmethod
    async def get(self, *, timeout=None):
        ...

    @abc.abstractmethod
    def empty(self):
        ...

    @abc.abstractmethod
    async def on_key_decode_error(self, exc, message):
        ...

    @abc.abstractmethod
    async def on_value_decode_error(self, exc, message):
        ...

    @abc.abstractmethod
    async def on_decode_error(self, exc, message):
        ...

    @abc.abstractmethod
    def on_stop_iteration(self):
        ...

    @abc.abstractmethod
    def __aiter__(self):
        ...

    @abc.abstractmethod
    def __anext__(self):
        ...

    @abc.abstractmethod
    async def throw(self, exc):
        ...

    @abc.abstractmethod
    def _throw(self, exc):
        ...

    @abc.abstractmethod
    def derive(self, **kwargs):
        ...

    @property
    @abc.abstractmethod
    def subscriber_count(self):
        ...

    @property
    @abc.abstractmethod
    def queue(self):
        ...