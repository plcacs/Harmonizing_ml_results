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
    def __init__(self, app: Union[models.ModelArg, int, asyncio.AbstractEventLoop], *, schema: Union[None, models.ModelArg, int, asyncio.AbstractEventLoop]=None, key_type: Union[None, models.ModelArg, int, asyncio.AbstractEventLoop]=None, value_type: Union[None, models.ModelArg, int, asyncio.AbstractEventLoop]=None, is_iterator: bool=False, queue: Union[None, models.ModelArg, int, asyncio.AbstractEventLoop]=None, maxsize: Union[None, models.ModelArg, int, asyncio.AbstractEventLoop]=None, root: Union[None, models.ModelArg, int, asyncio.AbstractEventLoop]=None, active_partitions: Union[None, models.ModelArg, int, asyncio.AbstractEventLoop]=None, loop: Union[None, models.ModelArg, int, asyncio.AbstractEventLoop]=None) -> None:
        ...

    @abc.abstractmethod
    def clone(self, *, is_iterator: Union[None, bool]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def clone_using_queue(self, queue: Union[asyncio.Queue, queues.AbstractQueue, typing.Iterable]) -> None:
        ...

    @abc.abstractmethod
    def stream(self, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def get_topic_name(self) -> None:
        ...

    @abc.abstractmethod
    async def send(self, *, key=None, value=None, partition=None, timestamp=None, headers=None, schema=None, key_serializer=None, value_serializer=None, callback=None, force=False):
        ...

    @abc.abstractmethod
    def send_soon(self, *, key: Union[None, bool, core.V]=None, value: Union[None, bool, core.V]=None, partition: Union[None, bool, core.V]=None, timestamp: Union[None, bool, core.V]=None, headers: Union[None, bool, core.V]=None, schema: Union[None, bool, core.V]=None, key_serializer: Union[None, bool, core.V]=None, value_serializer: Union[None, bool, core.V]=None, callback: Union[None, bool, core.V]=None, force: bool=False, eager_partitioning: bool=False) -> None:
        ...

    @abc.abstractmethod
    def as_future_message(self, key: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, value: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, partition: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, timestamp: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, headers: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, schema: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, key_serializer: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, value_serializer: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, callback: Union[None, codecs.CodecArg, serializers.SchemaT, bool]=None, eager_partitioning: bool=False) -> None:
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
    def prepare_key(self, key: Union[core.K, serializers.SchemaT, codecs.CodecArg], key_serializer: Union[core.K, serializers.SchemaT, codecs.CodecArg], schema: Union[None, core.K, serializers.SchemaT, codecs.CodecArg]=None) -> None:
        ...

    @abc.abstractmethod
    def prepare_value(self, value: Union[core.V, codecs.CodecArg, serializers.SchemaT], value_serializer: Union[core.V, codecs.CodecArg, serializers.SchemaT], schema: Union[None, core.V, codecs.CodecArg, serializers.SchemaT]=None) -> None:
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
    def empty(self) -> None:
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
    def on_stop_iteration(self) -> None:
        ...

    @abc.abstractmethod
    def __aiter__(self) -> None:
        ...

    @abc.abstractmethod
    def __anext__(self) -> None:
        ...

    @abc.abstractmethod
    async def throw(self, exc):
        ...

    @abc.abstractmethod
    def _throw(self, exc: Union[BaseException, Exception, None]) -> None:
        ...

    @abc.abstractmethod
    def derive(self, **kwargs) -> None:
        ...

    @property
    @abc.abstractmethod
    def subscriber_count(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def queue(self) -> None:
        ...