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
    def __init__(
        self,
        app: _AppT,
        *,
        schema: Optional[_SchemaT] = None,
        key_type: Optional[typing.Type[K]] = None,
        value_type: Optional[typing.Type[V]] = None,
        is_iterator: bool = False,
        queue: Optional[ThrowableQueue] = None,
        maxsize: Optional[int] = None,
        root: Any = None,
        active_partitions: Optional[Set[TP]] = None,
        loop: Optional[asyncio.BaseEventLoop] = None,
    ) -> None:
        ...

    def clone(
        self,
        *,
        is_iterator: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChannelT[_T]:
        ...

    def clone_using_queue(
        self,
        queue: ThrowableQueue,
    ) -> ChannelT[_T]:
        ...

    def stream(
        self,
        **kwargs: Any,
    ) -> _StreamT[_EventT[_T]]:
        ...

    def get_topic_name(self) -> str:
        ...

    async def send(
        self,
        *,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[TP] = None,
        timestamp: Optional[Seconds] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[_SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
    ) -> Awaitable[FutureMessage[_EventT[_T]]]:
        ...

    def send_soon(
        self,
        *,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[TP] = None,
        timestamp: Optional[Seconds] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[_SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
        eager_partitioning: bool = False,
    ) -> None:
        ...

    def as_future_message(
        self,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[TP] = None,
        timestamp: Optional[Seconds] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[_SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        eager_partitioning: bool = False,
    ) -> FutureMessage[_EventT[_T]]:
        ...

    async def publish_message(
        self,
        fut: FutureMessage[_EventT[_T]],
        wait: bool = True,
    ) -> None:
        ...

    @stampede
    async def maybe_declare(self) -> None:
        ...

    async def declare(self) -> None:
        ...

    def prepare_key(
        self,
        key: K,
        key_serializer: CodecArg,
        schema: Optional[_SchemaT] = None,
    ) -> bytes:
        ...

    def prepare_value(
        self,
        value: V,
        value_serializer: CodecArg,
        schema: Optional[_SchemaT] = None,
    ) -> bytes:
        ...

    async def decode(
        self,
        message: Message[_EventT[_T]],
        *,
        propagate: bool = False,
    ) -> _EventT[_T]:
        ...

    async def deliver(
        self,
        message: Message[_EventT[_T]],
    ) -> None:
        ...

    async def put(
        self,
        value: V,
    ) -> None:
        ...

    async def get(
        self,
        *,
        timeout: Optional[Seconds] = None,
    ) -> Optional[_EventT[_T]]:
        ...

    def empty(self) -> bool:
        ...

    async def on_key_decode_error(
        self,
        exc: Exception,
        message: Message[_EventT[_T]],
    ) -> None:
        ...

    async def on_value_decode_error(
        self,
        exc: Exception,
        message: Message[_EventT[_T]],
    ) -> None:
        ...

    async def on_decode_error(
        self,
        exc: Exception,
        message: Message[_EventT[_T]],
    ) -> None:
        ...

    def on_stop_iteration(self) -> None:
        ...

    def __aiter__(self) -> AsyncIterator[_EventT[_T]]:
        ...

    def __anext__(self) -> Awaitable[_EventT[_T]]:
        ...

    async def throw(
        self,
        exc: Exception,
    ) -> None:
        ...

    def _throw(
        self,
        exc: Exception,
    ) -> None:
        ...

    def derive(
        self,
        **kwargs: Any,
    ) -> ChannelT[_T]:
        ...

    @property
    def subscriber_count(self) -> int:
        ...

    @property
    def queue(self) -> Optional[ThrowableQueue]:
        ...
