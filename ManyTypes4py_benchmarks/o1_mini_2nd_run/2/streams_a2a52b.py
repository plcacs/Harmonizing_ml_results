import abc
import asyncio
import typing
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    no_type_check,
)
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

__all__ = ["Processor", "GroupByKeyArg", "StreamT", "T", "T_co", "T_contra"]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
Processor = Callable[[T], Union[T, Awaitable[T]]]
GroupByKeyArg = Union[FieldDescriptorT, Callable[[T], K]]


class JoinableT(abc.ABC):

    @abc.abstractmethod
    def combine(
        self, *nodes: NodeT[Any], **kwargs: Any
    ) -> "JoinableT":
        ...

    @abc.abstractmethod
    def join(self, *fields: Any) -> "JoinableT":
        ...

    @abc.abstractmethod
    def left_join(self, *fields: Any) -> "JoinableT":
        ...

    @abc.abstractmethod
    def inner_join(self, *fields: Any) -> "JoinableT":
        ...

    @abc.abstractmethod
    def outer_join(self, *fields: Any) -> "JoinableT":
        ...

    @abc.abstractmethod
    def __and__(self, other: "JoinableT") -> "JoinableT":
        ...

    @abc.abstractmethod
    def contribute_to_stream(self, active: bool) -> None:
        ...

    @abc.abstractmethod
    async def remove_from_stream(self, stream: "StreamT[Any]") -> None:
        ...

    @abc.abstractmethod
    def _human_channel(self) -> Any:
        ...


class StreamT(AsyncIterable[T_co], JoinableT, ServiceT):
    outbox: Optional[Any] = None
    join_strategy: Optional[Any] = None
    task_owner: Optional[Any] = None
    current_event: Optional[EventT] = None
    active_partitions: Optional[Any] = None
    concurrency_index: Optional[Any] = None
    enable_acks: bool = True
    prefix: str = ""
    _next: Optional["StreamT[Any]"] = None
    _prev: Optional["StreamT[Any]"] = None

    @abc.abstractmethod
    def __init__(
        self,
        channel: Optional[ChannelT] = None,
        *,
        app: Optional["_AppT"] = None,
        processors: Optional[List[Processor[T]]] = None,
        combined: Optional[Any] = None,
        on_start: Optional[Callable[[], Awaitable[None]]] = None,
        join_strategy: Optional[Any] = None,
        beacon: Optional[Any] = None,
        concurrency_index: Optional[Any] = None,
        prev: Optional["StreamT[Any]"] = None,
        active_partitions: Optional[Any] = None,
        enable_acks: bool = True,
        prefix: str = "",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def get_active_stream(self) -> Optional["StreamT[Any]"]:
        ...

    @abc.abstractmethod
    def add_processor(self, processor: Processor[T]) -> None:
        ...

    @abc.abstractmethod
    def info(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any) -> "StreamT[Any]":
        ...

    @abc.abstractmethod
    @no_type_check
    async def items(self) -> AsyncIterator[T_co]:
        ...

    @abc.abstractmethod
    @no_type_check
    async def events(self) -> AsyncIterator[EventT]:
        ...

    @abc.abstractmethod
    @no_type_check
    async def take(self, max_: int, within: Seconds) -> List[T_co]:
        ...

    @abc.abstractmethod
    def enumerate(self, start: int = 0) -> "StreamT[Tuple[int, T_co]]":
        ...

    @abc.abstractmethod
    def through(self, channel: ChannelT) -> "StreamT[Any]":
        ...

    @abc.abstractmethod
    def echo(self, *channels: ChannelT) -> "StreamT[Any]":
        ...

    @abc.abstractmethod
    def group_by(
        self,
        key: GroupByKeyArg,
        *,
        name: Optional[str] = None,
        topic: Optional[TopicT] = None,
    ) -> "StreamT[Any]":
        ...

    @abc.abstractmethod
    def derive_topic(
        self,
        name: str,
        *,
        schema: Optional["_SchemaT"] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        prefix: str = "",
        suffix: str = "",
    ) -> TopicT:
        ...

    @abc.abstractmethod
    async def throw(self, exc: BaseException) -> None:
        ...

    @abc.abstractmethod
    def __copy__(self) -> "StreamT[Any]":
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        ...

    @abc.abstractmethod
    def __next__(self) -> T_co:
        ...

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[T_co]:
        ...

    @abc.abstractmethod
    async def ack(self, event: EventT) -> None:
        ...
