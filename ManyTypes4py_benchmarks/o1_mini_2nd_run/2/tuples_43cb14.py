import asyncio
from time import time
import typing
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    MutableMapping,
    NamedTuple,
    Optional,
    Set,
    Union,
    cast,
)

from .codecs import CodecArg
from .core import HeadersArg, K, OpenHeadersArg, V

if typing.TYPE_CHECKING:
    from .channels import ChannelT as _ChannelT
    from .transports import ConsumerT as _ConsumerT
else:

    class _ChannelT:
        ...

    class _ConsumerT:
        ...


__all__ = [
    "ConsumerMessage",
    "FutureMessage",
    "Message",
    "MessageSentCallback",
    "PendingMessage",
    "RecordMetadata",
    "TP",
    "tp_set_to_map",
]


MessageSentCallback = Callable[["FutureMessage"], Union[None, Awaitable[None]]]


class TP(NamedTuple):
    topic: str
    partition: int


class RecordMetadata(NamedTuple):
    timestamp: Optional[float] = None
    timestamp_type: Optional[int] = None


class PendingMessage(NamedTuple):
    topic: Optional[str] = None
    offset: Optional[int] = None
    partition: Optional[int] = None
    timestamp: Optional[float] = None
    headers: Optional[HeadersArg] = None
    key: Optional[K] = None
    value: Optional[V] = None


def _PendingMessage_to_Message(p: PendingMessage) -> "Message":
    topic: str = cast(str, p.topic)
    partition: int = cast(int, p.partition) or 0
    tp: TP = TP(topic, partition)
    timestamp: float = cast(float, p.timestamp) or time()
    timestamp_type: int = 1 if p.timestamp else 0
    return Message(
        topic=topic,
        partition=partition,
        offset=-1,
        timestamp=timestamp,
        timestamp_type=timestamp_type,
        headers=p.headers,
        key=p.key,
        value=p.value,
        checksum=None,
        tp=tp,
    )


class FutureMessage(asyncio.Future, Awaitable[RecordMetadata]):
    message: "Message"

    def __init__(self, message: "Message") -> None:
        self.message = message
        super().__init__()

    def set_result(self, result: RecordMetadata) -> None:
        super().set_result(result)


def _get_len(s: Optional[bytes]) -> int:
    return len(s) if s is not None and isinstance(s, bytes) else 0


class Message:
    __slots__ = (
        "topic",
        "partition",
        "offset",
        "timestamp",
        "timestamp_type",
        "headers",
        "key",
        "value",
        "checksum",
        "serialized_key_size",
        "serialized_value_size",
        "acked",
        "refcount",
        "time_in",
        "time_out",
        "time_total",
        "tp",
        "tracked",
        "span",
        "__weakref__",
    )
    use_tracking: bool = False
    topic: str
    partition: int
    offset: int
    timestamp: float
    timestamp_type: int
    headers: HeadersArg
    key: Optional[K]
    value: Optional[V]
    checksum: Optional[Any]
    serialized_key_size: int
    serialized_value_size: int
    acked: bool
    refcount: int
    time_in: Optional[float]
    time_out: Optional[float]
    time_total: Optional[float]
    tp: TP
    tracked: bool
    span: Any

    def __init__(
        self,
        topic: str,
        partition: int,
        offset: int,
        timestamp: float,
        timestamp_type: int,
        headers: HeadersArg,
        key: Optional[K],
        value: Optional[V],
        checksum: Optional[Any],
        serialized_key_size: Optional[int] = None,
        serialized_value_size: Optional[int] = None,
        tp: Optional[TP] = None,
        time_in: Optional[float] = None,
        time_out: Optional[float] = None,
        time_total: Optional[float] = None,
    ) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.timestamp = timestamp
        self.timestamp_type = timestamp_type
        self.headers = headers
        self.key = key
        self.value = value
        self.checksum = checksum
        self.serialized_key_size = _get_len(key) if serialized_key_size is None else serialized_key_size
        self.serialized_value_size = (
            _get_len(value) if serialized_value_size is None else serialized_value_size
        )
        self.acked = False
        self.refcount = 0
        self.tp: TP = tp if tp is not None else TP(topic, partition)
        self.tracked = not self.use_tracking
        self.time_in = time_in
        self.time_out = time_out
        self.time_total = time_total

    def ack(self, consumer: "_ConsumerT", n: int = 1) -> bool:
        if not self.acked:
            if not self.decref(n):
                return self.on_final_ack(consumer)
        return False

    def on_final_ack(self, consumer: "_ConsumerT") -> bool:
        self.acked = True
        return True

    def incref(self, n: int = 1) -> None:
        self.refcount += n

    def decref(self, n: int = 1) -> int:
        self.refcount = max(self.refcount - n, 0)
        return self.refcount

    @classmethod
    def from_message(cls, message: "Message", tp: TP) -> "Message":
        return cls(
            topic=message.topic,
            partition=message.partition,
            offset=message.offset,
            timestamp=message.timestamp,
            timestamp_type=message.timestamp_type,
            headers=message.headers,
            key=message.key,
            value=message.value,
            checksum=message.checksum,
            serialized_key_size=message.serialized_key_size,
            serialized_value_size=message.serialized_value_size,
            tp=tp,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.tp} offset={self.offset}>"


class ConsumerMessage(Message):
    """Message type used by Kafka Consumer."""
    use_tracking: bool = True

    def on_final_ack(self, consumer: "_ConsumerT") -> bool:
        return consumer.ack(self)


def tp_set_to_map(tps: Set[TP]) -> MutableMapping[str, Set[TP]]:
    tpmap: Dict[str, Set[TP]] = defaultdict(set)
    for tp in tps:
        tpmap[tp.topic].add(tp)
    return tpmap

MessageSentCallback = Callable[[FutureMessage], Union[None, Awaitable[None]]]
