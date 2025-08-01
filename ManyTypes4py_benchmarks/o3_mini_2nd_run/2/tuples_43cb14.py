#!/usr/bin/env python3
import asyncio
from time import time
import typing
from collections import defaultdict
from typing import Any, Awaitable, Callable, DefaultDict, MutableMapping, NamedTuple, Optional, Set, Type, Union, cast

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
    'ConsumerMessage',
    'FutureMessage',
    'Message',
    'MessageSentCallback',
    'PendingMessage',
    'RecordMetadata',
    'TP',
    'tp_set_to_map',
]

MessageSentCallback = Callable[['FutureMessage'], Union[None, Awaitable[None]]]


class TP(NamedTuple):
    topic: str
    partition: int


class RecordMetadata(NamedTuple):
    timestamp: Optional[float]
    timestamp_type: Optional[int]


class PendingMessage(NamedTuple):
    topic: Optional[str]
    partition: Optional[int]
    timestamp: Optional[float]
    headers: Optional[Any]
    key: Optional[bytes]
    value: Optional[bytes]


def _PendingMessage_to_Message(p: PendingMessage) -> "Message":
    topic: str = cast(str, p.topic)
    partition: int = (cast(Optional[int], p.partition) or 0)
    tp: TP = TP(topic, partition)
    timestamp: float = cast(Optional[float], p.timestamp) or time()
    timestamp_type: int = 1 if p.timestamp else 0
    return Message(
        topic,
        partition,
        -1,
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
        'topic',
        'partition',
        'offset',
        'timestamp',
        'timestamp_type',
        'headers',
        'key',
        'value',
        'checksum',
        'serialized_key_size',
        'serialized_value_size',
        'acked',
        'refcount',
        'time_in',
        'time_out',
        'time_total',
        'tp',
        'tracked',
        'span',
        '__weakref__',
    )
    use_tracking: bool = False

    def __init__(
        self,
        topic: str,
        partition: int,
        offset: int,
        timestamp: float,
        timestamp_type: int,
        headers: Optional[HeadersArg],
        key: Optional[bytes],
        value: Optional[bytes],
        checksum: Optional[Any],
        serialized_key_size: Optional[int] = None,
        serialized_value_size: Optional[int] = None,
        tp: Optional[TP] = None,
        time_in: Optional[float] = None,
        time_out: Optional[float] = None,
        time_total: Optional[float] = None,
    ) -> None:
        self.topic: str = topic
        self.partition: int = partition
        self.offset: int = offset
        self.timestamp: float = timestamp
        self.timestamp_type: int = timestamp_type
        self.headers: Optional[HeadersArg] = headers
        self.key: Optional[bytes] = key
        self.value: Optional[bytes] = value
        self.checksum: Optional[Any] = checksum
        self.serialized_key_size: int = _get_len(key) if serialized_key_size is None else serialized_key_size
        self.serialized_value_size: int = _get_len(value) if serialized_value_size is None else serialized_value_size
        self.acked: bool = False
        self.refcount: int = 0
        self.tp: TP = tp if tp is not None else TP(topic, partition)
        self.tracked: bool = not self.use_tracking
        self.time_in: Optional[float] = time_in
        self.time_out: Optional[float] = time_out
        self.time_total: Optional[float] = time_total

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
    def from_message(cls: Type["Message"], message: "Message", tp: TP) -> "Message":
        return cls(
            message.topic,
            message.partition,
            message.offset,
            message.timestamp,
            message.timestamp_type,
            message.headers,
            message.key,
            message.value,
            message.checksum,
            message.serialized_key_size,
            message.serialized_value_size,
            tp,
        )

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.tp} offset={self.offset}>'


class ConsumerMessage(Message):
    """Message type used by Kafka Consumer."""
    use_tracking: bool = True

    def on_final_ack(self, consumer: "_ConsumerT") -> bool:
        return consumer.ack(self)


def tp_set_to_map(tps: Set[TP]) -> MutableMapping[str, Set[TP]]:
    tpmap: DefaultDict[str, Set[TP]] = defaultdict(set)
    for tp in tps:
        tpmap[tp.topic].add(tp)
    return tpmap
