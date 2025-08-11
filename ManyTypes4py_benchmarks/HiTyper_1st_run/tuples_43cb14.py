import asyncio
from time import time
import typing
from collections import defaultdict
from typing import Any, Awaitable, Callable, MutableMapping, NamedTuple, Optional, Set, Union, cast
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
__all__ = ['ConsumerMessage', 'FutureMessage', 'Message', 'MessageSentCallback', 'PendingMessage', 'RecordMetadata', 'TP', 'tp_set_to_map']
MessageSentCallback = Callable[['FutureMessage'], Union[None, Awaitable[None]]]

class TP(NamedTuple):
    pass

class RecordMetadata(NamedTuple):
    timestamp = None
    timestamp_type = None

class PendingMessage(NamedTuple):
    topic = None
    offset = None

def _PendingMessage_to_Message(p: Any) -> Message:
    topic = cast(str, p.topic)
    partition = cast(int, p.partition) or 0
    tp = TP(topic, partition)
    timestamp = cast(float, p.timestamp) or time()
    timestamp_type = 1 if p.timestamp else 0
    return Message(topic, partition, -1, timestamp=timestamp, timestamp_type=timestamp_type, headers=p.headers, key=p.key, value=p.value, checksum=None, tp=tp)

class FutureMessage(asyncio.Future, Awaitable[RecordMetadata]):

    def __init__(self, message) -> None:
        self.message = message
        super().__init__()

    def set_result(self, result: Union[typing.Any, None]) -> None:
        super().set_result(result)

def _get_len(s: Union[bytes, str, tuple]) -> int:
    return len(s) if s is not None and isinstance(s, bytes) else 0

class Message:
    __slots__ = ('topic', 'partition', 'offset', 'timestamp', 'timestamp_type', 'headers', 'key', 'value', 'checksum', 'serialized_key_size', 'serialized_value_size', 'acked', 'refcount', 'time_in', 'time_out', 'time_total', 'tp', 'tracked', 'span', '__weakref__')
    use_tracking = False

    def __init__(self, topic: Union[int, None, str], partition: Union[str, dict[str, typing.Any], bytes], offset: Union[int, str, None, float], timestamp: Union[int, list["PriceDraft"], None], timestamp_type: Union[str, None, int], headers: Union[bytes, str, dict], key: Union[bytes, None, str], value: Union[bytes, None, int, float], checksum: Union[bytes, int, str, None], serialized_key_size: Union[None, bytes, int]=None, serialized_value_size: Union[None, bytes, int]=None, tp: Union[None, faustypes.V, int, asyncio.AbstractEventLoop]=None, time_in: Union[None, int, float]=None, time_out: Union[None, int, float]=None, time_total: Union[None, int, str]=None) -> None:
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
        self.serialized_value_size = _get_len(value) if serialized_value_size is None else serialized_value_size
        self.acked = False
        self.refcount = 0
        self.tp = tp if tp is not None else TP(topic, partition)
        self.tracked = not self.use_tracking
        self.time_in = time_in
        self.time_out = time_out
        self.time_total = time_total

    def ack(self, consumer: Union[faustypes.transports.ConsumerT, int, asyncio.AbstractEventLoop], n: int=1) -> bool:
        if not self.acked:
            if not self.decref(n):
                return self.on_final_ack(consumer)
        return False

    def on_final_ack(self, consumer: Union[transports.ConsumerT, asyncio.AbstractEventLoop]) -> bool:
        self.acked = True
        return True

    def incref(self, n: int=1) -> None:
        self.refcount += n

    def decref(self, n: int=1) -> int:
        refcount = self.refcount = max(self.refcount - n, 0)
        return refcount

    @classmethod
    def from_message(cls: Union[bytes, str], message: Union[bytes, str], tp: Union[bytes, str]):
        return cls(message.topic, message.partition, message.offset, message.timestamp, message.timestamp_type, message.headers, message.key, message.value, message.checksum, message.serialized_key_size, message.serialized_value_size, tp)

    def __repr__(self) -> typing.Text:
        return f'<{type(self).__name__}: {self.tp} offset={self.offset}>'

class ConsumerMessage(Message):
    """Message type used by Kafka Consumer."""
    use_tracking = True

    def on_final_ack(self, consumer: Union[transports.ConsumerT, asyncio.AbstractEventLoop]) -> bool:
        return consumer.ack(self)

def tp_set_to_map(tps: str) -> Union[list[tuple[str]], str, frozenset]:
    tpmap = defaultdict(set)
    for tp in tps:
        tpmap[tp.topic].add(tp)
    return tpmap
MessageSentCallback = Callable[[FutureMessage], Union[None, Awaitable[None]]]