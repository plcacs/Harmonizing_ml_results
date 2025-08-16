import asyncio
from time import time
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

__all__: List[str] = ['ConsumerMessage', 'FutureMessage', 'Message', 'MessageSentCallback', 'PendingMessage', 'RecordMetadata', 'TP', 'tp_set_to_map']
MessageSentCallback: Callable[['FutureMessage'], Union[None, Awaitable[None]]]

class TP(NamedTuple):
    topic: str
    partition: int

class RecordMetadata(NamedTuple):
    timestamp: Optional[float] = None
    timestamp_type: Optional[int] = None

class PendingMessage(NamedTuple):
    topic: Optional[str] = None
    offset: Optional[int] = None

def _PendingMessage_to_Message(p: PendingMessage) -> Message:
    ...

class FutureMessage(asyncio.Future, Awaitable[RecordMetadata]):

    def __init__(self, message: Message):
        ...

    def set_result(self, result: RecordMetadata):
        ...

def _get_len(s: Optional[bytes]) -> int:
    ...

class Message:
    def __init__(self, topic: str, partition: int, offset: int, timestamp: float, timestamp_type: int, headers: HeadersArg, key: K, value: V, checksum: Optional[CodecArg], serialized_key_size: Optional[int] = None, serialized_value_size: Optional[int] = None, tp: Optional[TP] = None, time_in: Optional[float] = None, time_out: Optional[float] = None, time_total: Optional[float] = None):
        ...

    def ack(self, consumer: '_ConsumerT', n: int = 1) -> bool:
        ...

    def on_final_ack(self, consumer: '_ConsumerT') -> bool:
        ...

    def incref(self, n: int = 1):
        ...

    def decref(self, n: int = 1) -> int:
        ...

    @classmethod
    def from_message(cls, message: 'Message', tp: TP) -> 'Message':
        ...

    def __repr__(self) -> str:
        ...

class ConsumerMessage(Message):
    def on_final_ack(self, consumer: '_ConsumerT') -> bool:
        ...

def tp_set_to_map(tps: Set[TP]) -> MutableMapping[str, Set[TP]]:
    ...
