from typing import Any, AsyncIterator, Awaitable, ClassVar, Dict, Iterable, Iterator, List, Mapping, MutableMapping, MutableSet, NamedTuple, Optional, Set, Type, Tuple
from weakref import WeakSet

from mode import Service
from mode.threads import MethodQueue, QueueServiceThread
from mode.utils.locks import Event
from mode.utils.times import Seconds

from faust.exceptions import ProducerSendError
from faust.types import AppT, ConsumerMessage, Message, RecordMetadata, TP
from faust.types.core import HeadersArg
from faust.types.transports import ConsumerCallback, ConsumerT, PartitionsAssignedCallback, PartitionsRevokedCallback, ProducerT, TPorTopicSet, TransactionManagerT, TransportT
from faust.types.tuples import FutureMessage

RecordMap = Mapping[TP, List[Any]]
TopicPartitionGroup = NamedTuple('TopicPartitionGroup', [('topic', str), ('partition', int), ('group', str)])

def ensure_TP(tp: Any) -> TP:
    ...

def ensure_TPset(tps: Set[Any]) -> Set[TP]:
    ...

class Fetcher(Service):
    ...

class TransactionManager(Service, TransactionManagerT):
    ...

class Consumer(Service, ConsumerT):
    ...

class ConsumerThread(QueueServiceThread):
    ...

class ThreadDelegateConsumer(Consumer):
    ...
