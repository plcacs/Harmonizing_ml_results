import abc
import typing
from typing import Any, Dict, Iterable, Optional, Union
from mode import ServiceT
from . import web
from .assignor import PartitionAssignorT
from .events import EventT
from .streams import StreamT
from .tables import CollectionT
from .transports import ConsumerT, ProducerT
from .tuples import Message, PendingMessage, RecordMetadata, TP

if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
else:

    class _AppT:
        ...

__all__ = ['SensorInterfaceT', 'SensorT', 'SensorDelegateT']

class SensorInterfaceT(abc.ABC):

    @abc.abstractmethod
    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        ...

    @abc.abstractmethod
    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> None:
        ...

    @abc.abstractmethod
    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Dict[str, Any]] = None) -> None:
        ...

    @abc.abstractmethod
    def on_topic_buffer_full(self, tp: TP) -> None:
        ...

    @abc.abstractmethod
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        ...

    @abc.abstractmethod
    def on_table_get(self, table: CollectionT, key: Any) -> None:
        ...

    @abc.abstractmethod
    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        ...

    @abc.abstractmethod
    def on_table_del(self, table: CollectionT, key: Any) -> None:
        ...

    @abc.abstractmethod
    def on_commit_initiated(self, consumer: ConsumerT) -> None:
        ...

    @abc.abstractmethod
    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None:
        ...

    @abc.abstractmethod
    def on_send_initiated(self, producer: ProducerT, topic: str, message: Message, keysize: int, valsize: int) -> None:
        ...

    @abc.abstractmethod
    def on_send_completed(self, producer: ProducerT, state: Any, metadata: RecordMetadata) -> None:
        ...

    @abc.abstractmethod
    def on_send_error(self, producer: ProducerT, exc: Exception, state: Any) -> None:
        ...

    @abc.abstractmethod
    def on_assignment_start(self, assignor: PartitionAssignorT) -> None:
        ...

    @abc.abstractmethod
    def on_assignment_error(self, assignor: PartitionAssignorT, state: Any, exc: Exception) -> None:
        ...

    @abc.abstractmethod
    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Any) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_start(self, app: _AppT) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_return(self, app: _AppT, state: Any) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_end(self, app: _AppT, state: Any) -> None:
        ...

    @abc.abstractmethod
    def on_web_request_start(self, app: _AppT, request: web.Request, *, view: Optional[web.View] = None) -> None:
        ...

    @abc.abstractmethod
    def on_web_request_end(self, app: _AppT, request: web.Request, response: web.Response, state: Any, *, view: Optional[web.View] = None) -> None:
        ...

class SensorT(SensorInterfaceT, ServiceT):
    ...

class SensorDelegateT(SensorInterfaceT, Iterable[SensorInterfaceT]):

    @abc.abstractmethod
    def add(self, sensor: SensorInterfaceT) -> None:
        ...

    @abc.abstractmethod
    def remove(self, sensor: SensorInterfaceT) -> None:
        ...
