import abc
import typing
from typing import Any, Dict, Iterable, Optional
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
    def on_message_in(self, tp: Union[int, tuples.TP, tuples.Message], offset: Union[int, tuples.TP, tuples.Message], message: Union[int, tuples.TP, tuples.Message]) -> None:
        ...

    @abc.abstractmethod
    def on_stream_event_in(self, tp: Union[int, streams.StreamT, tuples.TP], offset: Union[int, streams.StreamT, tuples.TP], stream: Union[int, streams.StreamT, tuples.TP], event: Union[int, streams.StreamT, tuples.TP]) -> None:
        ...

    @abc.abstractmethod
    def on_stream_event_out(self, tp: Union[dict, int, faustypes.TP], offset: Union[dict, int, faustypes.TP], stream: Union[dict, int, faustypes.TP], event: Union[dict, int, faustypes.TP], state: Union[None, dict, int, faustypes.TP]=None) -> None:
        ...

    @abc.abstractmethod
    def on_topic_buffer_full(self, tp: Union[tuples.TP, typing.Optional]) -> None:
        ...

    @abc.abstractmethod
    def on_message_out(self, tp: Union[int, tuples.TP], offset: Union[int, tuples.TP], message: Union[int, tuples.TP]) -> None:
        ...

    @abc.abstractmethod
    def on_table_get(self, table: tables.CollectionT, key: tables.CollectionT) -> None:
        ...

    @abc.abstractmethod
    def on_table_set(self, table: tables.CollectionT, key: tables.CollectionT, value: tables.CollectionT) -> None:
        ...

    @abc.abstractmethod
    def on_table_del(self, table: tables.CollectionT, key: tables.CollectionT) -> None:
        ...

    @abc.abstractmethod
    def on_commit_initiated(self, consumer: Union[transports.ConsumerT, str]) -> None:
        ...

    @abc.abstractmethod
    def on_commit_completed(self, consumer: Union[transports.ConsumerT, typing.Callable], state: Union[transports.ConsumerT, typing.Callable]) -> None:
        ...

    @abc.abstractmethod
    def on_send_initiated(self, producer: Union[int, str, faustypes.transports.ProducerT], topic: Union[int, str, faustypes.transports.ProducerT], message: Union[int, str, faustypes.transports.ProducerT], keysize: Union[int, str, faustypes.transports.ProducerT], valsize: Union[int, str, faustypes.transports.ProducerT]) -> None:
        ...

    @abc.abstractmethod
    def on_send_completed(self, producer: Union[faustypes.tuples.RecordMetadata, faustypes.transports.ProducerT], state: Union[faustypes.tuples.RecordMetadata, faustypes.transports.ProducerT], metadata: Union[faustypes.tuples.RecordMetadata, faustypes.transports.ProducerT]) -> None:
        ...

    @abc.abstractmethod
    def on_send_error(self, producer: Union[BaseException, faustypes.transports.ProducerT], exc: Union[BaseException, faustypes.transports.ProducerT], state: Union[BaseException, faustypes.transports.ProducerT]) -> None:
        ...

    @abc.abstractmethod
    def on_assignment_start(self, assignor: assignor.PartitionAssignorT) -> None:
        ...

    @abc.abstractmethod
    def on_assignment_error(self, assignor: Union[BaseException, faustypes.assignor.PartitionAssignorT], state: Union[BaseException, faustypes.assignor.PartitionAssignorT], exc: Union[BaseException, faustypes.assignor.PartitionAssignorT]) -> None:
        ...

    @abc.abstractmethod
    def on_assignment_completed(self, assignor: Union[dict, assignor.PartitionAssignorT], state: Union[dict, assignor.PartitionAssignorT]) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_start(self, app: Union[faustypes.AppT, abilian.app.Application]) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_return(self, app: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict]) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_end(self, app: Union[dict, faustypes.AppT], state: Union[dict, faustypes.AppT]) -> None:
        ...

    @abc.abstractmethod
    def on_web_request_start(self, app: faustypes.AppT, request: faustypes.AppT, *, view: Union[None, faustypes.AppT]=None) -> None:
        ...

    @abc.abstractmethod
    def on_web_request_end(self, app: Union[faustypes.AppT, dict], request: Union[faustypes.AppT, dict], response: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict], *, view: Union[None, faustypes.AppT, dict]=None) -> None:
        ...

class SensorT(SensorInterfaceT, ServiceT):
    ...

class SensorDelegateT(SensorInterfaceT, Iterable):

    @abc.abstractmethod
    def add(self, sensor: typing.Callable[typing.Callable, None]) -> None:
        ...

    @abc.abstractmethod
    def remove(self, sensor: Union[routemaster.app.App, dict]) -> None:
        ...