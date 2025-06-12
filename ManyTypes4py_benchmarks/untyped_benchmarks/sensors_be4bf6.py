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
    def on_message_in(self, tp, offset, message):
        ...

    @abc.abstractmethod
    def on_stream_event_in(self, tp, offset, stream, event):
        ...

    @abc.abstractmethod
    def on_stream_event_out(self, tp, offset, stream, event, state=None):
        ...

    @abc.abstractmethod
    def on_topic_buffer_full(self, tp):
        ...

    @abc.abstractmethod
    def on_message_out(self, tp, offset, message):
        ...

    @abc.abstractmethod
    def on_table_get(self, table, key):
        ...

    @abc.abstractmethod
    def on_table_set(self, table, key, value):
        ...

    @abc.abstractmethod
    def on_table_del(self, table, key):
        ...

    @abc.abstractmethod
    def on_commit_initiated(self, consumer):
        ...

    @abc.abstractmethod
    def on_commit_completed(self, consumer, state):
        ...

    @abc.abstractmethod
    def on_send_initiated(self, producer, topic, message, keysize, valsize):
        ...

    @abc.abstractmethod
    def on_send_completed(self, producer, state, metadata):
        ...

    @abc.abstractmethod
    def on_send_error(self, producer, exc, state):
        ...

    @abc.abstractmethod
    def on_assignment_start(self, assignor):
        ...

    @abc.abstractmethod
    def on_assignment_error(self, assignor, state, exc):
        ...

    @abc.abstractmethod
    def on_assignment_completed(self, assignor, state):
        ...

    @abc.abstractmethod
    def on_rebalance_start(self, app):
        ...

    @abc.abstractmethod
    def on_rebalance_return(self, app, state):
        ...

    @abc.abstractmethod
    def on_rebalance_end(self, app, state):
        ...

    @abc.abstractmethod
    def on_web_request_start(self, app, request, *, view=None):
        ...

    @abc.abstractmethod
    def on_web_request_end(self, app, request, response, state, *, view=None):
        ...

class SensorT(SensorInterfaceT, ServiceT):
    ...

class SensorDelegateT(SensorInterfaceT, Iterable):

    @abc.abstractmethod
    def add(self, sensor):
        ...

    @abc.abstractmethod
    def remove(self, sensor):
        ...