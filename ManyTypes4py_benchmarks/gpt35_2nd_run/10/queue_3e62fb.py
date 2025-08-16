from typing import Any, Callable, Generic, TypeVar
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.channel import Channel
from pika.spec import Basic

ChannelT = TypeVar('ChannelT', Channel, BlockingChannel)
Consumer = Callable[[ChannelT, Basic.Deliver, pika.BasicProperties, bytes], None]

class QueueClient(Generic[ChannelT], ABC):

    def __init__(self, rabbitmq_heartbeat: int = 0, prefetch: int = 0) -> None:
        ...

    def _connect(self) -> None:
        ...

    def _reconnect(self) -> None:
        ...

    def _get_parameters(self) -> pika.ConnectionParameters:
        ...

    def _generate_ctag(self, queue_name: str) -> str:
        ...

    def _reconnect_consumer_callback(self, queue: str, consumer: Consumer) -> None:
        ...

    def _reconnect_consumer_callbacks(self) -> None:
        ...

    def ready(self) -> bool:
        ...

    def ensure_queue(self, queue_name: str, callback: Callable[[ChannelT], None]) -> None:
        ...

    def publish(self, queue_name: str, body: bytes) -> None:
        ...

    def json_publish(self, queue_name: str, body: Any) -> None:
        ...

class SimpleQueueClient(QueueClient[BlockingChannel]):

    def _connect(self) -> None:
        ...

    def _reconnect(self) -> None:
        ...

    def ensure_queue(self, queue_name: str, callback: Callable[[BlockingChannel], None]) -> None:
        ...

    def start_json_consumer(self, queue_name: str, callback: Callable[[Any], None], batch_size: int = 1, timeout: int = None) -> None:
        ...

    def local_queue_size(self) -> int:
        ...

    def stop_consuming(self) -> None:
        ...

class ExceptionFreeTornadoConnection(pika.adapters.tornado_connection.TornadoConnection):

    def _adapter_disconnect(self) -> None:
        ...

class TornadoQueueClient(QueueClient[Channel]):

    def __init__(self) -> None:
        ...

    def _connect(self) -> None:
        ...

    def _reconnect(self) -> None:
        ...

    def _on_connection_open_error(self, connection, reason) -> None:
        ...

    def _on_connection_closed(self, connection, reason) -> None:
        ...

    def _on_open(self, connection) -> None:
        ...

    def _on_channel_open(self, channel) -> None:
        ...

    def ensure_queue(self, queue_name: str, callback: Callable[[Channel], None]) -> None:
        ...

    def start_json_consumer(self, queue_name: str, callback: Callable[[Any], None], batch_size: int = 1, timeout: int = None) -> None:
        ...

def get_queue_client() -> QueueClient:
    ...

def set_queue_client(queue_client: QueueClient) -> None:
    ...

def queue_json_publish_rollback_unsafe(queue_name: str, event: Any, processor: Callable[[Any], None] = None) -> None:
    ...

def queue_event_on_commit(queue_name: str, event: Any) -> None:
    ...

def retry_event(queue_name: str, event: Any, failure_processor: Callable[[Any], None]) -> None:
    ...
