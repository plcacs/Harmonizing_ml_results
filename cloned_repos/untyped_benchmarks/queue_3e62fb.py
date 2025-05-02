import logging
import random
import ssl
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeAlias, TypeVar
import orjson
import pika
import pika.adapters.tornado_connection
import pika.connection
import pika.exceptions
from django.conf import settings
from django.db import transaction
from pika.adapters.blocking_connection import BlockingChannel
from pika.channel import Channel
from pika.spec import Basic
from tornado import ioloop
from typing_extensions import override
from zerver.lib.utils import assert_is_not_none
MAX_REQUEST_RETRIES = 3
ChannelT = TypeVar('ChannelT', Channel, BlockingChannel)
Consumer = Callable[[ChannelT, Basic.Deliver, pika.BasicProperties, bytes], None]

class QueueClient(Generic[ChannelT], ABC):

    def __init__(self, rabbitmq_heartbeat=0, prefetch=0):
        self.log = logging.getLogger('zulip.queue')
        self.queues = set()
        self.channel = None
        self.prefetch = prefetch
        self.consumers = defaultdict(set)
        self.rabbitmq_heartbeat = rabbitmq_heartbeat
        self.is_consuming = False
        self._connect()

    @abstractmethod
    def _connect(self):
        raise NotImplementedError

    @abstractmethod
    def _reconnect(self):
        raise NotImplementedError

    def _get_parameters(self):
        credentials = pika.PlainCredentials(settings.RABBITMQ_USERNAME, assert_is_not_none(settings.RABBITMQ_PASSWORD))
        tcp_options = None
        if self.rabbitmq_heartbeat == 0:
            tcp_options = dict(TCP_KEEPIDLE=60 * 5)
        ssl_options = pika.ConnectionParameters._DEFAULT
        if settings.RABBITMQ_USE_TLS:
            ssl_options = pika.SSLOptions(context=ssl.create_default_context())
        return pika.ConnectionParameters(settings.RABBITMQ_HOST, port=settings.RABBITMQ_PORT, virtual_host=settings.RABBITMQ_VHOST, heartbeat=self.rabbitmq_heartbeat, tcp_options=tcp_options, ssl_options=ssl_options, credentials=credentials)

    def _generate_ctag(self, queue_name):
        return f'{queue_name}_{random.getrandbits(16)}'

    def _reconnect_consumer_callback(self, queue, consumer):
        self.log.info('Queue reconnecting saved consumer %r to queue %s', consumer, queue)
        self.ensure_queue(queue, lambda channel: channel.basic_consume(queue, consumer, consumer_tag=self._generate_ctag(queue)))

    def _reconnect_consumer_callbacks(self):
        for queue, consumers in self.consumers.items():
            for consumer in consumers:
                self._reconnect_consumer_callback(queue, consumer)

    def ready(self):
        return self.channel is not None

    @abstractmethod
    def ensure_queue(self, queue_name, callback):
        raise NotImplementedError

    def publish(self, queue_name, body):

        def do_publish(channel):
            channel.basic_publish(exchange='', routing_key=queue_name, properties=pika.BasicProperties(delivery_mode=2), body=body)
        self.ensure_queue(queue_name, do_publish)

    def json_publish(self, queue_name, body):
        data = orjson.dumps(body)
        try:
            self.publish(queue_name, data)
            return
        except pika.exceptions.AMQPConnectionError:
            self.log.warning('Failed to send to rabbitmq, trying to reconnect and send again')
        self._reconnect()
        self.publish(queue_name, data)

class SimpleQueueClient(QueueClient[BlockingChannel]):

    @override
    def _connect(self):
        start = time.time()
        self.connection = pika.BlockingConnection(self._get_parameters())
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=self.prefetch)
        self.log.info('SimpleQueueClient connected (connecting took %.3fs)', time.time() - start)

    @override
    def _reconnect(self):
        self.connection = None
        self.channel = None
        self.queues = set()
        self._connect()

    def close(self):
        if self.connection is not None:
            self.connection.close()

    @override
    def ensure_queue(self, queue_name, callback):
        """Ensure that a given queue has been declared, and then call
        the callback with no arguments."""
        if self.connection is None or not self.connection.is_open:
            self._connect()
            assert self.channel is not None
        else:
            assert self.channel is not None
        if queue_name not in self.queues:
            self.channel.queue_declare(queue=queue_name, durable=True)
            self.queues.add(queue_name)
        callback(self.channel)

    def start_json_consumer(self, queue_name, callback, batch_size=1, timeout=None):
        if batch_size == 1:
            timeout = None

        def do_consume(channel):
            events = []
            last_process = time.time()
            max_processed = None
            self.is_consuming = True
            for method, properties, body in channel.consume(queue_name, inactivity_timeout=timeout):
                if body is not None:
                    assert method is not None
                    events.append(orjson.loads(body))
                    max_processed = method.delivery_tag
                now = time.time()
                if len(events) >= batch_size or (timeout and now >= last_process + timeout):
                    if events:
                        assert max_processed is not None
                        try:
                            callback(events)
                            channel.basic_ack(max_processed, multiple=True)
                        except BaseException:
                            if channel.is_open:
                                channel.basic_nack(max_processed, multiple=True)
                            raise
                        events = []
                    last_process = now
                if not self.is_consuming:
                    break
        self.ensure_queue(queue_name, do_consume)

    def local_queue_size(self):
        assert self.channel is not None
        return self.channel.get_waiting_message_count() + len(self.channel._pending_events)

    def stop_consuming(self):
        assert self.channel is not None
        assert self.is_consuming
        self.is_consuming = False
        self.channel.stop_consuming()

class ExceptionFreeTornadoConnection(pika.adapters.tornado_connection.TornadoConnection):

    def _adapter_disconnect(self):
        try:
            super()._adapter_disconnect()
        except (pika.exceptions.ProbableAuthenticationError, pika.exceptions.ProbableAccessDeniedError, pika.exceptions.IncompatibleProtocolError):
            logging.warning('Caught exception in ExceptionFreeTornadoConnection when calling _adapter_disconnect, ignoring', exc_info=True)

class TornadoQueueClient(QueueClient[Channel]):

    def __init__(self):
        super().__init__(rabbitmq_heartbeat=None, prefetch=100)
        self._on_open_cbs = []
        self._connection_failure_count = 0

    @override
    def _connect(self):
        self.log.info('Beginning TornadoQueueClient connection')
        self.connection = ExceptionFreeTornadoConnection(self._get_parameters(), on_open_callback=self._on_open, on_open_error_callback=self._on_connection_open_error, on_close_callback=self._on_connection_closed)

    @override
    def _reconnect(self):
        self.connection = None
        self.channel = None
        self.queues = set()
        self.log.warning('TornadoQueueClient attempting to reconnect to RabbitMQ')
        self._connect()
    CONNECTION_RETRY_SECS = 2
    CONNECTION_FAILURES_BEFORE_NOTIFY = 10

    def _on_connection_open_error(self, connection, reason):
        self._connection_failure_count += 1
        retry_secs = self.CONNECTION_RETRY_SECS
        self.log.log(logging.CRITICAL if self._connection_failure_count > self.CONNECTION_FAILURES_BEFORE_NOTIFY else logging.WARNING, "TornadoQueueClient couldn't connect to RabbitMQ, retrying in %d secs...", retry_secs)
        ioloop.IOLoop.current().call_later(retry_secs, self._reconnect)

    def _on_connection_closed(self, connection, reason):
        if self.connection is None:
            return
        self._connection_failure_count = 1
        retry_secs = self.CONNECTION_RETRY_SECS
        self.log.warning('TornadoQueueClient lost connection to RabbitMQ, reconnecting in %d secs...', retry_secs)
        ioloop.IOLoop.current().call_later(retry_secs, self._reconnect)

    def _on_open(self, connection):
        assert self.connection is not None
        self._connection_failure_count = 0
        try:
            self.connection.channel(on_open_callback=self._on_channel_open)
        except pika.exceptions.ConnectionClosed:
            self.log.warning("TornadoQueueClient couldn't open channel: connection already closed")

    def _on_channel_open(self, channel):
        self.channel = channel
        for callback in self._on_open_cbs:
            callback(channel)
        self._reconnect_consumer_callbacks()
        self.log.info('TornadoQueueClient connected')

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @override
    def ensure_queue(self, queue_name, callback):

        def set_qos(frame):
            assert self.channel is not None
            self.queues.add(queue_name)
            self.channel.basic_qos(prefetch_count=self.prefetch, callback=finish)

        def finish(frame):
            assert self.channel is not None
            callback(self.channel)
        if queue_name not in self.queues:
            if not self.ready():
                self._on_open_cbs.append(lambda channel: self.ensure_queue(queue_name, callback))
                return
            assert self.channel is not None
            self.channel.queue_declare(queue=queue_name, durable=True, callback=set_qos)
        else:
            assert self.channel is not None
            callback(self.channel)

    def start_json_consumer(self, queue_name, callback, batch_size=1, timeout=None):

        def wrapped_consumer(ch, method, properties, body):
            assert method.delivery_tag is not None
            callback([orjson.loads(body)])
            ch.basic_ack(delivery_tag=method.delivery_tag)
        assert batch_size == 1
        assert timeout is None
        self.consumers[queue_name].add(wrapped_consumer)
        if not self.ready():
            return
        self.ensure_queue(queue_name, lambda channel: channel.basic_consume(queue_name, wrapped_consumer, consumer_tag=self._generate_ctag(queue_name)))
thread_data = threading.local()

def get_queue_client():
    if not hasattr(thread_data, 'queue_client'):
        if not settings.USING_RABBITMQ:
            raise RuntimeError('Cannot get a queue client without USING_RABBITMQ')
        thread_data.queue_client = SimpleQueueClient()
    return thread_data.queue_client

def set_queue_client(queue_client):
    thread_data.queue_client = queue_client

def queue_json_publish_rollback_unsafe(queue_name, event, processor=None):
    if settings.USING_RABBITMQ:
        get_queue_client().json_publish(queue_name, event)
    elif processor:
        processor(event)
    else:
        from zerver.worker.queue_processors import get_worker
        get_worker(queue_name, disable_timeout=True).consume_single_event(event)

def queue_event_on_commit(queue_name, event):
    transaction.on_commit(lambda: queue_json_publish_rollback_unsafe(queue_name, event))

def retry_event(queue_name, event, failure_processor):
    if 'failed_tries' not in event:
        event['failed_tries'] = 0
    event['failed_tries'] += 1
    if event['failed_tries'] > MAX_REQUEST_RETRIES:
        failure_processor(event)
    else:
        queue_json_publish_rollback_unsafe(queue_name, event)