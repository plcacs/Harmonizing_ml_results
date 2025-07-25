"""Topic - Named channel using Kafka."""
import asyncio
import re
import typing
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Pattern, Sequence, Set, Tuple, Union, cast, no_type_check
from mode import Seconds, get_logger
from mode.utils.futures import stampede
from mode.utils.queues import ThrowableQueue
from .channels import SerializedChannel
from .events import Event
from .streams import current_event
from .types import AppT, CodecArg, EventT, FutureMessage, HeadersArg, K, MessageSentCallback, ModelArg, PendingMessage, RecordMetadata, SchemaT, TP, V
from .types.topics import ChannelT, TopicT
from .types.transports import ProducerT
if typing.TYPE_CHECKING:
    from .app import App as _App
else:
    class _App:
        ...
__all__ = ['Topic']
logger = get_logger(__name__)

class Topic(SerializedChannel, TopicT):
    _partitions: Optional[int] = None
    _pattern: Optional[Pattern] = None

    def __init__(
        self,
        app: AppT,
        *,
        topics: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        is_iterator: bool = False,
        partitions: Optional[int] = None,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        replicas: Optional[int] = None,
        acks: bool = True,
        internal: bool = False,
        config: Optional[Dict[str, Any]] = None,
        queue: Optional[ThrowableQueue] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        maxsize: Optional[int] = None,
        root: Optional[Any] = None,
        active_partitions: Optional[Set[TP]] = None,
        allow_empty: Optional[bool] = None,
        has_prefix: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        super().__init__(
            app,
            schema=schema,
            key_type=key_type,
            value_type=value_type,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            allow_empty=allow_empty,
            loop=loop,
            active_partitions=active_partitions,
            is_iterator=is_iterator,
            queue=queue,
            maxsize=maxsize
        )
        self.topics: List[str] = topics or []
        self.pattern = cast(Pattern, pattern)
        self.partitions = partitions
        self.retention = retention
        self.compacting = compacting
        self.deleting = deleting
        self.replicas = replicas
        self.acks = acks
        self.internal = internal
        self.config: Dict[str, Any] = config or {}
        self.has_prefix = has_prefix
        self._compile_decode()

    def _compile_decode(self) -> None:
        self.decode = self.schema.compile(
            self.app,
            on_key_decode_error=self.on_key_decode_error,
            on_value_decode_error=self.on_value_decode_error
        )

    async def send(
        self,
        *,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False
    ) -> Awaitable[RecordMetadata]:
        app = cast(_App, self.app)
        if app._attachments.enabled and (not force):
            event = current_event()
            if event is not None:
                return cast(Event, event)._attach(
                    self, key, value,
                    partition=partition,
                    timestamp=timestamp,
                    headers=headers,
                    schema=schema,
                    key_serializer=key_serializer,
                    value_serializer=value_serializer,
                    callback=callback
                )
        return await self._send_now(
            key, value,
            partition=partition,
            timestamp=timestamp,
            headers=headers,
            schema=schema,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            callback=callback
        )

    def send_soon(
        self,
        *,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
        eager_partitioning: bool = False
    ) -> FutureMessage:
        fut = self.as_future_message(
            key=key,
            value=value,
            partition=partition,
            timestamp=timestamp,
            headers=headers,
            schema=schema,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            callback=callback,
            eager_partitioning=eager_partitioning
        )
        self.app.producer.send_soon(fut)
        return fut

    async def put(self, event: EventT) -> None:
        if not self.is_iterator:
            raise RuntimeError(f'Cannot put on Topic channel before aiter({self})')
        await self.queue.put(event)

    @no_type_check
    def _clone_args(self) -> Dict[str, Any]:
        return {
            **super()._clone_args(),
            **{
                'topics': self.topics,
                'pattern': self.pattern,
                'partitions': self.partitions,
                'retention': self.retention,
                'compacting': self.compacting,
                'deleting': self.deleting,
                'replicas': self.replicas,
                'internal': self.internal,
                'acks': self.acks,
                'config': self.config,
                'active_partitions': self.active_partitions,
                'allow_empty': self.allow_empty,
                'has_prefix': self.has_prefix
            }
        }

    @property
    def pattern(self) -> Optional[Pattern]:
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: Optional[str]) -> None:
        if pattern and self.topics:
            raise TypeError('Cannot specify both topics and pattern')
        self._pattern = re.compile(pattern) if pattern else None

    @property
    def partitions(self) -> Optional[int]:
        return self._partitions

    @partitions.setter
    def partitions(self, partitions: Optional[int]) -> None:
        if partitions == 0:
            raise ValueError('Topic cannot have zero partitions')
        self._partitions = partitions

    def derive(self, **kwargs: Any) -> 'Topic':
        return self.derive_topic(**kwargs)

    def derive_topic(
        self,
        *,
        topics: Optional[List[str]] = None,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        partitions: Optional[int] = None,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        internal: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
        prefix: str = '',
        suffix: str = '',
        **kwargs: Any
    ) -> 'Topic':
        topics = self.topics if topics is None else topics
        if suffix or prefix:
            if self.pattern:
                raise ValueError('Cannot add prefix/suffix to Topic with pattern')
            topics = [f'{prefix}{topic}{suffix}' for topic in topics]
        return type(self)(
            self.app,
            topics=topics,
            pattern=self.pattern,
            schema=self.schema if schema is None else schema,
            key_type=key_type,
            value_type=value_type,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            partitions=self.partitions if partitions is None else partitions,
            retention=self.retention if retention is None else retention,
            compacting=self.compacting if compacting is None else compacting,
            deleting=self.deleting if deleting is None else deleting,
            config=self.config if config is None else config,
            internal=self.internal if internal is None else internal,
            has_prefix=bool(self.has_prefix or prefix)
        )

    def get_topic_name(self) -> str:
        if self.pattern:
            raise TypeError('Topic with pattern subscription cannot be identified')
        if self.topics:
            if len(self.topics) > 1:
                raise ValueError('Topic with multiple topic names cannot be identified')
            return self.topics[0]
        raise TypeError('Topic has no subscriptions (no pattern, no topics)')

    async def _get_producer(self) -> ProducerT:
        return await self.app.maybe_start_producer()

    async def publish_message(
        self,
        fut: FutureMessage,
        wait: bool = False
    ) -> Union[Awaitable[RecordMetadata], asyncio.Future]:
        app = self.app
        message = fut.message
        topic = self._topic_name_or_default(message.channel)
        key = cast(bytes, message.key)
        value = cast(bytes, message.value)
        partition = message.partition
        timestamp = cast(float, message.timestamp)
        headers = message.headers
        logger.debug(
            'send: topic=%r k=%r v=%r timestamp=%r partition=%r',
            topic, key, value, timestamp, partition
        )
        assert topic is not None
        producer = await self._get_producer()
        state = app.sensors.on_send_initiated(
            producer, topic, message=message,
            keysize=len(key) if key else 0,
            valsize=len(value) if value else 0
        )
        if wait:
            ret = await producer.send_and_wait(
                topic, key, value,
                partition=partition,
                timestamp=timestamp,
                headers=headers
            )
            app.sensors.on_send_completed(producer, state, ret)
            return await self._finalize_message(fut, ret)
        else:
            fut2 = cast(asyncio.Future, await producer.send(
                topic, key, value,
                partition=partition,
                timestamp=timestamp,
                headers=headers
            ))
            callback = partial(
                self._on_published,
                message=fut,
                state=state,
                producer=producer
            )
            fut2.add_done_callback(cast(Callable, callback))
            return fut2

    def _topic_name_or_default(self, obj: Union[str, TopicT]) -> str:
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, TopicT):
            return obj.get_topic_name()
        else:
            return self.get_topic_name()

    def _on_published(
        self,
        fut: asyncio.Future,
        message: FutureMessage,
        producer: ProducerT,
        state: Any
    ) -> None:
        try:
            res = fut.result()
        except Exception as exc:
            message.set_exception(exc)
            self.app.sensors.on_send_error(producer, exc, state)
        else:
            message.set_result(res)
            if message.message.callback:
                message.message.callback(message)
            self.app.sensors.on_send_completed(producer, state, res)

    @stampede
    async def maybe_declare(self) -> None:
        await self.declare()

    async def declare(self) -> None:
        partitions = self.partitions
        if partitions is None:
            partitions = self.app.conf.topic_partitions
        if self.replicas is None:
            replicas = self.app.conf.topic_replication_factor
        else:
            replicas = self.replicas
        if self.app.conf.topic_allow_declare:
            producer = await self._get_producer()
            for topic in self.topics:
                await producer.create_topic(
                    topic=topic,
                    partitions=partitions,
                    replication=replicas or 0,
                    config=self.config,
                    compacting=self.compacting,
                    deleting=self.deleting,
                    retention=self.retention
                )

    def __aiter__(self) -> 'Topic':
        if self.is_iterator:
            return self
        else:
            channel = self.clone(is_iterator=True)
            self.app.topics.add(cast(TopicT, channel))
            return channel

    def __str__(self) -> str:
        return str(self.pattern) if self.pattern else ','.join(self.topics)
