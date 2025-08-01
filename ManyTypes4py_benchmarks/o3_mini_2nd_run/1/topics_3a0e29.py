"""Topic - Named channel using Kafka."""
import asyncio
import re
import typing
from functools import partial
from typing import Any, Awaitable, Callable, Mapping, Optional, Pattern, Sequence, Set, Union, cast, no_type_check, Dict, AsyncIterator

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
    """Define new topic description.

    Arguments:
        app: App instance used to create this topic description.
        topics: List of topic names.
        partitions: Number of partitions for these topics.
                    On declaration, topics are created using this.
                    Note: If a message is produced before the topic is
                    declared, and ``autoCreateTopics`` is enabled on
                    the Kafka Server, the number of partitions used
                    will be specified by the server configuration.
        retention: Number of seconds (as float/:class:`~datetime.timedelta`)
                   to keep messages in the topic before they can
                   be expired by the server.
        pattern: Regular expression evaluated to decide what topics to
                 subscribe to. You cannot specify both topics and a pattern.
        schema: Schema used for serialization/deserialization.
        key_type: How to deserialize keys for messages in this topic.
                  Can be a :class:`faust.Model` type, :class:`str`,
                  :class:`bytes`, or :const:`None` for "autodetect"
                  (Overrides schema if one is defined).
        value_type: How to deserialize values for messages in this topic.
                  Can be a :class:`faust.Model` type, :class:`str`,
                  :class:`bytes`, or :const:`None` for "autodetect"
                  (Overrides schema if ones is defined).
        active_partitions: Set of :class:`faust.types.tuples.TP` that this
                  topic should be restricted to.

    Raises:
        TypeError: if both `topics` and `pattern` is provided.
    """
    _partitions: Optional[int] = None
    _pattern: Optional[Pattern] = None

    def __init__(
        self,
        app: _App,
        *,
        topics: Optional[Sequence[str]] = None,
        pattern: Optional[Union[str, Pattern]] = None,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        is_iterator: bool = False,
        partitions: Optional[int] = None,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        replicas: Optional[int] = None,
        acks: Union[bool, int] = True,
        internal: bool = False,
        config: Optional[Mapping[str, Any]] = None,
        queue: Optional[ThrowableQueue] = None,
        key_serializer: Optional[Callable[[Any], bytes]] = None,
        value_serializer: Optional[Callable[[Any], bytes]] = None,
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
            maxsize=maxsize,
        )
        self.topics: Sequence[str] = topics or []
        self.pattern = pattern  # Uses the property setter below.
        self.partitions = partitions
        self.retention = retention
        self.compacting = compacting
        self.deleting = deleting
        self.replicas = replicas
        self.acks = acks
        self.internal = internal
        self.config: Mapping[str, Any] = config or {}
        self.has_prefix = has_prefix
        self._compile_decode()

    def _compile_decode(self) -> None:
        self.decode = self.schema.compile(
            self.app,
            on_key_decode_error=self.on_key_decode_error,
            on_value_decode_error=self.on_value_decode_error,
        )

    async def send(
        self,
        *,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[Callable[[Any], bytes]] = None,
        value_serializer: Optional[Callable[[Any], bytes]] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False
    ) -> Any:
        """Send message to topic."""
        app: _App = cast(_App, self.app)
        if app._attachments.enabled and (not force):
            event: Optional[EventT] = current_event()
            if event is not None:
                return cast(Event, event)._attach(
                    self,
                    key,
                    value,
                    partition=partition,
                    timestamp=timestamp,
                    headers=headers,
                    schema=schema,
                    key_serializer=key_serializer,
                    value_serializer=value_serializer,
                    callback=callback,
                )
        return await self._send_now(
            key,
            value,
            partition=partition,
            timestamp=timestamp,
            headers=headers,
            schema=schema,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            callback=callback,
        )

    def send_soon(
        self,
        *,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[Callable[[Any], bytes]] = None,
        value_serializer: Optional[Callable[[Any], bytes]] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
        eager_partitioning: bool = False
    ) -> FutureMessage:
        """Produce message by adding to buffer.

        Notes:
            This method can be used by non-`async def` functions
            to produce messages.
        """
        fut: FutureMessage = self.as_future_message(
            key=key,
            value=value,
            partition=partition,
            timestamp=timestamp,
            headers=headers,
            schema=schema,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            callback=callback,
            eager_partitioning=eager_partitioning,
        )
        self.app.producer.send_soon(fut)
        return fut

    async def put(self, event: EventT) -> None:
        """Put event directly onto the underlying queue of this topic.

        This will only affect subscribers to a particular
        instance, in a particular process.
        """
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
                'has_prefix': self.has_prefix,
            },
        }

    @property
    def pattern(self) -> Optional[Pattern]:
        """Regular expression used by this topic (if any)."""
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: Optional[Union[str, Pattern]]) -> None:
        """Set the regular expression pattern this topic subscribes to."""
        if pattern and self.topics:
            raise TypeError('Cannot specify both topics and pattern')
        if pattern:
            if isinstance(pattern, str):
                self._pattern = re.compile(pattern)
            else:
                self._pattern = pattern
        else:
            self._pattern = None

    @property
    def partitions(self) -> Optional[int]:
        """Return the number of configured partitions for this topic.

        Notes:
            This is only active for internal topics, fully owned
            and managed by Faust itself.

            We never touch the configuration of a topic that exists in Kafka,
            and Kafka will sometimes automatically create topics
            when they don't exist.  In this case the number of
            partitions for the automatically created topic
            will depend on the Kafka server configuration
            (``num.partitions``).

            Always make sure your topics have the correct
            number of partitions.
        """
        return self._partitions

    @partitions.setter
    def partitions(self, partitions: Optional[int]) -> None:
        """Set the number of partitions for this topic.

        Only used for internal topics, see :attr:`partitions`.
        """
        if partitions == 0:
            raise ValueError('Topic cannot have zero partitions')
        self._partitions = partitions

    def derive(self, **kwargs: Any) -> "Topic":
        """Create topic derived from the configuration of this topic.

        Configuration will be copied from this topic, but any parameter
        overridden as a keyword argument.

        See Also:
            :meth:`derive_topic`: for a list of supported keyword arguments.
        """
        return self.derive_topic(**kwargs)

    def derive_topic(
        self,
        *,
        topics: Optional[Sequence[str]] = None,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[Callable[[Any], bytes]] = None,
        value_serializer: Optional[Callable[[Any], bytes]] = None,
        partitions: Optional[int] = None,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        internal: Optional[bool] = None,
        config: Optional[Mapping[str, Any]] = None,
        prefix: str = '',
        suffix: str = '',
        **kwargs: Any
    ) -> "Topic":
        """Create new topic with configuration derived from this topic."""
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
            has_prefix=bool(self.has_prefix or prefix),
        )

    def get_topic_name(self) -> str:
        """Return the main topic name of this topic description.

        As topic descriptions can have multiple topic names, this will only
        return when the topic has a singular topic name in the description.

        Raises:
            TypeError: if configured with a regular expression pattern.
            ValueError: if configured with multiple topic names.
            TypeError: if not configured with any names or patterns.
        """
        if self.pattern:
            raise TypeError('Topic with pattern subscription cannot be identified')
        if self.topics:
            if len(self.topics) > 1:
                raise ValueError('Topic with multiple topic names cannot be identified')
            return self.topics[0]
        raise TypeError('Topic has no subscriptions (no pattern, no topics)')

    async def _get_producer(self) -> ProducerT:
        return await self.app.maybe_start_producer()

    async def publish_message(self, fut: FutureMessage, wait: bool = False) -> Any:
        """Fulfill promise to publish message to topic."""
        app: _App = self.app
        message = fut.message
        topic: Optional[str] = self._topic_name_or_default(message.channel)
        key: bytes = cast(bytes, message.key)
        value: bytes = cast(bytes, message.value)
        partition: Optional[int] = message.partition
        timestamp: float = cast(float, message.timestamp)
        headers = message.headers
        logger.debug('send: topic=%r k=%r v=%r timestamp=%r partition=%r', topic, key, value, timestamp, partition)
        assert topic is not None
        producer: ProducerT = await self._get_producer()
        state: Any = app.sensors.on_send_initiated(
            producer, topic, message=message, keysize=len(key) if key else 0, valsize=len(value) if value else 0
        )
        if wait:
            ret = await producer.send_and_wait(topic, key, value, partition=partition, timestamp=timestamp, headers=headers)
            app.sensors.on_send_completed(producer, state, ret)
            return await self._finalize_message(fut, ret)
        else:
            fut2: asyncio.Future[Any] = cast(asyncio.Future, await producer.send(topic, key, value, partition=partition, timestamp=timestamp, headers=headers))
            callback = partial(self._on_published, fut, message=fut, producer=producer, state=state)
            fut2.add_done_callback(cast(Callable[[asyncio.Future[Any]], None], callback))
            return fut2

    def _topic_name_or_default(self, obj: Union[str, TopicT, Any]) -> str:
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, TopicT):
            return obj.get_topic_name()
        else:
            return self.get_topic_name()

    def _on_published(self, fut: asyncio.Future[Any], message: FutureMessage, producer: ProducerT, state: Any) -> None:
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
        """Declare/create this topic, only if it does not exist."""
        await self.declare()

    async def declare(self) -> None:
        """Declare/create this topic on the server."""
        partitions: Optional[int] = self.partitions
        if partitions is None:
            partitions = self.app.conf.topic_partitions
        if self.replicas is None:
            replicas: Optional[int] = self.app.conf.topic_replication_factor
        else:
            replicas = self.replicas
        if self.app.conf.topic_allow_declare:
            producer: ProducerT = await self._get_producer()
            for topic in self.topics:
                await producer.create_topic(
                    topic=topic,
                    partitions=partitions,
                    replication=replicas or 0,
                    config=self.config,
                    compacting=self.compacting,
                    deleting=self.deleting,
                    retention=self.retention,
                )

    def __aiter__(self) -> AsyncIterator[EventT]:
        if self.is_iterator:
            return self  # type: ignore
        else:
            channel: Topic = self.clone(is_iterator=True)
            self.app.topics.add(cast(TopicT, channel))
            return channel

    def __str__(self) -> str:
        return str(self.pattern) if self.pattern else ','.join(self.topics)
