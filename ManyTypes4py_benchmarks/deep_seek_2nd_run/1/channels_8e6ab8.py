"""Channel.

A channel is used to send values to streams.

The stream will iterate over incoming events in the channel.

"""
import asyncio
from typing import Any, Awaitable, Callable, Mapping, MutableSet, Optional, Set, Tuple, TypeVar, cast, no_type_check
from weakref import WeakSet
from mode import Seconds, get_logger, want_seconds
from mode.utils.futures import maybe_async, stampede
from mode.utils.queues import ThrowableQueue
from .types import AppT, ChannelT, CodecArg, EventT, FutureMessage, K, Message, MessageSentCallback, ModelArg, PendingMessage, RecordMetadata, SchemaT, StreamT, TP, V
from .types.core import HeadersArg, OpenHeadersArg, prepare_headers
from .types.tuples import _PendingMessage_to_Message
__all__ = ['Channel']
logger = get_logger(__name__)
T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)

class Channel(ChannelT[T]):
    """Create new channel.

    Arguments:
        app: The app that created this channel (``app.channel()``)

        schema: Schema used for serialization/deserialization
        key_type:  The Model used for keys in this channel.
           (overrides schema if one is defined)
        value_type: The Model used for values in this channel.
           (overrides schema if one is defined)
        maxsize: The maximum number of messages this channel can hold.
                 If exceeded any new ``put`` call will block until a message
                 is removed from the channel.
        is_iterator: When streams iterate over a channel they will call
            ``stream.clone(is_iterator=True)`` so this attribute
            denotes that this channel instance is currently being iterated
            over.
        active_partitions: Set of active topic partitions this
           channel instance is assigned to.
        loop: The :mod:`asyncio` event loop to use.
    """

    def __init__(
        self,
        app: AppT,
        *,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        is_iterator: bool = False,
        queue: Optional[ThrowableQueue] = None,
        maxsize: Optional[int] = None,
        root: Optional['Channel'] = None,
        active_partitions: Optional[Set[TP]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        self.app: AppT = app
        self.loop: Optional[asyncio.AbstractEventLoop] = loop
        self.is_iterator: bool = is_iterator
        self._queue: Optional[ThrowableQueue] = queue
        self.maxsize: Optional[int] = maxsize
        self.deliver: Callable[[Message], Awaitable[None]] = self._compile_deliver()
        self._root: 'Channel' = cast(Channel, root)
        self.active_partitions: Optional[Set[TP]] = active_partitions
        self._subscribers: MutableSet['Channel'] = WeakSet()
        if schema is None:
            self.schema: SchemaT = self._get_default_schema(key_type, value_type)
        else:
            self.schema: SchemaT = schema
            self.schema.update(key_type=key_type, value_type=value_type)
        self.key_type: Optional[ModelArg] = self.schema.key_type
        self.value_type: Optional[ModelArg] = self.schema.value_type

    def _get_default_schema(
        self,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None
    ) -> SchemaT:
        return cast(SchemaT, self.app.conf.Schema(key_type=key_type, value_type=value_type))

    @property
    def queue(self) -> ThrowableQueue:
        """Return the underlying queue/buffer backing this channel."""
        if self._queue is None:
            maxsize = self.maxsize
            if maxsize is None:
                maxsize = self.app.conf.stream_buffer_maxsize
            self._queue = self.app.FlowControlQueue(
                maxsize=maxsize,
                loop=self.loop,
                clear_on_resume=True
            )
        return self._queue

    def clone(
        self,
        *,
        is_iterator: Optional[bool] = None,
        **kwargs: Any
    ) -> 'Channel':
        """Create clone of this channel.

        Arguments:
            is_iterator: Set to True if this is now a channel
                that is being iterated over.

        Keyword Arguments:
            **kwargs: Any keyword arguments passed will override any
                of the arguments supported by
                :class:`Channel.__init__ <Channel>`.
        """
        is_it = is_iterator if is_iterator is not None else self.is_iterator
        subchannel = self._clone(is_iterator=is_it, **kwargs)
        if is_it:
            (self._root or self)._subscribers.add(cast(Channel, subchannel))
        subchannel.queue
        return subchannel

    def clone_using_queue(self, queue: ThrowableQueue) -> 'Channel':
        """Create clone of this channel using specific queue instance."""
        return self.clone(queue=queue, is_iterator=True)

    def _clone(self, **kwargs: Any) -> 'Channel':
        return type(self)(**{**self._clone_args(), **kwargs})

    def _clone_args(self) -> Mapping[str, Any]:
        return {
            'app': self.app,
            'loop': self.loop,
            'schema': self.schema,
            'key_type': self.key_type,
            'value_type': self.value_type,
            'maxsize': self.maxsize,
            'root': self._root if self._root is not None else self,
            'queue': None,
            'active_partitions': self.active_partitions
        }

    def stream(self, **kwargs: Any) -> StreamT:
        """Create stream reading from this channel."""
        return self.app.stream(self, **kwargs)

    def get_topic_name(self) -> str:
        """Get the topic name, or raise if this is not a named channel."""
        raise NotImplementedError('Channels are unnamed topics')

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
    ) -> FutureMessage:
        """Send message to channel."""
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
        """Produce message by adding to buffer.

        This method is only supported by :class:`~faust.Topic`.

        Raises:
            NotImplementedError: always for in-memory channel.
        """
        raise NotImplementedError()

    def as_future_message(
        self,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        eager_partitioning: bool = False
    ) -> FutureMessage:
        """Create promise that message will be transmitted."""
        open_headers = self.prepare_headers(headers)
        final_key, open_headers = self.prepare_key(key, key_serializer, schema, open_headers)
        final_value, open_headers = self.prepare_value(value, value_serializer, schema, open_headers)
        if partition is None and eager_partitioning:
            partition = self.app.producer.key_partition(
                self.get_topic_name(), final_key
            ).partition
        return FutureMessage(PendingMessage(
            self, final_key, final_value,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            partition=partition,
            timestamp=timestamp,
            headers=open_headers,
            callback=callback,
            topic=None,
            offset=None
        ))

    def prepare_headers(self, headers: Optional[HeadersArg]) -> OpenHeadersArg:
        """Prepare ``headers`` passed before publishing."""
        if headers is not None:
            return prepare_headers(headers)
        return {}

    async def _send_now(
        self,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None
    ) -> FutureMessage:
        return await self.publish_message(self.as_future_message(
            key, value,
            partition=partition,
            timestamp=timestamp,
            headers=headers,
            schema=schema,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            callback=callback
        ))

    async def publish_message(self, fut: FutureMessage, wait: bool = True) -> FutureMessage:
        """Publish message to channel."""
        event = self._future_message_to_event(fut)
        await self.put(event)
        topic, partition = tp = TP(fut.message.topic or '<anon>', fut.message.partition or -1)
        return await self._finalize_message(
            fut,
            RecordMetadata(
                topic=topic,
                partition=partition,
                topic_partition=tp,
                offset=-1,
                timestamp=fut.message.timestamp,
                timestamp_type=1
            )
        )

    def _future_message_to_event(self, fut: FutureMessage) -> EventT:
        return self._create_event(
            fut.message.key,
            fut.message.value,
            fut.message.headers,
            message=_PendingMessage_to_Message(fut.message)
        )

    async def _finalize_message(self, fut: FutureMessage, result: RecordMetadata) -> FutureMessage:
        fut.set_result(result)
        if fut.message.callback:
            await maybe_async(fut.message.callback(fut))
        return fut

    @stampede
    async def maybe_declare(self) -> None:
        """Declare/create this channel, but only if it doesn't exist."""
        ...

    async def declare(self) -> None:
        """Declare/create this channel.

        This is used to create this channel on a server,
        if that is required to operate it.
        """
        ...

    def prepare_key(
        self,
        key: Optional[K],
        key_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[OpenHeadersArg] = None
    ) -> Tuple[Optional[bytes], OpenHeadersArg]:
        """Prepare key before it is sent to this channel."""
        return (key, headers or {})

    def prepare_value(
        self,
        value: Optional[V],
        value_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[OpenHeadersArg] = None
    ) -> Tuple[Optional[bytes], OpenHeadersArg]:
        """Prepare value before it is sent to this channel."""
        return (value, headers or {})

    async def decode(self, message: Message, *, propagate: bool = False) -> EventT:
        """Decode message into event."""
        return self._create_event(message.key, message.value, message.headers, message=message)

    def _compile_deliver(self) -> Callable[[Message], Awaitable[None]]:
        put = None

        async def deliver(message: Message) -> None:
            nonlocal put
            if put is None:
                put = self.queue.put
            event = await self.decode(message)
            await put(event)
        return deliver

    def _create_event(
        self,
        key: Optional[K],
        value: Optional[V],
        headers: Optional[HeadersArg],
        message: Optional[Message] = None
    ) -> EventT:
        return self.app.create_event(key, value, headers, message)

    async def put(self, value: EventT) -> None:
        """Put event onto this channel."""
        root = self._root if self._root is not None else self
        for subscriber in root._subscribers:
            await subscriber.queue.put(value)

    async def get(self, *, timeout: Optional[Seconds] = None) -> EventT:
        """Get the next event received on this channel."""
        timeout_ = want_seconds(timeout)
        if timeout_:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout_)
        return await self.queue.get()

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return self.queue.empty()

    async def on_key_decode_error(self, exc: Exception, message: Message) -> None:
        """Unable to decode the key of an item in the queue."""
        await self.on_decode_error(exc, message)
        await self.throw(exc)

    async def on_value_decode_error(self, exc: Exception, message: Message) -> None:
        """Unable to decode the value of an item in the queue."""
        await self.on_decode_error(exc, message)
        await self.throw(exc)

    async def on_decode_error(self, exc: Exception, message: Message) -> None:
        """Signal that there was an error reading an event in the queue."""
        ...

    def on_stop_iteration(self) -> None:
        """Signal that iteration over this channel was stopped."""
        ...

    def derive(self, **kwargs: Any) -> 'Channel':
        """Derive new channel from this channel, using new configuration."""
        return self

    def __aiter__(self) -> 'Channel':
        return self if self.is_iterator else self.clone(is_iterator=True)

    async def __anext__(self) -> EventT:
        if not self.is_iterator:
            raise RuntimeError('Need to call channel.__aiter__()')
        return await self.queue.get()

    async def throw(self, exc: Exception) -> None:
        """Throw exception to be received by channel subscribers."""
        self.queue._throw(exc)

    def _throw(self, exc: Exception) -> None:
        """Non-async version of throw."""
        self.queue._throw(exc)

    def __repr__(self) -> str:
        s = f'<{self.label}@{self._object_id_as_hex()}'
        if self.active_partitions is not None:
            if self.active_partitions:
                active = '{' + ', '.join(sorted(
                    f'{tp.topic}:{tp.partition}' for tp in self.active_partitions
                )) + '}'
            else:
                active = '{<pending for assignment>}'
            s += f' active={active}'
        s += '>'
        return s

    def _object_id_as_hex(self) -> str:
        return f'{id(self):#x}'

    def __str__(self) -> str:
        return '<ANON>'

    @property
    def subscriber_count(self) -> int:
        """Return number of active subscribers to local channel."""
        return len(self._subscribers)

    @property
    def label(self) -> str:
        """Short textual description of channel."""
        sym = '(*)' if self.is_iterator else ''
        return f'{sym}{type(self).__name__}: {self}'


class SerializedChannel(Channel[T]):
    def __init__(
        self,
        app: AppT,
        *,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        self.app: AppT = app
        if schema is not None:
            self._contribute_to_schema(
                schema,
                key_type=key_type,
                value_type=value_type,
                key_serializer=key_serializer,
                value_serializer=value_serializer,
                allow_empty=allow_empty
            )
        else:
            schema = self._get_default_schema(
                key_type,
                value_type,
                key_serializer,
                value_serializer,
                allow_empty
            )
        super().__init__(
            app,
            schema=schema,
            key_type=key_type,
            value_type=value_type,
            **kwargs
        )
        self.key_serializer: Optional[CodecArg] = self.schema.key_serializer
        self.value_serializer: Optional[CodecArg] = self.schema.value_serializer
        self.allow_empty: Optional[bool] = self.schema.allow_empty

    def _contribute_to_schema(
        self,
        schema: SchemaT,
        *,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow