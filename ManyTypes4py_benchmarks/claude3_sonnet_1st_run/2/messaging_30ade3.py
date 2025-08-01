import asyncio
import json
import socket
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from functools import partial
from types import TracebackType
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union, cast
import orjson
from redis.asyncio import Redis
from redis.exceptions import ResponseError
from typing_extensions import Self
from prefect.logging import get_logger
from prefect.server.utilities.messaging import Cache as _Cache
from prefect.server.utilities.messaging import Consumer as _Consumer
from prefect.server.utilities.messaging import Message, MessageHandler, StopConsumer
from prefect.server.utilities.messaging import Publisher as _Publisher
from prefect_redis.client import get_async_redis_client
logger = get_logger(__name__)
M = TypeVar('M', bound=Message)
MESSAGE_DEDUPLICATION_LOOKBACK = timedelta(minutes=5)

class Cache(_Cache):

    def __init__(self, topic: str = 'messaging-cache') -> None:
        self.topic = topic
        self._client: Redis = get_async_redis_client()

    async def clear_recently_seen_messages(self) -> None:
        return

    async def without_duplicates(self, attribute: str, messages: List[Message]) -> List[Message]:
        messages_with_attribute: List[Message] = []
        messages_without_attribute: List[Message] = []
        async with self._client.pipeline() as p:
            for m in messages:
                if m.attributes is None or attribute not in m.attributes:
                    logger.warning('Message is missing deduplication attribute %r', attribute, extra={'event_message': m})
                    messages_without_attribute.append(m)
                    continue
                p.set(f'message:{self.topic}:{m.attributes[attribute]}', '1', nx=True, ex=MESSAGE_DEDUPLICATION_LOOKBACK)
                messages_with_attribute.append(m)
            results = await p.execute()
        return [m for i, m in enumerate(messages_with_attribute) if results[i]] + messages_without_attribute

    async def forget_duplicates(self, attribute: str, messages: List[Message]) -> None:
        async with self._client.pipeline() as p:
            for m in messages:
                if m.attributes is None or attribute not in m.attributes:
                    logger.warning('Message is missing deduplication attribute %r', attribute, extra={'event_message': m})
                    continue
                p.delete(f'message:{self.topic}:{m.attributes[attribute]}')
            await p.execute()

class RedisStreamsMessage(Message):
    """
    A message sent to a Redis stream.
    """

    def __init__(self, data: Union[str, bytes], attributes: Optional[Dict[str, Any]], acker: Optional[Callable[[], Awaitable[None]]] = None) -> None:
        self.data = data.decode() if isinstance(data, bytes) else data
        self.attributes = attributes
        self.acker = acker

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RedisStreamsMessage):
            return NotImplemented
        return self.data == other.data and self.attributes == other.attributes

    async def acknowledge(self) -> None:
        assert self.acker is not None
        await self.acker()

class Subscription:
    """
    A subscription-like object for Redis. We mimic the memory subscription interface
    so that we can set max_retries and handle dead letter queue storage in Redis.
    """

    def __init__(self, max_retries: int = 3, dlq_key: str = 'dlq') -> None:
        self.max_retries = max_retries
        self.dlq_key = dlq_key

class Publisher(_Publisher):

    def __init__(self, topic: str, cache: Cache, deduplicate_by: Optional[str] = None, batch_size: int = 5, publish_every: Optional[timedelta] = None) -> None:
        self.stream = topic
        self.cache = cache
        self.deduplicate_by = deduplicate_by
        self.batch_size = batch_size
        self.publish_every = publish_every
        self._periodic_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> Self:
        self._client: Redis = get_async_redis_client()
        self._batch: List[RedisStreamsMessage] = []
        if self.publish_every is not None:
            interval = self.publish_every.total_seconds()

            async def _publish_periodically() -> None:
                while True:
                    await asyncio.sleep(interval)
                    await asyncio.shield(self._publish_current_batch())
            self._periodic_task = asyncio.create_task(_publish_periodically())
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        if not hasattr(self, '_batch'):
            raise RuntimeError('Use this publisher as an async context manager')
        try:
            if self._periodic_task:
                self._periodic_task.cancel()
            await self._publish_current_batch()
        except Exception:
            if self.deduplicate_by:
                await self.cache.forget_duplicates(self.deduplicate_by, self._batch)
            raise

    async def publish_data(self, data: Union[str, bytes], attributes: Optional[Dict[str, Any]]) -> None:
        if not hasattr(self, '_batch'):
            raise RuntimeError('Use this publisher as an async context manager')
        self._batch.append(RedisStreamsMessage(data=data, attributes=attributes))
        if len(self._batch) >= self.batch_size:
            await asyncio.shield(self._publish_current_batch())

    async def _publish_current_batch(self) -> None:
        if not self._batch:
            return
        if self.deduplicate_by:
            to_publish = await self.cache.without_duplicates(self.deduplicate_by, self._batch)
        else:
            to_publish = list(self._batch)
        self._batch.clear()
        try:
            for message in to_publish:
                await self._client.xadd(self.stream, {'data': message.data, 'attributes': orjson.dumps(message.attributes)})
        except Exception:
            if self.deduplicate_by:
                await self.cache.forget_duplicates(self.deduplicate_by, to_publish)
            raise

class Consumer(_Consumer):
    """
    Consumer implementation for Redis Streams with DLQ support.
    """

    def __init__(
        self,
        topic: str,
        name: Optional[str] = None,
        group: Optional[str] = None,
        block: timedelta = timedelta(seconds=1),
        min_idle_time: timedelta = timedelta(seconds=0),
        should_process_pending_messages: bool = True,
        starting_message_id: str = '0',
        automatically_acknowledge: bool = True,
        max_retries: int = 3,
        trim_every: timedelta = timedelta(seconds=60)
    ) -> None:
        self.name = name or topic
        self.stream = topic
        self.group = group or topic
        self.block = block
        self.min_idle_time = min_idle_time
        self.should_process_pending_messages = should_process_pending_messages
        self.starting_message_id = starting_message_id
        self.automatically_acknowledge = automatically_acknowledge
        self.subscription = Subscription(max_retries=max_retries)
        self._retry_counts: Dict[str, int] = {}
        self.trim_every = trim_every
        self._last_trimmed: Optional[float] = None

    async def _ensure_stream_and_group(self, redis_client: Redis) -> None:
        """Ensure the stream and consumer group exist."""
        try:
            await redis_client.xgroup_create(self.stream, self.group, id=self.starting_message_id, mkstream=True)
        except ResponseError as e:
            if 'BUSYGROUP Consumer Group name already exists' not in str(e):
                raise
            logger.debug('Consumer group already exists: %s', e)

    async def process_pending_messages(
        self,
        handler: MessageHandler,
        redis_client: Redis,
        message_batch_size: int,
        start_id: str = '0-0'
    ) -> None:
        acker = partial(redis_client.xack, self.stream, self.group)
        while True:
            result = await redis_client.xautoclaim(
                name=self.stream,
                groupname=self.group,
                consumername=self.name,
                min_idle_time=int(self.min_idle_time.total_seconds() * 1000),
                start_id=start_id,
                count=message_batch_size
            )
            next_start_id, claimed_messages = (result[0], result[1])
            if not claimed_messages:
                break
            for message_id, message in claimed_messages:
                await self._handle_message(message_id, message, handler, acker)
            start_id = next_start_id

    async def run(self, handler: MessageHandler) -> None:
        redis_client: Redis = get_async_redis_client()
        await self._ensure_stream_and_group(redis_client)
        while True:
            if self.should_process_pending_messages:
                try:
                    await self.process_pending_messages(handler, redis_client, 1)
                except StopConsumer:
                    return
            try:
                stream_entries = await redis_client.xreadgroup(
                    groupname=self.group,
                    consumername=self.name,
                    streams={self.stream: '>'},
                    count=1,
                    block=int(self.block.total_seconds() * 1000)
                )
            except ResponseError as e:
                logger.error(f'Failed to read from stream: {e}')
                raise
            if not stream_entries:
                await self._trim_stream_if_necessary()
                continue
            acker = partial(redis_client.xack, self.stream, self.group)
            for _, messages in stream_entries:
                for message_id, message in messages:
                    try:
                        await self._handle_message(message_id, message, handler, acker)
                    except StopConsumer:
                        return

    async def _handle_message(
        self,
        message_id: Union[str, bytes],
        message: Dict[str, bytes],
        handler: MessageHandler,
        acker: Callable[[Union[str, bytes]], Awaitable[int]]
    ) -> None:
        redis_stream_message = RedisStreamsMessage(
            data=message[b'data'] if isinstance(message, dict) and b'data' in message else message['data'],
            attributes=orjson.loads(message[b'attributes'] if isinstance(message, dict) and b'attributes' in message else message['attributes']),
            acker=cast(Callable[[], Awaitable[None]], partial(acker, message_id))
        )
        msg_id_str = message_id.decode() if isinstance(message_id, bytes) else message_id
        try:
            await handler(redis_stream_message)
            if self.automatically_acknowledge:
                await redis_stream_message.acknowledge()
        except StopConsumer as e:
            if not e.ack:
                await self._on_message_failure(redis_stream_message, msg_id_str)
            elif self.automatically_acknowledge:
                await redis_stream_message.acknowledge()
            raise
        except Exception:
            await self._on_message_failure(redis_stream_message, msg_id_str)
        finally:
            await self._trim_stream_if_necessary()

    async def _on_message_failure(self, msg: RedisStreamsMessage, msg_id_str: str) -> None:
        current_count = self._retry_counts.get(msg_id_str, 0) + 1
        self._retry_counts[msg_id_str] = current_count
        if current_count > self.subscription.max_retries:
            await self._send_to_dlq(msg, current_count)
            await msg.acknowledge()
        else:
            pass

    async def _send_to_dlq(self, msg: RedisStreamsMessage, retry_count: int) -> None:
        """Store failed messages in Redis instead of filesystem"""
        redis_client: Redis = get_async_redis_client()
        data_str = msg.data.decode() if isinstance(msg.data, bytes) else msg.data
        dlq_message = {
            'data': data_str,
            'attributes': msg.attributes,
            'retry_count': retry_count,
            'timestamp': str(asyncio.get_event_loop().time()),
            'message_id': str(uuid.uuid4().hex)
        }
        message_id = f'dlq:{uuid.uuid4().hex}'
        await redis_client.hset(message_id, mapping={'data': json.dumps(dlq_message)})
        await redis_client.sadd(self.subscription.dlq_key, message_id)

    async def _trim_stream_if_necessary(self) -> None:
        now = time.monotonic()
        if self._last_trimmed is None:
            self._last_trimmed = now
        if now - self._last_trimmed > self.trim_every.total_seconds():
            await _trim_stream_to_lowest_delivered_id(self.stream)
            self._last_trimmed = now

@asynccontextmanager
async def ephemeral_subscription(
    topic: str,
    source: Optional[str] = None,
    group: Optional[str] = None
) -> AsyncGenerator[Dict[str, str], None]:
    source = source or topic
    group_name = group or f'ephemeral-{socket.gethostname()}-{uuid.uuid4().hex}'
    redis_client: Redis = get_async_redis_client()
    await redis_client.xgroup_create(source, group_name, id='0', mkstream=True)
    try:
        yield {'topic': topic, 'name': topic, 'group': group_name}
    finally:
        await redis_client.xgroup_destroy(source, group_name)

@asynccontextmanager
async def break_topic() -> AsyncGenerator[None, None]:
    from unittest import mock
    publishing_mock = mock.AsyncMock(side_effect=ValueError('oops'))
    with mock.patch('redis.asyncio.client.Redis.xadd', publishing_mock):
        yield

async def _trim_stream_to_lowest_delivered_id(stream_name: str) -> None:
    """
    Trims a Redis stream by removing all messages that have been delivered to and
    acknowledged by all consumer groups.

    This function finds the lowest last-delivered-id across all consumer groups and
    trims the stream up to that point, as we know all consumers have processed those
    messages.

    Args:
        stream_name: The name of the Redis stream to trim
    """
    redis_client: Redis = get_async_redis_client()
    groups = await redis_client.xinfo_groups(stream_name)
    if not groups:
        logger.debug(f'No consumer groups found for stream {stream_name}')
        return
    lowest_id = min((group['last-delivered-id'] for group in groups if group['last-delivered-id'] != '0-0'))
    if lowest_id == '0-0':
        logger.debug(f'No messages have been delivered in stream {stream_name}')
        return
    await redis_client.xtrim(stream_name, minid=lowest_id, approximate=True)
