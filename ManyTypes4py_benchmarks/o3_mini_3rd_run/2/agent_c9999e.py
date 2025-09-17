#!/usr/bin/env python3
"""Agent implementation."""
from __future__ import annotations

import asyncio
import typing
from contextlib import suppress
from contextvars import ContextVar
from time import time
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    MutableSet,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)
from uuid import uuid4
from weakref import WeakSet, WeakValueDictionary

from mode import CrashingSupervisor, Service, ServiceT, SupervisorStrategyT
from mode.utils.aiter import aenumerate, aiter
from mode.utils.compat import want_bytes, want_str
from mode.utils.futures import maybe_async
from mode.utils.objects import canonshortname, qualname
from mode.utils.text import shorten_fqdn
from mode.utils.types.trees import NodeT
from faust.exceptions import ImproperlyConfigured
from faust.utils.tracing import traced_from_parent_span
from faust.types import AppT, ChannelT, CodecArg, EventT, HeadersArg, K, Message, MessageSentCallback, ModelArg, ModelT, RecordMetadata, StreamT, TP, TopicT, V
from faust.types.agents import ActorRefT, ActorT, AgentErrorHandler, AgentFun, AgentT, AgentTestWrapperT, ReplyToArg, SinkT
from faust.types.core import merge_headers, prepare_headers
from faust.types.serializers import SchemaT
from .actor import AsyncIterableActor, AwaitableActor
from .models import ModelReqRepRequest, ModelReqRepResponse, ReqRepRequest, ReqRepResponse
from .replies import BarrierState, ReplyPromise

if typing.TYPE_CHECKING:
    from faust.app.base import App as _App
else:
    class _App:
        ...


__all__ = ['Agent']

_current_agent: ContextVar[Optional[Agent]] = ContextVar('current_agent')


def current_agent() -> Optional[Agent]:
    return _current_agent.get(None)


class Agent(AgentT, Service):
    """Agent.

    This is the type of object returned by the ``@app.agent`` decorator.
    """
    supervisor: Optional[SupervisorStrategyT] = None
    _channel: Optional[ChannelT] = None
    _channel_iterator: Optional[ChannelT] = None
    _pending_active_partitions: Optional[Set[Any]] = None
    _first_assignment_done: bool = False

    def __init__(
        self,
        fun: AgentFun,
        *,
        app: AppT,
        name: Optional[str] = None,
        channel: Optional[Union[str, ChannelT]] = None,
        concurrency: int = 1,
        sink: Optional[Iterable[Any]] = None,
        on_error: Optional[AgentErrorHandler] = None,
        supervisor_strategy: Optional[Type[SupervisorStrategyT]] = None,
        help: Optional[str] = None,
        schema: Optional[SchemaT] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        isolated_partitions: bool = False,
        use_reply_headers: Any = None,
        **kwargs: Any,
    ) -> None:
        self.app = app
        self.fun = fun
        self.name = name or canonshortname(self.fun)
        if schema is not None:
            assert channel is None or isinstance(channel, str)
        if key_type is not None:
            assert channel is None or isinstance(channel, str)
        self._key_type = key_type
        if value_type is not None:
            assert channel is None or isinstance(channel, str)
        self._schema = schema
        self._value_type = value_type
        self._channel_arg = channel
        self._channel_kwargs = kwargs
        self.concurrency = concurrency or 1
        self.isolated_partitions = isolated_partitions
        self.help = help or ''
        self._sinks: List[Any] = list(sink) if sink is not None else []
        self._on_error = on_error
        self.supervisor_strategy = supervisor_strategy
        self._actors: WeakSet[Any] = WeakSet()
        self._actor_by_partition: WeakValueDictionary[Any, Any] = WeakValueDictionary()
        if self.isolated_partitions and self.concurrency > 1:
            raise ImproperlyConfigured('Agent concurrency must be 1 when using isolated partitions')
        self.use_reply_headers = use_reply_headers
        Service.__init__(self)

    def on_init_dependencies(self) -> List[Service]:
        """Return list of services dependencies required to start agent."""
        self.beacon.reattach(self.app.agents.beacon)
        return []

    def actor_tracebacks(self) -> List[Any]:
        return [actor.traceback() for actor in self._actors]

    async def _start_one(
        self,
        *,
        index: Optional[int] = None,
        active_partitions: Optional[Set[Any]] = None,
        stream: Optional[StreamT] = None,
        channel: Optional[ChannelT] = None,
    ) -> ActorRefT:
        index = index if self.concurrency > 1 else None
        return await self._start_task(index=index, active_partitions=active_partitions, stream=stream, channel=channel, beacon=self.beacon)

    async def _start_one_supervised(
        self,
        index: Optional[int] = None,
        active_partitions: Optional[Set[Any]] = None,
        stream: Optional[StreamT] = None,
    ) -> ActorRefT:
        aref = await self._start_one(index=index, active_partitions=active_partitions, stream=stream)
        self.supervisor.add(aref)
        await aref.maybe_start()
        return aref

    async def _start_for_partitions(self, active_partitions: Set[Any]) -> ActorRefT:
        assert active_partitions
        self.log.info('Starting actor for partitions %s', active_partitions)
        return await self._start_one_supervised(active_partitions=active_partitions)

    async def on_start(self) -> None:
        """Call when an agent starts."""
        self.supervisor = self._new_supervisor()
        await self._on_start_supervisor()

    def _new_supervisor(self) -> SupervisorStrategyT:
        return self._get_supervisor_strategy()(max_restarts=100.0, over=1.0, replacement=self._replace_actor, loop=self.loop, beacon=self.beacon)

    async def _replace_actor(self, service: ServiceT, index: Optional[int]) -> ActorRefT:
        aref = cast(ActorRefT, service)
        return await self._start_one(index=index, active_partitions=aref.active_partitions, stream=aref.stream, channel=cast(ChannelT, aref.stream.channel))

    def _get_supervisor_strategy(self) -> Type[SupervisorStrategyT]:
        SupervisorStrategy = self.supervisor_strategy
        if SupervisorStrategy is None:
            return cast(Type[SupervisorStrategyT], self.app.conf.agent_supervisor)
        else:
            return SupervisorStrategy

    async def _on_start_supervisor(self) -> None:
        active_partitions: Optional[Set[Any]] = self._get_active_partitions()
        channel: Optional[ChannelT] = None
        for i in range(self.concurrency):
            res = await self._start_one(index=i, active_partitions=active_partitions, channel=channel)
            if channel is None:
                channel = res.stream.channel
            self.supervisor.add(res)
        await self.supervisor.start()

    def _get_active_partitions(self) -> Optional[Set[Any]]:
        active_partitions: Optional[Set[Any]] = None
        if self.isolated_partitions:
            active_partitions = self._pending_active_partitions = set()
        return active_partitions

    async def on_stop(self) -> None:
        """Call when an agent stops."""
        await self._stop_supervisor()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*[aref.actor_task for aref in self._actors if aref.actor_task is not None])
        self._actors.clear()

    async def _stop_supervisor(self) -> None:
        if self.supervisor:
            await self.supervisor.stop()
            self.supervisor = cast(SupervisorStrategyT, None)

    def cancel(self) -> None:
        """Cancel agent and its actor instances running in this process."""
        for aref in self._actors:
            aref.cancel()

    async def on_partitions_revoked(self, revoked: Iterable[Any]) -> None:
        """Call when partitions are revoked."""
        T = traced_from_parent_span()
        if self.isolated_partitions:
            await T(self.on_isolated_partitions_revoked)(revoked)
        else:
            await T(self.on_shared_partitions_revoked)(revoked)

    async def on_partitions_assigned(self, assigned: Iterable[Any]) -> None:
        """Call when partitions are assigned."""
        T = traced_from_parent_span()
        if self.isolated_partitions:
            await T(self.on_isolated_partitions_assigned)(assigned)
        else:
            await T(self.on_shared_partitions_assigned)(assigned)

    async def on_isolated_partitions_revoked(self, revoked: Iterable[Any]) -> None:
        """Call when isolated partitions are revoked."""
        self.log.dev('Partitions revoked')
        T = traced_from_parent_span()
        for tp in revoked:
            aref = self._actor_by_partition.pop(tp, None)
            if aref is not None:
                await T(aref.on_isolated_partition_revoked)(tp)

    async def on_isolated_partitions_assigned(self, assigned: Iterable[Any]) -> None:
        """Call when isolated partitions are assigned."""
        T = traced_from_parent_span()
        for tp in sorted(assigned):
            await T(self._assign_isolated_partition)(tp)

    async def _assign_isolated_partition(self, tp: Any) -> None:
        T = traced_from_parent_span()
        if not self._first_assignment_done and (not self._actor_by_partition):
            self._first_assignment_done = True
            T(self._on_first_isolated_partition_assigned)(tp)
        await T(self._maybe_start_isolated)(tp)

    def _on_first_isolated_partition_assigned(self, tp: Any) -> None:
        assert self._actors
        assert len(self._actors) == 1
        self._actor_by_partition[tp] = next(iter(self._actors))
        if self._pending_active_partitions is not None:
            assert not self._pending_active_partitions
            self._pending_active_partitions.add(tp)

    async def _maybe_start_isolated(self, tp: Any) -> None:
        try:
            aref = self._actor_by_partition[tp]
        except KeyError:
            aref = await self._start_isolated(tp)
            self._actor_by_partition[tp] = aref
        await aref.on_isolated_partition_assigned(tp)

    async def _start_isolated(self, tp: Any) -> ActorRefT:
        return await self._start_for_partitions({tp})

    async def on_shared_partitions_revoked(self, revoked: Iterable[Any]) -> None:
        """Call when non-isolated partitions are revoked."""
        ...

    async def on_shared_partitions_assigned(self, assigned: Iterable[Any]) -> None:
        """Call when non-isolated partitions are assigned."""
        ...

    def info(self) -> Dict[str, Any]:
        """Return agent attributes as a dictionary."""
        return {
            'app': self.app,
            'fun': self.fun,
            'name': self.name,
            'channel': self.channel,
            'concurrency': self.concurrency,
            'help': self.help,
            'sink': self._sinks,
            'on_error': self._on_error,
            'supervisor_strategy': self.supervisor_strategy,
            'isolated_partitions': self.isolated_partitions,
        }

    def clone(self, *, cls: Optional[Type[Agent]] = None, **kwargs: Any) -> Agent:
        """Create clone of this agent object.

        Keyword arguments can be passed to override any argument
        supported by :class:`Agent.__init__ <Agent>`.
        """
        return (cls or type(self))(**{**self.info(), **kwargs})

    def test_context(self, channel: Optional[ChannelT] = None, supervisor_strategy: Optional[Type[SupervisorStrategyT]] = None, on_error: Optional[AgentErrorHandler] = None, **kwargs: Any) -> AgentTestWrapperT:
        """Create new unit-testing wrapper for this agent."""
        self.app.flow_control.resume()

        async def on_agent_error(agent: Agent, exc: Exception) -> None:
            if on_error is not None:
                await on_error(agent, exc)
            await cast(AgentTestWrapper, agent).crash_test_agent(exc)
        return cast(
            AgentTestWrapperT,
            self.clone(
                cls=AgentTestWrapper,
                channel=channel if channel is not None else self.app.channel(),
                supervisor_strategy=supervisor_strategy or CrashingSupervisor,
                original_channel=self.channel,
                on_error=on_agent_error,
                **kwargs,
            ),
        )

    def _prepare_channel(
        self,
        channel: Optional[Union[str, ChannelT]] = None,
        internal: bool = True,
        schema: Optional[SchemaT] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChannelT:
        app = self.app
        has_prefix = False
        if channel is None:
            channel = f'{app.conf.id}-{self.name}'
            has_prefix = True
        if isinstance(channel, ChannelT):
            return channel
        elif isinstance(channel, str):
            return app.topic(channel, internal=internal, schema=schema, key_type=key_type, value_type=value_type, has_prefix=has_prefix, **kwargs)
        raise TypeError(f'Channel must be channel, topic, or str; not {type(channel)}')

    def __call__(self, *, index: Optional[int] = None, active_partitions: Optional[Set[Any]] = None, stream: Optional[StreamT] = None, channel: Optional[ChannelT] = None) -> ActorRefT:
        """Create new actor instance for this agent."""
        return self.actor_from_stream(stream, index=index, active_partitions=active_partitions, channel=channel)

    def actor_from_stream(self, stream: Optional[StreamT], *, index: Optional[int] = None, active_partitions: Optional[Set[Any]] = None, channel: Optional[ChannelT] = None) -> ActorRefT:
        """Create new actor from stream."""
        we_created_stream: bool = False
        if stream is None:
            actual_stream: StreamT = self.stream(channel=channel, active_partitions=active_partitions)
            we_created_stream = True
        else:
            assert stream.concurrency_index == index
            assert stream.active_partitions == active_partitions
            actual_stream = stream
        res = self.fun(actual_stream)
        if isinstance(res, AsyncIterable):
            if we_created_stream:
                actual_stream.add_processor(self._maybe_unwrap_reply_request)
            return cast(ActorRefT, AsyncIterableActor(self, actual_stream, res, index=actual_stream.concurrency_index, active_partitions=actual_stream.active_partitions, loop=self.loop, beacon=self.beacon))
        else:
            return cast(ActorRefT, AwaitableActor(self, actual_stream, res, index=actual_stream.concurrency_index, active_partitions=actual_stream.active_partitions, loop=self.loop, beacon=self.beacon))

    def add_sink(self, sink: Any) -> None:
        """Add new sink to further handle results from this agent."""
        if sink not in self._sinks:
            self._sinks.append(sink)

    def stream(self, channel: Optional[ChannelT] = None, active_partitions: Optional[Set[Any]] = None, **kwargs: Any) -> StreamT:
        """Create underlying stream used by this agent."""
        if channel is None:
            channel = cast(TopicT, self.channel_iterator).clone(is_iterator=False, active_partitions=active_partitions)
        if active_partitions is not None:
            assert channel.active_partitions == active_partitions
        s: StreamT = self.app.stream(channel, loop=self.loop, active_partitions=active_partitions, prefix=self.name, beacon=self.beacon, **kwargs)
        return s

    def _maybe_unwrap_reply_request(self, value: Any) -> Any:
        if isinstance(value, ReqRepRequest):
            return value.value
        return value

    async def _start_task(
        self,
        *,
        index: Optional[int],
        active_partitions: Optional[Set[Any]] = None,
        stream: Optional[StreamT] = None,
        channel: Optional[ChannelT] = None,
        beacon: Any,
    ) -> ActorRefT:
        actor: ActorRefT = self(index=index, active_partitions=active_partitions, stream=stream, channel=channel)
        return await self._prepare_actor(actor, beacon if beacon is not None else self.beacon)

    async def _prepare_actor(self, aref: ActorRefT, beacon: Any) -> ActorRefT:
        if isinstance(aref, Awaitable):
            coro = aref
            if self._sinks:
                raise ImproperlyConfigured('Agent must yield to use sinks')
        else:
            coro = self._slurp(aref, aiter(aref))
        task: asyncio.Task = asyncio.Task(self._execute_actor(coro, aref), loop=self.loop)
        task._beacon = beacon
        aref.actor_task = task
        self._actors.add(aref)
        return aref

    async def _execute_actor(self, coro: Awaitable[Any], aref: ActorRefT) -> None:
        _current_agent.set(self)
        try:
            await coro
        except asyncio.CancelledError:
            if self.should_stop:
                raise
        except Exception as exc:
            if self._on_error is not None:
                await self._on_error(self, exc)
            await aref.crash(exc)
            self.supervisor.wakeup()

    async def _slurp(self, res: Any, it: AsyncIterator[Any]) -> None:
        stream: Optional[StreamT] = None
        async for value in it:
            self.log.debug('%r yielded: %r', self.fun, value)
            if stream is None:
                stream = res.stream.get_active_stream()
            event = stream.current_event
            if event is not None:
                headers = event.headers
                reply_to: Optional[str] = None
                correlation_id: Optional[str] = None
                if isinstance(event.value, ReqRepRequest):
                    req = event.value
                    reply_to = req.reply_to
                    correlation_id = req.correlation_id
                elif headers:
                    reply_to_bytes = headers.get('Faust-Ag-ReplyTo')
                    if reply_to_bytes:
                        reply_to = want_str(reply_to_bytes)
                        correlation_id_bytes = headers.get('Faust-Ag-CorrelationId')
                        if correlation_id_bytes:
                            correlation_id = want_str(correlation_id_bytes)
                if reply_to is not None:
                    await self._reply(event.key, value, reply_to, cast(str, correlation_id))
            await self._delegate_to_sinks(value)

    async def _delegate_to_sinks(self, value: Any) -> None:
        for sink in self._sinks:
            if isinstance(sink, Agent):
                await sink.send(value=value)
            elif isinstance(sink, ChannelT):
                await cast(TopicT, sink).send(value=value)
            else:
                await maybe_async(cast(Callable, sink)(value))

    async def _reply(self, key: Any, value: Any, reply_to: str, correlation_id: str) -> None:
        assert reply_to
        response = self._response_class(value)(key=key, value=value, correlation_id=correlation_id)
        await self.app.send(reply_to, key=None, value=response)

    def _response_class(self, value: Any) -> Type:
        if isinstance(value, ModelT):
            return ModelReqRepResponse
        return ReqRepResponse

    async def cast(self, value: Any = None, *, key: Any = None, partition: Any = None, timestamp: Any = None, headers: Optional[HeadersArg] = None) -> None:
        """RPC operation: like :meth:`ask` but do not expect reply.

        Cast here is like "casting a spell", and will not expect
        a reply back from the agent.
        """
        await self.send(key=key, value=value, partition=partition, timestamp=timestamp, headers=headers)

    async def ask(self, value: Any = None, *, key: Any = None, partition: Any = None, timestamp: Any = None, headers: Optional[HeadersArg] = None, reply_to: Optional[ReplyToArg] = None, correlation_id: Optional[str] = None) -> Any:
        """RPC operation: ask agent for result of processing value.

        This version will wait until the result is available
        and return the processed value.
        """
        p: ReplyPromise = await self.ask_nowait(value, key=key, partition=partition, timestamp=timestamp, headers=headers, reply_to=reply_to or self.app.conf.reply_to, correlation_id=correlation_id, force=True)
        app = cast(_App, self.app)
        await app._reply_consumer.add(p.correlation_id, p)
        await app.maybe_start_client()
        return await p

    async def ask_nowait(self, value: Any = None, *, key: Any = None, partition: Any = None, timestamp: Any = None, headers: Optional[HeadersArg] = None, reply_to: Optional[ReplyToArg] = None, correlation_id: Optional[str] = None, force: bool = False) -> ReplyPromise:
        """RPC operation: ask agent for result of processing value.

        This version does not wait for the result to arrive,
        but instead returns a promise of future evaluation.
        """
        if reply_to is None:
            raise TypeError('Missing reply_to argument')
        reply_to_str: str = self._get_strtopic(reply_to)
        correlation_id = correlation_id or str(uuid4())
        value, headers = self._create_req(key, value, reply_to_str, correlation_id, headers)
        await self.channel.send(key=key, value=value, partition=partition, timestamp=timestamp, headers=headers, force=force)
        return ReplyPromise(reply_to_str, correlation_id)

    def _create_req(self, key: Any = None, value: Any = None, reply_to: str = '', correlation_id: Optional[str] = None, headers: Optional[Mapping[str, Any]] = None) -> Tuple[Any, Mapping[str, Any]]:
        if reply_to is None:
            raise TypeError('Missing reply_to argument')
        topic_name: str = self._get_strtopic(reply_to)
        correlation_id = correlation_id or str(uuid4())
        open_headers: MutableMapping[str, Any] = prepare_headers(headers or {})
        if self.use_reply_headers:
            merge_headers(open_headers, {'Faust-Ag-ReplyTo': want_bytes(topic_name), 'Faust-Ag-CorrelationId': want_bytes(correlation_id)})
            return (value, open_headers)
        else:
            req = self._request_class(value)(value=value, reply_to=topic_name, correlation_id=correlation_id)
            return (req, open_headers)

    def _request_class(self, value: Any) -> Type:
        if isinstance(value, ModelT):
            return ModelReqRepRequest
        return ReqRepRequest

    async def send(
        self,
        *,
        key: Any = None,
        value: Any = None,
        partition: Any = None,
        timestamp: Any = None,
        headers: Optional[HeadersArg] = None,
        key_serializer: Any = None,
        value_serializer: Any = None,
        callback: Optional[MessageSentCallback] = None,
        reply_to: Optional[ReplyToArg] = None,
        correlation_id: Optional[str] = None,
        force: bool = False,
    ) -> Any:
        """Send message to topic used by agent."""
        if reply_to:
            value, headers = self._create_req(key, value, reply_to, correlation_id, headers)
        return await self.channel.send(key=key, value=value, partition=partition, timestamp=timestamp, headers=headers, key_serializer=key_serializer, value_serializer=value_serializer, force=force)

    def _get_strtopic(self, topic: Any) -> str:
        if isinstance(topic, Agent):
            return self._get_strtopic(topic.channel)
        if isinstance(topic, TopicT):
            return topic.get_topic_name()
        if isinstance(topic, ChannelT):
            raise ValueError('Channels are unnamed topics')
        return topic

    async def map(self, values: AsyncIterable[Any], key: Any = None, reply_to: Optional[ReplyToArg] = None) -> AsyncIterator[Any]:
        """RPC map operation on a list of values.

        A map operation iterates over results as they arrive.
        See :meth:`join` and :meth:`kvjoin` if you want them in order.
        """
        async for value in self.kvmap(((key, v) async for v in aiter(values)), reply_to):
            yield value

    async def kvmap(self, items: AsyncIterable[Tuple[Any, Any]], reply_to: Optional[ReplyToArg] = None) -> AsyncIterator[Any]:
        """RPC map operation on a list of ``(key, value)`` pairs.

        A map operation iterates over results as they arrive.
        See :meth:`join` and :meth:`kvjoin` if you want them in order.
        """
        reply_to_str = self._get_strtopic(reply_to or self.app.conf.reply_to)
        barrier = BarrierState(reply_to_str)
        async for _ in self._barrier_send(barrier, items, reply_to_str):
            try:
                _, val = barrier.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                yield val
        barrier.finalize()
        async for _, value in barrier.iterate():
            yield value

    async def join(self, values: AsyncIterable[Any], key: Any = None, reply_to: Optional[ReplyToArg] = None) -> Any:
        """RPC map operation on a list of values.

        A join returns the results in order, and only returns once
        all values have been processed.
        """
        return await self.kvjoin(((key, value) async for value in aiter(values)), reply_to=reply_to)

    async def kvjoin(self, items: AsyncIterable[Tuple[Any, Any]], reply_to: Optional[ReplyToArg] = None) -> List[Any]:
        """RPC map operation on list of ``(key, value)`` pairs.

        A join returns the results in order, and only returns once
        all values have been processed.
        """
        reply_to_str = self._get_strtopic(reply_to or self.app.conf.reply_to)
        barrier = BarrierState(reply_to_str)
        posindex: Dict[str, int] = {cid: i async for i, cid in aenumerate(self._barrier_send(barrier, items, reply_to_str))}
        barrier.finalize()
        await barrier
        values: List[Any] = [None] * barrier.total
        async for correlation_id, value in barrier.iterate():
            values[posindex[correlation_id]] = value
        return values

    async def _barrier_send(self, barrier: BarrierState, items: AsyncIterable[Tuple[Any, Any]], reply_to: str) -> AsyncIterator[str]:
        async for key, value in aiter(items):
            correlation_id = str(uuid4())
            p = await self.ask_nowait(key=key, value=value, reply_to=reply_to, correlation_id=correlation_id)
            barrier.add(p)
            app = cast(_App, self.app)
            await app.maybe_start_client()
            await app._reply_consumer.add(p.correlation_id, barrier)
            yield correlation_id

    def _repr_info(self) -> str:
        return shorten_fqdn(self.name)

    def get_topic_names(self) -> List[str]:
        """Return list of topic names this agent subscribes to."""
        channel = self.channel
        if isinstance(channel, TopicT):
            return channel.topics
        return []

    @property
    def channel(self) -> ChannelT:
        """Return channel used by agent."""
        if self._channel is None:
            self._channel = self._prepare_channel(self._channel_arg, schema=self._schema, key_type=self._key_type, value_type=self._value_type, **self._channel_kwargs)
        return self._channel

    @channel.setter
    def channel(self, channel: ChannelT) -> None:
        self._channel = channel

    @property
    def channel_iterator(self) -> ChannelT:
        """Return channel agent iterates over."""
        if self._channel_iterator is None:
            self._channel_iterator = self.channel.clone(is_iterator=False)
        return self._channel_iterator

    @channel_iterator.setter
    def channel_iterator(self, it: ChannelT) -> None:
        self._channel_iterator = it

    @property
    def label(self) -> str:
        """Return human-readable description of agent."""
        return self._agent_label()

    def _agent_label(self, name_suffix: str = '') -> str:
        s = f'{type(self).__name__}{name_suffix}: '
        s += f'{shorten_fqdn(qualname(self.fun))}'
        return s

    @property
    def shortlabel(self) -> str:
        """Return short description of agent."""
        return self._agent_label()


class AgentTestWrapper(Agent, AgentTestWrapperT):
    def __init__(self, *args: Any, original_channel: Optional[ChannelT] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.results: Dict[int, Any] = {}
        self.new_value_processed: asyncio.Condition = asyncio.Condition(loop=self.loop)
        self.original_channel: ChannelT = cast(ChannelT, original_channel)
        self.add_sink(self._on_value_processed)
        self._stream: StreamT = self.channel.stream()
        self.sent_offset: int = 0
        self.processed_offset: int = 0

    async def on_stop(self) -> None:
        await self._stream.stop()
        await super().on_stop()

    def stream(self, *args: Any, **kwargs: Any) -> StreamT:
        return self._stream.get_active_stream()

    async def _on_value_processed(self, value: Any) -> None:
        async with self.new_value_processed:
            self.results[self.processed_offset] = value
            self.processed_offset += 1
            self.new_value_processed.notify_all()

    async def crash_test_agent(self, exc: Exception) -> None:
        self._crash(exc)
        async with self.new_value_processed:
            self.new_value_processed.notify_all()

    async def put(self, value: Any = None, key: Any = None, partition: Any = None, timestamp: Any = None, headers: Optional[HeadersArg] = None, key_serializer: Any = None, value_serializer: Any = None, *, reply_to: Optional[ReplyToArg] = None, correlation_id: Optional[str] = None, wait: bool = True) -> Message:
        if reply_to:
            value, headers = self._create_req(key, value, reply_to, correlation_id, headers)
        channel: ChannelT = cast(ChannelT, self.stream().channel)
        message: Message = self.to_message(key, value, partition=partition, offset=self.sent_offset, timestamp=timestamp, headers=headers)
        event = await channel.decode(message)
        await channel.put(event)
        self.sent_offset += 1
        if wait:
            async with self.new_value_processed:
                await self.new_value_processed.wait()
                if self._crash_reason:
                    raise self._crash_reason from self._crash_reason
        return event

    def to_message(self, key: Any, value: Any, *, partition: Optional[int] = None, offset: int = 0, timestamp: Optional[float] = None, timestamp_type: int = 0, headers: Optional[HeadersArg] = None) -> Message:
        try:
            topic_name: str = self._get_strtopic(self.original_channel)
        except ValueError:
            topic_name = '<internal>'
        return Message(topic=topic_name, partition=partition or 0, offset=offset, timestamp=timestamp or time(), timestamp_type=timestamp_type, headers=headers, key=key, value=value, checksum=b'', serialized_key_size=0, serialized_value_size=0)

    async def throw(self, exc: Exception) -> None:
        await self.stream().throw(exc)

    def __aiter__(self) -> AsyncIterator[Any]:
        return aiter(self._stream)
