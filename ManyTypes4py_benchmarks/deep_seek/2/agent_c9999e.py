"""Agent implementation."""
import asyncio
import typing
from contextlib import suppress
from contextvars import ContextVar
from time import time
from typing import (
    Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Dict, Iterable, 
    List, Mapping, MutableMapping, MutableSet, Optional, Set, Tuple, Type, 
    Union, cast, TypeVar, Generic, Coroutine
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
from faust.types import (
    AppT, ChannelT, CodecArg, EventT, HeadersArg, K, Message, MessageSentCallback, 
    ModelArg, ModelT, RecordMetadata, StreamT, TP, TopicT, V
)
from faust.types.agents import (
    ActorRefT, ActorT, AgentErrorHandler, AgentFun, AgentT, AgentTestWrapperT, 
    ReplyToArg, SinkT
)
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

_current_agent: ContextVar[Optional['Agent']] = ContextVar('current_agent')

def current_agent() -> Optional['Agent']:
    return _current_agent.get(None)

class Agent(AgentT, Service):
    """Agent.

    This is the type of object returned by the ``@app.agent`` decorator.
    """
    supervisor: SupervisorStrategyT = cast(SupervisorStrategyT, None)
    _channel: Optional[ChannelT] = None
    _channel_iterator: Optional[ChannelT] = None
    _pending_active_partitions: Optional[Set[TP]] = None
    _first_assignment_done: bool = False

    def __init__(
        self,
        fun: AgentFun,
        *,
        app: AppT,
        name: Optional[str] = None,
        channel: Union[str, ChannelT, None] = None,
        concurrency: int = 1,
        sink: Optional[Iterable[SinkT]] = None,
        on_error: Optional[AgentErrorHandler] = None,
        supervisor_strategy: Optional[Type[SupervisorStrategyT]] = None,
        help: Optional[str] = None,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        isolated_partitions: bool = False,
        use_reply_headers: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        self.app: AppT = app
        self.fun: AgentFun = fun
        self.name: str = name or canonshortname(self.fun)
        if schema is not None:
            assert channel is None or isinstance(channel, str)
        if key_type is not None:
            assert channel is None or isinstance(channel, str)
        self._key_type: Optional[ModelArg] = key_type
        if value_type is not None:
            assert channel is None or isinstance(channel, str)
        self._schema: Optional[SchemaT] = schema
        self._value_type: Optional[ModelArg] = value_type
        self._channel_arg: Union[str, ChannelT, None] = channel
        self._channel_kwargs: Dict[str, Any] = kwargs
        self.concurrency: int = concurrency or 1
        self.isolated_partitions: bool = isolated_partitions
        self.help: str = help or ''
        self._sinks: List[SinkT] = list(sink) if sink is not None else []
        self._on_error: Optional[AgentErrorHandler] = on_error
        self.supervisor_strategy: Optional[Type[SupervisorStrategyT]] = supervisor_strategy
        self._actors: WeakSet[ActorRefT] = WeakSet()
        self._actor_by_partition: WeakValueDictionary[TP, ActorRefT] = WeakValueDictionary()
        if self.isolated_partitions and self.concurrency > 1:
            raise ImproperlyConfigured('Agent concurrency must be 1 when using isolated partitions')
        self.use_reply_headers: Optional[bool] = use_reply_headers
        Service.__init__(self)

    def on_init_dependencies(self) -> List[ServiceT]:
        """Return list of services dependencies required to start agent."""
        self.beacon.reattach(self.app.agents.beacon)
        return []

    def actor_tracebacks(self) -> List[Optional[str]]:
        return [actor.traceback() for actor in self._actors]

    async def _start_one(
        self,
        *,
        index: Optional[int] = None,
        active_partitions: Optional[Set[TP]] = None,
        stream: Optional[StreamT] = None,
        channel: Optional[ChannelT] = None
    ) -> ActorRefT:
        index = index if self.concurrency > 1 else None
        return await self._start_task(
            index=index,
            active_partitions=active_partitions,
            stream=stream,
            channel=channel,
            beacon=self.beacon
        )

    async def _start_one_supervised(
        self,
        index: Optional[int] = None,
        active_partitions: Optional[Set[TP]] = None,
        stream: Optional[StreamT] = None
    ) -> ActorRefT:
        aref: ActorRefT = await self._start_one(
            index=index,
            active_partitions=active_partitions,
            stream=stream
        )
        self.supervisor.add(aref)
        await aref.maybe_start()
        return aref

    async def _start_for_partitions(self, active_partitions: Set[TP]) -> ActorRefT:
        assert active_partitions
        self.log.info('Starting actor for partitions %s', active_partitions)
        return await self._start_one_supervised(None, active_partitions)

    async def on_start(self) -> None:
        """Call when an agent starts."""
        self.supervisor = self._new_supervisor()
        await self._on_start_supervisor()

    def _new_supervisor(self) -> SupervisorStrategyT:
        return self._get_supervisor_strategy()(
            max_restarts=100.0,
            over=1.0,
            replacement=self._replace_actor,
            loop=self.loop,
            beacon=self.beacon
        )

    async def _replace_actor(self, service: ServiceT, index: Optional[int]) -> ActorRefT:
        aref = cast(ActorRefT, service)
        return await self._start_one(
            index=index,
            active_partitions=aref.active_partitions,
            stream=aref.stream,
            channel=cast(ChannelT, aref.stream.channel)
        )

    def _get_supervisor_strategy(self) -> Type[SupervisorStrategyT]:
        SupervisorStrategy = self.supervisor_strategy
        if SupervisorStrategy is None:
            return cast(Type[SupervisorStrategyT], self.app.conf.agent_supervisor)
        else:
            return SupervisorStrategy

    async def _on_start_supervisor(self) -> None:
        active_partitions: Optional[Set[TP]] = self._get_active_partitions()
        channel: Optional[ChannelT] = None
        for i in range(self.concurrency):
            res: ActorRefT = await self._start_one(
                index=i,
                active_partitions=active_partitions,
                channel=channel
            )
            if channel is None:
                channel = res.stream.channel
            self.supervisor.add(res)
        await self.supervisor.start()

    def _get_active_partitions(self) -> Optional[Set[TP]]:
        active_partitions: Optional[Set[TP]] = None
        if self.isolated_partitions:
            active_partitions = self._pending_active_partitions = set()
        return active_partitions

    async def on_stop(self) -> None:
        """Call when an agent stops."""
        await self._stop_supervisor()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*[
                aref.actor_task 
                for aref in self._actors 
                if aref.actor_task is not None
            ])
        self._actors.clear()

    async def _stop_supervisor(self) -> None:
        if self.supervisor:
            await self.supervisor.stop()
            self.supervisor = cast(SupervisorStrategyT, None)

    def cancel(self) -> None:
        """Cancel agent and its actor instances running in this process."""
        for aref in self._actors:
            aref.cancel()

    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        """Call when partitions are revoked."""
        T = traced_from_parent_span()
        if self.isolated_partitions:
            await T(self.on_isolated_partitions_revoked)(revoked)
        else:
            await T(self.on_shared_partitions_revoked)(revoked)

    async def on_partitions_assigned(self, assigned: Set[TP]) -> None:
        """Call when partitions are assigned."""
        T = traced_from_parent_span()
        if self.isolated_partitions:
            await T(self.on_isolated_partitions_assigned)(assigned)
        else:
            await T(self.on_shared_partitions_assigned)(assigned)

    async def on_isolated_partitions_revoked(self, revoked: Set[TP]) -> None:
        """Call when isolated partitions are revoked."""
        self.log.dev('Partitions revoked')
        T = traced_from_parent_span()
        for tp in revoked:
            aref = self._actor_by_partition.pop(tp, None)
            if aref is not None:
                await T(aref.on_isolated_partition_revoked)(tp)

    async def on_isolated_partitions_assigned(self, assigned: Set[TP]) -> None:
        """Call when isolated partitions are assigned."""
        T = traced_from_parent_span()
        for tp in sorted(assigned):
            await T(self._assign_isolated_partition)(tp)

    async def _assign_isolated_partition(self, tp: TP) -> None:
        T = traced_from_parent_span()
        if not self._first_assignment_done and (not self._actor_by_partition):
            self._first_assignment_done = True
            T(self._on_first_isolated_partition_assigned)(tp)
        await T(self._maybe_start_isolated)(tp)

    def _on_first_isolated_partition_assigned(self, tp: TP) -> None:
        assert self._actors
        assert len(self._actors) == 1
        self._actor_by_partition[tp] = next(iter(self._actors))
        if self._pending_active_partitions is not None:
            assert not self._pending_active_partitions
            self._pending_active_partitions.add(tp)

    async def _maybe_start_isolated(self, tp: TP) -> None:
        try:
            aref = self._actor_by_partition[tp]
        except KeyError:
            aref = await self._start_isolated(tp)
            self._actor_by_partition[tp] = aref
        await aref.on_isolated_partition_assigned(tp)

    async def _start_isolated(self, tp: TP) -> ActorRefT:
        return await self._start_for_partitions({tp})

    async def on_shared_partitions_revoked(self, revoked: Set[TP]) -> None:
        """Call when non-isolated partitions are revoked."""
        ...

    async def on_shared_partitions_assigned(self, assigned: Set[TP]) -> None:
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
            'isolated_partitions': self.isolated_partitions
        }

    def clone(self, *, cls: Optional[Type['Agent']] = None, **kwargs: Any) -> 'Agent':
        """Create clone of this agent object.

        Keyword arguments can be passed to override any argument
        supported by :class:`Agent.__init__ <Agent>`.
        """
        return (cls or type(self))(**{**self.info(), **kwargs})

    def test_context(
        self,
        channel: Optional[ChannelT] = None,
        supervisor_strategy: Optional[Type[SupervisorStrategyT]] = None,
        on_error: Optional[AgentErrorHandler] = None,
        **kwargs: Any
    ) -> AgentTestWrapperT:
        """Create new unit-testing wrapper for this agent."""
        self.app.flow_control.resume()

        async def on_agent_error(agent: 'Agent', exc: Exception) -> None:
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
                **kwargs
            )
        )

    def _prepare_channel(
        self,
        channel: Union[str, ChannelT, None] = None,
        internal: bool = True,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        **kwargs: Any
    ) -> ChannelT:
        app = self.app
        has_prefix = False
        if channel is None:
            channel = f'{app.conf.id}-{self.name}'
            has_prefix = True
        if isinstance(channel, ChannelT):
            return channel
        elif isinstance(channel, str):
            return app.topic(
                channel,
                internal=internal,
                schema=schema,
                key_type=key_type,
                value_type=value_type,
                has_prefix=has_prefix,
                **kwargs
            )
        raise TypeError(f'Channel must be channel, topic, or str; not {type(channel)}')

    def __call__(
        self,
        *,
        index: Optional[int] = None,
        active_partitions: Optional[Set[TP]] = None,
        stream: Optional[StreamT] = None,
        channel: Optional[ChannelT] = None
    ) -> ActorRefT:
        """Create new actor instance for this agent."""
        return self.actor_from_stream(
            stream,
            index=index,
            active_partitions=active_partitions,
            channel=channel
        )

    def actor_from_stream(
        self,
        stream: Optional[StreamT],
        *,
        index: Optional[int] = None,
        active_partitions: Optional[Set[TP]] = None,
        channel: Optional[ChannelT] = None
    ) -> ActorRefT:
        """Create new actor from stream."""
        we_created_stream = False
        if stream is None:
            actual_stream = self.stream(
                channel=channel,
                concurrency_index=index,
                active_partitions=active_partitions
            )
            we_created_stream = True
        else:
            assert stream.concurrency_index == index
            assert stream.active_partitions == active_partitions
            actual_stream = stream
        res = self.fun(actual_stream)
        if isinstance(res, AsyncIterable):
            if we_created_stream:
                actual_stream.add_processor(self._maybe_unwrap_reply_request)
            return cast(
                ActorRefT,
                AsyncIterableActor(
                    self,
                    actual_stream,
                    res,
                    index=actual_stream.concurrency_index,
                    active_partitions=actual_stream.active_partitions,
                    loop=self.loop,
                    beacon=self.beacon
                )
            )
        else:
            return cast(
                ActorRefT,
                AwaitableActor(
                    self,
                    actual_stream,
                    res,
                    index=actual_stream.concurrency_index,
                    active_partitions=actual_stream.active_partitions,
                    loop=self.loop,
                    beacon=self.beacon
                )
            )

    def add_sink(self, sink: SinkT) -> None:
        """Add new sink to further handle results from this agent."""
        if sink not in self._sinks:
            self._sinks.append(sink)

    def stream(
        self,
        channel: Optional[ChannelT] = None,
        active_partitions: Optional[Set[TP]] = None,
        **kwargs: Any
    ) -> StreamT:
        """Create underlying stream used by this agent."""
        if channel is None:
            channel = cast(TopicT, self.channel_iterator).clone(
                is_iterator=False,
                active_partitions=active_partitions
            )
        if active_partitions is not None:
            assert channel.active_partitions == active_partitions
        s = self.app.stream(
            channel,
            loop=self.loop,
            active_partitions=active_partitions,
            prefix=self.name,
            beacon=self.beacon,
            **kwargs
        )
        return s

    def _maybe_unwrap_reply_request(self, value: Any) -> Any:
        if isinstance(value, ReqRepRequest):
            return value.value
        return value

    async def _start_task(
        self,
        *,
        index: Optional[int],
        active_partitions: Optional[Set[TP]] = None,
        stream: Optional[StreamT] = None