import asyncio
import collections.abc
from typing import Optional, Iterator, Any, Type, TypeVar

import pytest
from faust.agents import Agent
from faust.agents.actor import Actor, AsyncIterableActor, AwaitableActor
from faust.types import TP
from mode.utils.mocks import AsyncMock, Mock


T = TypeVar('T', bound=Actor)


class FakeActor(Actor):

    def traceback(self) -> str:
        return ''


class test_Actor:
    ActorType: Type[T] = FakeActor

    @pytest.fixture
    def agent(self) -> Mock:
        agent = Mock(name='agent', autospec=Agent)
        agent.name = 'myagent'
        return agent

    @pytest.fixture
    def stream(self) -> Mock:
        stream = Mock(name='stream')
        stream.active_partitions = None
        return stream

    @pytest.fixture
    def it(self) -> Mock:
        it = Mock(name='it', autospec=collections.abc.Iterator[Any])
        it.__aiter__ = Mock(name='it.__aiter__')
        it.__await__ = Mock(name='it.__await__')
        return it

    @pytest.fixture
    def actor(self, *, agent: Mock, stream: Mock, it: Mock) -> Actor:
        return self.ActorType(agent, stream, it)

    def test_constructor(self, *, actor: Actor, agent: Mock, stream: Mock, it: Mock) -> None:
        assert actor.agent is agent
        assert actor.stream is stream
        assert actor.it is it
        assert actor.index is None
        assert actor.active_partitions is None
        assert actor.actor_task is None

    @pytest.mark.asyncio
    async def test_on_start(self, *, actor: Actor) -> None:
        actor.actor_task = Mock(name='actor_task', autospec=asyncio.Task)
        actor.add_future = Mock(name='add_future')
        await actor.on_start()
        actor.add_future.assert_called_once_with(actor.actor_task)

    @pytest.mark.asyncio
    async def test_on_stop(self, *, actor: Actor) -> None:
        actor.cancel = Mock(name='cancel')
        await actor.on_stop()
        actor.cancel.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_isolated_partition_revoked(self, *, actor: Actor) -> None:
        actor.cancel = Mock(name='cancel')
        actor.stop = AsyncMock(name='stop')
        await actor.on_isolated_partition_revoked(TP('foo', 0))
        actor.cancel.assert_called_once_with()
        actor.stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_isolated_partition_assigned(self, *, actor: Actor) -> None:
        await actor.on_isolated_partition_assigned(TP('foo', 0))

    def test_cancel(self, *, actor: Actor) -> None:
        actor.actor_task = Mock(name='actor_task', autospec=asyncio.Task)
        actor.cancel()
        actor.stream.channel._throw.assert_called_once()
        assert actor.stream.channel._throw.call_args[0][0] == StopAsyncIteration()

    def test_cancel__when_no_task(self, *, actor: Actor) -> None:
        actor.actor_task = None
        actor.cancel()

    def test_repr(self, *, actor: Actor) -> None:
        assert repr(actor)


class test_AsyncIterableActor(test_Actor):
    ActorType: Type[AsyncIterableActor] = AsyncIterableActor

    def test_aiter(self, *, actor: AsyncIterableActor, it: Mock) -> None:
        res = actor.__aiter__()
        it.__aiter__.assert_called_with()
        assert res is it.__aiter__()


class test_AwaitableActor(test_Actor):
    ActorType: Type[AwaitableActor] = AwaitableActor

    def test_await(self, *, actor: AwaitableActor, it: Mock) -> None:
        res = actor.__await__()
        it.__await__.assert_called_with()
        assert res is it.__await__()
