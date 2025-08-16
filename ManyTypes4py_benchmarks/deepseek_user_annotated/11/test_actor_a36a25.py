import asyncio
import collections.abc
from typing import Any, Iterator, Optional, cast
import pytest
from faust.agents import Agent
from faust.agents.actor import Actor, AsyncIterableActor, AwaitableActor
from faust.types import TP
from mode.utils.mocks import AsyncMock, Mock
from unittest.mock import Mock as MockType


class FakeActor(Actor):
    # need to implement traceback method to test this.

    def traceback(self) -> str:
        return ''


class test_Actor:

    ActorType = FakeActor

    @pytest.fixture()
    def agent(self) -> MockType:
        agent = Mock(name='agent', autospec=Agent)
        agent.name = 'myagent'
        return agent

    @pytest.fixture()
    def stream(self) -> MockType:
        stream = Mock(name='stream')
        stream.active_partitions = None
        return stream

    @pytest.fixture()
    def it(self) -> MockType:
        it = Mock(name='it', autospec=collections.abc.Iterator)
        it.__aiter__ = Mock(name='it.__aiter__')
        it.__await__ = Mock(name='it.__await__')
        return it

    @pytest.fixture()
    def actor(self, agent: MockType, stream: MockType, it: MockType) -> FakeActor:
        return self.ActorType(agent, stream, it)

    def test_constructor(self, *, actor: FakeActor, agent: MockType, stream: MockType, it: MockType) -> None:
        assert actor.agent is agent
        assert actor.stream is stream
        assert actor.it is it
        assert actor.index is None
        assert actor.active_partitions is None
        assert actor.actor_task is None

    @pytest.mark.asyncio
    async def test_on_start(self, *, actor: FakeActor) -> None:
        actor.actor_task = Mock(name='actor_task', autospec=asyncio.Task)
        actor.add_future = Mock(name='add_future')
        await actor.on_start()
        actor.add_future.assert_called_once_with(actor.actor_task)

    @pytest.mark.asyncio
    async def test_on_stop(self, *, actor: FakeActor) -> None:
        actor.cancel = Mock(name='cancel')
        await actor.on_stop()
        actor.cancel.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_isolated_partition_revoked(self, *, actor: FakeActor, tp: TP) -> None:
        actor.cancel = Mock(name='cancel')
        actor.stop = AsyncMock(name='stop')
        await actor.on_isolated_partition_revoked(TP('foo', 0))
        actor.cancel.assert_called_once_with()
        actor.stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_isolated_partition_assigned(self, *, actor: FakeActor, tp: TP) -> None:
        await actor.on_isolated_partition_assigned(TP('foo', 0))

    def test_cancel(self, *, actor: FakeActor) -> None:
        actor.actor_task = Mock(name='actor_task', autospec=asyncio.Task)
        actor.cancel()
        actor.stream.channel._throw.assert_called_once()
        actor.stream.channel._throw.call_args[0][0] == StopAsyncIteration()

    def test_cancel__when_no_task(self, *, actor: FakeActor) -> None:
        actor.actor_task = None
        actor.cancel()

    def test_repr(self, *, actor: FakeActor) -> None:
        assert repr(actor)


class test_AsyncIterableActor(test_Actor):

    ActorType = AsyncIterableActor

    def test_aiter(self, *, actor: AsyncIterableActor, it: MockType) -> None:
        res = actor.__aiter__()
        it.__aiter__.assert_called_with()
        assert res is it.__aiter__()


class test_AwaitableActor(test_Actor):

    ActorType = AwaitableActor

    def test_await(self, *, actor: AwaitableActor, it: MockType) -> None:
        res = actor.__await__()
        it.__await__.assert_called_with()
        assert res is it.__await__()
