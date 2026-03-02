import asyncio
import collections.abc
import pytest
from faust.agents import Agent
from faust.agents.actor import Actor, AsyncIterableActor, AwaitableActor
from faust.types import TP
from mode.utils.mocks import AsyncMock, Mock

class FakeActor(Actor):
    """Fake actor class for testing purposes."""
    def traceback(self) -> str:
        """Returns an empty string."""
        return ''

class TestActor:
    """Test class for Actor."""
    ActorType: type[Actor] = FakeActor

    @pytest.fixture()
    def agent(self) -> Mock:
        """Agent fixture."""
        agent = Mock(name='agent', autospec=Agent)
        agent.name = 'myagent'
        return agent

    @pytest.fixture()
    def stream(self) -> Mock:
        """Stream fixture."""
        stream = Mock(name='stream')
        stream.active_partitions = None
        return stream

    @pytest.fixture()
    def it(self) -> Mock:
        """Iterator fixture."""
        it = Mock(name='it', autospec=collections.abc.Iterator)
        it.__aiter__ = Mock(name='it.__aiter__')
        it.__await__ = Mock(name='it.__await__')
        return it

    @pytest.fixture()
    def actor(self, agent: Mock, stream: Mock, it: Mock) -> Actor:
        """Actor fixture."""
        return self.ActorType(agent, stream, it)

    def test_constructor(self, actor: Actor, agent: Mock, stream: Mock, it: Mock) -> None:
        """Tests the constructor of Actor."""
        assert actor.agent is agent
        assert actor.stream is stream
        assert actor.it is it
        assert actor.index is None
        assert actor.active_partitions is None
        assert actor.actor_task is None

    @pytest.mark.asyncio
    async def test_on_start(self, actor: Actor) -> None:
        """Tests the on_start method of Actor."""
        actor.actor_task = Mock(name='actor_task', autospec=asyncio.Task)
        actor.add_future = Mock(name='add_future')
        await actor.on_start()
        actor.add_future.assert_called_once_with(actor.actor_task)

    @pytest.mark.asyncio
    async def test_on_stop(self, actor: Actor) -> None:
        """Tests the on_stop method of Actor."""
        actor.cancel = Mock(name='cancel')
        await actor.on_stop()
        actor.cancel.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_isolated_partition_revoked(self, actor: Actor) -> None:
        """Tests the on_isolated_partition_revoked method of Actor."""
        actor.cancel = Mock(name='cancel')
        actor.stop = AsyncMock(name='stop')
        await actor.on_isolated_partition_revoked(TP('foo', 0))
        actor.cancel.assert_called_once_with()
        actor.stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_isolated_partition_assigned(self, actor: Actor) -> None:
        """Tests the on_isolated_partition_assigned method of Actor."""
        await actor.on_isolated_partition_assigned(TP('foo', 0))

    def test_cancel(self, actor: Actor) -> None:
        """Tests the cancel method of Actor."""
        actor.actor_task = Mock(name='actor_task', autospec=asyncio.Task)
        actor.cancel()
        actor.stream.channel._throw.assert_called_once()
        actor.stream.channel._throw.call_args[0][0] == StopAsyncIteration()

    def test_cancel__when_no_task(self, actor: Actor) -> None:
        """Tests the cancel method of Actor when no task is present."""
        actor.actor_task = None
        actor.cancel()

    def test_repr(self, actor: Actor) -> None:
        """Tests the representation of Actor."""
        assert repr(actor)

class TestAsyncIterableActor(TestActor):
    """Test class for AsyncIterableActor."""
    ActorType: type[AsyncIterableActor] = AsyncIterableActor

    def test_aiter(self, actor: AsyncIterableActor, it: Mock) -> None:
        """Tests the __aiter__ method of AsyncIterableActor."""
        res = actor.__aiter__()
        it.__aiter__.assert_called_with()
        assert res is it.__aiter__()

class TestAwaitableActor(TestActor):
    """Test class for AwaitableActor."""
    ActorType: type[AwaitableActor] = AwaitableActor

    def test_await(self, actor: AwaitableActor, it: Mock) -> None:
        """Tests the __await__ method of AwaitableActor."""
        res = actor.__await__()
        it.__await__.assert_called_with()
        assert res is it.__await__()
