from typing import Any, Optional

class FakeActor(Actor):
    def traceback(self) -> str:
        return ''

    def __init__(self, agent: Agent, stream: Any, it: Any) -> None:
        ...

    def on_start(self) -> Any:
        ...

    def on_stop(self) -> Any:
        ...

    async def on_isolated_partition_revoked(self, tp: TP) -> Any:
        ...

    async def on_isolated_partition_assigned(self, tp: TP) -> Any:
        ...

    def cancel(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

class test_Actor:
    ActorType: Any = FakeActor

    def __init__(self) -> None:
        ...

    def test_constructor(self, *, actor: FakeActor, agent: Agent, stream: Any, it: Any) -> None:
        ...

    async def test_on_start(self, *, actor: FakeActor) -> Any:
        ...

    async def test_on_stop(self, *, actor: FakeActor) -> Any:
        ...

    async def test_on_isolated_partition_revoked(self, *, actor: FakeActor) -> Any:
        ...

    async def test_on_isolated_partition_assigned(self, *, actor: FakeActor) -> Any:
        ...

    def test_cancel(self, *, actor: FakeActor) -> None:
        ...

    def test_cancel__when_no_task(self, *, actor: FakeActor) -> None:
        ...

    def test_repr(self, *, actor: FakeActor) -> str:
        ...

class test_AsyncIterableActor(test_Actor):
    ActorType: Any = AsyncIterableActor

    def test_aiter(self, *, actor: AsyncIterableActor, it: Any) -> None:
        ...

class test_AwaitableActor(test_Actor):
    ActorType: Any = AwaitableActor

    def test_await(self, *, actor: AwaitableActor, it: Any) -> None:
        ...
