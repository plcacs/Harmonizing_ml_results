from typing import List, Union
from uuid import UUID
import pytest
from prefect import flow, task
from prefect.concurrency.v1.sync import concurrency

def test_concurrency_orchestrates_api(concurrency_limit: ConcurrencyLimit) -> None:
    ...

def test_concurrency_emits_events(v1_concurrency_limit: ConcurrencyLimit, other_v1_concurrency_limit: ConcurrencyLimit, asserting_events_worker: EventsWorker, mock_should_emit_events: bool, reset_worker_events: bool) -> None:
    ...

def test_concurrency_can_be_used_within_a_flow(concurrency_limit: ConcurrencyLimit) -> None:
    ...

async def test_concurrency_can_be_used_while_event_loop_is_running(concurrency_limit: ConcurrencyLimit) -> None:
    ...

@pytest.fixture
def mock_increment_concurrency_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

def test_concurrency_respects_timeout() -> None:
    ...

def test_concurrency_without_limit_names_sync(names: Union[List[str], None]) -> None:
    ...
