#!/usr/bin/env python3
from typing import Any, Callable, List, Optional, Union
from unittest import mock

import pytest
from httpx import HTTPStatusError, Request, Response
from starlette import status
from prefect import flow, task
from prefect.concurrency.asyncio import ConcurrencySlotAcquisitionError
from prefect.concurrency.sync import (
    _acquire_concurrency_slots,
    _release_concurrency_slots,
    concurrency,
    rate_limit,
)
from prefect.events.clients import AssertingEventsClient
from prefect.events.worker import EventsWorker
from prefect.server.schemas.core import ConcurrencyLimitV2


def test_concurrency_orchestrates_api(concurrency_limit: ConcurrencyLimitV2) -> None:
    executed: bool = False

    def resource_heavy() -> None:
        nonlocal executed
        with concurrency(concurrency_limit.name, occupy=1):
            executed = True

    assert not executed
    with mock.patch(
        'prefect.concurrency.sync._acquire_concurrency_slots',
        wraps=_acquire_concurrency_slots,
    ) as acquire_spy:
        with mock.patch(
            'prefect.concurrency.sync._release_concurrency_slots',
            wraps=_release_concurrency_slots,
        ) as release_spy:
            resource_heavy()
            acquire_spy.assert_called_once_with(
                [concurrency_limit.name],
                1,
                timeout_seconds=None,
                create_if_missing=None,
                max_retries=None,
                strict=False,
            )
            names, occupy, occupy_seconds = release_spy.call_args[0]
            assert names == [concurrency_limit.name]
            assert occupy == 1
            assert occupy_seconds > 0
    assert executed


def test_concurrency_emits_events(
    concurrency_limit: ConcurrencyLimitV2,
    other_concurrency_limit: ConcurrencyLimitV2,
    asserting_events_worker: EventsWorker,
    mock_should_emit_events: Any,
    reset_worker_events: Any,
) -> None:
    def resource_heavy() -> None:
        with concurrency([concurrency_limit.name, other_concurrency_limit.name], occupy=1):
            pass

    resource_heavy()
    asserting_events_worker.drain()
    assert isinstance(asserting_events_worker._client, AssertingEventsClient)
    assert len(asserting_events_worker._client.events) == 4
    for phase in ['acquired', 'released']:
        event = next(
            filter(
                lambda e: e.event == f'prefect.concurrency-limit.{phase}' and e.resource.id == f'prefect.concurrency-limit.{concurrency_limit.id}',
                asserting_events_worker._client.events,
            )
        )
        assert dict(event.resource) == {
            'prefect.resource.id': f'prefect.concurrency-limit.{concurrency_limit.id}',
            'prefect.resource.name': concurrency_limit.name,
            'slots-acquired': '1',
            'limit': str(concurrency_limit.limit),
        }
        assert len(event.related) == 1
        assert dict(event.related[0]) == {
            'prefect.resource.id': f'prefect.concurrency-limit.{other_concurrency_limit.id}',
            'prefect.resource.role': 'concurrency-limit',
        }
    for phase in ['acquired', 'released']:
        event = next(
            filter(
                lambda e: e.event == f'prefect.concurrency-limit.{phase}' and e.resource.id == f'prefect.concurrency-limit.{other_concurrency_limit.id}',
                asserting_events_worker._client.events,
            )
        )
        assert dict(event.resource) == {
            'prefect.resource.id': f'prefect.concurrency-limit.{other_concurrency_limit.id}',
            'prefect.resource.name': other_concurrency_limit.name,
            'slots-acquired': '1',
            'limit': str(other_concurrency_limit.limit),
        }
        assert len(event.related) == 1
        assert dict(event.related[0]) == {
            'prefect.resource.id': f'prefect.concurrency-limit.{concurrency_limit.id}',
            'prefect.resource.role': 'concurrency-limit',
        }


def test_concurrency_can_be_used_within_a_flow(concurrency_limit: ConcurrencyLimitV2) -> None:
    executed: bool = False

    @task
    def resource_heavy() -> None:
        nonlocal executed
        with concurrency(concurrency_limit.name, occupy=1):
            executed = True

    @flow
    def my_flow() -> None:
        resource_heavy()

    assert not executed
    my_flow()
    assert executed


def test_concurrency_strict_within_a_flow() -> None:
    @task
    def resource_heavy() -> None:
        with concurrency('doesnt-exist-for-sure', occupy=1, strict=True):
            return

    @flow
    def my_flow() -> None:
        resource_heavy()

    state = my_flow(return_state=True)
    assert state.is_failed()
    with pytest.raises(ConcurrencySlotAcquisitionError):
        state.result()


@pytest.mark.parametrize('names', [[], None])
def test_rate_limit_without_limit_names_sync(names: Optional[List[Any]]) -> None:
    executed: bool = False

    def resource_heavy() -> None:
        nonlocal executed
        rate_limit(names=names, occupy=1)
        executed = True

    assert not executed
    with mock.patch(
        'prefect.concurrency.sync._acquire_concurrency_slots',
        wraps=lambda *args, **kwargs: None,
    ) as acquire_spy:
        with mock.patch(
            'prefect.concurrency.sync._release_concurrency_slots',
            wraps=lambda *args, **kwargs: None,
        ) as release_spy:
            resource_heavy()
            acquire_spy.assert_not_called()
            release_spy.assert_not_called()
    assert executed


@pytest.mark.asyncio
async def test_concurrency_can_be_used_while_event_loop_is_running(
    concurrency_limit: ConcurrencyLimitV2,
) -> None:
    executed: bool = False

    def resource_heavy() -> None:
        nonlocal executed
        with concurrency(concurrency_limit.name, occupy=1):
            executed = True

    assert not executed
    resource_heavy()
    assert executed


@pytest.fixture
def mock_increment_concurrency_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mocked_increment_concurrency_slots(*args: Any, **kwargs: Any) -> Any:
        response = Response(status_code=status.HTTP_423_LOCKED, headers={'Retry-After': '0.01'})
        raise HTTPStatusError(
            message='Locked',
            request=Request('GET', 'http://test.com'),
            response=response,
        )
    monkeypatch.setattr(
        'prefect.client.orchestration.PrefectClient.increment_concurrency_slots',
        mocked_increment_concurrency_slots,
    )


@pytest.mark.usefixtures('concurrency_limit', 'mock_increment_concurrency_slots')
def test_concurrency_respects_timeout() -> None:
    with pytest.raises(TimeoutError, match='.*timed out after 0.01 second(s)*.'):
        with concurrency('test', occupy=1, timeout_seconds=0.01):
            print('should not be executed')


def test_rate_limit_orchestrates_api(concurrency_limit_with_decay: ConcurrencyLimitV2) -> None:
    executed: bool = False

    def resource_heavy() -> None:
        nonlocal executed
        rate_limit(concurrency_limit_with_decay.name, 1)
        executed = True

    assert not executed
    with mock.patch(
        'prefect.concurrency.sync._acquire_concurrency_slots',
        wraps=_acquire_concurrency_slots,
    ) as acquire_spy:
        with mock.patch(
            'prefect.concurrency.sync._release_concurrency_slots',
            wraps=_release_concurrency_slots,
        ) as release_spy:
            resource_heavy()
            acquire_spy.assert_called_once_with(
                [concurrency_limit_with_decay.name],
                1,
                mode='rate_limit',
                timeout_seconds=None,
                create_if_missing=None,
                strict=False,
            )
            release_spy.assert_not_called()
    assert executed


def test_rate_limit_can_be_used_within_a_flow(concurrency_limit_with_decay: ConcurrencyLimitV2) -> None:
    executed: bool = False

    @task
    def resource_heavy() -> None:
        nonlocal executed
        rate_limit(concurrency_limit_with_decay.name, occupy=1)
        executed = True

    @flow
    def my_flow() -> None:
        resource_heavy()

    assert not executed
    my_flow()
    assert executed


def test_rate_limit_can_be_used_within_a_flow_with_strict() -> None:
    @task
    def resource_heavy() -> None:
        rate_limit(['definitely-doesnt-exist'], occupy=1, strict=True)

    @flow
    def my_flow() -> None:
        resource_heavy()

    state = my_flow(return_state=True)
    assert state.is_failed()
    with pytest.raises(ConcurrencySlotAcquisitionError):
        state.result()


def test_rate_limit_mixed_sync_async(concurrency_limit_with_decay: ConcurrencyLimitV2) -> None:
    executed: bool = False

    @task
    def resource_heavy() -> None:
        nonlocal executed
        rate_limit(concurrency_limit_with_decay.name, occupy=1)
        executed = True

    @flow
    def my_flow() -> None:
        resource_heavy()

    assert not executed
    my_flow()
    assert executed


def test_rate_limit_emits_events(
    concurrency_limit_with_decay: ConcurrencyLimitV2,
    other_concurrency_limit_with_decay: ConcurrencyLimitV2,
    asserting_events_worker: EventsWorker,
    mock_should_emit_events: Any,
    reset_worker_events: Any,
) -> None:
    def resource_heavy() -> None:
        rate_limit([concurrency_limit_with_decay.name, other_concurrency_limit_with_decay.name], occupy=1)

    resource_heavy()
    asserting_events_worker.drain()
    assert isinstance(asserting_events_worker._client, AssertingEventsClient)
    assert len(asserting_events_worker._client.events) == 2
    event = next(
        filter(
            lambda e: e.resource.id == f'prefect.concurrency-limit.{concurrency_limit_with_decay.id}',
            asserting_events_worker._client.events,
        )
    )
    assert event.event == 'prefect.concurrency-limit.acquired'
    assert dict(event.resource) == {
        'prefect.resource.id': f'prefect.concurrency-limit.{concurrency_limit_with_decay.id}',
        'prefect.resource.name': concurrency_limit_with_decay.name,
        'slots-acquired': '1',
        'limit': str(concurrency_limit_with_decay.limit),
    }
    assert len(event.related) == 1
    assert dict(event.related[0]) == {
        'prefect.resource.id': f'prefect.concurrency-limit.{other_concurrency_limit_with_decay.id}',
        'prefect.resource.role': 'concurrency-limit',
    }
    event = next(
        filter(
            lambda e: e.resource.id == f'prefect.concurrency-limit.{other_concurrency_limit_with_decay.id}',
            asserting_events_worker._client.events,
        )
    )
    assert event.event == 'prefect.concurrency-limit.acquired'
    assert dict(event.resource) == {
        'prefect.resource.id': f'prefect.concurrency-limit.{other_concurrency_limit_with_decay.id}',
        'prefect.resource.name': other_concurrency_limit_with_decay.name,
        'slots-acquired': '1',
        'limit': str(other_concurrency_limit_with_decay.limit),
    }
    assert len(event.related) == 1
    assert dict(event.related[0]) == {
        'prefect.resource.id': f'prefect.concurrency-limit.{concurrency_limit_with_decay.id}',
        'prefect.resource.role': 'concurrency-limit',
    }


@pytest.mark.parametrize('names', [[], None])
def test_concurrency_without_limit_names_sync(names: Optional[List[Any]]) -> None:
    executed: bool = False

    def resource_heavy() -> None:
        nonlocal executed
        with concurrency(names=names, occupy=1):
            executed = True

    assert not executed
    with mock.patch(
        'prefect.concurrency.sync._acquire_concurrency_slots',
        wraps=lambda *args, **kwargs: None,
    ) as acquire_spy:
        with mock.patch(
            'prefect.concurrency.sync._release_concurrency_slots',
            wraps=lambda *args, **kwargs: None,
        ) as release_spy:
            resource_heavy()
            acquire_spy.assert_not_called()
            release_spy.assert_not_called()
    assert executed
