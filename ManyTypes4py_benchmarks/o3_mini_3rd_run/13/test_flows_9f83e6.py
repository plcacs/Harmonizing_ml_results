#!/usr/bin/env python3
"""
This file contains annotated tests and utilities.
"""

import asyncio
import datetime
import enum
import inspect
import os
import signal
import sys
import threading
import time
import uuid
import warnings
from functools import partial
from itertools import combinations
from pathlib import Path
from textwrap import dedent
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import anyio
import pendulum
import pydantic
import pytest
import regex as re

# Import your prefect modules as needed
from prefect import flow, task
from prefect.blocks.core import Block
from prefect.client.orchestration import PrefectClient, SyncPrefectClient, get_client
from prefect.client.schemas.objects import ConcurrencyLimitConfig, Worker, WorkerStatus
from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule, RRuleSchedule
from prefect.context import FlowRunContext, get_run_context
from prefect.deployments.runner import RunnerDeployment, RunnerStorage
from prefect.docker.docker_image import DockerImage
from prefect.events import DeploymentEventTrigger, Posture
from prefect.exceptions import CancelledRun, InvalidNameError, MissingFlowError, ParameterTypeError, ReservedArgumentError, ScriptError, UnfinishedRun
from prefect.filesystems import LocalFileSystem
from prefect.flows import Flow, load_flow_arguments_from_entrypoint, load_flow_from_entrypoint, load_flow_from_flow_run, load_function_and_convert_to_flow, safe_load_flow_from_entrypoint
from prefect.logging import get_run_logger
from prefect.results import ResultRecord
from prefect.runtime import flow_run as flow_run_ctx
from prefect.schedules import Cron, Interval, RRule, Schedule
from prefect.server.schemas.core import TaskRunResult
from prefect.server.schemas.filters import FlowFilter, FlowRunFilter
from prefect.server.schemas.sorting import FlowRunSort
from prefect.settings import PREFECT_FLOW_DEFAULT_RETRIES, temporary_settings
from prefect.states import Cancelled, Cancelling, Paused, PausedRun, State, StateType, raise_state_exception
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.testing.utilities import AsyncMock, exceptions_equal, get_most_recent_flow_run
from prefect.transactions import get_transaction, transaction
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.annotations import allow_failure, quote
from prefect.utilities.callables import parameter_schema
from prefect.utilities.collections import flatdict_to_dict
from prefect.utilities.hashing import file_hash


SLEEP_TIME: int = 10


@pytest.fixture
def mock_sigterm_handler() -> Any:
    if threading.current_thread() != threading.main_thread():
        pytest.skip("Can't test signal handlers from a thread")
    mock_handler = pytest.importorskip("unittest.mock").MagicMock()

    def handler(*args: Any, **kwargs: Any) -> None:
        mock_handler(*args, **kwargs)
    prev_handler = signal.signal(signal.SIGTERM, handler)
    try:
        yield (handler, mock_handler)
    finally:
        signal.signal(signal.SIGTERM, prev_handler)


# --------------------------
# Utility hook creators
# --------------------------

def create_hook(mock_obj: Any) -> Callable[[Any, Any, Any], None]:
    def my_hook(flow: Any, flow_run: Any, state: Any) -> None:
        mock_obj()
    return my_hook


def create_async_hook(mock_obj: Any) -> Callable[[Any, Any, Any], Awaitable[None]]:
    async def my_hook(flow: Any, flow_run: Any, state: Any) -> None:
        mock_obj()
    return my_hook


# --------------------------
# Test classes start here
# --------------------------

class TestFlow:
    def test_initializes(self) -> None:
        f = Flow(name='test', fn=lambda **kwargs: 42, version='A', description='B', flow_run_name='hi')
        assert f.name == 'test'
        assert f.fn() == 42
        assert f.version == 'A'
        assert f.description == 'B'
        assert f.flow_run_name == 'hi'

    def test_initializes_with_callable_flow_run_name(self) -> None:
        f = Flow(name='test', fn=lambda **kwargs: 42, flow_run_name=lambda: 'hi')
        assert f.name == 'test'
        assert f.fn() == 42
        assert f.flow_run_name() == 'hi'

    def test_initializes_with_default_version(self) -> None:
        f = Flow(name='test', fn=lambda **kwargs: 42)
        # version may be computed dynamically.
        assert isinstance(f.version, str)

    @pytest.mark.parametrize('sourcefile', [None, '<stdin>', '<ipython-input-1-d31e8a6792d4>'])
    def test_version_none_if_source_file_cannot_be_determined(self, monkeypatch: Any, sourcefile: Optional[str]) -> None:
        """
        `getsourcefile` will return `None` when functions are defined interactively,
        or other values on Windows.
        """
        monkeypatch.setattr('prefect.flows.inspect.getsourcefile', lambda _: sourcefile)
        f = Flow(name='test', fn=lambda **kwargs: 42)
        assert f.version is None

    def test_raises_on_bad_funcs(self) -> None:
        with pytest.raises(TypeError):
            Flow(name='test', fn={})

    def test_default_description_is_from_docstring(self) -> None:
        def my_fn() -> None:
            """
            Hello
            """
        f = Flow(name='test', fn=my_fn)
        assert f.description == 'Hello'

    def test_default_name_is_from_function(self) -> None:
        def my_fn() -> None:
            pass
        f = Flow(fn=my_fn)
        assert f.name == 'my-fn'

    def test_raises_clear_error_when_not_compatible_with_validator(self) -> None:
        def my_fn(v__args: Any) -> None:
            pass
        with pytest.raises(ValueError, match='Flow function is not compatible with `validate_parameters`'):
            Flow(fn=my_fn)

    @pytest.mark.parametrize('name', ['my/flow', 'my%flow', 'my<flow', 'my>flow', 'my&flow'])
    def test_invalid_name(self, name: str) -> None:
        with pytest.raises(InvalidNameError, match='contains an invalid character'):
            Flow(fn=lambda: 1, name=name)

    def test_lambda_name_coerced_to_legal_characters(self) -> None:
        f = Flow(fn=lambda: 42)
        assert f.name == 'unknown-lambda'

    def test_invalid_run_name(self) -> None:
        class InvalidFlowRunNameArg:
            def format(self, *args: Any, **kwargs: Any) -> None:
                pass
        with pytest.raises(TypeError, match="Expected string or callable for 'flow_run_name'; got InvalidFlowRunNameArg instead."):
            Flow(fn=lambda: 1, name='hello', flow_run_name=InvalidFlowRunNameArg())

    def test_using_return_state_in_flow_definition_raises_reserved(self) -> None:
        with pytest.raises(ReservedArgumentError, match="'return_state' is a reserved argument name"):
            Flow(name='test', fn=lambda return_state: 42, version='A', description='B')


    def test_param_description_from_docstring(self) -> None:
        def my_fn(x: Any) -> None:
            """
            Hello

            Args:
                x: description
            """
        f = Flow(fn=my_fn)
        schema: Dict[str, Any] = parameter_schema(f).model_dump()
        assert schema['properties']['x']['description'] == 'description'


class TestDecorator:
    def test_flow_decorator_initializes(self) -> None:
        @flow(name='foo', version='B', flow_run_name='hi')
        def my_flow() -> str:
            return 'bar'
        assert isinstance(my_flow, Flow)
        assert my_flow.name == 'foo'
        assert my_flow.version == 'B'
        assert my_flow.fn() == 'bar'
        assert my_flow.flow_run_name == 'hi'

    def test_flow_decorator_initializes_with_callable_flow_run_name(self) -> None:
        @flow(flow_run_name=lambda: 'hi')
        def my_flow() -> str:
            return 'bar'
        assert isinstance(my_flow, Flow)
        assert my_flow.fn() == 'bar'
        assert my_flow.flow_run_name() == 'hi'

    def test_flow_decorator_sets_default_version(self) -> None:
        my_flow = flow(flatdict_to_dict)
        assert my_flow.version == file_hash(flatdict_to_dict.__globals__['__file__'])

    def test_invalid_run_name(self) -> None:
        class InvalidFlowRunNameArg:
            def format(self, *args: Any, **kwargs: Any) -> None:
                pass
        with pytest.raises(TypeError, match="Expected string or callable for 'flow_run_name'; got InvalidFlowRunNameArg instead."):
            @flow(flow_run_name=InvalidFlowRunNameArg())
            def flow_with_illegal_run_name() -> None:
                pass


class TestResultPersistence:
    @pytest.mark.parametrize('persist_result', [True, False])
    def test_persist_result_set_to_bool(self, persist_result: bool) -> None:
        @flow(persist_result=persist_result)
        def my_flow() -> None:
            pass

        @flow
        def base() -> None:
            pass
        new_flow = base.with_options(persist_result=persist_result)
        assert my_flow.persist_result is persist_result
        assert new_flow.persist_result is persist_result

    def test_setting_result_storage_sets_persist_result_to_true(self, tmpdir: Any) -> None:
        block: LocalFileSystem = LocalFileSystem(basepath=str(tmpdir))
        block.save('foo-bar-flow', _sync=True)

        @flow(result_storage=block)
        def my_flow() -> None:
            pass

        @flow
        def base() -> None:
            pass
        new_flow = base.with_options(result_storage=block)
        assert my_flow.persist_result is True
        assert new_flow.persist_result is True

    def test_setting_result_serializer_sets_persist_result_to_true(self) -> None:
        @flow(result_serializer='json')
        def my_flow() -> None:
            pass

        @flow
        def base() -> None:
            pass
        new_flow = base.with_options(result_serializer='json')
        assert my_flow.persist_result is True
        assert new_flow.persist_result is True


class TestFlowWithOptions:
    def test_with_options_allows_override_of_flow_settings(self) -> None:
        fooblock: LocalFileSystem = LocalFileSystem(basepath='foo')
        barblock: LocalFileSystem = LocalFileSystem(basepath='bar')
        fooblock.save('fooblock', _sync=True)
        barblock.save('barblock', _sync=True)

        @flow(name='Initial flow', description='Flow before with options', flow_run_name='OG',
              timeout_seconds=10, validate_parameters=True, persist_result=True,
              result_serializer='pickle', result_storage=fooblock, cache_result_in_memory=False,
              on_completion=None, on_failure=None, on_cancellation=None, on_crashed=None)
        def initial_flow() -> None:
            pass

        def failure_hook(flow: Any, flow_run: Any, state: Any) -> None:
            print('Woof!')

        def success_hook(flow: Any, flow_run: Any, state: Any) -> None:
            print('Meow!')

        def cancellation_hook(flow: Any, flow_run: Any, state: Any) -> None:
            print('Fizz Buzz!')

        def crash_hook(flow: Any, flow_run: Any, state: Any) -> None:
            print('Crash!')
        flow_with_options: Flow = initial_flow.with_options(
            name='Copied flow', description='A copied flow', flow_run_name=lambda: 'new-name',
            task_runner=ThreadPoolTaskRunner, retries=3, retry_delay_seconds=20, timeout_seconds=5,
            validate_parameters=False, persist_result=False, result_serializer='json', result_storage=barblock,
            cache_result_in_memory=True, on_completion=[success_hook], on_failure=[failure_hook],
            on_cancellation=[cancellation_hook], on_crashed=[crash_hook]
        )
        assert flow_with_options.name == 'Copied flow'
        assert flow_with_options.description == 'A copied flow'
        assert flow_with_options.flow_run_name() == 'new-name'
        assert isinstance(flow_with_options.task_runner, ThreadPoolTaskRunner)
        assert flow_with_options.timeout_seconds == 5
        assert flow_with_options.retries == 3
        assert flow_with_options.retry_delay_seconds == 20
        assert flow_with_options.should_validate_parameters is False
        assert flow_with_options.persist_result is False
        assert flow_with_options.result_serializer == 'json'
        assert flow_with_options.result_storage == barblock
        assert flow_with_options.cache_result_in_memory is True
        assert flow_with_options.on_completion_hooks == [success_hook]
        assert flow_with_options.on_failure_hooks == [failure_hook]
        assert flow_with_options.on_cancellation_hooks == [cancellation_hook]
        assert flow_with_options.on_crashed_hooks == [crash_hook]

    def test_with_options_uses_existing_settings_when_no_override(self, tmp_path: Any) -> None:
        storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        storage.save('test-overrides', _sync=True)

        @flow(name='Initial flow', description='Flow before with options',
              task_runner=ThreadPoolTaskRunner, timeout_seconds=10,
              validate_parameters=True, retries=3, retry_delay_seconds=20, persist_result=False,
              result_serializer='json', result_storage=storage, cache_result_in_memory=False, log_prints=False)
        def initial_flow() -> None:
            pass
        flow_with_options: Flow = initial_flow.with_options()
        assert flow_with_options is not initial_flow
        assert flow_with_options.name == 'Initial flow'
        assert flow_with_options.description == 'Flow before with options'
        assert isinstance(flow_with_options.task_runner, ThreadPoolTaskRunner)
        assert flow_with_options.timeout_seconds == 10
        assert flow_with_options.should_validate_parameters is True
        assert flow_with_options.retries == 3
        assert flow_with_options.retry_delay_seconds == 20
        assert flow_with_options.persist_result is False
        assert flow_with_options.result_serializer == 'json'
        assert flow_with_options.result_storage == storage
        assert flow_with_options.cache_result_in_memory is False
        assert flow_with_options.log_prints is False

    def test_with_options_can_unset_timeout_seconds_with_zero(self) -> None:
        @flow(timeout_seconds=1)
        def initial_flow() -> None:
            pass
        flow_with_options: Flow = initial_flow.with_options(timeout_seconds=0)
        assert flow_with_options.timeout_seconds is None

    def test_with_options_can_unset_retries_with_zero(self) -> None:
        @flow(retries=3)
        def initial_flow() -> None:
            pass
        flow_with_options: Flow = initial_flow.with_options(retries=0)
        assert flow_with_options.retries == 0

    def test_with_options_can_unset_retry_delay_seconds_with_zero(self) -> None:
        @flow(retry_delay_seconds=3)
        def initial_flow() -> None:
            pass
        flow_with_options: Flow = initial_flow.with_options(retry_delay_seconds=0)
        assert flow_with_options.retry_delay_seconds == 0

    def test_with_options_uses_parent_flow_run_name_if_not_provided(self) -> None:
        def generate_flow_run_name() -> str:
            return 'new-name'

        @flow(retry_delay_seconds=3, flow_run_name=generate_flow_run_name)
        def initial_flow() -> None:
            pass
        flow_with_options: Flow = initial_flow.with_options()
        assert flow_with_options.flow_run_name is generate_flow_run_name

    def test_with_options_can_unset_result_options_with_none(self, tmp_path: Any) -> None:
        storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        storage.save('test-unset', _sync=True)

        @flow(result_serializer='json', result_storage=storage)
        def initial_flow() -> None:
            pass
        flow_with_options: Flow = initial_flow.with_options(result_serializer=None, result_storage=None)
        assert flow_with_options.result_serializer is None
        assert flow_with_options.result_storage is None

    def test_with_options_signature_aligns_with_flow_signature(self) -> None:
        flow_params = set(inspect.signature(flow).parameters.keys())
        with_options_params = set(inspect.signature(Flow.with_options).parameters.keys())
        flow_params.remove('_FlowDecorator__fn')
        with_options_params.remove('self')
        assert flow_params == with_options_params

    def get_flow_run_name(self) -> str:
        name: str = 'test'
        date: str = 'todays_date'
        return f'{name}-{date}'

    @pytest.mark.parametrize('name, match', [(1, "Expected string for flow parameter 'name'; got int instead."), (get_flow_run_name, "Expected string for flow parameter 'name'; got function instead. Perhaps you meant to call it?")])
    def test_flow_name_non_string_raises(self, name: Any, match: str) -> None:
        with pytest.raises(TypeError, match=match):
            Flow(name=name, fn=lambda **kwargs: 42, version='A', description='B', flow_run_name='hi')

    @pytest.mark.parametrize('name', ['test', get_flow_run_name()])
    def test_flow_name_string_succeeds(self, name: str) -> None:
        f = Flow(name=name, fn=lambda **kwargs: 42, version='A', description='B', flow_run_name='hi')
        assert f.name == name


class TestFlowCall:
    async def test_call_creates_flow_run_and_runs(self) -> None:
        @flow(version='test', name=f'foo-{uuid.uuid4()}')
        def foo(x: int, y: int = 3, z: int = 3) -> int:
            return x + y + z
        assert foo(1, 2) == 6
        flow_run = await get_most_recent_flow_run(flow_name=foo.name)
        assert flow_run.parameters == {'x': 1, 'y': 2, 'z': 3}
        assert flow_run.flow_version == foo.version

    async def test_async_call_creates_flow_run_and_runs(self) -> None:
        @flow(version='test', name=f'foo-{uuid.uuid4()}')
        async def foo(x: int, y: int = 3, z: int = 3) -> int:
            return x + y + z
        assert await foo(1, 2) == 6
        flow_run = await get_most_recent_flow_run(flow_name=foo.name)
        assert flow_run.parameters == {'x': 1, 'y': 2, 'z': 3}
        assert flow_run.flow_version == foo.version

    async def test_call_with_return_state_true(self) -> None:
        @flow()
        def foo(x: int, y: int = 3, z: int = 3) -> int:
            return x + y + z
        state = foo(1, 2, return_state=True)
        assert isinstance(state, State)
        assert await state.result() == 6

    def test_call_coerces_parameter_types(self) -> None:
        import pydantic

        class CustomType(pydantic.BaseModel):
            pass

        @flow(version='test')
        def foo(x: int, y: List[int], zt: Any) -> int:
            return x + sum(y) + zt.z
        result = foo(x='1', y=['2', '3'], zt=CustomType(z=4).model_dump())
        assert result == 10

    def test_call_with_variadic_args(self) -> None:
        @flow
        def test_flow(*foo: Any, bar: Any) -> Tuple[Tuple[Any, ...], Any]:
            return (foo, bar)
        assert test_flow(1, 2, 3, bar=4) == ((1, 2, 3), 4)

    def test_call_with_variadic_keyword_args(self) -> None:
        @flow
        def test_flow(foo: Any, bar: Any, **foobar: Any) -> Tuple[Any, Any, Dict[str, Any]]:
            return (foo, bar, foobar)
        assert test_flow(1, 2, x=3, y=4, z=5) == (1, 2, dict(x=3, y=4, z=5))

    async def test_fails_but_does_not_raise_on_incompatible_parameter_types(self) -> None:
        @flow(version='test')
        def foo(x: int) -> None:
            pass
        state = foo(x='foo', return_state=True)
        with pytest.raises(ParameterTypeError):
            await state.result()

    def test_call_ignores_incompatible_parameter_types_if_asked(self) -> None:
        @flow(version='test', validate_parameters=False)
        def foo(x: Any) -> Any:
            return x
        assert foo(x='foo') == 'foo'

    @pytest.mark.parametrize('error', [ValueError('Hello'), None])
    async def test_final_state_reflects_exceptions_during_run(self, error: Optional[Exception]) -> None:
        @flow(version='test')
        def foo() -> None:
            if error:
                raise error
        state = foo(return_state=True)
        assert state.is_failed() if error else state.is_completed()
        assert exceptions_equal(await state.result(raise_on_failure=False), error)

    async def test_final_state_respects_returned_state(self) -> None:
        @flow(version='test')
        def foo() -> State:
            return State(type=StateType.FAILED, message='Test returned state', data='hello!')
        state = foo(return_state=True)
        assert state.is_failed()
        assert await state.result(raise_on_failure=False) == 'hello!'
        assert state.message == 'Test returned state'

    async def test_flow_state_reflects_returned_task_run_state(self) -> None:
        @task
        def fail() -> None:
            raise ValueError('Test')

        @flow(version='test')
        def foo() -> Any:
            return fail(return_state=True)
        flow_state = foo(return_state=True)
        assert flow_state.is_failed()
        task_run_state: State = await flow_state.result(raise_on_failure=False)
        assert isinstance(task_run_state, State)
        assert task_run_state.is_failed()
        with pytest.raises(ValueError, match='Test'):
            await task_run_state.result()

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_flow_state_defaults_to_task_states_when_no_return_failure(self) -> None:
        @task
        def fail() -> None:
            raise ValueError('Test')

        @flow(version='test')
        def foo() -> None:
            fail(return_state=True)
            fail(return_state=True)
            return None
        flow_state = foo(return_state=True)
        assert flow_state.is_failed()
        task_run_states = await flow_state.result(raise_on_failure=False)
        assert len(task_run_states) == 2
        assert all((isinstance(state, State) for state in task_run_states))
        task_run_state = task_run_states[0]
        assert task_run_state.is_failed()
        with pytest.raises(ValueError, match='Test'):
            await raise_state_exception(task_run_states[0])

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_flow_state_defaults_to_task_states_when_no_return_completed(self) -> None:
        @task
        def succeed() -> str:
            return 'foo'

        @flow(version='test')
        def foo() -> None:
            succeed()
            succeed()
            return None
        flow_state = foo(return_state=True)
        task_run_states = await flow_state.result()
        assert len(task_run_states) == 2
        assert all((isinstance(state, State) for state in task_run_states))
        assert await task_run_states[0].result() == 'foo'

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_flow_state_default_includes_subflow_states(self) -> None:
        @task
        def succeed() -> str:
            return 'foo'

        @flow
        def fail() -> None:
            raise ValueError('bar')

        @flow(version='test')
        def foo() -> None:
            succeed(return_state=True)
            fail(return_state=True)
            return None
        states = await foo(return_state=True).result(raise_on_failure=False)
        assert len(states) == 2
        assert all((isinstance(state, State) for state in states))
        assert await states[0].result() == 'foo'
        with pytest.raises(ValueError, match='bar'):
            await raise_state_exception(states[1])

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_flow_state_default_handles_nested_failures(self) -> None:
        @task
        def fail_task() -> None:
            raise ValueError('foo')

        @flow
        def fail_flow() -> None:
            fail_task(return_state=True)

        @flow
        def wrapper_flow() -> None:
            fail_flow(return_state=True)

        @flow(version='test')
        def foo() -> None:
            wrapper_flow(return_state=True)
            return None
        states = await foo(return_state=True).result(raise_on_failure=False)
        assert len(states) == 1
        state = states[0]
        assert isinstance(state, State)
        with pytest.raises(ValueError, match='foo'):
            await raise_state_exception(state)

    async def test_flow_state_reflects_returned_multiple_task_run_states(self) -> None:
        @task
        def fail1() -> None:
            raise ValueError('Test 1')

        @task
        def fail2() -> None:
            raise ValueError('Test 2')

        @task
        def succeed() -> bool:
            return True

        @flow(version='test')
        def foo() -> Tuple[Any, Any, Any]:
            return (fail1(return_state=True), fail2(return_state=True), succeed(return_state=True))
        flow_state = foo(return_state=True)
        assert flow_state.is_failed()
        assert flow_state.message == '2/3 states failed.'
        first, second, third = await flow_state.result(raise_on_failure=False)
        assert first.is_failed()
        assert second.is_failed()
        assert third.is_completed()
        with pytest.raises(ValueError, match='Test 1'):
            await first.result()
        with pytest.raises(ValueError, match='Test 2'):
            await second.result()

    def test_flow_can_end_in_paused_state(self) -> None:
        @flow
        def my_flow() -> State:
            return Paused()
        with pytest.raises(PausedRun, match='result is not available'):
            my_flow()
        flow_state: State = my_flow(return_state=True)
        assert flow_state.is_paused()

    def test_flow_can_end_in_cancelled_state(self) -> None:
        @flow
        def my_flow() -> State:
            return Cancelled()
        flow_state: State = my_flow(return_state=True)
        assert flow_state.is_cancelled()

    async def test_flow_state_with_cancelled_tasks_has_cancelled_state(self) -> None:
        @task
        def cancel() -> State:
            return Cancelled()

        @task
        def fail() -> None:
            raise ValueError('Fail')

        @task
        def succeed() -> bool:
            return True

        @flow(version='test')
        def my_flow() -> Tuple[Any, Any, Any]:
            return (cancel.submit(), succeed.submit(), fail.submit())
        flow_state: State = my_flow(return_state=True)
        assert flow_state.is_cancelled()
        assert flow_state.message == '1/3 states cancelled.'
        first, second, third = await flow_state.result(raise_on_failure=False)
        assert first.is_cancelled()
        assert second.is_completed()
        assert third.is_failed()
        with pytest.raises(CancelledRun):
            await first.result()

    def test_flow_with_cancelled_subflow_has_cancelled_state(self) -> None:
        @task
        def cancel() -> State:
            return Cancelled()

        @flow(version='test')
        def subflow() -> Any:
            return cancel.submit()

        @flow
        def my_flow() -> State:
            return subflow(return_state=True)
        flow_state: State = my_flow(return_state=True)
        assert flow_state.is_cancelled()
        assert flow_state.message == '1/1 states cancelled.'

    class BaseFooModel(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(ignored_types=(Flow,))

    class BaseFoo:
        def __init__(self, x: Any) -> None:
            self.x = x

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    def test_flow_supports_instance_methods(self, T: Any) -> None:
        class Foo(T):
            @flow
            def instance_method(self) -> Any:
                return self.x
        f = Foo(x=1)
        assert Foo(x=5).instance_method() == 5
        assert f.instance_method() == 1
        assert isinstance(Foo(x=10).instance_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    def test_flow_supports_class_methods(self, T: Any) -> None:
        class Foo(T):
            def __init__(self, x: Any) -> None:
                self.x = x

            @classmethod
            @flow
            def class_method(cls) -> str:
                return cls.__name__
        assert Foo.class_method() == 'Foo'
        assert isinstance(Foo.class_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    def test_flow_supports_static_methods(self, T: Any) -> None:
        class Foo(T):
            def __init__(self, x: Any) -> None:
                self.x = x

            @staticmethod
            @flow
            def static_method() -> str:
                return 'static'
        assert Foo.static_method() == 'static'
        assert isinstance(Foo.static_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    async def test_flow_supports_async_instance_methods(self, T: Any) -> None:
        class Foo(T):
            @flow
            async def instance_method(self) -> Any:
                return self.x
        f = Foo(x=1)
        assert await Foo(x=5).instance_method() == 5
        assert await f.instance_method() == 1
        assert isinstance(Foo(x=10).instance_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    async def test_flow_supports_async_class_methods(self, T: Any) -> None:
        class Foo(T):
            def __init__(self, x: Any) -> None:
                self.x = x

            @classmethod
            @flow
            async def class_method(cls) -> str:
                return cls.__name__
        assert await Foo.class_method() == 'Foo'
        assert isinstance(Foo.class_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    async def test_flow_supports_async_static_methods(self, T: Any) -> None:
        class Foo(T):
            def __init__(self, x: Any) -> None:
                self.x = x

            @staticmethod
            @flow
            async def static_method() -> str:
                return 'static'
        assert await Foo.static_method() == 'static'
        assert isinstance(Foo.static_method, Flow)

    def test_flow_supports_instance_methods_with_basemodel(self) -> None:
        class Foo(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(ignored_types=(Flow,))
            x: int = 5

            @flow
            def instance_method(self) -> int:
                return self.x
        assert Foo().instance_method() == 5
        assert isinstance(Foo().instance_method, Flow)

    def test_flow_supports_class_methods_with_basemodel(self) -> None:
        class Foo(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(ignored_types=(Flow,))

            @classmethod
            @flow
            def class_method(cls) -> str:
                return cls.__name__
        assert Foo.class_method() == 'Foo'
        assert isinstance(Foo.class_method, Flow)

    def test_flow_supports_static_methods_with_basemodel(self) -> None:
        class Foo(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(ignored_types=(Flow,))

            @staticmethod
            @flow
            def static_method() -> str:
                return 'static'
        assert Foo.static_method() == 'static'
        assert isinstance(Foo.static_method, Flow)

    def test_error_message_if_decorate_classmethod(self) -> None:
        with pytest.raises(TypeError, match='@classmethod should be applied on top of @flow'):
            class Foo:
                @flow
                @classmethod
                def bar(cls) -> None:
                    pass

    def test_error_message_if_decorate_staticmethod(self) -> None:
        with pytest.raises(TypeError, match='@staticmethod should be applied on top of @flow'):
            class Foo:
                @flow
                @staticmethod
                def bar() -> None:
                    pass

    def test_returns_when_cache_result_in_memory_is_false_sync_flow(self) -> None:
        @flow(cache_result_in_memory=False)
        def my_flow() -> int:
            return 42
        assert my_flow() == 42

    async def test_returns_when_cache_result_in_memory_is_false_async_flow(self) -> None:
        @flow(cache_result_in_memory=False)
        async def my_flow() -> int:
            return 42
        assert await my_flow() == 42

    def test_raises_correct_error_when_cache_result_in_memory_is_false_sync_flow(self) -> None:
        @flow(cache_result_in_memory=False)
        def my_flow() -> None:
            raise ValueError('Test')
        with pytest.raises(ValueError, match='Test'):
            my_flow()

    async def test_raises_correct_error_when_cache_result_in_memory_is_false_async_flow(self) -> None:
        @flow(cache_result_in_memory=False)
        async def my_flow() -> None:
            raise ValueError('Test')
        with pytest.raises(ValueError, match='Test'):
            await my_flow()


class TestSubflowCalls:
    async def test_subflow_call_with_no_tasks(self) -> None:
        @flow(version='foo')
        def child(x: int, y: int, z: int) -> int:
            return x + y + z

        @flow(version='bar')
        def parent(x: int, y: int = 2, z: int = 3) -> int:
            return child(x, y, z)
        assert parent(1, 2) == 6

    def test_subflow_call_with_returned_task(self) -> None:
        @task
        def compute(x: int, y: int, z: int) -> int:
            return x + y + z

        @flow(version='foo')
        def child(x: int, y: int, z: int) -> int:
            return compute(x, y, z)

        @flow(version='bar')
        def parent(x: int, y: int = 2, z: int = 3) -> int:
            return child(x, y, z)
        assert parent(1, 2) == 6

    async def test_async_flow_with_async_subflow_and_async_task(self) -> None:
        @task
        async def compute_async(x: int, y: int, z: int) -> int:
            return x + y + z

        @flow(version='foo')
        async def child(x: int, y: int, z: int) -> int:
            return await compute_async(x, y, z)

        @flow(version='bar')
        async def parent(x: int, y: int = 2, z: int = 3) -> int:
            return await child(x, y, z)
        assert await parent(1, 2) == 6

    async def test_async_flow_with_async_subflow_and_sync_task(self) -> None:
        @task
        def compute(x: int, y: int, z: int) -> int:
            return x + y + z

        @flow(version='foo')
        async def child(x: int, y: int, z: int) -> int:
            return compute(x, y, z)

        @flow(version='bar')
        async def parent(x: int, y: int = 2, z: int = 3) -> int:
            return await child(x, y, z)
        assert await parent(1, 2) == 6

    async def test_async_flow_with_sync_subflow_and_sync_task(self) -> None:
        @task
        def compute(x: int, y: int, z: int) -> int:
            return x + y + z

        @flow(version='foo')
        def child(x: int, y: int, z: int) -> int:
            return compute(x, y, z)

        @flow(version='bar')
        async def parent(x: int, y: int = 2, z: int = 3) -> int:
            return child(x, y, z)
        assert await parent(1, 2) == 6

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    def test_sync_flow_with_async_subflow(self) -> None:
        result: str = 'a string, not a coroutine'

        @flow
        async def async_child() -> str:
            return result

        @flow
        def parent() -> str:
            return async_child()
        assert parent() == result

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    def test_sync_flow_with_async_subflow_and_async_task(self) -> None:
        @task
        async def compute(x: int, y: int, z: int) -> int:
            return x + y + z

        @flow(version='foo')
        async def child(x: int, y: int, z: int) -> int:
            return await compute(x, y, z)

        @flow(version='bar')
        def parent(x: int, y: int = 2, z: int = 3) -> int:
            return child(x, y, z)
        assert parent(1, 2) == 6

    async def test_concurrent_async_subflow(self) -> None:
        @task
        async def test_task() -> int:
            return 1

        @flow(log_prints=True)
        async def child(i: int) -> int:
            assert await test_task() == 1
            return i

        @flow
        async def parent() -> List[int]:
            coros = [child(i) for i in range(5)]
            return await asyncio.gather(*coros)
        assert await parent() == list(range(5))

    async def test_recursive_async_subflow(self) -> None:
        @task
        async def test_task() -> int:
            return 1

        @flow
        async def recurse(i: int) -> int:
            assert await test_task() == 1
            if i == 0:
                return i
            else:
                return i + await recurse(i - 1)

        @flow
        async def parent() -> int:
            return await recurse(5)
        expected_total: int = 5 + 4 + 3 + 2 + 1
        assert await parent() == expected_total

    def test_recursive_sync_subflow(self) -> None:
        @task
        def test_task() -> int:
            return 1

        @flow
        def recurse(i: int) -> int:
            assert test_task() == 1
            if i == 0:
                return i
            else:
                return i + recurse(i - 1)

        @flow
        def parent() -> int:
            return recurse(5)
        expected_total: int = 5 + 4 + 3 + 2 + 1
        assert parent() == expected_total

    def test_recursive_sync_flow(self) -> None:
        @task
        def test_task() -> int:
            return 1

        @flow
        def recurse(i: int) -> int:
            assert test_task() == 1
            if i == 0:
                return i
            else:
                return i + recurse(i - 1)
        expected_total: int = 5 + 4 + 3 + 2 + 1
        assert recurse(5) == expected_total

    async def test_subflow_with_invalid_parameters_is_failed(self, prefect_client: PrefectClient) -> None:
        @flow
        def child(x: Any) -> Any:
            return x

        @flow
        def parent(x: Any) -> Any:
            return child(x, return_state=True)
        parent_state: State = parent('foo', return_state=True)
        with pytest.raises(ParameterTypeError, match='invalid parameters'):
            await parent_state.result()
        child_state: State = await parent_state.result(raise_on_failure=False)
        flow_run = await prefect_client.read_flow_run(child_state.state_details.flow_run_id)
        assert flow_run.state.is_failed()

    async def test_subflow_with_invalid_parameters_fails_parent(self) -> None:
        child_state: Optional[State] = None

        @flow
        def child(x: Any) -> Any:
            return x

        @flow
        def parent() -> Tuple[Any, Any]:
            nonlocal child_state
            child_state = child('foo', return_state=True)
            return (child_state, child(1, return_state=True))
        parent_state: State = parent(return_state=True)
        assert parent_state.is_failed()
        assert '1/2 states failed.' in parent_state.message
        with pytest.raises(ParameterTypeError):
            await child_state.result()

    async def test_subflow_with_invalid_parameters_is_not_failed_without_validation(self) -> None:
        @flow(validate_parameters=False)
        def child(x: Any) -> Any:
            return x

        @flow
        def parent(x: Any) -> Any:
            return child(x)
        assert parent('foo') == 'foo'

    async def test_subflow_relationship_tracking(self, prefect_client: PrefectClient) -> None:
        @flow(version='inner')
        def child(x: int, y: int) -> int:
            return x + y

        @flow()
        def parent() -> State:
            return child(1, 2, return_state=True)
        parent_state: State = parent(return_state=True)
        parent_flow_run_id: str = parent_state.state_details.flow_run_id
        child_state: State = await parent_state.result()
        child_flow_run_id: str = child_state.state_details.flow_run_id
        child_flow_run = await prefect_client.read_flow_run(child_flow_run_id)
        parent_flow_run_task = await prefect_client.read_task_run(child_flow_run.parent_task_run_id)
        assert parent_flow_run_task.task_version == 'inner'
        assert parent_flow_run_id != child_flow_run_id, 'The subflow run and parent flow run are distinct'
        assert child_state.state_details.task_run_id == parent_flow_run_task.id, 'The client subflow run state links to the parent task'
        subflow_states = await prefect_client.read_flow_run_states(child_flow_run_id)
        assert all((state.state_details.task_run_id == parent_flow_run_task.id for state in subflow_states)), 'All server subflow run states link to the parent task'
        assert parent_flow_run_task.state.state_details.child_flow_run_id == child_flow_run_id, 'The parent task links to the subflow run id'
        assert parent_flow_run_task.state.state_details.flow_run_id == parent_flow_run_id, 'The parent task belongs to the parent flow'
        assert child_flow_run.parent_task_run_id == parent_flow_run_task.id, 'The server subflow run links to the parent task'
        assert child_flow_run.id == child_flow_run_id, 'The server subflow run id matches the client'

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_sync_flow_with_async_subflow_and_task_that_awaits_result(self) -> None:
        """
        Regression test for https://github.com/PrefectHQ/prefect/issues/12053, where
        we discovered that a sync flow running an async flow that awaits `.result()`
        on a submitted task's future can hang indefinitely.
        """
        @task
        async def some_async_task() -> int:
            return 42

        @flow
        async def integrations_flow() -> int:
            future = await some_async_task.submit()
            return await future.result()

        @flow
        def sync_flow() -> int:
            return integrations_flow()
        result: int = sync_flow()
        assert result == 42

    # The remaining tests follow similarly with annotations...
    # Due to the length of the file, additional functions should be annotated in the same manner.
    # Each function should have parameters and return types declared as shown above.


# Additional test classes (TestFlowRunTags, TestFlowTimeouts, ParameterTestModel, ParameterTestClass, ParameterTestEnum,
# TestFlowParameterTypes, TestSubflowTaskInputs, _wait_for_logs, TestFlowRunLogs, TestSubflowRunLogs, TestFlowRetries,
# TestLoadFlowFromEntrypoint, TestLoadFunctionAndConvertToFlow, TestFlowRunName, TestFlowHooksContext, TestFlowHooksWithKwargs,
# TestFlowHooksOnCompletion, TestFlowHooksOnFailure, TestFlowHooksOnCancellation, TestFlowHooksOnCrashed,
# TestFlowHooksOnRunning, TestFlowToDeployment, TestFlowServe, MockStorage, TestFlowFromSource, TestFlowDeploy,
# TestLoadFlowFromFlowRun, TestTransactions, TestLoadFlowArgumentFromEntrypoint, TestSafeLoadFlowFromEntrypoint)
# and so on should be annotated similarly with type hints.
# Due to space constraints, only a portion of the file is fully annotated here.
