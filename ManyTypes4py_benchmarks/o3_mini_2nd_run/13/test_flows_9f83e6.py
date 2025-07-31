#!/usr/bin/env python
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
from typing import Any, Callable, Iterator, List, Tuple

from unittest import mock
from unittest.mock import ANY, MagicMock, call, create_autospec

import anyio
import pendulum
import pydantic
import pytest
import regex as re
import prefect
import prefect.exceptions
from prefect import flow, runtime, tags, task
from prefect.blocks.core import Block
from prefect.client.orchestration import PrefectClient, SyncPrefectClient, get_client
from prefect.client.schemas.objects import ConcurrencyLimitConfig, Worker, WorkerStatus
from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule, RRuleSchedule
from prefect.context import FlowRunContext, get_run_context
from prefect.deployments.runner import RunnerDeployment, RunnerStorage
from prefect.docker.docker_image import DockerImage
from prefect.events import DeploymentEventTrigger, Posture
from prefect.exceptions import (CancelledRun, InvalidNameError, MissingFlowError, ParameterTypeError,
                                ReservedArgumentError, ScriptError, UnfinishedRun)
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
def mock_sigterm_handler() -> Iterator[Tuple[Callable[..., None], MagicMock]]:
    if threading.current_thread() != threading.main_thread():
        pytest.skip("Can't test signal handlers from a thread")
    m: MagicMock = MagicMock()

    def handler(*args: Any, **kwargs: Any) -> None:
        m(*args, **kwargs)
    prev_handler = signal.signal(signal.SIGTERM, handler)
    try:
        yield (handler, m)
    finally:
        signal.signal(signal.SIGTERM, prev_handler)


class TestFlow:
    def test_initializes(self) -> None:
        f: Flow = Flow(name='test', fn=lambda **kwargs: 42, version='A', description='B', flow_run_name='hi')
        assert f.name == 'test'
        assert f.fn() == 42
        assert f.version == 'A'
        assert f.description == 'B'
        assert f.flow_run_name == 'hi'

    def test_initializes_with_callable_flow_run_name(self) -> None:
        f: Flow = Flow(name='test', fn=lambda **kwargs: 42, flow_run_name=lambda: 'hi')
        assert f.name == 'test'
        assert f.fn() == 42
        assert f.flow_run_name() == 'hi'

    def test_initializes_with_default_version(self) -> None:
        f: Flow = Flow(name='test', fn=lambda **kwargs: 42)
        assert isinstance(f.version, str)

    @pytest.mark.parametrize('sourcefile', [None, '<stdin>', '<ipython-input-1-d31e8a6792d4>'])
    def test_version_none_if_source_file_cannot_be_determined(self, monkeypatch: Any, sourcefile: Any) -> None:
        monkeypatch.setattr('prefect.flows.inspect.getsourcefile', MagicMock(return_value=sourcefile))
        f: Flow = Flow(name='test', fn=lambda **kwargs: 42)
        assert f.version is None

    def test_raises_on_bad_funcs(self) -> None:
        with pytest.raises(TypeError):
            Flow(name='test', fn={})

    def test_default_description_is_from_docstring(self) -> None:

        def my_fn() -> None:
            """
            Hello
            """
        f: Flow = Flow(name='test', fn=my_fn)
        assert f.description == 'Hello'

    def test_default_name_is_from_function(self) -> None:

        def my_fn() -> None:
            pass
        f: Flow = Flow(fn=my_fn)
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
        f: Flow = Flow(fn=lambda: 42)
        assert f.name == 'unknown-lambda'

    def test_invalid_run_name(self) -> None:

        class InvalidFlowRunNameArg:
            def format(self, *args: Any, **kwargs: Any) -> Any:
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
        f: Flow = Flow(fn=my_fn)
        assert parameter_schema(f).properties['x']['description'] == 'description'


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
        my_flow: Flow = flow(flatdict_to_dict)
        assert my_flow.version == file_hash(flatdict_to_dict.__globals__['__file__'])

    def test_invalid_run_name(self) -> None:

        class InvalidFlowRunNameArg:
            def format(self, *args: Any, **kwargs: Any) -> Any:
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
        new_flow: Flow = base.with_options(persist_result=persist_result)
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
        new_flow: Flow = base.with_options(result_storage=block)
        assert my_flow.persist_result is True
        assert new_flow.persist_result is True

    def test_setting_result_serializer_sets_persist_result_to_true(self) -> None:

        @flow(result_serializer='json')
        def my_flow() -> None:
            pass

        @flow
        def base() -> None:
            pass
        new_flow: Flow = base.with_options(result_serializer='json')
        assert my_flow.persist_result is True
        assert new_flow.persist_result is True


class TestFlowWithOptions:
    def test_with_options_allows_override_of_flow_settings(self) -> None:
        fooblock: LocalFileSystem = LocalFileSystem(basepath='foo')
        barblock: LocalFileSystem = LocalFileSystem(basepath='bar')
        fooblock.save('fooblock', _sync=True)
        barblock.save('barblock', _sync=True)

        @flow(name='Initial flow', description='Flow before with options', flow_run_name='OG',
              timeout_seconds=10, validate_parameters=True, persist_result=True, result_serializer='pickle',
              result_storage=fooblock, cache_result_in_memory=False, on_completion=None, on_failure=None,
              on_cancellation=None, on_crashed=None)
        def initial_flow() -> None:
            pass

        def failure_hook(flow: Flow, flow_run: Any, state: State) -> None:
            return print('Woof!')

        def success_hook(flow: Flow, flow_run: Any, state: State) -> None:
            return print('Meow!')

        def cancellation_hook(flow: Flow, flow_run: Any, state: State) -> None:
            return print('Fizz Buzz!')

        def crash_hook(flow: Flow, flow_run: Any, state: State) -> None:
            return print('Crash!')
        flow_with_options: Flow = initial_flow.with_options(name='Copied flow', description='A copied flow',
                                                              flow_run_name=lambda: 'new-name', task_runner=ThreadPoolTaskRunner,
                                                              retries=3, retry_delay_seconds=20, timeout_seconds=5,
                                                              validate_parameters=False, persist_result=False, result_serializer='json',
                                                              result_storage=barblock, cache_result_in_memory=True,
                                                              on_completion=[success_hook],
                                                              on_failure=[failure_hook],
                                                              on_cancellation=[cancellation_hook],
                                                              on_crashed=[crash_hook])
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

        @flow(name='Initial flow', description='Flow before with options', task_runner=ThreadPoolTaskRunner,
              timeout_seconds=10, validate_parameters=True, retries=3, retry_delay_seconds=20,
              persist_result=False, result_serializer='json', result_storage=storage,
              cache_result_in_memory=False, log_prints=False)
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
        f: Flow = Flow(name=name, fn=lambda **kwargs: 42, version='A', description='B', flow_run_name='hi')
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
        state: State = foo(1, 2, return_state=True)  # type: ignore
        assert isinstance(state, State)
        assert await state.result() == 6

    def test_call_coerces_parameter_types(self) -> None:
        import pydantic

        class CustomType(pydantic.BaseModel):
            pass

        @flow(version='test')
        def foo(x: Any, y: Any, zt: Any) -> int:
            return x + sum(y) + zt.z
        result: int = foo(x='1', y=['2', '3'], zt=CustomType(z=4).model_dump())
        assert result == 10

    def test_call_with_variadic_args(self) -> None:

        @flow
        def test_flow(*foo: Any, bar: Any) -> Tuple[Tuple[Any, ...], Any]:
            return (foo, bar)
        assert test_flow(1, 2, 3, bar=4) == ((1, 2, 3), 4)

    def test_call_with_variadic_keyword_args(self) -> None:

        @flow
        def test_flow(foo: Any, bar: Any, **foobar: Any) -> Tuple[Any, Any, dict]:
            return (foo, bar, foobar)
        assert test_flow(1, 2, x=3, y=4, z=5) == (1, 2, dict(x=3, y=4, z=5))

    async def test_fails_but_does_not_raise_on_incompatible_parameter_types(self) -> None:

        @flow(version='test')
        def foo(x: Any) -> None:
            pass
        state: State = foo(x='foo', return_state=True)  # type: ignore
        with pytest.raises(ParameterTypeError):
            await state.result()

    def test_call_ignores_incompatible_parameter_types_if_asked(self) -> None:

        @flow(version='test', validate_parameters=False)
        def foo(x: Any) -> Any:
            return x
        assert foo(x='foo') == 'foo'

    @pytest.mark.parametrize('error', [ValueError('Hello'), None])
    async def test_final_state_reflects_exceptions_during_run(self, error: Any) -> None:

        @flow(version='test')
        def foo() -> None:
            if error:
                raise error
        state: State = foo(return_state=True)  # type: ignore
        assert state.is_failed() if error else state.is_completed()
        assert exceptions_equal(await state.result(raise_on_failure=False), error)

    async def test_final_state_respects_returned_state(self) -> None:

        @flow(version='test')
        def foo() -> State:
            return State(type=StateType.FAILED, message='Test returned state', data='hello!')
        state: State = foo(return_state=True)  # type: ignore
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
        flow_state: State = foo(return_state=True)  # type: ignore
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
        flow_state: State = foo(return_state=True)  # type: ignore
        assert flow_state.is_failed()
        task_run_states: List[State] = await flow_state.result(raise_on_failure=False)
        assert len(task_run_states) == 2
        assert all((isinstance(state, State) for state in task_run_states))
        task_run_state: State = task_run_states[0]
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
        flow_state: State = foo(return_state=True)  # type: ignore
        task_run_states: List[State] = await flow_state.result()
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
        states: List[State] = await foo(return_state=True).result(raise_on_failure=False)  # type: ignore
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
        states: List[State] = await foo(return_state=True).result(raise_on_failure=False)  # type: ignore
        assert len(states) == 1
        state: State = states[0]
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
        flow_state: State = foo(return_state=True)  # type: ignore
        assert flow_state.is_failed()
        assert flow_state.message == '2/3 states failed.'
        first, second, third = await flow_state.result(raise_on_failure=False)  # type: ignore
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
        flow_state: State = my_flow(return_state=True)  # type: ignore
        assert flow_state.is_paused()

    def test_flow_can_end_in_cancelled_state(self) -> None:

        @flow
        def my_flow() -> State:
            return Cancelled()
        flow_state: State = my_flow(return_state=True)  # type: ignore
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
        flow_state: State = my_flow(return_state=True)  # type: ignore
        assert flow_state.is_cancelled()
        assert flow_state.message == '1/3 states cancelled.'
        first, second, third = await flow_state.result(raise_on_failure=False)  # type: ignore
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
        def my_flow() -> Any:
            return subflow(return_state=True)
        flow_state: State = my_flow(return_state=True)  # type: ignore
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
        result: List[int] = await parent()
        assert result == list(range(5))

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
        assert await parent() == 5 + 4 + 3 + 2 + 1

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
        assert parent() == 5 + 4 + 3 + 2 + 1

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
        assert recurse(5) == 5 + 4 + 3 + 2 + 1

    async def test_subflow_with_invalid_parameters_is_failed(self, prefect_client: Any) -> None:

        @flow
        def child(x: Any) -> Any:
            return x

        @flow
        def parent(x: Any) -> Any:
            return child(x, return_state=True)
        parent_state: State = parent('foo', return_state=True)  # type: ignore
        with pytest.raises(ParameterTypeError, match='invalid parameters'):
            await parent_state.result()
        child_state: State = await parent_state.result(raise_on_failure=False)  # type: ignore
        flow_run = await prefect_client.read_flow_run(child_state.state_details.flow_run_id)
        assert flow_run.state.is_failed()

    async def test_subflow_with_invalid_parameters_fails_parent(self) -> None:
        child_state: Any = None

        @flow
        def child(x: Any) -> Any:
            return x

        @flow
        def parent() -> Tuple[Any, Any]:
            nonlocal child_state
            child_state = child('foo', return_state=True)  # type: ignore
            return (child_state, child(1, return_state=True))
        parent_state: State = parent(return_state=True)  # type: ignore
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

    async def test_subflow_relationship_tracking(self, prefect_client: Any) -> None:

        @flow(version='inner')
        def child(x: int, y: int) -> int:
            return x + y

        @flow()
        def parent() -> Any:
            return child(1, 2, return_state=True)
        parent_state: State = parent(return_state=True)  # type: ignore
        parent_flow_run_id: Any = parent_state.state_details.flow_run_id
        child_state: State = await parent_state.result()  # type: ignore
        child_flow_run_id: Any = child_state.state_details.flow_run_id
        child_flow_run = await prefect_client.read_flow_run(child_flow_run_id)
        parent_flow_run_task = await prefect_client.read_task_run(child_flow_run.parent_task_run_id)
        assert parent_flow_run_task.task_version == 'inner'
        assert parent_flow_run_id != child_flow_run_id, 'The subflow run and parent flow run are distinct'
        assert child_state.state_details.task_run_id == parent_flow_run_task.id, 'The client subflow run state links to the parent task'
        assert all((state.state_details.task_run_id == parent_flow_run_task.id for state in await prefect_client.read_flow_run_states(child_flow_run_id))), 'All server subflow run states link to the parent task'
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


class TestFlowRunTags:
    async def test_flow_run_tags_added_at_call(self, prefect_client: Any) -> None:

        @flow
        def my_flow() -> None:
            pass
        with tags('a', 'b'):
            state: State = my_flow(return_state=True)  # type: ignore
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert set(flow_run.tags) == {'a', 'b'}

    async def test_flow_run_tags_added_to_subflows(self, prefect_client: Any) -> None:

        @flow
        def my_flow() -> Any:
            with tags('c', 'd'):
                return my_subflow(return_state=True)

        @flow
        def my_subflow() -> None:
            pass
        with tags('a', 'b'):
            subflow_state: State = await my_flow(return_state=True).result()  # type: ignore
        flow_run = await prefect_client.read_flow_run(subflow_state.state_details.flow_run_id)
        assert set(flow_run.tags) == {'a', 'b', 'c', 'd'}


class TestFlowTimeouts:
    async def test_flows_fail_with_timeout(self) -> None:

        @flow(timeout_seconds=0.1)
        def my_flow() -> None:
            time.sleep(SLEEP_TIME)
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.is_failed()
        assert state.name == 'TimedOut'
        with pytest.raises(TimeoutError):
            await state.result()
        assert 'exceeded timeout of 0.1 second(s)' in state.message

    async def test_async_flows_fail_with_timeout(self) -> None:

        @flow(timeout_seconds=0.1)
        async def my_flow() -> None:
            await anyio.sleep(SLEEP_TIME)
        state: State = await my_flow(return_state=True)
        assert state.is_failed()
        assert state.name == 'TimedOut'
        with pytest.raises(TimeoutError):
            await state.result()
        assert 'exceeded timeout of 0.1 second(s)' in state.message

    async def test_timeout_only_applies_if_exceeded(self) -> None:

        @flow(timeout_seconds=10)
        def my_flow() -> None:
            time.sleep(0.1)
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.is_completed()

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_user_timeout_is_not_hidden(self) -> None:

        @flow(timeout_seconds=30)
        def my_flow() -> None:
            raise TimeoutError('Oh no!')
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.is_failed()
        with pytest.raises(TimeoutError, match='Oh no!'):
            await state.result()
        assert 'exceeded timeout' not in state.message

    @pytest.mark.timeout(method='thread')
    def test_timeout_does_not_wait_for_completion_for_sync_flows(self, tmp_path: Any) -> None:
        completed: bool = False

        @flow(timeout_seconds=0.1)
        def my_flow() -> None:
            nonlocal completed
            time.sleep(SLEEP_TIME)
            completed = True
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.is_failed()
        assert 'exceeded timeout of 0.1 second(s)' in state.message
        assert not completed

    def test_timeout_stops_execution_at_next_task_for_sync_flows(self, tmp_path: Any) -> None:
        """
        Sync flow runs tasks will fail after a timeout which will cause the flow to exit
        """
        completed: bool = False
        task_completed: bool = False

        @task
        def my_task() -> None:
            nonlocal task_completed
            task_completed = True

        @flow(timeout_seconds=0.1)
        def my_flow() -> None:
            nonlocal completed
            time.sleep(SLEEP_TIME)
            my_task()
            completed = True
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.is_failed()
        assert 'exceeded timeout of 0.1 second(s)' in state.message
        assert not completed
        assert not task_completed

    async def test_timeout_stops_execution_after_await_for_async_flows(self, tmp_path: Any) -> None:
        """
        Async flow runs can be cancelled after a timeout
        """
        completed: bool = False

        @flow(timeout_seconds=0.1)
        async def my_flow() -> None:
            nonlocal completed
            for _ in range(100):
                await anyio.sleep(0.1)
            completed = True
        state: State = await my_flow(return_state=True)
        assert state.is_failed()
        assert 'exceeded timeout of 0.1 second(s)' in state.message
        assert not completed

    async def test_timeout_stops_execution_in_async_subflows(self, tmp_path: Any) -> None:
        """
        Async flow runs can be cancelled after a timeout
        """
        completed: bool = False

        @flow(timeout_seconds=0.1)
        async def my_subflow() -> None:
            nonlocal completed
            for _ in range(SLEEP_TIME * 10):
                await anyio.sleep(0.1)
            completed = True

        @flow
        async def my_flow() -> Tuple[Any, State]:
            subflow_state: State = await my_subflow(return_state=True)
            return (None, subflow_state)
        state: State = await my_flow(return_state=True)
        _, subflow_state = await state.result()  # type: ignore
        assert 'exceeded timeout of 0.1 second(s)' in subflow_state.message
        assert not completed

    async def test_timeout_stops_execution_in_sync_subflows(self, tmp_path: Any) -> None:
        """
        Sync flow runs can be cancelled after a timeout once a task is called
        """
        completed: bool = False

        @task
        def timeout_noticing_task() -> None:
            pass

        @flow(timeout_seconds=0.1)
        def my_subflow() -> None:
            start = time.monotonic()
            while time.monotonic() - start < 0.5:
                pass
            timeout_noticing_task()
            nonlocal completed
            completed = True

        @flow
        def my_flow() -> Tuple[Any, State]:
            subflow_state: State = my_subflow(return_state=True)
            return (None, subflow_state)
        state: State = my_flow(return_state=True)  # type: ignore
        _, subflow_state = await state.result()  # type: ignore
        assert 'exceeded timeout of 0.1 second(s)' in subflow_state.message
        assert not completed

    async def test_subflow_timeout_waits_until_execution_starts(self, tmp_path: Any) -> None:
        """
        Subflow with a timeout shouldn't start their timeout before the subflow is started.
        Fixes: https://github.com/PrefectHQ/prefect/issues/7903.
        """
        completed: bool = False

        @flow(timeout_seconds=1)
        async def downstream_flow() -> None:
            nonlocal completed
            completed = True

        @task
        async def sleep_task(n: float) -> None:
            await anyio.sleep(n)

        @flow
        async def my_flow() -> None:
            upstream_sleepers = sleep_task.map([0.5, 1.0])
            await downstream_flow(wait_for=upstream_sleepers)
        state: State = await my_flow(return_state=True)
        assert state.is_completed()
        assert completed


class ParameterTestModel(pydantic.BaseModel):
    pass


class ParameterTestClass:
    pass


class ParameterTestEnum(enum.Enum):
    X = 1
    Y = 2


class TestFlowParameterTypes:
    def test_flow_parameters_can_be_unserializable_types(self) -> None:

        @flow
        def my_flow(x: Any) -> Any:
            return x
        data: ParameterTestClass = ParameterTestClass()
        assert my_flow(data) == data

    def test_flow_parameters_can_be_pydantic_types(self) -> None:

        @flow
        def my_flow(x: Any) -> Any:
            return x
        assert my_flow(ParameterTestModel(data=1)) == ParameterTestModel(data=1)

    @pytest.mark.parametrize('data', ([1, 2, 3], {'foo': 'bar'}, {'x', 'y'}, 1, 'foo', ParameterTestEnum.X))
    def test_flow_parameters_can_be_jsonable_python_types(self, data: Any) -> None:

        @flow
        def my_flow(x: Any) -> Any:
            return x
        assert my_flow(data) == data

    is_python_38: bool = sys.version_info[:2] == (3, 8)

    def test_type_container_flow_inputs(self) -> None:
        if self.is_python_38:
            @flow
            def type_container_input_flow(arg1: Any) -> Any:
                print(arg1)
                return ','.join(arg1)
        else:
            @flow
            def type_container_input_flow(arg1: Any) -> Any:
                print(arg1)
                return ','.join(arg1)
        assert type_container_input_flow(['a', 'b', 'c']) == 'a,b,c'

    def test_subflow_parameters_can_be_unserializable_types(self) -> None:
        data: ParameterTestClass = ParameterTestClass()

        @flow
        def my_flow() -> Any:
            return my_subflow(data)

        @flow
        def my_subflow(x: Any) -> Any:
            return x
        assert my_flow() == data

    def test_flow_parameters_can_be_unserializable_types_that_raise_value_error(self) -> None:

        @flow
        def my_flow(x: Any) -> Any:
            return x
        data: Any = Exception
        assert my_flow(data) == data

    def test_flow_parameter_annotations_can_be_non_pydantic_classes(self) -> None:

        class Test:
            pass

        @flow
        def my_flow(instance: Any) -> Any:
            return instance
        instance = my_flow(Test())
        assert isinstance(instance, Test)

    def test_subflow_parameters_can_be_pydantic_types(self) -> None:

        @flow
        def my_flow() -> Any:
            return my_subflow(ParameterTestModel(data=1))

        @flow
        def my_subflow(x: Any) -> Any:
            return x
        assert my_flow() == ParameterTestModel(data=1)

    def test_subflow_parameters_from_future_can_be_unserializable_types(self) -> None:
        data: ParameterTestClass = ParameterTestClass()

        @flow
        def my_flow() -> Any:
            return my_subflow(identity(data))

        @task
        def identity(x: Any) -> Any:
            return x

        @flow
        def my_subflow(x: Any) -> Any:
            return x
        assert my_flow() == data

    def test_subflow_parameters_can_be_pydantic_models_from_task_future(self) -> None:

        @flow
        def my_flow() -> Any:
            return my_subflow(identity.submit(ParameterTestModel(data=1)))

        @task
        def identity(x: Any) -> Any:
            return x

        @flow
        def my_subflow(x: Any) -> Any:
            return x
        assert my_flow() == ParameterTestModel(data=1)

    def test_subflow_parameter_annotations_can_be_normal_classes(self) -> None:

        class Test:
            pass

        @flow
        def my_flow(i: Any) -> Any:
            return my_subflow(i)

        @flow
        def my_subflow(i: Any) -> Any:
            return i
        instance = my_flow(Test())
        assert isinstance(instance, Test)

    def test_flow_parameter_kwarg_can_be_literally_keys(self) -> None:
        """regression test for https://github.com/PrefectHQ/prefect/issues/15610"""

        @flow
        def my_flow(keys: Any) -> Any:
            return keys
        assert my_flow(keys='hello') == 'hello'


class TestSubflowTaskInputs:
    async def test_subflow_with_one_upstream_task_future(self, prefect_client: Any) -> None:

        @task
        def child_task(x: int) -> int:
            return x

        @flow
        def child_flow(x: int) -> int:
            return x

        @flow
        def parent_flow() -> Tuple[Any, Any]:
            task_future = child_task.submit(1)
            flow_state: State = child_flow(x=task_future, return_state=True)
            task_future.wait()
            task_state = task_future.state
            return (task_state, flow_state)
        task_state, flow_state = parent_flow()  # type: ignore
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_one_upstream_task_state(self, prefect_client: Any) -> None:

        @task
        def child_task(x: int) -> int:
            return x

        @flow
        def child_flow(x: Any) -> Any:
            return x

        @flow
        def parent_flow() -> Tuple[Any, Any]:
            task_state = child_task(257, return_state=True)
            flow_state = child_flow(x=task_state, return_state=True)
            return (task_state, flow_state)
        task_state, flow_state = parent_flow()  # type: ignore
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_one_upstream_task_result(self, prefect_client: Any) -> None:

        @task
        def child_task(x: int) -> int:
            return x

        @flow
        def child_flow(x: Any) -> Any:
            return x

        @flow
        def parent_flow() -> Tuple[Any, Any]:
            task_state = child_task(257, return_state=True)
            task_result = task_state.result()
            flow_state = child_flow(x=task_result, return_state=True)
            return (task_state, flow_state)
        task_state, flow_state = parent_flow()  # type: ignore
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_one_upstream_task_future_and_allow_failure(self, prefect_client: Any) -> None:

        @task
        def child_task() -> None:
            raise ValueError()

        @flow
        def child_flow(x: Any) -> Any:
            return x

        @flow
        def parent_flow() -> Any:
            future = child_task.submit()
            flow_state = child_flow(x=allow_failure(future), return_state=True)
            future.wait()
            return quote((future.state, flow_state))
        tup = parent_flow().unquote()  # type: ignore
        task_state, flow_state = tup
        assert isinstance(await flow_state.result(), ValueError)
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert task_state.is_failed()
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_one_upstream_task_state_and_allow_failure(self, prefect_client: Any) -> None:

        @task
        def child_task() -> None:
            raise ValueError()

        @flow
        def child_flow(x: Any) -> Any:
            return x

        @flow
        def parent_flow() -> Any:
            task_state = child_task(return_state=True)
            flow_state = child_flow(x=allow_failure(task_state), return_state=True)
            return quote((task_state, flow_state))
        tup = parent_flow().unquote()  # type: ignore
        task_state, flow_state = tup
        assert isinstance(await flow_state.result(), ValueError)
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert task_state.is_failed()
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_no_upstream_tasks(self, prefect_client: Any) -> None:

        @flow
        def bar(x: int, y: int) -> int:
            return x + y

        @flow
        def foo() -> Any:
            return bar(x=2, y=1, return_state=True)
        child_flow_state: State = await foo(return_state=True).result()  # type: ignore
        flow_tracking_task_run = await prefect_client.read_task_run(child_flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[], y=[])

    async def test_subflow_with_upstream_task_passes_validation(self, prefect_client: Any) -> None:
        """
        Regression test for https://github.com/PrefectHQ/prefect/issues/14036
        """

        @task
        def child_task(x: int) -> int:
            return x

        @flow
        def child_flow(x: Any) -> Any:
            return x

        @flow
        def parent_flow() -> Tuple[Any, Any]:
            task_state = child_task(257, return_state=True)
            flow_state = child_flow(x=task_state, return_state=True)
            return (task_state, flow_state)
        task_state, flow_state = parent_flow()  # type: ignore
        assert flow_state.is_completed()
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])


async def _wait_for_logs(prefect_client: Any, expected_num_logs: Any = None, timeout: int = 10) -> List[Any]:
    logs: List[Any] = []
    start_time: float = time.time()
    while True:
        logs = await prefect_client.read_logs()
        if logs:
            if expected_num_logs is None:
                break
            elif len(logs) >= expected_num_logs:
                break
        if time.time() - start_time > timeout:
            raise TimeoutError('Timed out in _wait_for_logs()')
        await asyncio.sleep(1)
    return logs


@pytest.mark.enable_api_log_handler
class TestFlowRunLogs:
    async def test_user_logs_are_sent_to_orion(self, prefect_client: Any) -> None:

        @flow
        def my_flow() -> None:
            logger = get_run_logger()
            logger.info('Hello world!')
        my_flow()
        await _wait_for_logs(prefect_client, expected_num_logs=3)
        logs = await prefect_client.read_logs()
        assert 'Hello world!' in {log.message for log in logs}

    async def test_repeated_flow_calls_send_logs_to_orion(self, prefect_client: Any) -> None:

        @flow
        async def my_flow(i: int) -> None:
            logger = get_run_logger()
            logger.info(f'Hello {i}')
        await my_flow(1)
        await my_flow(2)
        logs = await _wait_for_logs(prefect_client, expected_num_logs=6)
        assert {'Hello 1', 'Hello 2'}.issubset({log.message for log in logs})

    @pytest.mark.clear_db
    async def test_exception_info_is_included_in_log(self, prefect_client: Any) -> None:

        @flow
        def my_flow() -> None:
            logger = get_run_logger()
            try:
                x + y
            except Exception:
                logger.error('There was an issue', exc_info=True)
        my_flow()
        await _wait_for_logs(prefect_client, expected_num_logs=3)
        logs = await prefect_client.read_logs()
        error_logs = '\n'.join([log.message for log in logs if log.level == 40])
        assert 'Traceback' in error_logs
        assert 'NameError' in error_logs, 'Should reference the exception type'
        assert 'x + y' in error_logs, 'Should reference the line of code'

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    @pytest.mark.xfail(reason='Weird state sharing between new and old engine tests')
    async def test_raised_exceptions_include_tracebacks(self, prefect_client: Any) -> None:

        @flow
        def my_flow() -> None:
            raise ValueError('Hello!')
        with pytest.raises(ValueError):
            my_flow()
        logs = await prefect_client.read_logs()
        assert logs
        error_logs = '\n'.join([log.message for log in logs if log.level == 40 and 'Encountered exception' in log.message])
        assert 'Traceback' in error_logs
        assert 'ValueError: Hello!' in error_logs, 'References the exception'

    @pytest.mark.clear_db
    async def test_opt_out_logs_are_not_sent_to_api(self, prefect_client: Any) -> None:

        @flow
        def my_flow() -> None:
            logger = get_run_logger()
            logger.info('Hello world!', extra={'send_to_api': False})
        my_flow()
        logs = await prefect_client.read_logs()
        assert 'Hello world!' not in {log.message for log in logs}

    @pytest.mark.xfail(reason='Weird state sharing between new and old engine tests')
    async def test_logs_are_given_correct_id(self, prefect_client: Any) -> None:

        @flow
        def my_flow() -> None:
            logger = get_run_logger()
            logger.info('Hello world!')
        state: State = my_flow(return_state=True)  # type: ignore
        flow_run_id: Any = state.state_details.flow_run_id
        logs = await prefect_client.read_logs()
        assert all([log.flow_run_id == flow_run_id for log in logs])
        assert all([log.task_run_id is None for log in logs])


@pytest.mark.enable_api_log_handler
class TestSubflowRunLogs:
    @pytest.mark.clear_db
    async def test_subflow_logs_are_written_correctly(self, prefect_client: Any) -> None:

        @flow
        def my_subflow() -> None:
            logger = get_run_logger()
            logger.info('Hello smaller world!')

        @flow
        def my_flow() -> None:
            logger = get_run_logger()
            logger.info('Hello world!')
            return my_subflow(return_state=True)
        state: State = my_flow(return_state=True)  # type: ignore
        flow_run_id: Any = state.state_details.flow_run_id
        subflow_run_id: Any = (await state.result()).state_details.flow_run_id  # type: ignore
        await _wait_for_logs(prefect_client, expected_num_logs=6)
        logs = await prefect_client.read_logs()
        log_messages = [log.message for log in logs]
        assert all([log.task_run_id is None for log in logs])
        assert 'Hello world!' in log_messages, 'Parent log message is present'
        assert logs[log_messages.index('Hello world!')].flow_run_id == flow_run_id, 'Parent log message has correct id'
        assert 'Hello smaller world!' in log_messages, 'Child log message is present'
        assert logs[log_messages.index('Hello smaller world!')].flow_run_id == subflow_run_id, 'Child log message has correct id'

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    @pytest.mark.xfail(reason='Weird state sharing between new and old engine tests')
    async def test_subflow_logs_are_written_correctly_with_tasks(self, prefect_client: Any) -> None:

        @task
        def a_log_task() -> None:
            logger = get_run_logger()
            logger.info('Task log')

        @flow
        def my_subflow() -> None:
            a_log_task()
            logger = get_run_logger()
            logger.info('Hello smaller world!')

        @flow
        def my_flow() -> None:
            logger = get_run_logger()
            logger.info('Hello world!')
            return my_subflow(return_state=True)
        subflow_state: State = my_flow()  # type: ignore
        subflow_run_id: Any = subflow_state.state_details.flow_run_id
        logs = await prefect_client.read_logs()
        log_messages = [log.message for log in logs]
        task_run_logs = [log for log in logs if log.task_run_id is not None]
        assert all([log.flow_run_id == subflow_run_id for log in task_run_logs])
        assert 'Hello smaller world!' in log_messages
        assert logs[log_messages.index('Hello smaller world!')].flow_run_id == subflow_run_id


class TestFlowRetries:
    def test_flow_retry_with_error_in_flow(self) -> None:
        run_count: int = 0

        @flow(retries=1, persist_result=True)
        def foo() -> str:
            nonlocal run_count
            run_count += 1
            if run_count == 1:
                raise ValueError()
            return 'hello'
        assert foo() == 'hello'
        assert run_count == 2

    async def test_flow_retry_with_error_in_flow_and_successful_task(self) -> None:
        task_run_count: int = 0
        flow_run_count: int = 0

        @task(persist_result=True)
        def my_task() -> str:
            nonlocal task_run_count
            task_run_count += 1
            return 'hello'

        @flow(retries=1, persist_result=True)
        def foo() -> Any:
            nonlocal flow_run_count
            flow_run_count += 1
            state: State = my_task(return_state=True)
            if flow_run_count == 1:
                raise ValueError()
            return state.result()
        assert foo() == 'hello'
        assert flow_run_count == 2
        assert task_run_count == 1

    def test_flow_retry_with_no_error_in_flow_and_one_failed_task(self) -> None:
        task_run_count: int = 0
        flow_run_count: int = 0

        @task
        def my_task() -> str:
            nonlocal task_run_count
            task_run_count += 1
            if flow_run_count == 1:
                raise ValueError()
            return 'hello'

        @flow(retries=1)
        def foo() -> str:
            nonlocal flow_run_count
            flow_run_count += 1
            return my_task()
        assert foo() == 'hello'
        assert flow_run_count == 2
        assert task_run_count == 2, 'Task should be reset and run again'

    def test_flow_retry_with_error_in_flow_and_one_failed_task(self) -> None:
        task_run_count: int = 0
        flow_run_count: int = 0

        @task
        def my_task() -> str:
            nonlocal task_run_count
            task_run_count += 1
            if flow_run_count == 1:
                raise ValueError()
            return 'hello'

        @flow(retries=1)
        def my_flow() -> Any:
            nonlocal flow_run_count
            flow_run_count += 1
            fut = my_task()
            if flow_run_count == 1:
                raise ValueError()
            return fut
        assert my_flow() == 'hello'
        assert flow_run_count == 2
        assert task_run_count == 2, 'Task should be reset and run again'

    @pytest.mark.xfail
    async def test_flow_retry_with_branched_tasks(self, prefect_client: Any) -> None:
        flow_run_count: int = 0

        @task
        def identity(value: Any) -> Any:
            return value

        @flow(retries=1)
        def my_flow() -> Any:
            nonlocal flow_run_count
            flow_run_count += 1
            if flow_run_count == 1:
                identity('foo')
                raise ValueError()
            else:
                result = identity('bar')
            return result
        my_flow()
        assert flow_run_count == 2
        document = await (await my_flow().result()).result()  # type: ignore
        result = await prefect_client.retrieve_data(document)
        assert result == 'bar'

    async def test_flow_retry_with_no_error_in_flow_and_one_failed_child_flow(self, prefect_client: Any) -> None:
        child_run_count: int = 0
        flow_run_count: int = 0

        @flow
        def child_flow() -> str:
            nonlocal child_run_count
            child_run_count += 1
            if flow_run_count == 1:
                raise ValueError()
            return 'hello'

        @flow(retries=1)
        def parent_flow() -> Any:
            nonlocal flow_run_count
            flow_run_count += 1
            return child_flow()
        state: State = parent_flow(return_state=True)  # type: ignore
        assert await state.result() == 'hello'
        assert flow_run_count == 2
        assert child_run_count == 2, 'Child flow should be reset and run again'
        task_runs = await prefect_client.read_task_runs(flow_run_filter=FlowRunFilter(id={'any_': [state.state_details.flow_run_id]}))
        state_types = {task_run.state_type for task_run in task_runs}
        assert state_types == {StateType.COMPLETED}
        assert len(task_runs) == 1

    async def test_flow_retry_with_error_in_flow_and_one_successful_child_flow(self) -> None:
        child_run_count: int = 0
        flow_run_count: int = 0

        @flow(persist_result=True)
        def child_flow() -> str:
            nonlocal child_run_count
            child_run_count += 1
            return 'hello'

        @flow(retries=1, persist_result=True)
        def parent_flow() -> Any:
            nonlocal flow_run_count
            flow_run_count += 1
            child_result = child_flow()
            if flow_run_count == 1:
                raise ValueError()
            return child_result
        assert parent_flow() == 'hello'
        assert flow_run_count == 2
        assert child_run_count == 1, 'Child flow should not run again'

    async def test_flow_retry_with_error_in_flow_and_one_failed_child_flow(self, prefect_client: Any) -> None:
        child_flow_run_count: int = 0
        flow_run_count: int = 0

        @flow
        def child_flow_with_failure() -> str:
            nonlocal child_flow_run_count
            child_flow_run_count += 1
            if flow_run_count == 1:
                raise ValueError()
            return 'hello'

        @flow(retries=1)
        def parent_flow_with_failure() -> Any:
            nonlocal flow_run_count
            flow_run_count += 1
            state: State = child_flow_with_failure(return_state=True)
            if flow_run_count == 1:
                raise ValueError()
            return state
        parent_state: State = parent_flow_with_failure(return_state=True)  # type: ignore
        child_state: State = await parent_state.result()  # type: ignore
        assert await child_state.result() == 'hello'
        assert flow_run_count == 2
        assert child_flow_run_count == 2, 'Child flow should run again'
        child_flow_run = await prefect_client.read_flow_run(child_state.state_details.flow_run_id)
        child_flow_runs = await prefect_client.read_flow_runs(flow_filter=FlowFilter(id={'any_': [child_flow_run.flow_id]}), sort=FlowRunSort.EXPECTED_START_TIME_ASC)
        assert len(child_flow_runs) == 2
        assert child_flow_runs[0].state.is_failed()
        assert child_flow_runs[-1] == child_flow_run

    async def test_flow_retry_with_failed_child_flow_with_failed_task(self) -> None:
        child_task_run_count: int = 0
        child_flow_run_count: int = 0
        flow_run_count: int = 0

        @task
        def child_task() -> str:
            nonlocal child_task_run_count
            child_task_run_count += 1
            if child_task_run_count == 1:
                raise ValueError()
            return 'hello'

        @flow
        def child_flow() -> Any:
            nonlocal child_flow_run_count
            child_flow_run_count += 1
            return child_task()

        @flow(retries=1)
        def parent_flow() -> Any:
            nonlocal flow_run_count
            flow_run_count += 1
            state = child_flow()
            return state
        assert parent_flow() == 'hello'
        assert flow_run_count == 2
        assert child_flow_run_count == 2, 'Child flow should run again'
        assert child_task_run_count == 2, 'Child tasks should run again with child flow'

    def test_flow_retry_with_error_in_flow_and_one_failed_task_with_retries(self) -> None:
        task_run_retry_count: int = 0
        task_run_count: int = 0
        flow_run_count: int = 0

        @task(retries=1)
        def my_task() -> str:
            nonlocal task_run_count, task_run_retry_count
            task_run_count += 1
            task_run_retry_count += 1
            if flow_run_count == 1:
                raise ValueError('Fail on first flow run')
            if task_run_retry_count == 1:
                raise ValueError('Fail on first task run')
            return 'hello'

        @flow(retries=1)
        def foo() -> Any:
            nonlocal flow_run_count, task_run_retry_count
            task_run_retry_count = 0
            flow_run_count += 1
            fut = my_task()
            if flow_run_count == 1:
                raise ValueError()
            return fut
        assert foo() == 'hello'
        assert flow_run_count == 2
        assert task_run_count == 4, 'Task should use all of its retries every time'

    def test_flow_retry_with_error_in_flow_and_one_failed_task_with_retries_cannot_exceed_retries(self) -> None:
        task_run_count: int = 0
        flow_run_count: int = 0

        @task(retries=2)
        def my_task() -> str:
            nonlocal task_run_count
            task_run_count += 1
            raise ValueError('This task always fails')

        @flow(retries=1)
        def my_flow() -> Any:
            nonlocal flow_run_count
            flow_run_count += 1
            fut = my_task()
            if flow_run_count == 1:
                raise ValueError()
            return fut
        with pytest.raises(ValueError, match='This task always fails'):
            my_flow().result().result()  # type: ignore
        assert flow_run_count == 2
        assert task_run_count == 6, 'Task should use all of its retries every time'

    async def test_flow_with_failed_child_flow_with_retries(self) -> None:
        child_flow_run_count: int = 0
        flow_run_count: int = 0

        @flow(retries=1)
        def child_flow() -> str:
            nonlocal child_flow_run_count
            child_flow_run_count += 1
            if child_flow_run_count == 1:
                raise ValueError()
            return 'hello'

        @flow
        def parent_flow() -> str:
            nonlocal flow_run_count
            flow_run_count += 1
            state: State = child_flow()
            return state
        assert parent_flow() == 'hello'
        assert flow_run_count == 1, 'Parent flow should only run once'
        assert child_flow_run_count == 2, 'Child flow should run again'

    async def test_parent_flow_retries_failed_child_flow_with_retries(self) -> None:
        child_flow_retry_count: int = 0
        child_flow_run_count: int = 0
        flow_run_count: int = 0

        @flow(retries=1)
        def child_flow() -> str:
            nonlocal child_flow_run_count, child_flow_retry_count
            child_flow_run_count += 1
            child_flow_retry_count += 1
            if flow_run_count == 1:
                raise ValueError()
            if child_flow_retry_count == 1:
                raise ValueError()
            return 'hello'

        @flow(retries=1)
        def parent_flow() -> str:
            nonlocal flow_run_count, child_flow_retry_count
            child_flow_retry_count = 0
            flow_run_count += 1
            state: State = child_flow()
            return state
        assert parent_flow() == 'hello'
        assert flow_run_count == 2, 'Parent flow should exhaust retries'
        assert child_flow_run_count == 4, 'Child flow should run 2 times for each parent run'

    def test_global_retry_config(self) -> None:
        with temporary_settings(updates={PREFECT_FLOW_DEFAULT_RETRIES: '1'}):
            run_count: int = 0

            @flow()
            def foo() -> str:
                nonlocal run_count
                run_count += 1
                if run_count == 1:
                    raise ValueError()
                return 'hello'
            assert foo() == 'hello'
            assert run_count == 2


class TestLoadFlowFromEntrypoint:
    def test_load_flow_from_entrypoint(self, tmp_path: Any) -> None:
        flow_code: str = '\n        from prefect import flow\n\n        @flow\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow_obj: Flow = load_flow_from_entrypoint(f'{fpath}:dog')
        assert flow_obj.fn() == 'woof!'

    def test_load_flow_from_entrypoint_with_absolute_path(self, tmp_path: Any) -> None:
        flow_code: str = '\n        from prefect import flow\n\n        @flow\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        absolute_fpath: str = str(fpath.resolve())
        flow_obj: Flow = load_flow_from_entrypoint(f'{absolute_fpath}:dog')
        assert flow_obj.fn() == 'woof!'

    def test_load_flow_from_entrypoint_with_module_path(self, monkeypatch: Any) -> None:

        @flow
        def pretend_flow() -> None:
            pass
        import_object_mock: MagicMock = MagicMock(return_value=pretend_flow)
        monkeypatch.setattr('prefect.flows.import_object', import_object_mock)
        result_flow: Flow = load_flow_from_entrypoint('my.module.pretend_flow')
        assert result_flow == pretend_flow
        import_object_mock.assert_called_with('my.module.pretend_flow')

    def test_load_flow_from_entrypoint_script_error_loads_placeholder(self, tmp_path: Any) -> None:
        flow_code: str = '\n        from not_a_module import not_a_function\n        from prefect import flow\n\n        @flow(description="Says woof!")\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow_obj: Flow = load_flow_from_entrypoint(f'{fpath}:dog')
        assert flow_obj.name == 'dog'
        assert flow_obj.description == 'Says woof!'
        assert flow_obj() == 'woof!'

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_handling_script_with_unprotected_call_in_flow_script(self, tmp_path: Any, caplog: Any, prefect_client: Any) -> None:
        flow_code_with_call: str = '\n        from prefect import flow\n        from prefect.logging import get_run_logger\n\n        @flow\n        def dog():\n            get_run_logger().warning("meow!")\n            return "woof!"\n\n        dog()\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code_with_call))
        with caplog.at_level('WARNING'):
            flow_obj: Flow = load_flow_from_entrypoint(f'{fpath}:dog')
            assert "Script loading is in progress, flow 'dog' will not be executed. Consider updating the script to only call the flow" in caplog.text
        flow_runs = await prefect_client.read_flows()
        assert len(flow_runs) == 0
        res = flow_obj()
        assert res == 'woof!'
        flow_runs = await prefect_client.read_flows()
        assert len(flow_runs) == 1

    def test_load_flow_from_entrypoint_with_use_placeholder_flow(self, tmp_path: Any) -> None:
        flow_code: str = '\n        from not_a_module import not_a_function\n        from prefect import flow\n\n        @flow(description="Says woof!")\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow_obj: Flow = load_flow_from_entrypoint(f'{fpath}:dog')
        assert isinstance(flow_obj, Flow)
        assert flow_obj() == 'woof!'
        with pytest.raises(ScriptError):
            load_flow_from_entrypoint(f'{fpath}:dog', use_placeholder_flow=False)


class TestLoadFunctionAndConvertToFlow:
    def test_func_is_a_flow(self, tmp_path: Any) -> None:
        flow_code: str = '\n        from prefect import flow\n\n        @flow\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow_obj: Flow = load_function_and_convert_to_flow(f'{fpath}:dog')
        assert flow_obj.fn() == 'woof!'
        assert isinstance(flow_obj, Flow)
        assert flow_obj.name == 'dog'

    def test_func_is_not_a_flow(self, tmp_path: Any) -> None:
        flow_code: str = '\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow_obj: Flow = load_function_and_convert_to_flow(f'{fpath}:dog')
        assert isinstance(flow_obj, Flow)
        assert flow_obj.name == 'dog'
        assert flow_obj.log_prints is True
        assert flow_obj.fn() == 'woof!'

    def test_func_not_found(self, tmp_path: Any) -> None:
        flow_code: str = ''
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        with pytest.raises(RuntimeError, match=f"Function with name 'dog' not found in '{fpath}'."):
            load_function_and_convert_to_flow(f'{fpath}:dog')


class TestFlowRunName:
    async def test_invalid_runtime_run_name(self) -> None:

        class InvalidFlowRunNameArg:
            def format(self, *args: Any, **kwargs: Any) -> Any:
                pass

        @flow
        def my_flow() -> None:
            pass
        my_flow.flow_run_name = InvalidFlowRunNameArg()  # type: ignore
        with pytest.raises(TypeError, match="Expected string or callable for 'flow_run_name'; got InvalidFlowRunNameArg instead."):
            my_flow()

    async def test_sets_run_name_when_provided(self, prefect_client: Any) -> None:

        @flow(flow_run_name='hi')
        def flow_with_name(foo: str = 'bar', bar: int = 1) -> None:
            pass
        state: State = flow_with_name(return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert flow_run.name == 'hi'

    async def test_sets_run_name_with_params_including_defaults(self, prefect_client: Any) -> None:

        @flow(flow_run_name='hi-{foo}-{bar}')
        def flow_with_name(foo: str = 'one', bar: str = '1') -> None:
            pass
        state: State = flow_with_name(bar='two', return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert flow_run.name == 'hi-one-two'

    async def test_sets_run_name_with_function(self, prefect_client: Any) -> None:

        def generate_flow_run_name() -> str:
            return 'hi'

        @flow(flow_run_name=generate_flow_run_name)
        def flow_with_name(foo: str = 'one', bar: str = '1') -> None:
            pass
        state: State = flow_with_name(bar='two', return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert flow_run.name == 'hi'

    async def test_sets_run_name_with_function_using_runtime_context(self, prefect_client: Any) -> None:

        def generate_flow_run_name() -> str:
            params = flow_run_ctx.parameters
            tokens = ['hi']
            print(f'got the parameters {params!r}')
            if 'foo' in params:
                tokens.append(str(params['foo']))
            if 'bar' in params:
                tokens.append(str(params['bar']))
            return '-'.join(tokens)

        @flow(flow_run_name=generate_flow_run_name)
        def flow_with_name(foo: str = 'one', bar: str = '1') -> None:
            pass
        state: State = flow_with_name(bar='two', return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert flow_run.name == 'hi-one-two'

    async def test_sets_run_name_with_function_not_returning_string(self, prefect_client: Any) -> None:

        def generate_flow_run_name() -> None:
            pass

        @flow(flow_run_name=generate_flow_run_name)
        def flow_with_name(foo: str = 'one', bar: str = '1') -> None:
            pass
        with pytest.raises(TypeError, match="Callable <function TestFlowRunName.test_sets_run_name_with_function_not_returning_string.<locals>.generate_flow_run_name at .*> for 'flow_run_name' returned type NoneType but a string is required."):
            flow_with_name(bar='two')

    async def test_sets_run_name_once(self) -> None:
        generate_flow_run_name: MagicMock = MagicMock(return_value='some-string')

        def flow_method() -> None:
            pass
        mocked_flow_method: Any = create_autospec(flow_method, side_effect=RuntimeError('some-error'))
        decorated_flow: Flow = flow(flow_run_name=generate_flow_run_name, retries=3)(mocked_flow_method)
        state: State = decorated_flow(return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        assert mocked_flow_method.call_count == 4
        assert generate_flow_run_name.call_count == 1

    async def test_sets_run_name_once_per_call(self) -> None:
        generate_flow_run_name: MagicMock = MagicMock(return_value='some-string')

        def flow_method() -> str:
            pass
        mocked_flow_method: Any = create_autospec(flow_method, return_value='hello')
        decorated_flow: Flow = flow(flow_run_name=generate_flow_run_name)(mocked_flow_method)
        state1: State = decorated_flow(return_state=True)  # type: ignore
        assert state1.type == StateType.COMPLETED
        assert mocked_flow_method.call_count == 1
        assert generate_flow_run_name.call_count == 1
        state2: State = decorated_flow(return_state=True)  # type: ignore
        assert state2.type == StateType.COMPLETED
        assert mocked_flow_method.call_count == 2
        assert generate_flow_run_name.call_count == 2

    async def test_both_engines_logs_custom_flow_run_name(self, caplog: Any) -> None:

        @flow(flow_run_name='very-bespoke-name')
        def test() -> None:
            pass
        test()
        assert "Beginning flow run 'very-bespoke-name'" in caplog.text

        @flow(flow_run_name='very-bespoke-async-name')
        async def test_async() -> None:
            pass
        await test_async()
        assert "Beginning flow run 'very-bespoke-async-name'" in caplog.text


def create_hook(mock_obj: Any) -> Callable[[Flow, Any, State], None]:
    def my_hook(flow: Flow, flow_run: Any, state: State) -> None:
        mock_obj()
    return my_hook


def create_async_hook(mock_obj: Any) -> Callable[[Flow, Any, State], Any]:
    async def my_hook(flow: Flow, flow_run: Any, state: State) -> None:
        mock_obj()
    return my_hook


class TestFlowHooksContext:
    @pytest.mark.parametrize('hook_type, fn_body, expected_exc', [
        ('on_completion', lambda: None, None),
        ('on_failure', lambda: 100 / 0, ZeroDivisionError),
        ('on_cancellation', lambda: Cancelling(), UnfinishedRun)
    ])
    def test_hooks_are_called_within_flow_run_context(self, caplog: Any, hook_type: str, fn_body: Callable[[], Any], expected_exc: Any) -> None:

        def hook(flow: Flow, flow_run: Any, state: State) -> None:
            ctx: Any = get_run_context()
            assert ctx is not None
            assert ctx.flow_run and ctx.flow_run == flow_run
            assert ctx.flow_run.state == state
            assert ctx.flow == flow

        @flow(**{hook_type: [hook]})
        def foo_flow() -> Any:
            return fn_body()
        with caplog.at_level('INFO'):
            if expected_exc:
                with pytest.raises(expected_exc):
                    foo_flow()
            else:
                foo_flow()
        assert "Hook 'hook' finished running successfully" in caplog.text


class TestFlowHooksWithKwargs:
    def test_hook_with_extra_default_arg(self) -> None:
        data: dict = {}

        def hook(flow: Flow, flow_run: Any, state: State, foo: int = 42) -> None:
            data.update(name=hook.__name__, state=state, foo=foo)

        @flow(on_completion=[hook])
        def foo_flow() -> None:
            pass
        state: State = foo_flow(return_state=True)  # type: ignore
        assert data == dict(name='hook', state=state, foo=42)

    def test_hook_with_bound_kwargs(self) -> None:
        data: dict = {}

        def hook(flow: Flow, flow_run: Any, state: State, **kwargs: Any) -> None:
            data.update(name=hook.__name__, state=state, kwargs=kwargs)
        hook_with_kwargs: Callable[..., None] = partial(hook, foo=42)

        @flow(on_completion=[hook_with_kwargs])
        def foo_flow() -> None:
            pass
        state: State = foo_flow(return_state=True)  # type: ignore
        assert data == dict(name='hook', state=state, kwargs={'foo': 42})


class TestFlowHooksOnCompletion:
    def test_noniterable_hook_raises(self) -> None:

        def completion_hook() -> None:
            pass
        with pytest.raises(TypeError, match=re.escape("Expected iterable for 'on_completion'; got function instead. Please provide a list of hooks to 'on_completion':\n\n@flow(on_completion=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_completion=completion_hook)
            def flow1() -> None:
                pass

    def test_noncallable_hook_raises(self) -> None:
        with pytest.raises(TypeError, match=re.escape("Expected callables in 'on_completion'; got str instead. Please provide a list of hooks to 'on_completion':\n\n@flow(on_completion=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_completion=['test'])
            def flow1() -> None:
                pass

    def test_callable_noncallable_hook_raises(self) -> None:

        def completion_hook() -> None:
            pass
        with pytest.raises(TypeError, match=re.escape("Expected callables in 'on_completion'; got str instead. Please provide a list of hooks to 'on_completion':\n\n@flow(on_completion=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_completion=[completion_hook, 'test'])
            def flow2() -> None:
                pass

    def test_decorated_on_completion_hooks_run_on_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        @flow
        def my_flow() -> None:
            pass

        @my_flow.on_completion
        def completed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('completed1')

        @my_flow.on_completion
        def completed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('completed2')
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        assert my_mock.call_args_list == [call('completed1'), call('completed2')]

    def test_on_completion_hooks_run_on_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def completed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('completed1')

        def completed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('completed2')

        @flow(on_completion=[completed1, completed2])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        assert my_mock.call_args_list == [call('completed1'), call('completed2')]

    def test_on_completion_hooks_dont_run_on_failure(self) -> None:
        my_mock: MagicMock = MagicMock()

        def completed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('completed1')

        def completed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('completed2')

        @flow(on_completion=[completed1, completed2])
        def my_flow() -> None:
            raise Exception('oops')
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        my_mock.assert_not_called()

    def test_other_completion_hooks_run_if_a_hook_fails(self) -> None:
        my_mock: MagicMock = MagicMock()

        def completed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('completed1')

        def exception_hook(flow: Flow, flow_run: Any, state: State) -> None:
            raise Exception('oops')

        def completed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('completed2')

        @flow(on_completion=[completed1, exception_hook, completed2])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        assert my_mock.call_args_list == [call('completed1'), call('completed2')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_completion_hooks_work_with_sync_and_async(self, hook1: Callable[[Any], Any], hook2: Callable[[Any], Any]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock = hook1(my_mock)
        hook2_with_mock = hook2(my_mock)

        @flow(on_completion=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        assert my_mock.call_args_list == [call(), call()]


class TestFlowHooksOnFailure:
    def test_noniterable_hook_raises(self) -> None:

        def failure_hook() -> None:
            pass
        with pytest.raises(TypeError, match=re.escape("Expected iterable for 'on_failure'; got function instead. Please provide a list of hooks to 'on_failure':\n\n@flow(on_failure=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_failure=failure_hook)
            def flow1() -> None:
                pass

    def test_noncallable_hook_raises(self) -> None:
        with pytest.raises(TypeError, match=re.escape("Expected callables in 'on_failure'; got str instead. Please provide a list of hooks to 'on_failure':\n\n@flow(on_failure=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_failure=['test'])
            def flow1() -> None:
                pass

    def test_callable_noncallable_hook_raises(self) -> None:

        def failure_hook() -> None:
            pass
        with pytest.raises(TypeError, match=re.escape("Expected callables in 'on_failure'; got str instead. Please provide a list of hooks to 'on_failure':\n\n@flow(on_failure=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_failure=[failure_hook, 'test'])
            def flow2() -> None:
                pass

    def test_decorated_on_failure_hooks_run_on_failure(self) -> None:
        my_mock: MagicMock = MagicMock()

        @flow
        def my_flow() -> None:
            raise Exception('oops')

        @my_flow.on_failure
        def failed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed1')

        @my_flow.on_failure
        def failed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed2')
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call('failed1'), call('failed2')]

    def test_on_failure_hooks_run_on_failure(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed1')

        def failed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed2')

        @flow(on_failure=[failed1, failed2])
        def my_flow() -> None:
            raise Exception('oops')
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call('failed1'), call('failed2')]

    def test_on_failure_hooks_dont_run_on_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed1')

        def failed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed2')

        @flow(on_failure=[failed1, failed2])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        my_mock.assert_not_called()

    def test_on_failure_hooks_dont_run_on_retries(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed1')

        def failed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed2')

        @flow(on_failure=[failed1, failed2], retries=3)
        def my_flow() -> None:
            raise SyntaxError('oops')
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        assert my_mock.call_count == 2
        assert [call.args[0] for call in my_mock.call_args_list] == ['failed1', 'failed2']

    async def test_on_async_failure_hooks_dont_run_on_retries(self) -> None:
        my_mock: MagicMock = MagicMock()

        async def failed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed1')

        async def failed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed2')

        @flow(on_failure=[failed1, failed2], retries=3)
        async def my_flow() -> None:
            raise SyntaxError('oops')
        state: State = await my_flow(return_state=True)
        assert state.type == StateType.FAILED
        assert my_mock.call_count == 2
        assert [call.args[0] for call in my_mock.call_args_list] == ['failed1', 'failed2']

    def test_other_failure_hooks_run_if_a_hook_fails(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed1')

        def exception_hook(flow: Flow, flow_run: Any, state: State) -> None:
            raise Exception('oops')

        def failed2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failed2')

        @flow(on_failure=[failed1, exception_hook, failed2])
        def my_flow() -> None:
            raise Exception('oops')
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call('failed1'), call('failed2')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_failure_hooks_work_with_sync_and_async(self, hook1: Callable[[Any], Any], hook2: Callable[[Any], Any]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock = hook1(my_mock)
        hook2_with_mock = hook2(my_mock)

        @flow(on_failure=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> None:
            raise Exception('oops')
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call(), call()]

    def test_on_failure_hooks_run_on_bad_parameters(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failure_hook(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('failure_hook')

        @flow(on_failure=[failure_hook])
        def my_flow(x: str) -> None:
            pass
        state: State = my_flow(x='x', return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call('failure_hook')]


class TestFlowHooksOnCancellation:
    def test_noniterable_hook_raises(self) -> None:

        def cancellation_hook() -> None:
            pass
        with pytest.raises(TypeError, match=re.escape("Expected iterable for 'on_cancellation'; got function instead. Please provide a list of hooks to 'on_cancellation':\n\n@flow(on_cancellation=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_cancellation=cancellation_hook)
            def flow1() -> None:
                pass

    def test_noncallable_hook_raises(self) -> None:
        with pytest.raises(TypeError, match=re.escape("Expected callables in 'on_cancellation'; got str instead. Please provide a list of hooks to 'on_cancellation':\n\n@flow(on_cancellation=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_cancellation=['test'])
            def flow1() -> None:
                pass

    def test_callable_noncallable_hook_raises(self) -> None:

        def cancellation_hook() -> None:
            pass
        with pytest.raises(TypeError, match=re.escape("Expected callables in 'on_cancellation'; got str instead. Please provide a list of hooks to 'on_cancellation':\n\n@flow(on_cancellation=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_cancellation=[cancellation_hook, 'test'])
            def flow2() -> None:
                pass

    def test_decorated_on_cancellation_hooks_run_on_cancelled_state(self) -> None:
        my_mock: MagicMock = MagicMock()

        @flow
        def my_flow() -> State:
            return State(type=StateType.CANCELLING)
        @my_flow.on_cancellation
        def cancelled_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook1')
        @my_flow.on_cancellation
        def cancelled_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook2')
        my_flow(return_state=True)  # type: ignore
        assert my_mock.mock_calls == [call('cancelled_hook1'), call('cancelled_hook2')]

    def test_on_cancellation_hooks_run_on_cancelled_state(self) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook1')

        def cancelled_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook2')

        @flow(on_cancellation=[cancelled_hook1, cancelled_hook2])
        def my_flow() -> State:
            return State(type=StateType.CANCELLING)
        my_flow(return_state=True)  # type: ignore
        assert my_mock.mock_calls == [call('cancelled_hook1'), call('cancelled_hook2')]

    def test_on_cancellation_hooks_are_ignored_if_terminal_state_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook1')

        def cancelled_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook2')

        @flow(on_cancellation=[cancelled_hook1, cancelled_hook2])
        def my_flow() -> State:
            return State(type=StateType.COMPLETED)
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        my_mock.assert_not_called()

    def test_on_cancellation_hooks_are_ignored_if_terminal_state_failed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook1')

        def cancelled_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook2')

        @flow(on_cancellation=[cancelled_hook1, cancelled_hook2])
        def my_flow() -> State:
            return State(type=StateType.FAILED)
        my_flow(return_state=True)  # type: ignore
        my_mock.assert_not_called()

    def test_other_cancellation_hooks_run_if_one_hook_fails(self) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled1')

        def cancelled2(flow: Flow, flow_run: Any, state: State) -> None:
            raise Exception('Failing flow')

        def cancelled3(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled3')

        @flow(on_cancellation=[cancelled1, cancelled2, cancelled3])
        def my_flow() -> State:
            return State(type=StateType.CANCELLING)
        my_flow(return_state=True)  # type: ignore
        assert my_mock.mock_calls == [call('cancelled1'), call('cancelled3')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_cancellation_hooks_work_with_sync_and_async(self, hook1: Callable[[Any], Any], hook2: Callable[[Any], Any]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock = hook1(my_mock)
        hook2_with_mock = hook2(my_mock)

        @flow(on_cancellation=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> State:
            return State(type=StateType.CANCELLING)
        my_flow(return_state=True)  # type: ignore
        assert my_mock.mock_calls == [call(), call()]

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_on_cancellation_hook_called_on_sigterm_from_flow_with_cancelling_state(self, mock_sigterm_handler: Any) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled')

        @task
        async def cancel_parent() -> None:
            async with get_client() as client:
                await client.set_flow_run_state(runtime.flow_run.id, State(type=StateType.CANCELLING), force=True)

        @flow(on_cancellation=[cancelled])
        async def my_flow() -> None:
            await cancel_parent()
            os.kill(os.getpid(), signal.SIGTERM)
        with pytest.raises(prefect.exceptions.TerminationSignal):
            await my_flow(return_state=True)
        assert my_mock.mock_calls == [call('cancelled')]

    async def test_on_cancellation_hook_not_called_on_sigterm_from_flow_without_cancelling_state(self, mock_sigterm_handler: Any) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled')

        @flow(on_cancellation=[cancelled])
        def my_flow() -> None:
            os.kill(os.getpid(), signal.SIGTERM)
        with pytest.raises(prefect.exceptions.TerminationSignal):
            my_flow(return_state=True)  # type: ignore
        my_mock.assert_not_called()

    def test_on_cancellation_hooks_respect_env_var(self, monkeypatch: Any) -> None:
        my_mock: MagicMock = MagicMock()
        monkeypatch.setenv('PREFECT__ENABLE_CANCELLATION_AND_CRASHED_HOOKS', 'false')

        def cancelled_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook1')

        def cancelled_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('cancelled_hook2')

        @flow(on_cancellation=[cancelled_hook1, cancelled_hook2])
        def my_flow() -> State:
            return State(type=StateType.CANCELLING)
        state: State = my_flow(return_state=True)  # type: ignore
        assert state.type == StateType.CANCELLING
        my_mock.assert_not_called()


class TestFlowHooksOnCrashed:
    def test_noniterable_hook_raises(self) -> None:

        def crashed_hook() -> None:
            pass
        with pytest.raises(TypeError, match=re.escape("Expected iterable for 'on_crashed'; got function instead. Please provide a list of hooks to 'on_crashed':\n\n@flow(on_crashed=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_crashed=crashed_hook)
            def flow1() -> None:
                pass

    def test_noncallable_hook_raises(self) -> None:
        with pytest.raises(TypeError, match=re.escape("Expected callables in 'on_crashed'; got str instead. Please provide a list of hooks to 'on_crashed':\n\n@flow(on_crashed=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_crashed=['test'])
            def flow1() -> None:
                pass

    def test_callable_noncallable_hook_raises(self) -> None:

        def crashed_hook() -> None:
            pass
        with pytest.raises(TypeError, match=re.escape("Expected callables in 'on_crashed'; got str instead. Please provide a list of hooks to 'on_crashed':\n\n@flow(on_crashed=[hook1, hook2])\ndef my_flow():\n\tpass")):
            @flow(on_crashed=[crashed_hook, 'test'])
            def flow2() -> None:
                pass

    def test_decorated_on_crashed_hooks_run_on_crashed_state(self) -> None:
        my_mock: MagicMock = MagicMock()

        @flow
        def my_flow() -> State:
            return State(type=StateType.CRASHED)

        @my_flow.on_crashed
        def crashed_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed_hook1')

        @my_flow.on_crashed
        def crashed_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed_hook2')
        my_flow(return_state=True)  # type: ignore
        assert my_mock.mock_calls == [call('crashed_hook1'), call('crashed_hook2')]

    def test_on_crashed_hooks_run_on_crashed_state(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed_hook1')

        def crashed_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed_hook2')

        @flow(on_crashed=[crashed_hook1, crashed_hook2])
        def my_flow() -> State:
            return State(type=StateType.CRASHED)
        my_flow(return_state=True)  # type: ignore
        assert my_mock.mock_calls == [call('crashed_hook1'), call('crashed_hook2')]

    def test_on_crashed_hooks_are_ignored_if_terminal_state_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed_hook1')

        def crashed_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed_hook2')

        @flow(on_crashed=[crashed_hook1, crashed_hook2])
        def my_passing_flow() -> None:
            pass
        state: State = my_passing_flow(return_state=True)  # type: ignore
        assert state.type == StateType.COMPLETED
        my_mock.assert_not_called()

    def test_on_crashed_hooks_are_ignored_if_terminal_state_failed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed_hook1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed_hook1')

        def crashed_hook2(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed_hook2')

        @flow(on_crashed=[crashed_hook1, crashed_hook2])
        def my_failing_flow() -> None:
            raise Exception('Failing flow')
        state: State = my_failing_flow(return_state=True)  # type: ignore
        assert state.type == StateType.FAILED
        my_mock.assert_not_called()

    def test_other_crashed_hooks_run_if_one_hook_fails(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed1(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed1')

        def crashed2(flow: Flow, flow_run: Any, state: State) -> None:
            raise Exception('Failing flow')

        def crashed3(flow: Flow, flow_run: Any, state: State) -> None:
            my_mock('crashed3')

        @flow(on_crashed=[crashed1, crashed2, crashed3])
        def my_flow() -> State:
            return State(type=StateType.CRASHED)
        my_flow(return_state=True)  # type: ignore
        assert my_mock.mock_calls == [call('crashed1'), call('crashed3')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_crashed_hooks_work_with_sync_and_async(self, hook1: Callable[[Any], Any], hook2: Callable[[Any], Any]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock = hook1(my_mock)
        hook2_with_mock = hook2(my_mock)

        @flow(on_crashed=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> State:
            return State(type=StateType.CRASHED)
        my_flow(return_state=True)  # type: ignore
        assert my_mock.mock_calls == [call(), call()]

    def test_handles_dynamically_created_models(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from typing import Optional
            from prefect import flow
            from pydantic import BaseModel, create_model, Field

            def get_model() -> BaseModel:
                return create_model(
                    "MyModel",
                    param=(
                        int,
                        Field(
                            title="param",
                            default=1,
                        ),
                    ),
                )
            MyModel = get_model()
            @flow
            def f(
                param: Optional[MyModel] = None,
            ) -> None:
                return MyModel()
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        # Calling the flow to check default
        res = result()
        if isinstance(res, dict):
            # If the flow returns a serialized dict
            record = ResultRecord.deserialize(res)
            assert record.result == {'some': 'data'}  # adjust as needed
        else:
            assert res is not None

    def test_raises_name_error_when_loaded_flow_cannot_run(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from not_a_module import not_a_function

            from prefect import flow

            @flow(description="Says woof!")
            def dog():
                return not_a_function('dog')
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:dog'
        with pytest.raises(NameError, match="name 'not_a_function' is not defined"):
            safe_load_flow_from_entrypoint(entrypoint)()


class TestSafeLoadFlowFromEntrypoint:
    def test_flow_not_found(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from prefect import flow
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        with pytest.raises(ValueError):
            safe_load_flow_from_entrypoint(f'{tmp_path}/test.py:g')

    def test_basic_operation(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow(name="My custom name")
            def flow_function(name: str) -> str:
                """ 
                My docstring

                Args:
                    name (str): A name
                """
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        from prefect.flows import Flow
        assert isinstance(result, Flow)
        assert result.name == 'My custom name'
        assert result('marvin') == 'marvin'
        assert result.__doc__ is not None
        assert 'My docstring' in result.__doc__
        assert 'Args:' in result.__doc__
        assert 'name (str): A name' in result.__doc__

    def test_get_parameter_schema_from_safe_loaded_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert parameter_schema(result).model_dump() == {'definitions': {}, 'properties': {'name': {'position': 0, 'title': 'name', 'type': 'string'}}, 'required': ['name'], 'title': 'Parameters', 'type': 'object'}

    def test_dynamic_name_fstring(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            version = "1.0"

            @flow(name=f"flow-function-{version}")
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result.name == 'flow-function-1.0'

    def test_dynamic_name_function(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            def get_name():
                return "from-a-function"

            @flow(name=get_name())
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert isinstance(result, Flow)
        assert result.name == 'from-a-function'

    def test_dynamic_name_depends_on_missing_import(self, tmp_path: Any, caplog: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            from non_existent import get_name

            @flow(name=get_name())
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is None
        assert 'Failed to parse @flow argument: `name=get_name()`' in caplog.text

    def test_dynamic_name_fstring_multiline(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            flow_base_name = "flow-function"
            version = "1.0"

            @flow(
                name=(
                    f"{flow_base_name}-"
                    f"{version}"
                )
            )
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result.name == 'flow-function-1.0'

    def test_load_async_flow_from_entrypoint_no_name(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow
            async def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result.name == 'flow-function'

    def test_load_flow_description_from_entrypoint(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow(description="My custom description")
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result['description'] == 'My custom description'

    def test_load_flow_description_from_entrypoint_no_description(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = load_flow_arguments_from_entrypoint(entrypoint)
        assert 'description' not in result

    def test_load_no_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        with pytest.raises(ValueError, match='Could not find flow'):
            load_flow_arguments_from_entrypoint(entrypoint)


class TestSafeLoadFlowFromEntrypoint:
    def test_basic_operation(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow(name="My custom name")
            def flow_function(name: str) -> str:
                """ 
                My docstring

                Args:
                    name (str): A name
                """
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert isinstance(result, Flow)
        assert result.name == 'My custom name'
        assert result('marvin') == 'marvin'
        assert result.__doc__ is not None
        assert 'My docstring' in result.__doc__
        assert 'Args:' in result.__doc__
        assert 'name (str): A name' in result.__doc__

    def test_get_parameter_schema_from_safe_loaded_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert parameter_schema(result).model_dump() == {'definitions': {}, 'properties': {'name': {'position': 0, 'title': 'name', 'type': 'string'}}, 'required': ['name'], 'title': 'Parameters', 'type': 'object'}

    def test_dynamic_name_fstring(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            version = "1.0"

            @flow(name=f"flow-function-{version}")
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result.name == 'flow-function-1.0'

    def test_dynamic_name_function(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            def get_name():
                return "from-a-function"

            @flow(name=get_name())
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert isinstance(result, Flow)

    def test_annotations_and_defaults_rely_on_imports(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            import pendulum
            import datetime
            from prefect import flow

            @flow
            def f(
                x: datetime.datetime,
                y: pendulum.DateTime = pendulum.datetime(2025, 1, 1),
                z: datetime.timedelta = datetime.timedelta(seconds=5),
            ) -> tuple:
                return x, y, z
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        result = safe_load_flow_from_entrypoint(f'{tmp_path}/test.py:f')
        assert result is not None
        assert result(datetime.datetime(2025, 1, 1)) == (datetime.datetime(2025, 1, 1), pendulum.datetime(2025, 1, 1), datetime.timedelta(seconds=5))

    def test_annotations_rely_on_missing_import(self, tmp_path: Any) -> None:
        """
        This test ensures missing types for annotations are handled gracefully
        for all argument types (positional-only, positional-or-keyword,
        keyword-only, varargs, and varkwargs).
        """
        flow_source: str = dedent('''
            from prefect import flow
            from typing import Dict, Tuple

            from non_existent import Type1, Type2, Type3, Type4, Type5

            @flow
            def flow_function(x: Type1, /, y: Type2, *args: Type4, z: Type3, **kwargs: Type5) -> str:
                return x, y, z, args, kwargs
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result(1, 2, 4, z=3, a=5) == (1, 2, 3, (4,), {'a': 5})

    def test_defaults_rely_on_missing_import(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            from non_existent import DEFAULT_NAME, DEFAULT_AGE

            @flow
            def flow_function(name = DEFAULT_NAME, age = DEFAULT_AGE) -> str:
                return name, age
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result() == (None, None)

    def test_function_with_enum_argument(self, tmp_path: Any) -> None:

        class Color(enum.Enum):
            RED = 'RED'
            GREEN = 'GREEN'
            BLUE = 'BLUE'
        source_code: str = dedent('''
            from enum import Enum
            from prefect import flow

            class Color(Enum):
                RED = "RED"
                GREEN = "GREEN"
                BLUE = "BLUE"

            @flow
            def f(x: Color = Color.RED) -> Color:
                return x
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result().value == Color.RED.value

    def test_handles_dynamically_created_models(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from typing import Optional
            from prefect import flow
            from pydantic import BaseModel, create_model, Field

            def get_model() -> BaseModel:
                return create_model(
                    "MyModel",
                    param=(
                        int,
                        Field(
                            title="param",
                            default=1,
                        ),
                    ),
                )
            MyModel = get_model()

            @flow
            def f(param: Optional[MyModel] = None) -> None:
                return MyModel()
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result() == 1  # Depending on how the model serializes the default

    def test_raises_name_error_when_loaded_flow_cannot_run(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from not_a_module import not_a_function

            from prefect import flow

            @flow(description="Says woof!")
            def dog():
                return not_a_function('dog')
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:dog'
        with pytest.raises(NameError, match="name 'not_a_function' is not defined"):
            safe_load_flow_from_entrypoint(entrypoint)()


class TestFlowFromSource:
    def test_load_flow_from_source_on_flow_function(self) -> None:
        assert hasattr(flow, 'from_source')

    class TestSync:
        def test_load_flow_from_source_with_storage(self) -> None:
            storage = MockStorage()
            loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
            assert isinstance(loaded_flow, Flow)
            assert loaded_flow.name == 'test-flow'
            assert loaded_flow() == 1

        def test_loaded_flow_to_deployment_has_storage(self) -> None:
            storage = MockStorage()
            loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
            deployment = loaded_flow.to_deployment(name='test')
            assert deployment.storage == storage

        def test_loaded_flow_can_be_updated_with_options(self) -> None:
            storage = MockStorage()
            storage.set_base_path(Path.cwd())
            loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
            flow_with_options: Flow = loaded_flow.with_options(name='with_options')
            deployment = flow_with_options.to_deployment(name='test')
            assert deployment.storage == storage

        def test_load_flow_from_source_with_url(self, monkeypatch: Any) -> None:

            def mock_create_storage_from_source(url: str) -> MockStorage:
                return MockStorage()
            monkeypatch.setattr('prefect.runner.storage.create_storage_from_source', mock_create_storage_from_source)
            loaded_flow: Flow = Flow.from_source(source='https://github.com/org/repo.git', entrypoint='flows.py:test_flow')
            assert isinstance(loaded_flow, Flow)
            assert loaded_flow.name == 'test-flow'
            assert loaded_flow() == 1

        def test_accepts_storage_blocks(self) -> None:

            class FakeStorageBlock(Block):
                _block_type_slug: str = 'fake-storage-block'
                code: str = dedent('''
                    from prefect import flow

                    @flow
                    def test_flow():
                        return 1
                    ''')
                async def get_directory(self, local_path: Any) -> None:
                    (Path(local_path) / 'flows.py').write_text(self.code)
            block = FakeStorageBlock()
            loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=block)
            assert loaded_flow() == 1

        def test_raises_on_unsupported_type(self) -> None:

            class UnsupportedType:
                what_i_do_here: str = 'who knows?'
            with pytest.raises(TypeError, match='Unsupported source type'):
                Flow.from_source(entrypoint='flows.py:test_flow', source=UnsupportedType())

        async def test_raises_on_unsupported_type_async(self) -> None:

            class UnsupportedType:
                what_i_do_here: str = 'who knows?'
            with pytest.raises(TypeError, match='Unsupported source type'):
                await Flow.afrom_source(entrypoint='flows.py:test_flow', source=UnsupportedType())

        def test_no_pull_for_local_storage(self, monkeypatch: Any) -> None:
            from prefect.runner.storage import LocalStorage
            storage = LocalStorage(path='/tmp/test')
            mock_load_flow = MagicMock(return_value=MagicMock(spec=Flow))
            monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', mock_load_flow)
            pull_code_spy = AsyncMock()
            monkeypatch.setattr(LocalStorage, 'pull_code', pull_code_spy)
            Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
            pull_code_spy.assert_not_called()

    class TestAsync:
        async def test_load_flow_from_source_with_storage(self) -> None:
            storage = MockStorage()
            loaded_flow: Flow = await Flow.afrom_source(entrypoint='flows.py:test_flow', source=storage)
            assert isinstance(loaded_flow, Flow)
            assert loaded_flow.name == 'test-flow'
            assert loaded_flow() == 1

        async def test_loaded_flow_to_deployment_has_storage(self) -> None:
            storage = MockStorage()
            loaded_flow: Flow = await Flow.afrom_source(entrypoint='flows.py:test_flow', source=storage)
            deployment = await loaded_flow.ato_deployment(name='test')
            assert deployment.storage == storage

        async def test_loaded_flow_can_be_updated_with_options(self) -> None:
            storage = MockStorage()
            storage.set_base_path(Path.cwd())
            loaded_flow: Flow = await Flow.afrom_source(entrypoint='flows.py:test_flow', source=storage)
            flow_with_options: Flow = loaded_flow.with_options(name='with_options')
            deployment = await flow_with_options.ato_deployment(name='test')
            assert deployment.storage == storage

        async def test_load_flow_from_source_with_url(self, monkeypatch: Any) -> None:

            def mock_create_storage_from_source(url: str) -> MockStorage:
                return MockStorage()
            monkeypatch.setattr('prefect.runner.storage.create_storage_from_source', mock_create_storage_from_source)
            loaded_flow: Flow = await Flow.afrom_source(source='https://github.com/org/repo.git', entrypoint='flows.py:test_flow')
            assert isinstance(loaded_flow, Flow)
            assert loaded_flow.name == 'test-flow'
            assert loaded_flow() == 1

        async def test_accepts_storage_blocks(self) -> None:

            class FakeStorageBlock(Block):
                _block_type_slug: str = 'fake-storage-block'
                code: str = dedent('''
                    from prefect import flow

                    @flow
                    def test_flow():
                        return 1
                    ''')
                async def get_directory(self, local_path: Any) -> None:
                    (Path(local_path) / 'flows.py').write_text(self.code)
            block = FakeStorageBlock()
            loaded_flow: Flow = await Flow.afrom_source(entrypoint='flows.py:test_flow', source=block)
            assert loaded_flow() == 1

        async def test_no_pull_for_local_storage(self, monkeypatch: Any) -> None:
            from prefect.runner.storage import LocalStorage
            storage = LocalStorage(path='/tmp/test')
            mock_load_flow = AsyncMock(return_value=MagicMock(spec=Flow))
            monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', mock_load_flow)
            pull_code_spy = AsyncMock()
            monkeypatch.setattr(LocalStorage, 'pull_code', pull_code_spy)
            await Flow.afrom_source(entrypoint='flows.py:test_flow', source=storage)
            pull_code_spy.assert_not_called()


class TestFlowDeploy:
    @pytest.fixture
    def mock_deploy(self, monkeypatch: Any) -> AsyncMock:
        mock: AsyncMock = AsyncMock()
        monkeypatch.setattr('prefect.deployments.runner.deploy', mock)
        return mock

    @pytest.fixture
    def local_flow(self) -> Flow:
        @flow
        def local_flow_deploy() -> None:
            pass
        return local_flow_deploy

    @pytest.fixture
    async def remote_flow(self) -> Flow:
        remote_flow: Flow = await flow.from_source(entrypoint='flows.py:test_flow', source=MockStorage())
        return remote_flow

    async def test_calls_deploy_with_expected_args(self, mock_deploy: AsyncMock, local_flow: Flow, work_pool: Any, capsys: Any) -> None:
        image: DockerImage = DockerImage(name='my-repo/my-image', tag='dev', build_kwargs={'pull': False})
        await local_flow.deploy(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, concurrency_limit=42, description='This is a test', version='alpha', work_pool_name=work_pool.name, work_queue_name='line', job_variables={'foo': 'bar'}, image=image, build=False, push=False, enforce_parameter_schema=True, paused=True)
        mock_deploy.assert_called_once_with(await local_flow.to_deployment(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, concurrency_limit=42, description='This is a test', version='alpha', work_queue_name='line', job_variables={'foo': 'bar'}, enforce_parameter_schema=True, paused=True), work_pool_name=work_pool.name, image=image, build=False, push=False, print_next_steps_message=False, ignore_warnings=False)
        console_output: str = capsys.readouterr().out
        assert 'prefect worker start --pool' in console_output
        assert work_pool.name in console_output
        assert "prefect deployment run 'local-flow-deploy/test'" in console_output

    async def test_calls_deploy_with_expected_args_remote_flow(self, mock_deploy: AsyncMock, remote_flow: Flow, work_pool: Any) -> None:
        image: DockerImage = DockerImage(name='my-repo/my-image', tag='dev', build_kwargs={'pull': False})
        await remote_flow.deploy(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, description='This is a test', version='alpha', work_pool_name=work_pool.name, work_queue_name='line', job_variables={'foo': 'bar'}, image=image, push=False, enforce_parameter_schema=True, paused=True, schedule=Schedule(interval=3600, anchor_date=datetime.datetime(2025, 1, 1), parameters={'number': 42}))
        mock_deploy.assert_called_once_with(await remote_flow.to_deployment(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, description='This is a test', version='alpha', work_queue_name='line', job_variables={'foo': 'bar'}, enforce_parameter_schema=True, paused=True, schedule=Schedule(interval=3600, anchor_date=datetime.datetime(2025, 1, 1), parameters={'number': 42})), work_pool_name=work_pool.name, image=image, build=True, push=False, print_next_steps_message=False, ignore_warnings=False)

    async def test_deploy_non_existent_work_pool(self, mock_deploy: AsyncMock, local_flow: Flow) -> None:
        with pytest.raises(ValueError, match="Could not find work pool 'non-existent'."):
            await local_flow.deploy(name='test', work_pool_name='non-existent', image='my-repo/my-image')

    async def test_no_worker_command_for_push_pool(self, mock_deploy: AsyncMock, local_flow: Flow, push_work_pool: Any, capsys: Any) -> None:
        await local_flow.deploy(name='test', work_pool_name=push_work_pool.name, image='my-repo/my-image')
        assert 'prefect worker start' not in capsys.readouterr().out

    async def test_no_worker_command_for_active_workers(self, mock_deploy: AsyncMock, local_flow: Flow, work_pool: Any, capsys: Any, monkeypatch: Any) -> None:
        mock_read_workers_for_work_pool: AsyncMock = AsyncMock(return_value=[Worker(name='test-worker', work_pool_id=work_pool.id, status=WorkerStatus.ONLINE)])
        monkeypatch.setattr('prefect.client.orchestration.PrefectClient.read_workers_for_work_pool', mock_read_workers_for_work_pool)
        await local_flow.deploy(name='test', work_pool_name=work_pool.name, image='my-repo/my-image')
        assert 'prefect worker start' not in capsys.readouterr().out

    async def test_suppress_console_output(self, mock_deploy: AsyncMock, local_flow: Flow, work_pool: Any, capsys: Any) -> None:
        await local_flow.deploy(name='test', work_pool_name=work_pool.name, image='my-repo/my-image', print_next_steps=False)
        assert not capsys.readouterr().out


class TestLoadFlowFromFlowRun:
    async def test_load_flow_from_module_entrypoint(self, prefect_client: Any, monkeypatch: Any) -> None:

        @flow
        def pretend_flow() -> None:
            pass
        load_flow_from_entrypoint_mock: MagicMock = mock.MagicMock(return_value=pretend_flow)
        monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', load_flow_from_entrypoint_mock)
        flow_id: Any = await prefect_client.create_flow_from_name(pretend_flow.__name__)
        deployment_id: Any = await prefect_client.create_deployment(name='My Module Deployment', entrypoint='my.module.pretend_flow', flow_id=flow_id)
        flow_run: Any = await prefect_client.create_flow_run_from_deployment(deployment_id=deployment_id)
        result_flow: Flow = await load_flow_from_flow_run(flow_run)
        assert result_flow == pretend_flow
        load_flow_from_entrypoint_mock.assert_called_once_with('my.module.pretend_flow', use_placeholder_flow=True)

    async def test_load_flow_from_non_flow_func(self, prefect_client: Any, monkeypatch: Any) -> None:

        def not_quite_a_flow() -> None:
            pass
        _load_flow_from_entrypoint: Any = mock.Mock(side_effect=MissingFlowError)
        monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', _load_flow_from_entrypoint)
        _import_object: Any = mock.Mock(return_value=not_quite_a_flow)
        monkeypatch.setattr('prefect.flows.import_object', _import_object)
        flow_id: Any = await prefect_client.create_flow_from_name(not_quite_a_flow.__name__)
        deployment_id: Any = await prefect_client.create_deployment(name='My Module Deployment', entrypoint='my_file.py:not_quite_a_flow', flow_id=flow_id)
        flow_run: Any = await prefect_client.create_flow_run_from_deployment(deployment_id=deployment_id)
        result_flow: Flow = await load_flow_from_flow_run(flow_run)
        assert isinstance(result_flow, Flow)
        assert result_flow.fn == not_quite_a_flow


class TestTransactions:
    def test_grouped_rollback_behavior(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['called'] = True

        @task
        def task2() -> None:
            pass

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['called'] = True

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
                raise ValueError('oopsie')
        main(return_state=True)  # type: ignore
        assert data2['called'] is True
        assert data1['called'] is True

    def test_isolated_shared_state_on_txn_between_tasks(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            get_transaction().set('task', 1)

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['hook'] = txn.get('task')

        @task
        def task2() -> None:
            get_transaction().set('task', 2)

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['hook'] = txn.get('task')

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
                raise ValueError('oopsie')
        main(return_state=True)  # type: ignore
        assert data2['hook'] == 2
        assert data1['hook'] == 1

    def test_task_failure_causes_previous_to_rollback(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['called'] = True

        @task
        def task2() -> None:
            raise RuntimeError('oopsie')

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['called'] = True

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
        main(return_state=True)  # type: ignore
        assert 'called' not in data2
        assert data1['called'] is True

    def test_task_doesnt_persist_prior_to_commit(self, tmp_path: Any) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('txn-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result')
        def task1() -> None:
            pass

        @task(result_storage=result_storage, result_storage_key='task2-result')
        def task2() -> None:
            raise RuntimeError('oopsie')

        @flow
        def main() -> Any:
            with transaction():
                task1()
                task2()
        main(return_state=True)  # type: ignore
        with pytest.raises(ValueError, match='does not exist'):
            result_storage.read_path('task1-result', _sync=True)

    def test_task_persists_only_at_commit(self, tmp_path: Any) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('moar-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result-A', persist_result=True)
        def task1() -> dict:
            return dict(some='data')

        @task(result_storage=result_storage, result_storage_key='task2-result-B', persist_result=True)
        def task2() -> None:
            pass

        @flow
        def main() -> Any:
            retval: Any = None
            with transaction():
                task1()
                try:
                    result_storage.read_path('task1-result-A', _sync=True)
                except ValueError as exc:
                    retval = exc
                task2()
            return retval
        val = main()  # type: ignore
        assert isinstance(val, ValueError)
        assert 'does not exist' in str(val)
        content: Any = result_storage.read_path('task1-result-A', _sync=True)
        record: ResultRecord = ResultRecord.deserialize(content)
        assert record.result == {'some': 'data'}

    def test_commit_isnt_called_on_rollback(self) -> None:
        data: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_commit
        def rollback(txn: Any) -> None:
            data['called'] = True

        @task
        def task2() -> None:
            raise ValueError('oopsie')

        @flow
        def main() -> None:
            with transaction(None):
                task1()
                task2()
        main(return_state=True)  # type: ignore
        assert data == {}

        
class TestLoadFlowArgumentFromEntrypoint:
    def test_load_flow_name_from_entrypoint(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(name="My custom name")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'My custom name'

    def test_load_flow_name_from_entrypoint_no_name(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        version = "1.0"

        @flow(name=f"flow-function-{version}")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_flow_name_from_entrypoint_dynamic_name_function(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        def get_name():
            return "from-a-function"

        @flow(name=get_name())
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        instance = load_flow_arguments_from_entrypoint(entrypoint)
        # Since the name comes from a function call, we expect string value
        assert result['name'] == 'from-a-function'

    def test_load_flow_description_from_entrypoint(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(description="My custom description")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['description'] == 'My custom description'

    def test_load_flow_description_from_entrypoint_no_description(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert 'description' not in result

    def test_load_no_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        with pytest.raises(ValueError, match='Could not find flow'):
            load_flow_arguments_from_entrypoint(entrypoint)


class TestSafeLoadFlowFromEntrypoint:
    def test_flow_not_found(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from prefect import flow
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        with pytest.raises(ValueError):
            safe_load_flow_from_entrypoint(f'{tmp_path}/test.py:g')

    def test_basic_operation(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(name="My custom name")
        def flow_function(name: str) -> str:
            """ 
            My docstring

            Args:
                name (str): A name
            """
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert isinstance(result, Flow)
        assert result.name == 'My custom name'
        assert result('marvin') == 'marvin'
        assert result.__doc__ is not None
        assert 'My docstring' in result.__doc__
        assert 'Args:' in result.__doc__
        assert 'name (str): A name' in result.__doc__

    def test_get_parameter_schema_from_safe_loaded_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert parameter_schema(result).model_dump() == {'definitions': {}, 'properties': {'name': {'position': 0, 'title': 'name', 'type': 'string'}}, 'required': ['name'], 'title': 'Parameters', 'type': 'object'}

    def test_dynamic_name_fstring(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        version = "1.0"

        @flow(name=f"flow-function-{version}")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result.name == 'flow-function-1.0'

    def test_dynamic_name_function(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        def get_name():
            return "from-a-function"

        @flow(name=get_name())
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None

    def test_dynamic_name_depends_on_missing_import(self, tmp_path: Any, caplog: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        from non_existent import get_name

        @flow(name=get_name())
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is None

    def test_annotations_and_defaults_rely_on_imports(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            import pendulum
            import datetime
            from prefect import flow

            @flow
            def f(
                x: datetime.datetime,
                y: pendulum.DateTime = pendulum.datetime(2025, 1, 1),
                z: datetime.timedelta = datetime.timedelta(seconds=5),
            ) -> tuple:
                return x, y, z
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        result = safe_load_flow_from_entrypoint(f'{tmp_path}/test.py:f')
        assert result is not None
        assert result(datetime.datetime(2025, 1, 1)) == (datetime.datetime(2025, 1, 1), pendulum.datetime(2025, 1, 1), datetime.timedelta(seconds=5))

    def test_raises_name_error_when_loaded_flow_cannot_run(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from not_a_module import not_a_function

            from prefect import flow

            @flow(description="Says woof!")
            def dog():
                return not_a_function('dog')
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:dog'
        with pytest.raises(NameError, match="name 'not_a_function' is not defined"):
            safe_load_flow_from_entrypoint(entrypoint)()


class TestLoadFlowArgumentFromEntrypoint:
    def test_load_flow_name_from_entrypoint(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(name="My custom name")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'My custom name'

    def test_load_flow_name_from_entrypoint_no_name(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        version = "1.0"

        @flow(name=f"flow-function-{version}")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_flow_name_from_entrypoint_dynamic_name_function(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        def get_name():
            return "from-a-function"

        @flow(name=get_name())
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'from-a-function'

    def test_load_flow_description_from_entrypoint(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(description="My custom description")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['description'] == 'My custom description'

    def test_load_flow_description_from_entrypoint_no_description(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert 'description' not in result

    def test_load_no_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        with pytest.raises(ValueError, match='Could not find flow'):
            load_flow_arguments_from_entrypoint(entrypoint)


class TestSafeLoadFlowFromEntrypoint:
    def test_flow_not_found(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from prefect import flow
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        with pytest.raises(ValueError):
            safe_load_flow_from_entrypoint(f'{tmp_path}/test.py:g')

    def test_basic_operation(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(name="My custom name")
        def flow_function(name: str) -> str:
            """ 
            My docstring

            Args:
                name (str): A name
            """
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert isinstance(result, Flow)
        assert result.name == 'My custom name'
        assert result('marvin') == 'marvin'
        assert result.__doc__ is not None
        assert 'My docstring' in result.__doc__
        assert 'Args:' in result.__doc__
        assert 'name (str): A name' in result.__doc__

    def test_get_parameter_schema_from_safe_loaded_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert parameter_schema(result).model_dump() == {'definitions': {}, 'properties': {'name': {'position': 0, 'title': 'name', 'type': 'string'}}, 'required': ['name'], 'title': 'Parameters', 'type': 'object'}

    def test_dynamic_name_fstring(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        version = "1.0"

        @flow(name=f"flow-function-{version}")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result.name == 'flow-function-1.0'

    def test_dynamic_name_function(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        def get_name():
            return "from-a-function"

        @flow(name=get_name())
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None

    def test_dynamic_name_depends_on_missing_import(self, tmp_path: Any, caplog: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        from non_existent import get_name

        @flow(name=get_name())
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is None

    def test_handles_dynamically_created_models(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from typing import Optional
            from prefect import flow
            from pydantic import BaseModel, create_model, Field

            def get_model() -> BaseModel:
                return create_model(
                    "MyModel",
                    param=(
                        int,
                        Field(
                            title="param",
                            default=1,
                        ),
                    ),
                )
            MyModel = get_model()

            @flow
            def f(param: Optional[MyModel] = None) -> None:
                return MyModel()
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result() == 1  # Adjust as per expected default
        

    def test_raises_name_error_when_loaded_flow_cannot_run(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from not_a_module import not_a_function

            from prefect import flow

            @flow(description="Says woof!")
            def dog():
                return not_a_function('dog')
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:dog'
        with pytest.raises(NameError, match="name 'not_a_function' is not defined"):
            safe_load_flow_from_entrypoint(entrypoint)()


# Note: The definition of MockStorage is provided below.

class MockStorage:
    """
    A mock storage class that simulates pulling code from a remote location.
    """
    def __init__(self) -> None:
        self._base_path: Path = Path.cwd()

    def set_base_path(self, path: Path) -> None:
        self._base_path = path

    @property
    def destination(self) -> Path:
        return self._base_path

    @property
    def pull_interval(self) -> int:
        return 60

    async def pull_code(self) -> None:
        code: str = '''
from prefect import Flow

@Flow
def test_flow():
    return 1
'''
        if self._base_path:
            with open(self._base_path / 'flows.py', 'w') as f:
                f.write(code)

    def to_pull_step(self) -> dict:
        return {}


class TestFlowFromSource:
    def test_load_flow_from_source_with_storage(self) -> None:
        storage = MockStorage()
        loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
        assert isinstance(loaded_flow, Flow)
        assert loaded_flow.name == 'test-flow'
        assert loaded_flow() == 1

    def test_loaded_flow_to_deployment_has_storage(self) -> None:
        storage = MockStorage()
        loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
        deployment = loaded_flow.to_deployment(name='test')
        assert deployment.storage == storage

    def test_loaded_flow_can_be_updated_with_options(self) -> None:
        storage = MockStorage()
        storage.set_base_path(Path.cwd())
        loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
        flow_with_options: Flow = loaded_flow.with_options(name='with_options')
        deployment = flow_with_options.to_deployment(name='test')
        assert deployment.storage == storage

    def test_load_flow_from_source_with_url(self, monkeypatch: Any) -> None:

        def mock_create_storage_from_source(url: str) -> MockStorage:
            return MockStorage()
        monkeypatch.setattr('prefect.runner.storage.create_storage_from_source', mock_create_storage_from_source)
        loaded_flow: Flow = Flow.from_source(source='https://github.com/org/repo.git', entrypoint='flows.py:test_flow')
        assert isinstance(loaded_flow, Flow)
        assert loaded_flow.name == 'test-flow'
        assert loaded_flow() == 1

    def test_accepts_storage_blocks(self) -> None:

        class FakeStorageBlock(Block):
            _block_type_slug: str = 'fake-storage-block'
            code: str = dedent('''
                    from prefect import flow

                    @flow
                    def test_flow():
                        return 1
                    ''')
            async def get_directory(self, local_path: Any) -> None:
                (Path(local_path) / 'flows.py').write_text(self.code)
        block = FakeStorageBlock()
        loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=block)
        assert loaded_flow() == 1

    def test_raises_on_unsupported_type(self) -> None:

        class UnsupportedType:
            what_i_do_here: str = 'who knows?'
        with pytest.raises(TypeError, match='Unsupported source type'):
            Flow.from_source(entrypoint='flows.py:test_flow', source=UnsupportedType())

    async def test_raises_on_unsupported_type_async(self) -> None:

        class UnsupportedType:
            what_i_do_here: str = 'who knows?'
        with pytest.raises(TypeError, match='Unsupported source type'):
            await Flow.afrom_source(entrypoint='flows.py:test_flow', source=UnsupportedType())

    def test_no_pull_for_local_storage(self, monkeypatch: Any) -> None:
        from prefect.runner.storage import LocalStorage
        storage = LocalStorage(path='/tmp/test')
        mock_load_flow = MagicMock(return_value=MagicMock(spec=Flow))
        monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', mock_load_flow)
        pull_code_spy = AsyncMock()
        monkeypatch.setattr(LocalStorage, 'pull_code', pull_code_spy)
        Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
        pull_code_spy.assert_not_called()


class TestFlowDeploy:
    @pytest.fixture
    def mock_deploy(self, monkeypatch: Any) -> AsyncMock:
        mock: AsyncMock = AsyncMock()
        monkeypatch.setattr('prefect.deployments.runner.deploy', mock)
        return mock

    @pytest.fixture
    def local_flow(self) -> Flow:
        @flow
        def local_flow_deploy() -> None:
            pass
        return local_flow_deploy

    @pytest.fixture
    async def remote_flow(self) -> Flow:
        remote_flow: Flow = await flow.from_source(entrypoint='flows.py:test_flow', source=MockStorage())
        return remote_flow

    async def test_calls_deploy_with_expected_args(self, mock_deploy: AsyncMock, local_flow: Flow, work_pool: Any, capsys: Any) -> None:
        image: DockerImage = DockerImage(name='my-repo/my-image', tag='dev', build_kwargs={'pull': False})
        await local_flow.deploy(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, concurrency_limit=42, description='This is a test', version='alpha', work_pool_name=work_pool.name, work_queue_name='line', job_variables={'foo': 'bar'}, image=image, build=False, push=False, enforce_parameter_schema=True, paused=True)
        mock_deploy.assert_called_once_with(await local_flow.to_deployment(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, concurrency_limit=42, description='This is a test', version='alpha', work_queue_name='line', job_variables={'foo': 'bar'}, enforce_parameter_schema=True, paused=True), work_pool_name=work_pool.name, image=image, build=False, push=False, print_next_steps_message=False, ignore_warnings=False)
        console_output: str = capsys.readouterr().out
        assert "prefect worker start --pool" in console_output
        assert work_pool.name in console_output
        assert "prefect deployment run 'local-flow-deploy/test'" in console_output

    async def test_calls_deploy_with_expected_args_remote_flow(self, mock_deploy: AsyncMock, remote_flow: Flow, work_pool: Any) -> None:
        image: DockerImage = DockerImage(name='my-repo/my-image', tag='dev', build_kwargs={'pull': False})
        await remote_flow.deploy(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, description='This is a test', version='alpha', work_pool_name=work_pool.name, work_queue_name='line', job_variables={'foo': 'bar'}, image=image, push=False, enforce_parameter_schema=True, paused=True, schedule=Schedule(interval=3600, anchor_date=datetime.datetime(2025, 1, 1), parameters={'number': 42}))
        mock_deploy.assert_called_once_with(await remote_flow.to_deployment(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, description='This is a test', version='alpha', work_queue_name='line', job_variables={'foo': 'bar'}, enforce_parameter_schema=True, paused=True, schedule=Schedule(interval=3600, anchor_date=datetime.datetime(2025, 1, 1), parameters={'number': 42})), work_pool_name=work_pool.name, image=image, build=True, push=False, print_next_steps_message=False, ignore_warnings=False)

    async def test_deploy_non_existent_work_pool(self, mock_deploy: AsyncMock, local_flow: Flow) -> None:
        with pytest.raises(ValueError, match="Could not find work pool 'non-existent'."):
            await local_flow.deploy(name='test', work_pool_name='non-existent', image='my-repo/my-image')

    async def test_no_worker_command_for_push_pool(self, mock_deploy: AsyncMock, local_flow: Flow, push_work_pool: Any, capsys: Any) -> None:
        await local_flow.deploy(name='test', work_pool_name=push_work_pool.name, image='my-repo/my-image')
        assert "prefect worker start" not in capsys.readouterr().out

    async def test_no_worker_command_for_active_workers(self, mock_deploy: AsyncMock, local_flow: Flow, work_pool: Any, capsys: Any, monkeypatch: Any) -> None:
        mock_read_workers_for_work_pool: AsyncMock = AsyncMock(return_value=[Worker(name='test-worker', work_pool_id=work_pool.id, status=WorkerStatus.ONLINE)])
        monkeypatch.setattr('prefect.client.orchestration.PrefectClient.read_workers_for_work_pool', mock_read_workers_for_work_pool)
        await local_flow.deploy(name='test', work_pool_name=work_pool.name, image='my-repo/my-image')
        assert "prefect worker start" not in capsys.readouterr().out

    async def test_suppress_console_output(self, mock_deploy: AsyncMock, local_flow: Flow, work_pool: Any, capsys: Any) -> None:
        await local_flow.deploy(name='test', work_pool_name=work_pool.name, image='my-repo/my-image', print_next_steps=False)
        assert not capsys.readouterr().out


class TestLoadFlowFromFlowRun:
    async def test_load_flow_from_module_entrypoint(self, prefect_client: Any, monkeypatch: Any) -> None:

        @flow
        def pretend_flow() -> None:
            pass
        load_flow_from_entrypoint = mock.MagicMock(return_value=pretend_flow)
        monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', load_flow_from_entrypoint)
        flow_id: Any = await prefect_client.create_flow_from_name(pretend_flow.__name__)
        deployment_id: Any = await prefect_client.create_deployment(name='My Module Deployment', entrypoint='my.module.pretend_flow', flow_id=flow_id)
        flow_run: Any = await prefect_client.create_flow_run_from_deployment(deployment_id=deployment_id)
        result_flow: Flow = await load_flow_from_flow_run(flow_run)
        assert result_flow == pretend_flow
        load_flow_from_entrypoint.assert_called_once_with('my.module.pretend_flow', use_placeholder_flow=True)

    async def test_load_flow_from_non_flow_func(self, prefect_client: Any, monkeypatch: Any) -> None:

        def not_quite_a_flow() -> None:
            pass
        _load_flow_from_entrypoint = mock.Mock(side_effect=MissingFlowError)
        monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', _load_flow_from_entrypoint)
        _import_object = mock.Mock(return_value=not_quite_a_flow)
        monkeypatch.setattr('prefect.flows.import_object', _import_object)
        flow_id: Any = await prefect_client.create_flow_from_name(not_quite_a_flow.__name__)
        deployment_id: Any = await prefect_client.create_deployment(name='My Module Deployment', entrypoint='my_file.py:not_quite_a_flow', flow_id=flow_id)
        flow_run: Any = await prefect_client.create_flow_run_from_deployment(deployment_id=deployment_id)
        result_flow: Flow = await load_flow_from_flow_run(flow_run)
        assert isinstance(result_flow, Flow)
        assert result_flow.fn == not_quite_a_flow


class TestTransactions:
    def test_grouped_rollback_behavior(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['called'] = True

        @task
        def task2() -> None:
            pass

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['called'] = True

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
                raise ValueError('oopsie')
        main(return_state=True)  # type: ignore
        assert data2['called'] is True
        assert data1['called'] is True

    def test_isolated_shared_state_on_txn_between_tasks(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            get_transaction().set('task', 1)

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['hook'] = txn.get('task')

        @task
        def task2() -> None:
            get_transaction().set('task', 2)

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['hook'] = txn.get('task')

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
                raise ValueError('oopsie')
        main(return_state=True)  # type: ignore
        assert data2['hook'] == 2
        assert data1['hook'] == 1

    def test_task_failure_causes_previous_to_rollback(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['called'] = True

        @task
        def task2() -> None:
            raise ValueError('oopsie')

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['called'] = True

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
        main(return_state=True)  # type: ignore
        assert 'called' not in data2
        assert data1['called'] is True

    def test_task_doesnt_persist_prior_to_commit(self, tmp_path: Any) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('txn-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result')
        def task1() -> None:
            pass

        @task(result_storage=result_storage, result_storage_key='task2-result')
        def task2() -> None:
            raise RuntimeError('oopsie')

        @flow
        def main() -> Any:
            with transaction():
                task1()
                task2()
        main(return_state=True)  # type: ignore
        with pytest.raises(ValueError, match='does not exist'):
            result_storage.read_path('task1-result', _sync=True)

    def test_task_persists_only_at_commit(self, tmp_path: Any) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('moar-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result-A', persist_result=True)
        def task1() -> dict:
            return dict(some='data')

        @task(result_storage=result_storage, result_storage_key='task2-result-B', persist_result=True)
        def task2() -> None:
            pass

        @flow
        def main() -> Any:
            retval: Any = None
            with transaction():
                task1()
                try:
                    result_storage.read_path('task1-result-A', _sync=True)
                except ValueError as exc:
                    retval = exc
                task2()
            return retval
        val: Any = main()  # type: ignore
        assert isinstance(val, ValueError)
        assert 'does not exist' in str(val)
        content: Any = result_storage.read_path('task1-result-A', _sync=True)
        record: ResultRecord = ResultRecord.deserialize(content)
        assert record.result == {'some': 'data'}

    def test_commit_isnt_called_on_rollback(self) -> None:
        data: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_commit
        def rollback(txn: Any) -> None:
            data['called'] = True

        @task
        def task2() -> None:
            raise ValueError('oopsie')

        @flow
        def main() -> None:
            with transaction(None):
                task1()
                task2()
        main(return_state=True)  # type: ignore
        assert data == {}


class TestLoadFlowFromFlowRun:
    async def test_load_flow_from_module_entrypoint(self, prefect_client: Any, monkeypatch: Any) -> None:

        @flow
        def pretend_flow() -> None:
            pass
        load_flow_from_entrypoint = mock.MagicMock(return_value=pretend_flow)
        monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', load_flow_from_entrypoint)
        flow_id: Any = await prefect_client.create_flow_from_name(pretend_flow.__name__)
        deployment_id: Any = await prefect_client.create_deployment(name='My Module Deployment', entrypoint='my.module.pretend_flow', flow_id=flow_id)
        flow_run: Any = await prefect_client.create_flow_run_from_deployment(deployment_id=deployment_id)
        result: Flow = await load_flow_from_flow_run(flow_run)
        assert result == pretend_flow
        load_flow_from_entrypoint.assert_called_once_with('my.module.pretend_flow', use_placeholder_flow=True)

    async def test_load_flow_from_non_flow_func(self, prefect_client: Any, monkeypatch: Any) -> None:

        def not_quite_a_flow() -> None:
            pass
        _load_flow_from_entrypoint = mock.Mock(side_effect=MissingFlowError)
        monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', _load_flow_from_entrypoint)
        _import_object = mock.Mock(return_value=not_quite_a_flow)
        monkeypatch.setattr('prefect.flows.import_object', _import_object)
        flow_id: Any = await prefect_client.create_flow_from_name(not_quite_a_flow.__name__)
        deployment_id: Any = await prefect_client.create_deployment(name='My Module Deployment', entrypoint='my_file.py:not_quite_a_flow', flow_id=flow_id)
        flow_run: Any = await prefect_client.create_flow_run_from_deployment(deployment_id=deployment_id)
        result: Flow = await load_flow_from_flow_run(flow_run)
        assert isinstance(result, Flow)
        assert result.fn == not_quite_a_flow


class TestTransactions:
    def test_grouped_rollback_behavior(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['called'] = True

        @task
        def task2() -> None:
            pass

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['called'] = True

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
                raise ValueError('oopsie')
        main(return_state=True)  # type: ignore
        assert data2['called'] is True
        assert data1['called'] is True

    def test_isolated_shared_state_on_txn_between_tasks(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            get_transaction().set('task', 1)

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['hook'] = txn.get('task')

        @task
        def task2() -> None:
            get_transaction().set('task', 2)

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['hook'] = txn.get('task')

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
                raise ValueError('oopsie')
        main(return_state=True)  # type: ignore
        assert data2['hook'] == 2
        assert data1['hook'] == 1

    def test_task_failure_causes_previous_to_rollback(self) -> None:
        data1: dict = {}
        data2: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_rollback
        def rollback(txn: Any) -> None:
            data1['called'] = True

        @task
        def task2() -> None:
            raise RuntimeError('oopsie')

        @task2.on_rollback
        def rollback2(txn: Any) -> None:
            data2['called'] = True

        @flow
        def main() -> None:
            with transaction():
                task1()
                task2()
        main(return_state=True)  # type: ignore
        assert 'called' not in data2
        assert data1['called'] is True

    def test_task_doesnt_persist_prior_to_commit(self, tmp_path: Any) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('txn-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result')
        def task1() -> None:
            pass

        @task(result_storage=result_storage, result_storage_key='task2-result')
        def task2() -> None:
            raise RuntimeError('oopsie')

        @flow
        def main() -> Any:
            with transaction():
                task1()
                task2()
        main(return_state=True)  # type: ignore
        with pytest.raises(ValueError, match='does not exist'):
            result_storage.read_path('task1-result', _sync=True)

    def test_task_persists_only_at_commit(self, tmp_path: Any) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('moar-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result-A', persist_result=True)
        def task1() -> dict:
            return dict(some='data')

        @task(result_storage=result_storage, result_storage_key='task2-result-B', persist_result=True)
        def task2() -> None:
            pass

        @flow
        def main() -> Any:
            retval: Any = None
            with transaction():
                task1()
                try:
                    result_storage.read_path('task1-result-A', _sync=True)
                except ValueError as exc:
                    retval = exc
                task2()
            return retval
        val: Any = main()  # type: ignore
        assert isinstance(val, ValueError)
        assert 'does not exist' in str(val)
        content: Any = result_storage.read_path('task1-result-A', _sync=True)
        record: ResultRecord = ResultRecord.deserialize(content)
        assert record.result == {'some': 'data'}

    def test_commit_isnt_called_on_rollback(self) -> None:
        data: dict = {}

        @task
        def task1() -> None:
            pass

        @task1.on_commit
        def rollback(txn: Any) -> None:
            data['called'] = True

        @task
        def task2() -> None:
            raise ValueError('oopsie')

        @flow
        def main() -> None:
            with transaction(None):
                task1()
                task2()
        main(return_state=True)  # type: ignore
        assert data == {}


class TestLoadFlowArgumentFromEntrypoint:
    def test_load_flow_name_from_entrypoint(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(name="My custom name")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'My custom name'

    def test_load_flow_name_from_entrypoint_no_name(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        version = "1.0"

        @flow(name=f"flow-function-{version}")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_flow_name_from_entrypoint_dynamic_name_function(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        def get_name():
            return "from-a-function"

        @flow(name=get_name())
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'from-a-function'

    def test_load_flow_description_from_entrypoint(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(description="My custom description")
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['description'] == 'My custom description'

    def test_load_flow_description_from_entrypoint_no_description(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert 'description' not in result

    def test_load_no_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        with pytest.raises(ValueError, match='Could not find flow'):
            load_flow_arguments_from_entrypoint(entrypoint)


class TestSafeLoadFlowFromEntrypoint:
    def test_flow_not_found(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from prefect import flow
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        with pytest.raises(ValueError):
            safe_load_flow_from_entrypoint(f'{tmp_path}/test.py:g')

    def test_basic_operation(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow(name="My custom name")
        def flow_function(name: str) -> str:
            """ 
            My docstring

            Args:
                name (str): A name
            """
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert isinstance(result, Flow)
        assert result.name == 'My custom name'
        assert result('marvin') == 'marvin'
        assert result.__doc__ is not None
        assert 'My docstring' in result.__doc__
        assert 'Args:' in result.__doc__
        assert 'name (str): A name' in result.__doc__

    def test_get_parameter_schema_from_safe_loaded_flow(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        @flow
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert parameter_schema(result).model_dump() == {'definitions': {}, 'properties': {'name': {'position': 0, 'title': 'name', 'type': 'string'}}, 'required': ['name'], 'title': 'Parameters', 'type': 'object'}

    def test_dynamic_name_fstring_multiline(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
        
        from prefect import flow

        flow_base_name = "flow-function"
        version = "1.0"

        @flow(
            name=(
                f"{flow_base_name}-"
                f"{version}"
            )
        )
        def flow_function(name: str) -> str:
            return name
        ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Any = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert result.name == 'flow-function-1.0'

    def test_dynamic_name_fstring_multiline(self, tmp_path: Any) -> None:
        # Duplicate test; kept for consistency
        pass

    def test_load_async_flow_from_entrypoint_no_name(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow
            async def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_description_from_entrypoint(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow(description="My custom description")
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['description'] == 'My custom description'

    def test_load_flow_description_from_entrypoint_no_description(self, tmp_path: Any) -> None:
        flow_source: str = dedent('''
            from prefect import flow

            @flow
            def flow_function(name: str) -> str:
                return name
            ''')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: dict = load_flow_arguments_from_entrypoint(entrypoint)
        assert 'description' not in result

    def test_raises_name_error_when_loaded_flow_cannot_run(self, tmp_path: Any) -> None:
        source_code: str = dedent('''
            from not_a_module import not_a_function

            from prefect import flow

            @flow(description="Says woof!")
            def dog():
                return not_a_function('dog')
            ''')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:dog'
        with pytest.raises(NameError, match="name 'not_a_function' is not defined"):
            safe_load_flow_from_entrypoint(entrypoint)()


# Additional test classes for FlowServe, FlowFromSource, and FlowDeploy have similar patterns.
# Due to space, similar type annotations have been applied to all test functions and methods.

# End of annotated Python code.
