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
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Type, Union
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

from unittest.mock import MagicMock

from typing import Union

from prefect.deployments.runner import RunnerDeployment, RunnerStorage

SLEEP_TIME: int = 10

@pytest.fixture
def mock_sigterm_handler() -> Generator[Tuple[Callable[..., None], MagicMock], None, None]:
    if threading.current_thread() != threading.main_thread():
        pytest.skip("Can't test signal handlers from a thread")
    mock: MagicMock = MagicMock()

    def handler(*args: Any, **kwargs: Any) -> None:
        mock(*args, **kwargs)
    prev_handler: Any = signal.signal(signal.SIGTERM, handler)
    try:
        yield (handler, mock)
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
    def test_version_none_if_source_file_cannot_be_determined(self, monkeypatch: Any, sourcefile: Optional[str]) -> None:
        """
        `getsourcefile` will return `None` when functions are defined interactively,
        or other values on Windows.
        """
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
            pass
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

            @staticmethod
            def format(*args: Any, **kwargs: Any) -> None:
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
            pass
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

            @staticmethod
            def format(*args: Any, **kwargs: Any) -> None:
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
        def base() -> Flow:
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
        def base() -> Flow:
            pass
        new_flow: Flow = base.with_options(result_storage=block)
        assert my_flow.persist_result is True
        assert new_flow.persist_result is True

    def test_setting_result_serializer_sets_persist_result_to_true(self) -> None:

        @flow(result_serializer='json')
        def my_flow() -> None:
            pass

        @flow
        def base() -> Flow:
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

        @flow(
            name='Initial flow',
            description='Flow before with options',
            flow_run_name='OG',
            timeout_seconds=10,
            validate_parameters=True,
            persist_result=True,
            result_serializer='pickle',
            result_storage=fooblock,
            cache_result_in_memory=False,
            on_completion=None,
            on_failure=None,
            on_cancellation=None,
            on_crashed=None
        )
        def initial_flow() -> None:
            pass

        def failure_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            print('Woof!')

        def success_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            print('Meow!')

        def cancellation_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            print('Fizz Buzz!')

        def crash_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            print('Crash!')
        flow_with_options: Flow = initial_flow.with_options(
            name='Copied flow',
            description='A copied flow',
            flow_run_name=lambda: 'new-name',
            task_runner=ThreadPoolTaskRunner,
            retries=3,
            retry_delay_seconds=20,
            timeout_seconds=5,
            validate_parameters=False,
            persist_result=False,
            result_serializer='json',
            result_storage=barblock,
            cache_result_in_memory=True,
            on_completion=[success_hook],
            on_failure=[failure_hook],
            on_cancellation=[cancellation_hook],
            on_crashed=[crash_hook]
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

    def test_with_options_uses_existing_settings_when_no_override(self, tmp_path: Path) -> None:
        storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        storage.save('test-overrides', _sync=True)

        @flow(
            name='Initial flow',
            description='Flow before with options',
            task_runner=ThreadPoolTaskRunner,
            timeout_seconds=10,
            validate_parameters=True,
            retries=3,
            retry_delay_seconds=20,
            persist_result=False,
            result_serializer='json',
            result_storage=storage,
            cache_result_in_memory=False,
            log_prints=False
        )
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
            name: str = 'test'
            date: str = 'todays_date'
            return f'{name}-{date}'

        @flow(retry_delay_seconds=3, flow_run_name=generate_flow_run_name)
        def initial_flow() -> None:
            pass
        flow_with_options: Flow = initial_flow.with_options()
        assert flow_with_options.flow_run_name is generate_flow_run_name

    def test_with_options_can_unset_result_options_with_none(self, tmp_path: Path) -> None:
        storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        storage.save('test-unset', _sync=True)

        @flow(result_serializer='json', result_storage=storage)
        def initial_flow() -> None:
            pass
        flow_with_options: Flow = initial_flow.with_options(result_serializer=None, result_storage=None)
        assert flow_with_options.result_serializer is None
        assert flow_with_options.result_storage is None

    def test_with_options_signature_aligns_with_flow_signature(self) -> None:
        flow_params: set[str] = set(inspect.signature(flow).parameters.keys())
        with_options_params: set[str] = set(inspect.signature(Flow.with_options).parameters.keys())
        flow_params.remove('_FlowDecorator__fn')
        with_options_params.remove('self')
        assert flow_params == with_options_params

    def get_flow_run_name() -> str:
        name: str = 'test'
        date: str = 'todays_date'
        return f'{name}-{date}'

    @pytest.mark.parametrize('name, match', [
        (1, "Expected string for flow parameter 'name'; got int instead."),
        (get_flow_run_name, "Expected string for flow parameter 'name'; got function instead. Perhaps you meant to call it?")
    ])
    def test_flow_name_non_string_raises(self, name: Union[int, Callable[[], str]], match: str) -> None:
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
        state: State = foo(1, 2, return_state=True)
        assert isinstance(state, State)
        assert await state.result() == 6

    def test_call_coerces_parameter_types(self) -> None:
        import pydantic

        class CustomType(pydantic.BaseModel):
            z: int

        @flow(version='test')
        def foo(x: str, y: List[str], zt: Dict[str, int]) -> int:
            return int(x) + sum(map(int, y)) + zt['z']

        result: int = foo(x='1', y=['2', '3'], zt=CustomType(z=4).model_dump())
        assert result == 10

    def test_call_with_variadic_args(self) -> None:

        @flow
        def test_flow(*foo: int, bar: int) -> Tuple[Tuple[int, ...], int]:
            return (foo, bar)
        assert test_flow(1, 2, 3, bar=4) == ((1, 2, 3), 4)

    def test_call_with_variadic_keyword_args(self) -> None:

        @flow
        def test_flow(foo: int, bar: int, **foobar: int) -> Tuple[int, int, Dict[str, int]]:
            return (foo, bar, foobar)
        assert test_flow(1, 2, x=3, y=4, z=5) == (1, 2, dict(x=3, y=4, z=5))

    async def test_fails_but_does_not_raise_on_incompatible_parameter_types(self) -> None:

        @flow(version='test')
        def foo(x: int) -> None:
            pass
        state: State = foo(x='foo', return_state=True)
        with pytest.raises(ParameterTypeError):
            await state.result()

    def test_call_ignores_incompatible_parameter_types_if_asked(self) -> None:

        @flow(version='test', validate_parameters=False)
        def foo(x: int) -> Any:
            return x
        assert foo(x='foo') == 'foo'

    @pytest.mark.parametrize('error', [ValueError('Hello'), None])
    async def test_final_state_reflects_exceptions_during_run(self, error: Optional[Exception]) -> None:

        @flow(version='test')
        def foo() -> Any:
            if error:
                raise error
        state: State = foo(return_state=True)
        assert state.is_failed() if error else state.is_completed()
        assert exceptions_equal(await state.result(raise_on_failure=False), error)

    async def test_final_state_respects_returned_state(self) -> None:

        @flow(version='test')
        def foo() -> Any:
            return State(type=StateType.FAILED, message='Test returned state', data='hello!')
        state: State = foo(return_state=True)
        assert state.is_failed()
        assert await state.result(raise_on_failure=False) == 'hello!'
        assert state.message == 'Test returned state'

    async def test_flow_state_reflects_returned_task_run_state(self) -> None:

        @task
        def fail() -> Exception:
            raise ValueError('Test')

        @flow(version='test')
        def foo() -> Any:
            return fail(return_state=True)
        flow_state: State = foo(return_state=True)
        assert flow_state.is_failed()
        task_run_state: State = await flow_state.result(raise_on_failure=False)
        assert isinstance(task_run_state, State)
        assert task_run_state.is_failed()
        with pytest.raises(ValueError, match='Test'):
            await task_run_state.result()

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_flow_state_defaults_to_task_states_when_no_return_failure(self) -> None:

        @task
        def fail() -> Exception:
            raise ValueError('Test')

        @flow(version='test')
        def foo() -> Any:
            fail(return_state=True)
            fail(return_state=True)
            return None
        flow_state: State = foo(return_state=True)
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
        def foo() -> Any:
            succeed()
            succeed()
            return None
        flow_state: State = foo(return_state=True)
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
        def foo() -> Any:
            succeed(return_state=True)
            fail(return_state=True)
            return None
        states: List[State] = await foo(return_state=True).result(raise_on_failure=False)
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
        def foo() -> Any:
            wrapper_flow(return_state=True)
            return None
        states: List[State] = await foo(return_state=True).result(raise_on_failure=False)
        assert len(states) == 1
        state: State = states[0]
        assert isinstance(state, State)
        with pytest.raises(ValueError, match='foo'):
            await raise_state_exception(state)

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_flow_state_reflects_returned_multiple_task_run_states(self) -> None:

        @task
        def fail1() -> Exception:
            raise ValueError('Test 1')

        @task
        def fail2() -> Exception:
            raise ValueError('Test 2')

        @task
        def succeed() -> bool:
            return True

        @flow(version='test')
        def foo() -> Any:
            return (fail1(return_state=True), fail2(return_state=True), succeed(return_state=True))
        flow_state: State = foo(return_state=True)
        assert flow_state.is_failed()
        assert flow_state.message == '2/3 states failed.'
        first: State
        second: State
        third: State
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
        def cancel() -> Cancelled:
            return Cancelled()

        @task
        def fail() -> Exception:
            raise ValueError('Fail')

        @task
        def succeed() -> bool:
            return True

        @flow(version='test')
        def my_flow() -> Tuple[State, State, State]:
            return (cancel.submit(), succeed.submit(), fail.submit())
        flow_state: State = my_flow(return_state=True)
        assert flow_state.is_cancelled()
        assert flow_state.message == '1/3 states cancelled.'
        first: State
        second: State
        third: State
        first, second, third = await flow_state.result(raise_on_failure=False)
        assert first.is_cancelled()
        assert second.is_completed()
        assert third.is_failed()
        with pytest.raises(CancelledRun):
            await first.result()

    def test_flow_with_cancelled_subflow_has_cancelled_state(self) -> None:

        @task
        def cancel() -> Cancelled:
            return Cancelled()

        @flow(version='test')
        def subflow() -> State:
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

        def __init__(self, x: int) -> None:
            self.x: int = x

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    def test_flow_supports_instance_methods(self, T: Type[Union[BaseFoo, BaseFooModel]]) -> None:

        class Foo(T):

            @flow
            def instance_method(self) -> int:
                return self.x
        f: Foo = Foo(x=1)
        assert Foo(x=5).instance_method() == 5
        assert f.instance_method() == 1
        assert isinstance(Foo(x=10).instance_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    def test_flow_supports_class_methods(self, T: Type[Union[BaseFoo, BaseFooModel]]) -> None:

        class Foo(T):

            def __init__(self, x: int) -> None:
                self.x: int = x

            @classmethod
            @flow
            def class_method(cls) -> str:
                return cls.__name__
        assert Foo.class_method() == 'Foo'
        assert isinstance(Foo.class_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    def test_flow_supports_static_methods(self, T: Type[Union[BaseFoo, BaseFooModel]]) -> None:

        class Foo(T):

            def __init__(self, x: int) -> None:
                self.x: int = x

            @staticmethod
            @flow
            def static_method() -> str:
                return 'static'
        assert Foo.static_method() == 'static'
        assert isinstance(Foo.static_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    async def test_flow_supports_async_instance_methods(self, T: Type[Union[BaseFoo, BaseFooModel]]) -> None:

        class Foo(T):

            @flow
            async def instance_method(self) -> int:
                return self.x
        f: Foo = Foo(x=1)
        assert await Foo(x=5).instance_method() == 5
        assert await f.instance_method() == 1
        assert isinstance(Foo(x=10).instance_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    async def test_flow_supports_async_class_methods(self, T: Type[Union[BaseFoo, BaseFooModel]]) -> None:

        class Foo(T):

            def __init__(self, x: int) -> None:
                self.x: int = x

            @classmethod
            @flow
            async def class_method(cls) -> str:
                return cls.__name__
        assert await Foo.class_method() == 'Foo'
        assert isinstance(Foo.class_method, Flow)

    @pytest.mark.parametrize('T', [BaseFoo, BaseFooModel])
    async def test_flow_supports_async_static_methods(self, T: Type[Union[BaseFoo, BaseFooModel]]) -> None:

        class Foo(T):

            def __init__(self, x: int) -> None:
                self.x: int = x

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
        def child(x: int, y: int, z: int) -> MagicMock:
            return compute(x, y, z)

        @flow(version='bar')
        def parent(x: int, y: int = 2, z: int = 3) -> MagicMock:
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
    async def test_sync_flow_with_async_subflow(self) -> None:
        result: str = 'a string, not a coroutine'

        @flow
        async def async_child() -> str:
            return result

        @flow
        def parent() -> str:
            return async_child()
        assert parent() == result

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_sync_flow_with_async_subflow_and_async_task(self) -> None:

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
            coros: List[Callable[..., Any]] = [child(i) for i in range(5)]
            assert await asyncio.gather(*coros) == list(range(5))
        await parent()

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
        assert await parent() == 15

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
        assert parent() == 15

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
        assert recurse(5) == 15

    async def test_subflow_with_invalid_parameters_is_failed(self, prefect_client: PrefectClient) -> None:

        @flow
        def child(x: int) -> int:
            return x

        @flow
        def parent(x: Any) -> State:
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
        def child(x: int) -> int:
            return x

        @flow
        def parent() -> Tuple[Optional[State], State]:
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
        parent_state: State = await parent(return_state=True).result()
        parent_flow_run_id: uuid.UUID = parent_state.state_details.flow_run_id
        child_state: State = await parent_state.result()
        child_flow_run_id: uuid.UUID = child_state.state_details.flow_run_id
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
    async def test_sync_flow_with_async_subflow_and_task_that_awaits_result(self, prefect_client: PrefectClient) -> None:
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
            future: Any = await some_async_task.submit()
            return await future.result()

        @flow
        def sync_flow() -> int:
            return integrations_flow()
        result: int = sync_flow()
        assert result == 42

class TestFlowRunTags:

    async def test_flow_run_tags_added_at_call(self, prefect_client: PrefectClient) -> None:

        @flow
        def my_flow() -> None:
            pass
        with tags('a', 'b'):
            state: State = my_flow(return_state=True)
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert set(flow_run.tags) == {'a', 'b'}

    async def test_flow_run_tags_added_to_subflows(self, prefect_client: PrefectClient) -> None:

        @flow
        def my_flow() -> State:
            with tags('c', 'd'):
                return my_subflow(return_state=True)

        @flow
        def my_subflow() -> None:
            pass
        with tags('a', 'b'):
            subflow_state: State = await my_flow(return_state=True).result()
        flow_run = await prefect_client.read_flow_run(subflow_state.state_details.flow_run_id)
        assert set(flow_run.tags) == {'a', 'b', 'c', 'd'}

class TestFlowTimeouts:

    async def test_flows_fail_with_timeout(self) -> None:

        @flow(timeout_seconds=0.1)
        def my_flow() -> None:
            time.sleep(SLEEP_TIME)
        state: State = my_flow(return_state=True)
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
        state: State = my_flow(return_state=True)
        assert state.is_completed()

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_user_timeout_is_not_hidden(self) -> None:

        @flow(timeout_seconds=30)
        def my_flow() -> None:
            raise TimeoutError('Oh no!')
        state: State = my_flow(return_state=True)
        assert state.is_failed()
        with pytest.raises(TimeoutError, match='Oh no!'):
            await state.result()
        assert 'exceeded timeout' not in state.message

    @pytest.mark.timeout(method='thread')
    def test_timeout_does_not_wait_for_completion_for_sync_flows(self, tmp_path: Path) -> None:
        completed: bool = False

        @flow(timeout_seconds=0.1)
        def my_flow() -> None:
            time.sleep(SLEEP_TIME)
            nonlocal completed
            completed = True
        state: State = my_flow(return_state=True)
        assert state.is_failed()
        assert 'exceeded timeout of 0.1 second(s)' in state.message
        assert not completed

    def test_timeout_stops_execution_at_next_task_for_sync_flows(self, tmp_path: Path) -> None:
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
            time.sleep(SLEEP_TIME)
            my_task()
            nonlocal completed
            completed = True
        state: State = my_flow(return_state=True)
        assert state.is_failed()
        assert 'exceeded timeout of 0.1 second(s)' in state.message
        assert not completed
        assert not task_completed

    async def test_timeout_stops_execution_after_await_for_async_flows(self) -> None:
        """
        Async flow runs can be cancelled after a timeout
        """
        completed: bool = False

        @flow(timeout_seconds=0.1)
        async def my_flow() -> None:
            for _ in range(100):
                await anyio.sleep(0.1)
            nonlocal completed
            completed = True
        state: State = await my_flow(return_state=True)
        assert state.is_failed()
        assert 'exceeded timeout of 0.1 second(s)' in state.message
        assert not completed

    async def test_timeout_stops_execution_in_async_subflows(self) -> None:
        """
        Async flow runs can be cancelled after a timeout
        """
        completed: bool = False

        @flow(timeout_seconds=0.1)
        async def my_subflow() -> State:
            for _ in range(SLEEP_TIME * 10):
                await anyio.sleep(0.1)
            nonlocal completed
            completed = True

        @flow
        async def my_flow() -> Tuple[Optional[Any], State]:
            subflow_state: State = await my_subflow(return_state=True)
            return (None, subflow_state)
        state: State = await my_flow(return_state=True)
        _, subflow_state = await state.result()
        assert 'exceeded timeout of 0.1 second(s)' in subflow_state.message
        assert not completed

    async def test_timeout_stops_execution_in_sync_subflows(self) -> None:
        """
        Sync flow runs can be cancelled after a timeout once a task is called
        """
        completed: bool = False

        @task
        def timeout_noticing_task() -> None:
            pass

        @flow(timeout_seconds=0.1)
        def my_subflow() -> State:
            start: float = time.monotonic()
            while time.monotonic() - start < 0.5:
                pass
            timeout_noticing_task()
            nonlocal completed
            completed = True

        @flow
        def my_flow() -> Tuple[Optional[Any], State]:
            subflow_state: State = my_subflow(return_state=True)
            return (None, subflow_state)
        state: State = my_flow(return_state=True)
        _, subflow_state = await state.result()
        assert 'exceeded timeout of 0.1 second(s)' in subflow_state.message
        assert not completed

    async def test_subflow_timeout_waits_until_execution_starts(self, tmp_path: Path) -> None:
        """
        Subflow with a timeout shouldn't start their timeout before the subflow is started.
        Fixes: https://github.com/PrefectHQ/prefect/issues/7903.
        """
        completed: bool = False

        @flow(timeout_seconds=1)
        async def downstream_flow() -> bool:
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
    data: int

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
        def my_flow(x: ParameterTestModel) -> ParameterTestModel:
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
            def type_container_input_flow(arg1: List[str]) -> str:
                print(arg1)
                return ','.join(arg1)
        else:

            @flow
            def type_container_input_flow(arg1: List[str]) -> str:
                print(arg1)
                return ','.join(arg1)
        assert type_container_input_flow(['a', 'b', 'c']) == 'a,b,c'

    def test_subflow_parameters_can_be_unserializable_types(self) -> None:
        data: ParameterTestClass = ParameterTestClass()

        @flow
        def my_flow() -> ParameterTestClass:
            return my_subflow(data)

        @flow
        def my_subflow(x: Any) -> Any:
            return x
        assert my_flow() == data

    def test_flow_parameters_can_be_unserializable_types_that_raise_value_error(self) -> None:

        @flow
        def my_flow(x: Any) -> Any:
            return x
        data: Type[Exception] = Exception
        assert my_flow(data) == data

    def test_flow_parameter_annotations_can_be_non_pydantic_classes(self) -> None:

        class Test:
            pass

        @flow
        def my_flow(instance: Test) -> Test:
            return instance
        instance: Test = my_flow(Test())
        assert isinstance(instance, Test)

    def test_subflow_parameters_can_be_pydantic_types(self) -> None:

        @flow
        def my_flow() -> ParameterTestModel:
            return my_subflow(ParameterTestModel(data=1))

        @flow
        def my_subflow(x: ParameterTestModel) -> ParameterTestModel:
            return x
        assert my_flow() == ParameterTestModel(data=1)

    def test_subflow_parameters_from_future_can_be_unserializable_types(self) -> None:
        data: ParameterTestClass = ParameterTestClass()

        @flow
        def my_flow() -> ParameterTestClass:
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
        def my_flow() -> ParameterTestModel:
            return my_subflow(identity.submit(ParameterTestModel(data=1)))

        @task
        def identity(x: Any) -> Any:
            return x

        @flow
        def my_subflow(x: ParameterTestModel) -> ParameterTestModel:
            return x
        assert my_flow() == ParameterTestModel(data=1)

    def test_subflow_parameter_annotations_can_be_normal_classes(self) -> None:

        class Test:
            pass

        @flow
        def my_flow(i: Test) -> Test:
            return my_subflow(i)

        @flow
        def my_subflow(i: Test) -> Test:
            return i
        instance: Test = my_flow(Test())
        assert isinstance(instance, Test)

    def test_flow_parameter_kwarg_can_be_literally_keys(self) -> None:
        """regression test for https://github.com/PrefectHQ/prefect/issues/15610"""

        @flow
        def my_flow(keys: str) -> str:
            return keys
        assert my_flow(keys='hello') == 'hello'

class TestSubflowTaskInputs:

    async def test_subflow_with_one_upstream_task_future(self, prefect_client: PrefectClient) -> None:

        @task
        def child_task(x: int) -> int:
            return x

        @flow
        def child_flow(x: int) -> int:
            return x

        @flow
        def parent_flow() -> Tuple[State, State]:
            task_future: State = child_task.submit(1)
            flow_state: State = child_flow(x=task_future, return_state=True)
            task_future.wait()
            task_state: State = task_future.state
            return (task_state, flow_state)
        task_state, flow_state = parent_flow()
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_one_upstream_task_state(self, prefect_client: PrefectClient) -> None:

        @task
        def child_task(x: int) -> int:
            return x

        @flow
        def child_flow(x: int) -> int:
            return x

        @flow
        def parent_flow() -> Tuple[State, State]:
            task_state: State = child_task(257, return_state=True)
            flow_state: State = child_flow(x=task_state, return_state=True)
            return (task_state, flow_state)
        task_state, flow_state = parent_flow()
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_one_upstream_task_result(self, prefect_client: PrefectClient) -> None:

        @task
        def child_task(x: int) -> int:
            return x

        @flow
        def child_flow(x: Any) -> Any:
            return x

        @flow
        def parent_flow() -> Tuple[State, State]:
            task_state: State = child_task(257, return_state=True)
            task_result: Any = task_state.result()
            flow_state: State = child_flow(x=task_result, return_state=True)
            return (task_state, flow_state)
        task_state, flow_state = parent_flow()
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_one_upstream_task_future_and_allow_failure(self, prefect_client: PrefectClient) -> None:

        @task
        def child_task() -> Exception:
            raise ValueError()

        @flow
        def child_flow(x: Any) -> Any:
            return x

        @flow
        def parent_flow() -> Any:
            future: State = child_task.submit()
            flow_state: State = child_flow(x=allow_failure(future), return_state=True)
            future.wait()
            return quote((future.state, flow_state))
        task_state, flow_state = parent_flow().unquote()
        assert isinstance(await flow_state.result(), ValueError)
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert task_state.is_failed()
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_one_upstream_task_state_and_allow_failure(self, prefect_client: PrefectClient) -> None:

        @task
        def child_task() -> Exception:
            raise ValueError()

        @flow
        def child_flow(x: Any) -> Any:
            return x

        @flow
        def parent_flow() -> Any:
            task_state: State = child_task(return_state=True)
            flow_state: State = child_flow(x=allow_failure(task_state), return_state=True)
            return quote((task_state, flow_state))
        task_state, flow_state = parent_flow().unquote()
        assert isinstance(await flow_state.result(), ValueError)
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert task_state.is_failed()
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

    async def test_subflow_with_no_upstream_tasks(self, prefect_client: PrefectClient) -> None:

        @flow
        def bar(x: int, y: int) -> int:
            return x + y

        @flow
        def foo() -> State:
            return bar(x=2, y=1, return_state=True)
        child_flow_state: State = await foo(return_state=True).result()
        flow_tracking_task_run = await prefect_client.read_task_run(child_flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[], y=[])

    async def test_subflow_with_upstream_task_passes_validation(self, prefect_client: PrefectClient) -> None:
        """
        Regression test for https://github.com/PrefectHQ/prefect/issues/14036
        """

        @task
        def child_task(x: int) -> int:
            return x

        @flow
        def child_flow(x: int) -> int:
            return x

        @flow
        def parent_flow() -> Tuple[State, State]:
            task_state: State = child_task(257, return_state=True)
            flow_state: State = child_flow(x=task_state, return_state=True)
            return (task_state, flow_state)
        task_state, flow_state = parent_flow()
        assert flow_state.is_completed()
        flow_tracking_task_run = await prefect_client.read_task_run(flow_state.state_details.task_run_id)
        assert flow_tracking_task_run.task_inputs == dict(x=[TaskRunResult(id=task_state.state_details.task_run_id)])

async def _wait_for_logs(prefect_client: PrefectClient, expected_num_logs: Optional[int] = None, timeout: float = 10.0) -> List[Any]:
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

    async def test_user_logs_are_sent_to_orion(self, prefect_client: PrefectClient) -> None:

        @flow
        def my_flow() -> None:
            logger: Any = get_run_logger()
            logger.info('Hello world!')
        my_flow()
        await _wait_for_logs(prefect_client, expected_num_logs=3)
        logs: List[Any] = await prefect_client.read_logs()
        assert 'Hello world!' in {log.message for log in logs}

    async def test_repeated_flow_calls_send_logs_to_orion(self, prefect_client: PrefectClient) -> None:

        @flow
        async def my_flow(i: int) -> None:
            logger: Any = get_run_logger()
            logger.info(f'Hello {i}')
        await my_flow(1)
        await my_flow(2)
        logs: List[Any] = await _wait_for_logs(prefect_client, expected_num_logs=6)
        assert {'Hello 1', 'Hello 2'}.issubset({log.message for log in logs})

    @pytest.mark.clear_db
    async def test_exception_info_is_included_in_log(self, prefect_client: PrefectClient) -> None:

        @flow
        def my_flow() -> None:
            logger: Any = get_run_logger()
            try:
                x + y
            except Exception:
                logger.error('There was an issue', exc_info=True)
        my_flow()
        await _wait_for_logs(prefect_client, expected_num_logs=3)
        logs: List[Any] = await prefect_client.read_logs()
        error_logs: str = '\n'.join([log.message for log in logs if log.level == 40])
        assert 'Traceback' in error_logs
        assert 'NameError' in error_logs, 'Should reference the exception type'
        assert 'x + y' in error_logs, 'Should reference the line of code'

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    @pytest.mark.xfail(reason='Weird state sharing between new and old engine tests')
    async def test_raised_exceptions_include_tracebacks(self, prefect_client: PrefectClient) -> None:

        @flow
        def my_flow() -> None:
            raise ValueError('Hello!')
        with pytest.raises(ValueError):
            my_flow()
        logs: List[Any] = await prefect_client.read_logs()
        assert logs
        error_logs: str = '\n'.join([log.message for log in logs if log.level == 40 and 'Encountered exception' in log.message])
        assert 'Traceback' in error_logs
        assert 'ValueError: Hello!' in error_logs, 'References the exception'

    @pytest.mark.clear_db
    async def test_opt_out_logs_are_not_sent_to_api(self, prefect_client: PrefectClient) -> None:

        @flow
        def my_flow() -> None:
            logger: Any = get_run_logger()
            logger.info('Hello world!', extra={'send_to_api': False})
        my_flow()
        logs: List[Any] = await prefect_client.read_logs()
        assert 'Hello world!' not in {log.message for log in logs}

    @pytest.mark.xfail(reason='Weird state sharing between new and old engine tests')
    async def test_logs_are_given_correct_id(self, prefect_client: PrefectClient) -> None:

        @flow
        def my_flow() -> None:
            logger: Any = get_run_logger()
            logger.info('Hello world!')
        state: State = my_flow(return_state=True)
        flow_run_id: uuid.UUID = state.state_details.flow_run_id
        logs: List[Any] = await prefect_client.read_logs()
        assert all([log.flow_run_id == flow_run_id for log in logs])
        assert all([log.task_run_id is None for log in logs])

@pytest.mark.enable_api_log_handler
class TestSubflowRunLogs:

    @pytest.mark.clear_db
    async def test_subflow_logs_are_written_correctly(self, prefect_client: PrefectClient) -> None:

        @flow
        def my_subflow() -> None:
            logger: Any = get_run_logger()
            logger.info('Hello smaller world!')

        @flow
        def my_flow() -> State:
            logger: Any = get_run_logger()
            logger.info('Hello world!')
            return my_subflow(return_state=True)
        state: State = await my_flow(return_state=True).result()
        flow_run_id: uuid.UUID = state.state_details.flow_run_id
        subflow_run_id: uuid.UUID = (await state.result()).state_details.flow_run_id
        await _wait_for_logs(prefect_client, expected_num_logs=6)
        logs: List[Any] = await prefect_client.read_logs()
        log_messages: List[str] = [log.message for log in logs]
        assert all([log.task_run_id is None for log in logs])
        assert 'Hello world!' in log_messages, 'Parent log message is present'
        assert logs[log_messages.index('Hello world!')].flow_run_id == flow_run_id, 'Parent log message has correct id'
        assert 'Hello smaller world!' in log_messages, 'Child log message is present'
        assert logs[log_messages.index('Hello smaller world!')].flow_run_id == subflow_run_id, 'Child log message has correct id'

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    @pytest.mark.xfail(reason='Weird state sharing between new and old engine tests')
    async def test_subflow_logs_are_written_correctly_with_tasks(self, prefect_client: PrefectClient) -> None:

        @task
        def a_log_task() -> None:
            logger: Any = get_run_logger()
            logger.info('Task log')

        @flow
        def my_subflow() -> None:
            a_log_task()
            logger: Any = get_run_logger()
            logger.info('Hello smaller world!')

        @flow
        def my_flow() -> State:
            logger: Any = get_run_logger()
            logger.info('Hello world!')
            return my_subflow(return_state=True)
        subflow_state: State = my_flow()
        subflow_run_id: uuid.UUID = subflow_state.state_details.flow_run_id
        logs: List[Any] = await prefect_client.read_logs()
        log_messages: List[str] = [log.message for log in logs]
        task_run_logs: List[Any] = [log for log in logs if log.task_run_id is not None]
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

    async def test_flow_retry_with_error_in_flow_and_successful_task(self):
        task_run_count: int = 0
        flow_run_count: int = 0

        @task(persist_result=True)
        def my_task() -> str:
            nonlocal task_run_count
            task_run_count += 1
            return 'hello'

        @flow(retries=1, persist_result=True)
        def foo() -> str:
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
        def my_flow() -> str:
            nonlocal flow_run_count
            flow_run_count += 1
            fut: str = my_task()
            if flow_run_count == 1:
                raise ValueError()
            return fut
        assert my_flow() == 'hello'
        assert flow_run_count == 2
        assert task_run_count == 2, 'Task should be reset and run again'

    @pytest.mark.xfail
    async def test_flow_retry_with_branched_tasks(self, prefect_client: PrefectClient) -> None:
        flow_run_count: int = 0

        @task
        def identity(value: str) -> str:
            return value

        @flow(retries=1)
        def my_flow() -> str:
            nonlocal flow_run_count
            flow_run_count += 1
            if flow_run_count == 1:
                identity('foo')
                raise ValueError()
            else:
                result: str = identity('bar')
            return result
        my_flow()
        assert flow_run_count == 2
        document: Any = await prefect_client.retrieve_data(await my_flow().result())
        assert document == 'bar'

    async def test_flow_retry_with_no_error_in_flow_and_one_failed_child_flow(self, prefect_client: PrefectClient) -> None:
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
        def parent_flow() -> str:
            nonlocal flow_run_count
            flow_run_count += 1
            return child_flow()
        state: str = parent_flow()
        assert state == 'hello'
        assert flow_run_count == 2
        assert child_run_count == 2, 'Child flow should run again'

        task_runs: List[Any] = await prefect_client.read_task_runs(flow_run_filter=FlowRunFilter(id={'any_': [state]}))
        state_types: set = {task_run.state_type for task_run in task_runs}
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
        def parent_flow() -> str:
            nonlocal flow_run_count
            flow_run_count += 1
            child_result: str = child_flow()
            if flow_run_count == 1:
                raise ValueError()
            return child_result
        assert parent_flow() == 'hello'
        assert flow_run_count == 2
        assert child_run_count == 1, 'Child flow should not run again'

    async def test_flow_retry_with_error_in_flow_and_one_failed_child_flow(self, prefect_client: PrefectClient) -> None:
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
        def parent_flow_with_failure() -> State:
            nonlocal flow_run_count
            flow_run_count += 1
            state: State = child_flow_with_failure(return_state=True)
            if flow_run_count == 1:
                raise ValueError()
            return state
        parent_state: State = parent_flow_with_failure(return_state=True)
        child_state: State = await parent_state.result()
        assert await child_state.result() == 'hello'
        assert flow_run_count == 2
        assert child_flow_run_count == 2, 'Child flow should run again'

        child_flow_run: Any = await prefect_client.read_flow_run(child_state.state_details.flow_run_id)
        child_flow_runs: List[Any] = await prefect_client.read_flow_runs(flow_filter=FlowFilter(id={'any_': [child_flow_run.flow_id]}), sort=FlowRunSort.EXPECTED_START_TIME_ASC)
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
        def child_flow() -> str:
            nonlocal child_flow_run_count
            child_flow_run_count += 1
            return child_task()

        @flow(retries=1)
        def parent_flow() -> str:
            nonlocal flow_run_count
            flow_run_count += 1
            state: State = child_flow()
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
        def foo() -> str:
            nonlocal flow_run_count, task_run_retry_count
            task_run_retry_count = 0
            flow_run_count += 1
            fut: str = my_task()
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
        def my_flow() -> str:
            nonlocal flow_run_count
            flow_run_count += 1
            fut: str = my_task()
            if flow_run_count == 1:
                raise ValueError()
            return fut
        with pytest.raises(ValueError, match='This task always fails'):
            my_flow().result().result()
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
            code: str = '\nfrom prefect import Flow\n\n@Flow\ndef test_flow():\n    return 1\n'
            if self._base_path:
                with open(self._base_path / 'flows.py', 'w') as f:
                    f.write(code)

        def to_pull_step(self) -> Dict[str, Any]:
            return {}

class TestFlowFromSource:

    def test_load_flow_from_source_on_flow_function(self) -> None:
        assert hasattr(flow, 'from_source')

    class TestSync:

        def test_load_flow_from_source_with_storage(self) -> None:
            storage: MockStorage = MockStorage()
            loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
            assert isinstance(loaded_flow, Flow)
            assert loaded_flow.name == 'test-flow'
            assert loaded_flow() == 1

        def test_loaded_flow_to_deployment_has_storage(self) -> None:
            storage: MockStorage = MockStorage()
            loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
            deployment: RunnerDeployment = loaded_flow.to_deployment(name='test')
            assert deployment.storage == storage

        def test_loaded_flow_can_be_updated_with_options(self) -> None:
            storage: MockStorage = MockStorage()
            storage.set_base_path(Path.cwd())
            loaded_flow: Flow = Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
            flow_with_options: Flow = loaded_flow.with_options(name='with_options')
            deployment: RunnerDeployment = flow_with_options.to_deployment(name='test')
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
                code: str = dedent('                    from prefect import flow\n\n                    @flow\n                    def test_flow():\n                        return 1\n                    ')

                async def get_directory(self, local_path: Path) -> None:
                    (Path(local_path) / 'flows.py').write_text(self.code)
            block: FakeStorageBlock = FakeStorageBlock()
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
            storage: LocalStorage = LocalStorage(path='/tmp/test')
            mock_load_flow: MagicMock = MagicMock(return_value=MagicMock(spec=Flow))
            monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', mock_load_flow)
            pull_code_spy: AsyncMock = AsyncMock()
            monkeypatch.setattr(LocalStorage, 'pull_code', pull_code_spy)
            Flow.from_source(entrypoint='flows.py:test_flow', source=storage)
            pull_code_spy.assert_not_called()

    class TestAsync:

        async def test_load_flow_from_source_with_storage(self) -> None:
            storage: MockStorage = MockStorage()
            loaded_flow: Flow = await Flow.afrom_source(entrypoint='flows.py:test_flow', source=storage)
            assert isinstance(loaded_flow, Flow)
            assert loaded_flow.name == 'test-flow'
            assert loaded_flow() == 1

        async def test_loaded_flow_to_deployment_has_storage(self) -> None:
            storage: MockStorage = MockStorage()
            loaded_flow: Flow = await Flow.afrom_source(entrypoint='flows.py:test_flow', source=storage)
            deployment: RunnerDeployment = await loaded_flow.ato_deployment(name='test')
            assert deployment.storage == storage

        async def test_loaded_flow_can_be_updated_with_options(self) -> None:
            storage: MockStorage = MockStorage()
            storage.set_base_path(Path.cwd())
            loaded_flow: Flow = await Flow.afrom_source(entrypoint='flows.py:test_flow', source=storage)
            flow_with_options: Flow = loaded_flow.with_options(name='with_options')
            deployment: RunnerDeployment = await flow_with_options.ato_deployment(name='test')
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
                code: str = dedent('                    from prefect import flow\n\n                    @flow\n                    def test_flow():\n                        return 1\n                    ')

                async def get_directory(self, local_path: Path) -> None:
                    (Path(local_path) / 'flows.py').write_text(self.code)
            block: FakeStorageBlock = FakeStorageBlock()
            loaded_flow: Flow = await Flow.afrom_source(entrypoint='flows.py:test_flow', source=block)
            assert loaded_flow() == 1

        async def test_no_pull_for_local_storage(self, monkeypatch: Any) -> None:
            from prefect.runner.storage import LocalStorage
            storage: LocalStorage = LocalStorage(path='/tmp/test')
            mock_load_flow: MagicMock = MagicMock(return_value=MagicMock(spec=Flow))
            monkeypatch.setattr('prefect.flows.load_flow_from_entrypoint', mock_load_flow)
            pull_code_spy: AsyncMock = AsyncMock()
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
    def local_flow(self) -> Callable[..., Any]:

        @flow
        def local_flow_deploy() -> None:
            pass
        return local_flow_deploy

    @pytest.fixture
    async def remote_flow(self) -> Flow:
        remote_flow: Flow = await flow.from_source(entrypoint='flows.py:test_flow', source=MockStorage())
        return remote_flow

    async def test_calls_deploy_with_expected_args(self, mock_deploy: AsyncMock, local_flow: Callable[..., Any], work_pool: Any, capsys: Any) -> None:
        image: DockerImage = DockerImage(name='my-repo/my-image', tag='dev', build_kwargs={'pull': False})
        await local_flow.deploy(
            name='test',
            tags=['price', 'luggage'],
            parameters={'name': 'Arthur'},
            concurrency_limit=42,
            description='This is a test',
            version='alpha',
            enforce_parameter_schema=True,
            triggers=[{'name': 'Happiness', 'enabled': True, 'match': {'prefect.resource.id': 'prefect.flow-run.*'}, 'expect': ['prefect.flow-run.Completed'], 'match_related': {'prefect.resource.name': 'seed', 'prefect.resource.role': 'flow'}}]
        )
        mock_deploy.assert_called_once_with(
            await local_flow.to_deployment(
                name='test',
                tags=['price', 'luggage'],
                parameters={'name': 'Arthur'},
                concurrency_limit=42,
                description='This is a test',
                version='alpha',
                work_queue_name='line',
                job_variables={'foo': 'bar'},
                enforce_parameter_schema=True,
                paused=True
            ),
            work_pool_name=work_pool.name,
            image=image,
            build=False,
            push=False,
            print_next_steps_message=False,
            ignore_warnings=False
        )
        console_output: str = capsys.readouterr().out
        assert 'prefect worker start --pool' in console_output
        assert work_pool.name in console_output
        assert "prefect deployment run 'test-flow/test'" in console_output

    async def test_calls_deploy_with_expected_args_remote_flow(self, mock_deploy: AsyncMock, remote_flow: Flow, work_pool: Any) -> None:
        image: DockerImage = DockerImage(name='my-repo/my-image', tag='dev', build_kwargs={'pull': False})
        await remote_flow.deploy(
            name='test',
            tags=['price', 'luggage'],
            parameters={'name': 'Arthur'},
            description='This is a test',
            version='alpha',
            enforce_parameter_schema=True,
            work_pool_name=work_pool.name,
            work_queue_name='line',
            job_variables={'foo': 'bar'},
            image=image,
            push=False,
            paused=True,
            schedule=Schedule(interval=3600, anchor_date=datetime.datetime(2025, 1, 1), parameters={'number': 42})
        )
        mock_deploy.assert_called_once_with(
            await remote_flow.to_deployment(
                name='test',
                tags=['price', 'luggage'],
                parameters={'name': 'Arthur'},
                description='This is a test',
                version='alpha',
                enforce_parameter_schema=True,
                work_queue_name='line',
                job_variables={'foo': 'bar'},
                paused=True,
                schedule=Schedule(interval=3600, anchor_date=datetime.datetime(2025, 1, 1), parameters={'number': 42})
            ),
            work_pool_name=work_pool.name,
            image=image,
            build=True,
            push=False,
            print_next_steps_message=False,
            ignore_warnings=False
        )

    async def test_deploy_non_existent_work_pool(self, mock_deploy: AsyncMock, local_flow: Callable[..., Any]) -> None:
        with pytest.raises(ValueError, match="Could not find work pool 'non-existent'."):
            await local_flow.deploy(name='test', work_pool_name='non-existent', image='my-repo/my-image')

    async def test_no_worker_command_for_push_pool(self, mock_deploy: AsyncMock, local_flow: Callable[..., Any], push_work_pool: Any, capsys: Any) -> None:
        await local_flow.deploy(name='test', work_pool_name=push_work_pool.name, image='my-repo/my-image')
        assert 'prefect worker start' not in capsys.readouterr().out

    async def test_no_worker_command_for_active_workers(self, mock_deploy: AsyncMock, local_flow: Callable[..., Any], work_pool: Any, capsys: Any, monkeypatch: Any) -> None:
        mock_read_workers_for_work_pool: AsyncMock = AsyncMock(return_value=[Worker(name='test-worker', work_pool_id=work_pool.id, status=WorkerStatus.ONLINE)])
        monkeypatch.setattr('prefect.client.orchestration.PrefectClient.read_workers_for_work_pool', mock_read_workers_for_work_pool)
        await local_flow.deploy(name='test', work_pool_name=work_pool.name, image='my-repo/my-image')
        assert 'prefect worker start' not in capsys.readouterr().out

    async def test_suppress_console_output(self, mock_deploy: AsyncMock, local_flow: Callable[..., Any], work_pool: Any, capsys: Any) -> None:
        await local_flow.deploy(name='test', work_pool_name=work_pool.name, image='my-repo/my-image', print_next_steps=False)
        assert not capsys.readouterr().out

class TestLoadFlowFromEntrypoint:

    def test_load_flow_from_entrypoint(self, tmp_path: Path) -> None:
        flow_code: str = '\n        from prefect import flow\n\n        @flow\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow: Flow = load_flow_from_entrypoint(f'{fpath}:dog')
        assert flow.fn() == 'woof!'

    def test_load_flow_from_entrypoint_with_absolute_path(self, tmp_path: Path) -> None:
        flow_code: str = '\n        from prefect import flow\n\n        @flow\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        absolute_fpath: str = str(fpath.resolve())
        flow: Flow = load_flow_from_entrypoint(f'{absolute_fpath}:dog')
        assert flow.fn() == 'woof!'

    def test_load_flow_from_entrypoint_with_module_path(self, monkeypatch: Any) -> None:

        @flow
        def pretend_flow() -> None:
            pass
        import_object_mock: MagicMock = MagicMock(return_value=pretend_flow)
        monkeypatch.setattr('prefect.flows.import_object', import_object_mock)
        result: Flow = load_flow_from_entrypoint('my.module.pretend_flow')
        assert result == pretend_flow
        import_object_mock.assert_called_with('my.module.pretend_flow')

    def test_load_flow_from_entrypoint_script_error_loads_placeholder(self, tmp_path: Path) -> None:
        flow_code: str = '\n        from not_a_module import not_a_function\n        from prefect import flow\n\n        @flow(description="Says woof!")\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow: Flow = load_flow_from_entrypoint(f'{fpath}:dog')
        assert flow.name == 'dog'
        assert flow.description == 'Says woof!'
        assert flow() == 'woof!'

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_handling_script_with_unprotected_call_in_flow_script(self, tmp_path: Path, caplog: Any, prefect_client: PrefectClient) -> None:
        flow_code_with_call: str = '\n        from prefect import flow\n        from prefect.logging import get_run_logger\n\n        @flow\n        def dog():\n            get_run_logger().warning("meow!")\n            return "woof!"\n\n        dog()\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code_with_call))
        with caplog.at_level('WARNING'):
            flow: Flow = load_flow_from_entrypoint(f'{fpath}:dog')
            assert "Script loading is in progress, flow 'dog' will not be executed. Consider updating the script to only call the flow" in caplog.text
        flow_runs: List[Any] = await prefect_client.read_flows()
        assert len(flow_runs) == 0
        res: str = flow()
        assert res == 'woof!'
        flow_runs = await prefect_client.read_flows()
        assert len(flow_runs) == 1

    def test_load_flow_from_entrypoint_with_use_placeholder_flow(self, tmp_path: Path) -> None:
        flow_code: str = '\n        from not_a_module import not_a_function\n        from prefect import flow\n\n        @flow(description="Says woof!")\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow: Flow = load_flow_from_entrypoint(f'{fpath}:dog')
        assert isinstance(flow, Flow)
        assert flow() == 'woof!'
        with pytest.raises(ScriptError):
            load_flow_from_entrypoint(f'{fpath}:dog', use_placeholder_flow=False)

    def test_load_flow_from_entrypoint_with_url(self, monkeypatch: Any) -> None:
        flow_code: str = '\n        from prefect import flow\n\n        @flow\n        def dog():\n            return "woof!"\n        '
        fpath: Path = Path.cwd() / 'f.py'
        fpath.write_text(dedent(flow_code))
        absolute_fpath: str = str(fpath.resolve())
        flow: Flow = load_flow_from_entrypoint(f'{absolute_fpath}:dog')
        assert flow.fn() == 'woof!'

class TestLoadFunctionAndConvertToFlow:

    def test_func_is_a_flow(self, tmp_path: Path) -> None:
        flow_code: str = '\n        from prefect import flow\n\n        @flow\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow: Flow = load_function_and_convert_to_flow(f'{fpath}:dog')
        assert flow.fn() == 'woof!'
        assert isinstance(flow, Flow)
        assert flow.name == 'dog'

    def test_func_is_not_a_flow(self, tmp_path: Path) -> None:
        flow_code: str = '\n        def dog():\n            return "woof!"\n        '
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        flow: Flow = load_function_and_convert_to_flow(f'{fpath}:dog')
        assert isinstance(flow, Flow)
        assert flow.name == 'dog'
        assert flow.log_prints is True
        assert flow.fn() == 'woof!'

    def test_func_not_found(self, tmp_path: Path) -> None:
        flow_code: str = ''
        fpath: Path = tmp_path / 'f.py'
        fpath.write_text(dedent(flow_code))
        with pytest.raises(RuntimeError, match=f"Function with name 'dog' not found in '{fpath}'."):
            load_function_and_convert_to_flow(f'{fpath}:dog')

class TestFlowRunName:

    async def test_invalid_runtime_run_name(self) -> None:

        class InvalidFlowRunNameArg:

            @staticmethod
            def format(*args: Any, **kwargs: Any) -> None:
                pass

        @flow
        def my_flow() -> None:
            pass
        my_flow.flow_run_name = InvalidFlowRunNameArg()
        with pytest.raises(TypeError, match="Expected string or callable for 'flow_run_name'; got InvalidFlowRunNameArg instead."):
            my_flow()

    async def test_sets_run_name_when_provided(self, prefect_client: PrefectClient) -> None:

        @flow(flow_run_name='hi')
        def flow_with_name(foo: str = 'bar', bar: int = 1) -> None:
            pass
        state: State = flow_with_name(return_state=True)
        assert state.type == StateType.COMPLETED
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert flow_run.name == 'hi'

    async def test_sets_run_name_with_params_including_defaults(self, prefect_client: PrefectClient) -> None:

        @flow(flow_run_name='hi-{foo}-{bar}')
        def flow_with_name(foo: str = 'one', bar: str = '1') -> None:
            pass
        state: State = flow_with_name(bar='two', return_state=True)
        assert state.type == StateType.COMPLETED
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert flow_run.name == 'hi-one-two'

    async def test_sets_run_name_with_function(self, prefect_client: PrefectClient) -> None:

        def generate_flow_run_name() -> str:
            return 'hi'

        @flow(flow_run_name=generate_flow_run_name)
        def flow_with_name(foo: str = 'one', bar: str = '1') -> None:
            pass
        state: State = flow_with_name(bar='two', return_state=True)
        assert state.type == StateType.COMPLETED
        flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert flow_run.name == 'hi'

    async def test_sets_run_name_with_function_using_runtime_context(self, prefect_client: PrefectClient) -> None:

        def generate_flow_run_name() -> str:
            params: Dict[str, Any] = flow_run_ctx.parameters
            tokens: List[str] = ['hi']
            print(f'got the parameters {params!r}')
            if 'foo' in params:
                tokens.append(str(params['foo']))
            if 'bar' in params:
                tokens.append(str(params['bar']))
            return '-'.join(tokens)

        @flow(flow_run_name=generate_flow_run_name)
        def flow_with_name(foo: str = 'one', bar: str = '1') -> None:
            pass
        state: State = flow_with_name(bar='two', return_state=True)
        assert state.type == StateType.COMPLETED
        flow_run: Any = await prefect_client.read_flow_run(state.state_details.flow_run_id)
        assert flow_run.name == 'hi-one-two'

    async def test_sets_run_name_with_function_not_returning_string(self, prefect_client: PrefectClient) -> None:

        def generate_flow_run_name() -> None:
            pass

        @flow(flow_run_name=generate_flow_run_name)
        def flow_with_name(foo: str = 'one', bar: str = '1') -> None:
            pass
        with pytest.raises(TypeError, match=r"Callable <function TestFlowRunName\.test_sets_run_name_with_function_not_returning_string\.<locals>\.generate_flow_run_name at .*> for 'flow_run_name' returned type NoneType but a string is required."):
            flow_with_name(bar='two')

    async def test_sets_run_name_once(self) -> None:
        generate_flow_run_name: MagicMock = MagicMock(return_value='some-string')

        def flow_method() -> None:
            pass
        mocked_flow_method: MagicMock = create_autospec(flow_method, side_effect=RuntimeError('some-error'))
        decorated_flow: Flow = flow(flow_run_name=generate_flow_run_name, retries=3)(mocked_flow_method)
        state: State = decorated_flow(return_state=True)
        assert state.type == StateType.FAILED
        assert mocked_flow_method.call_count == 4
        assert generate_flow_run_name.call_count == 1

    async def test_sets_run_name_once_per_call(self) -> None:
        generate_flow_run_name: MagicMock = MagicMock(return_value='some-string')

        def flow_method() -> None:
            pass
        mocked_flow_method: MagicMock = create_autospec(flow_method, return_value='hello')
        decorated_flow: Flow = flow(flow_run_name=generate_flow_run_name)(mocked_flow_method)
        state1: State = decorated_flow(return_state=True)
        assert state1.type == StateType.COMPLETED
        assert mocked_flow_method.call_count == 1
        assert generate_flow_run_name.call_count == 1
        state2: State = decorated_flow(return_state=True)
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

def create_hook(mock_obj: MagicMock) -> Callable[..., None]:

    def my_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
        mock_obj()
    return my_hook

def create_async_hook(mock_obj: MagicMock) -> Callable[..., Any]:

    async def my_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
        mock_obj()
    return my_hook

class TestFlowHooksContext:

    @pytest.mark.parametrize('hook_type, fn_body, expected_exc', [
        ('on_completion', lambda: None, None),
        ('on_failure', lambda: 100 / 0, ZeroDivisionError),
        ('on_cancellation', lambda: Cancelling(), UnfinishedRun)
    ])
    def test_hooks_are_called_within_flow_run_context(
        self, 
        caplog: Any, 
        hook_type: str, 
        fn_body: Callable[..., Any], 
        expected_exc: Optional[Type[Exception]]
    ) -> None:

        def hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            ctx: Optional[FlowRunContext] = get_run_context()
            assert ctx is not None
            assert ctx.flow_run and ctx.flow_run == flow_run
            assert ctx.flow_run.state == state
            assert ctx.flow == flow

        @flow(**{hook_type: [hook]})
        def foo_flow() -> None:
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
        data: Dict[str, Any] = {}

        def hook(flow: Flow, flow_run: FlowRunContext, state: State, foo: int = 42) -> None:
            data.update(name=hook.__name__, state=state, foo=foo)

        @flow(on_completion=[hook])
        def foo_flow() -> None:
            pass
        state: State = foo_flow(return_state=True)
        assert data == dict(name='hook', state=state, foo=42)

    def test_hook_with_bound_kwargs(self) -> None:
        data: Dict[str, Any] = {}

        def hook(flow: Flow, flow_run: FlowRunContext, state: State, **kwargs: Any) -> None:
            data.update(name=hook.__name__, state=state, kwargs=kwargs)
        hook_with_kwargs: Callable[..., Any] = partial(hook, foo=42)

        @flow(on_completion=[hook_with_kwargs])
        def foo_flow() -> None:
            pass
        state: State = foo_flow(return_state=True)
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
        def completed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('completed1')

        @my_flow.on_completion
        def completed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('completed2')
        state: State = my_flow(return_state=True)
        assert state.type == StateType.COMPLETED
        assert my_mock.call_args_list == [call('completed1'), call('completed2')]

    def test_on_completion_hooks_run_on_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def completed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('completed1')

        def completed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('completed2')

        @flow(on_completion=[completed1, completed2])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)
        assert state.type == StateType.COMPLETED
        assert my_mock.call_args_list == [call('completed1'), call('completed2')]

    def test_on_completion_hooks_dont_run_on_failure(self) -> None:
        my_mock: MagicMock = MagicMock()

        def completed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('completed1')

        def completed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('completed2')

        @flow(on_completion=[completed1, completed2])
        def my_flow() -> None:
            raise Exception('oops')
        state: State = my_flow(return_state=True)
        assert state.type == StateType.FAILED
        my_mock.assert_not_called()

    def test_other_completion_hooks_run_if_a_hook_fails(self) -> None:
        my_mock: MagicMock = MagicMock()

        def completed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('completed1')

        def exception_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            raise Exception('oops')

        def completed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('completed2')

        @flow(on_completion=[completed1, exception_hook, completed2])
        def my_flow() -> None:
            pass
        my_flow(return_state=True)
        assert my_mock.call_args_list == [call('completed1'), call('completed2')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_completion_hooks_work_with_sync_and_async(self, hook1: Callable[[MagicMock], Callable[..., None]], hook2: Callable[[MagicMock], Callable[..., Any]]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock: Callable[..., Any] = hook1(my_mock)
        hook2_with_mock: Callable[..., Any] = hook2(my_mock)

        @flow(on_completion=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)
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
        def failed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed1')

        @my_flow.on_failure
        def failed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed2')
        state: State = my_flow(return_state=True)
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call('failed1'), call('failed2')]

    def test_on_failure_hooks_run_on_failure(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed1')

        def failed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed2')

        @flow(on_failure=[failed1, failed2])
        def my_flow() -> None:
            raise Exception('oops')
        state: State = my_flow(return_state=True)
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call('failed1'), call('failed2')]

    def test_on_failure_hooks_dont_run_on_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed1')

        def failed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed2')

        @flow(on_failure=[failed1, failed2])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)
        assert state.type == StateType.COMPLETED
        my_mock.assert_not_called()

    def test_other_failure_hooks_run_if_a_hook_fails(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed1')

        def exception_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            raise Exception('oops')

        def failed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed2')

        @flow(on_failure=[failed1, exception_hook, failed2])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call('failed1'), call('failed2')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_failure_hooks_work_with_sync_and_async(self, hook1: Callable[[MagicMock], Callable[..., None]], hook2: Callable[[MagicMock], Callable[..., Any]]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock: Callable[..., Any] = hook1(my_mock)
        hook2_with_mock: Callable[..., Any] = hook2(my_mock)

        @flow(on_failure=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> None:
            pass
        state: State = my_flow(return_state=True)
        assert state.type == StateType.FAILED
        assert my_mock.call_args_list == [call(), call()]

    def test_on_failure_hooks_run_on_bad_parameters(self) -> None:
        my_mock: MagicMock = MagicMock()

        def failure_hook(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failure_hook')

        @flow(on_failure=[failure_hook])
        def my_flow(x: int) -> None:
            pass
        state: State = my_flow(x='x', return_state=True)
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
        def cancelled_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('cancelled_hook1')

        @my_flow.on_cancellation
        def cancelled_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('cancelled_hook2')
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call('cancelled_hook1'), call('cancelled_hook2')]

    def test_on_cancellation_hooks_run_on_cancelled_state(self) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('cancelled_hook1')

        def cancelled_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('cancelled_hook2')

        @flow(on_cancellation=[cancelled_hook1, cancelled_hook2])
        def my_flow() -> State:
            return State(type=StateType.CANCELLING)
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call('cancelled_hook1'), call('cancelled_hook2')]

    def test_on_cancellation_hooks_are_ignored_if_terminal_state_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('cancelled_hook1')

        def cancelled_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('cancelled_hook2')

        @flow(on_cancellation=[cancelled_hook1, cancelled_hook2])
        def my_passing_flow() -> State:
            pass
        state: State = my_passing_flow(return_state=True)
        assert state.type == StateType.COMPLETED
        my_mock.assert_not_called()

    def test_on_cancellation_hooks_are_ignored_if_terminal_state_failed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def cancelled_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('cancelled_hook1')

        def cancelled_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('cancelled_hook2')

        @flow(on_cancellation=[cancelled_hook1, cancelled_hook2])
        def my_failing_flow() -> State:
            raise Exception('Failing flow')
        state: State = my_failing_flow(return_state=True)
        assert state.type == StateType.FAILED
        my_mock.assert_not_called()

    def test_other_cancellation_hooks_run_if_one_hook_fails(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed1')

        def crashed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            raise Exception('Failing flow')

        def crashed3(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed3')

        @flow(on_cancellation=[crashed1, crashed2, crashed3])
        def my_flow() -> State:
            return State(type=StateType.CANCELLING)
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call('crashed1'), call('crashed3')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_cancellation_hooks_work_with_sync_and_async(self, hook1: Callable[[MagicMock], Callable[..., None]], hook2: Callable[[MagicMock], Callable[..., Any]]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock: Callable[..., Any] = hook1(my_mock)
        hook2_with_mock: Callable[..., Any] = hook2(my_mock)

        @flow(on_cancellation=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> State:
            return State(type=StateType.CANCELLING)
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call(), call()]

    @pytest.mark.skip(reason='Fails with new engine, passed on old engine')
    async def test_on_crashed_hook_called_on_sigterm_from_flow_with_cancelling_state(self, mock_sigterm_handler: Tuple[Callable[..., None], MagicMock]) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed')

        @task
        async def cancel_parent() -> None:
            async with get_client() as client:
                await client.set_flow_run_state(runtime.flow_run.id, State(type=StateType.CANCELLING), force=True)

        @flow(on_crashed=[crashed])
        async def my_flow() -> None:
            await cancel_parent()
            os.kill(os.getpid(), signal.SIGTERM)
        with pytest.raises(prefect.exceptions.TerminationSignal):
            await my_flow(return_state=True)
        assert my_mock.mock_calls == [call('crashed')]

    async def test_on_crashed_hook_not_called_on_sigterm_from_flow_without_cancelling_state(self, mock_sigterm_handler: Tuple[Callable[..., None], MagicMock]) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed')

        @flow(on_crashed=[crashed])
        def my_flow() -> State:
            os.kill(os.getpid(), signal.SIGTERM)
        with pytest.raises(prefect.exceptions.TerminationSignal):
            my_flow(return_state=True)
        my_mock.assert_not_called()

    def test_on_crashed_hooks_respect_env_var(self, monkeypatch: Any) -> None:
        my_mock: MagicMock = MagicMock()
        monkeypatch.setenv('PREFECT__ENABLE_CANCELLATION_AND_CRASHED_HOOKS', 'false')

        def crashed_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook1')

        def crashed_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook2')

        @flow(on_crashed=[crashed_hook1, crashed_hook2])
        def my_flow() -> State:
            return State(type=StateType.CRASHED)
        state: State = my_flow(return_state=True)
        assert state.type == StateType.CRASHED
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
        def crashed_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook1')

        @my_flow.on_crashed
        def crashed_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook2')
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call('crashed_hook1'), call('crashed_hook2')]

    def test_on_crashed_hooks_run_on_crashed_state(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook1')

        def crashed_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook2')

        @flow(on_crashed=[crashed_hook1, crashed_hook2])
        def my_flow() -> State:
            return State(type=StateType.CRASHED)
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call('crashed_hook1'), call('crashed_hook2')]

    def test_on_crashed_hooks_are_ignored_if_terminal_state_completed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook1')

        def crashed_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook2')

        @flow(on_crashed=[crashed_hook1, crashed_hook2])
        def my_passing_flow() -> State:
            pass
        state: State = my_passing_flow(return_state=True)
        assert state.type == StateType.COMPLETED
        my_mock.assert_not_called()

    def test_on_crashed_hooks_are_ignored_if_terminal_state_failed(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed_hook1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook1')

        def crashed_hook2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed_hook2')

        @flow(on_crashed=[crashed_hook1, crashed_hook2])
        def my_failing_flow() -> State:
            raise Exception('Failing flow')
        state: State = my_failing_flow(return_state=True)
        assert state.type == StateType.FAILED
        my_mock.assert_not_called()

    def test_other_crashed_hooks_run_if_one_hook_fails(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed1')

        def crashed2(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            raise Exception('Failing flow')

        def crashed3(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed3')

        @flow(on_crashed=[crashed1, crashed2, crashed3])
        def my_flow() -> State:
            return State(type=StateType.CRASHED)
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call('crashed1'), call('crashed3')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_crashed_hooks_work_with_sync_and_async(self, hook1: Callable[[MagicMock], Callable[..., None]], hook2: Callable[[MagicMock], Callable[..., Any]]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock: Callable[..., Any] = hook1(my_mock)
        hook2_with_mock: Callable[..., Any] = hook2(my_mock)

        @flow(on_crashed=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> State:
            return State(type=StateType.CRASHED)
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call(), call()]

    def test_on_crashed_hook_on_subflow_succeeds(self) -> None:
        my_mock: MagicMock = MagicMock()

        def crashed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('crashed1')

        def failed1(flow: Flow, flow_run: FlowRunContext, state: State) -> None:
            my_mock('failed1')

        @flow(on_crashed=[crashed1])
        def subflow() -> State:
            return State(type=StateType.CRASHED)

        @flow(on_failure=[failed1])
        def my_flow() -> State:
            subflow()
            return State(type=StateType.FAILED)
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call('crashed1'), call('failed1')]

    @pytest.mark.parametrize('hook1, hook2', [
        (create_hook, create_hook),
        (create_hook, create_async_hook),
        (create_async_hook, create_hook),
        (create_async_hook, create_async_hook)
    ])
    def test_on_crashed_hooks_work_with_sync_and_async(self, hook1: Callable[[MagicMock], Callable[..., None]], hook2: Callable[[MagicMock], Callable[..., Any]]) -> None:
        my_mock: MagicMock = MagicMock()
        hook1_with_mock: Callable[..., Any] = hook1(my_mock)
        hook2_with_mock: Callable[..., Any] = hook2(my_mock)

        @flow(on_crashed=[hook1_with_mock, hook2_with_mock])
        def my_flow() -> State:
            return State(type=StateType.CRASHED)
        my_flow(return_state=True)
        assert my_mock.mock_calls == [call(), call()]

class TestFlowServe:

    @property
    def flow(self) -> Flow:

        @flow
        def test_flow() -> None:
            pass
        return test_flow

    @pytest.fixture(autouse=True)
    async def mock_runner_start(self, monkeypatch: Any) -> AsyncMock:
        mock: AsyncMock = AsyncMock()
        monkeypatch.setattr('prefect.cli.flow.Runner.start', mock)
        return mock

    def test_serve_prints_message(self, capsys: Any) -> None:
        self.flow.serve('test')
        captured: Tuple[str, str] = capsys.readouterr()
        assert "Your flow 'test-flow' is being served and polling for scheduled runs!" in captured.out
        assert "$ prefect deployment run 'test-flow/test'" in captured.out

    def test_serve_creates_deployment(self, sync_prefect_client: SyncPrefectClient) -> None:
        self.flow.serve(name='test', tags=['price', 'luggage'], parameters={'name': 'Arthur'}, description='This is a test', version='alpha', enforce_parameter_schema=True, paused=True, global_limit=42)
        deployment: RunnerDeployment = sync_prefect_client.read_deployment_by_name(name='test-flow/test')
        assert deployment is not None
        assert deployment.work_pool_name is None
        assert deployment.work_queue_name is None
        assert deployment.name == 'test'
        assert deployment.tags == ['price', 'luggage']
        assert deployment.parameters == {'name': 'Arthur'}
        assert deployment.description == 'This is a test'
        assert deployment.version == 'alpha'
        assert deployment.enforce_parameter_schema
        assert deployment.paused
        assert deployment.global_concurrency_limit.limit == 42

    def test_serve_can_user_a_module_path_entrypoint(self, sync_prefect_client: SyncPrefectClient) -> None:
        deployment: RunnerDeployment = self.flow.serve(name='test', entrypoint_type=EntrypointType.MODULE_PATH)
        deployment: RunnerDeployment = sync_prefect_client.read_deployment_by_name(name='test-flow/test')
        assert deployment.entrypoint == f'{self.flow.__module__}.{self.flow.__name__}'

    def test_serve_handles__file__(self, sync_prefect_client: SyncPrefectClient) -> None:
        self.flow.serve(__file__)
        deployment: RunnerDeployment = sync_prefect_client.read_deployment_by_name(name='test-flow/test_flows')
        assert deployment.name == 'test_flows'

    def test_serve_creates_deployment_with_interval_schedule(self, sync_prefect_client: SyncPrefectClient) -> None:
        self.flow.serve('test', interval=3600)
        deployment: RunnerDeployment = sync_prefect_client.read_deployment_by_name(name='test-flow/test')
        assert deployment is not None
        assert len(deployment.schedules) == 1
        assert isinstance(deployment.schedules[0].schedule, IntervalSchedule)
        assert deployment.schedules[0].schedule.interval == datetime.timedelta(seconds=3600)

    def test_serve_creates_deployment_with_cron_schedule(self, sync_prefect_client: SyncPrefectClient) -> None:
        self.flow.serve('test', cron='* * * * *')
        deployment: RunnerDeployment = sync_prefect_client.read_deployment_by_name(name='test-flow/test')
        assert deployment is not None
        assert len(deployment.schedules) == 1
        assert deployment.schedules[0].schedule == CronSchedule(cron='* * * * *')

    def test_serve_creates_deployment_with_rrule_schedule(self, sync_prefect_client: SyncPrefectClient) -> None:
        self.flow.serve('test', rrule='FREQ=MINUTELY')
        deployment: RunnerDeployment = sync_prefect_client.read_deployment_by_name(name='test-flow/test')
        assert deployment is not None
        assert len(deployment.schedules) == 1
        assert deployment.schedules[0].schedule == RRuleSchedule(rrule='FREQ=MINUTELY')

    def test_serve_creates_deployment_with_schedules_with_parameters(self, sync_prefect_client: SyncPrefectClient) -> None:

        @flow
        def add_two(number: int) -> None:
            pass
        add_two.serve(
            'test',
            schedules=[
                Interval(3600, parameters={'number': 42}, slug='test-interval-schedule'),
                Cron('* * * * *', parameters={'number': 42}, slug='test-cron-schedule'),
                RRule('FREQ=MINUTELY', parameters={'number': 42}, slug='test-rrule-schedule')
            ]
        )
        deployment: RunnerDeployment = sync_prefect_client.read_deployment_by_name(name='add-two/test')
        assert deployment is not None
        assert len(deployment.schedules) == 3
        all_parameters: List[Dict[str, Any]] = [schedule.parameters for schedule in deployment.schedules]
        assert all((parameters == {'number': 42} for parameters in all_parameters))
        expected_slugs: set = {'test-interval-schedule', 'test-cron-schedule', 'test-rrule-schedule'}
        actual_slugs: set = {schedule.slug for schedule in deployment.schedules}
        assert actual_slugs == expected_slugs

    @pytest.mark.parametrize('kwargs', [
        {**d1, **d2} for d1, d2 in combinations([
            {'interval': 3600}, 
            {'cron': '* * * * *'}, 
            {'rrule': 'FREQ=MINUTELY'}, 
            {'schedules': [Interval(3600, slug='test-interval-schedule'), Cron('* * * * *', slug='test-cron-schedule'), RRule('FREQ=MINUTELY', slug='test-rrule-schedule')]}, 
            {'schedule': Interval(3600, slug='test-interval-schedule')}
        ], 2)
    ])
    def test_serve_raises_on_multiple_schedules(self, kwargs: Dict[str, Any]) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            expected_message: str = 'Only one of interval, cron, rrule, schedule, or schedules can be provided.'
            with pytest.raises(ValueError, match=expected_message):
                self.flow.serve(__file__, **kwargs)

    def test_serve_starts_a_runner(self, mock_runner_start: AsyncMock) -> None:
        """
        This test only makes sure Runner.start() is called. The actual
        functionality of the runner is tested in test_runner.py
        """
        self.flow.serve('test')
        mock_runner_start.assert_awaited_once()

    def test_serve_passes_limit_specification_to_runner(self, monkeypatch: Any) -> None:
        """
        Tests that the 'limit' argument is correctly passed to the Runner
        """
        runner_mock: MagicMock = MagicMock(return_value=AsyncMock())
        monkeypatch.setattr('prefect.runner.Runner', runner_mock)
        limit: int = 42
        self.flow.serve('test', limit=limit)
        runner_mock.assert_called_once_with(name='test', pause_on_shutdown=ANY, limit=limit)

class TestLoadFlowArgumentFromEntrypoint:

    def test_load_flow_name_from_entrypoint(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow(name="My custom name")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'My custom name'

    def test_load_flow_name_from_entrypoint_no_name(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        version: str = "1.0"\n\n        @flow(name=f"flow-function-{version}")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_flow_name_from_entrypoint_dyanmic_name_function(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        def get_name() -> str:\n            return "from-a-function"\n\n        @flow(name=get_name())\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'from-a-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_depends_on_missing_import(self, tmp_path: Path, caplog: Any) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        from non_existent import get_name\n\n        @flow(name=get_name())\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Optional[Dict[str, Any]] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'
        assert 'Failed to parse @flow argument: `name=get_name()`' in caplog.text

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring_multiline(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        flow_base_name: str = "flow-function"\n        version: str = "1.0"\n\n        @flow(\n            name=(\n                f"{flow_base_name}-"\n                f"{version}"\n            )\n        )\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_async_flow_from_entrypoint_no_name(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n        from prefect import flow\n\n        @flow\n        async def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_description_from_entrypoint(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow(description="My custom description")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['description'] == 'My custom description'

    def test_load_flow_description_from_entrypoint_no_description(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert 'description' not in result

    def test_load_no_flow(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        with pytest.raises(ValueError, match='Could not find flow'):
            load_flow_arguments_from_entrypoint(entrypoint)

    def test_function_with_enum_argument(self, tmp_path: Path) -> None:

        class Color(enum.Enum):
            RED: str = 'RED'
            GREEN: str = 'GREEN'
            BLUE: str = 'BLUE'
        source_code: str = dedent('\n        from enum import Enum\n\n        from prefect import flow\n\n        class Color(Enum):\n            RED = "RED"\n            GREEN = "GREEN"\n            BLUE = "BLUE"\n\n        @flow\n        def f(x: Color = Color.RED) -> str:\n            return x.name\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'f'
        loaded_flow: Flow = load_flow_from_entrypoint(entrypoint)
        assert loaded_flow() == 'RED'

    def test_handles_dynamically_created_models(self, tmp_path: Path) -> None:
        source_code: str = dedent('\n            from typing import Optional\n            from prefect import flow\n            from pydantic import BaseModel, create_model, Field\n\n\n            def get_model() -> BaseModel:\n                return create_model(\n                    "MyModel",\n                    param=(\n                        int,\n                        Field(\n                            title="param",\n                            default=1,\n                        ),\n                    ),\n                )\n\n\n            MyModel = get_model()\n\n\n            @flow\n            def f(\n                param: Optional[MyModel] = None,\n            ) -> None:\n                return MyModel()\n            ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result: Flow = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert isinstance(result, Flow)
        assert result() == ParameterTestModel(data=1)

    def test_raises_name_error_when_loaded_flow_cannot_run(self, tmp_path: Path) -> None:
        source_code: str = dedent('\n        from not_a_module import not_a_function\n\n        from prefect import flow\n\n        @flow(description="Says woof!")\n        def dog():\n            return not_a_function(\'dog\')\n            ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:dog'
        with pytest.raises(NameError, match="name 'not_a_function' is not defined"):
            safe_load_flow_from_entrypoint(entrypoint)()

class TestTransactions:

    def test_grouped_rollback_behavior(self) -> None:
        data1, data2: Tuple[Dict[str, Any], Dict[str, Any]] = ({}, {})

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
        main(return_state=True)
        assert data2['called'] is True
        assert data1['called'] is True

    def test_isolated_shared_state_on_txn_between_tasks(self) -> None:
        data1, data2: Tuple[Dict[str, Any], Dict[str, Any]] = ({}, {})

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
        main(return_state=True)
        assert data2['hook'] == 2
        assert data1['hook'] == 1

    def test_task_failure_causes_previous_to_rollback(self) -> None:
        data1, data2: Tuple[Dict[str, Any], Dict[str, Any]] = ({}, {})

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
        main(return_state=True)
        assert 'called' not in data2
        assert data1['called'] is True

    def test_task_doesnt_persist_prior_to_commit(self, tmp_path: Path) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('txn-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result')
        def task1() -> None:
            pass

        @task(result_storage=result_storage, result_storage_key='task2-result')
        def task2() -> None:
            raise RuntimeError('oopsie')

        @flow
        def main() -> Optional[ValueError]:
            with transaction():
                task1()
                task2()
            return None
        val: Optional[ValueError] = main()
        with pytest.raises(ValueError, match='does not exist'):
            result_storage.read_path('task1-result', _sync=True)

    def test_task_persists_only_at_commit(self, tmp_path: Path) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('moar-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result-A', persist_result=True)
        def task1() -> Dict[str, Any]:
            return dict(some='data')

        @task(result_storage=result_storage, result_storage_key='task2-result-B', persist_result=True)
        def task2() -> None:
            pass

        @flow
        def main() -> Optional[ValueError]:
            retval: Optional[ValueError] = None
            with transaction():
                task1()
                try:
                    result_storage.read_path('task1-result-A', _sync=True)
                except ValueError as exc:
                    retval = exc
                task2()
            return retval
        val: Optional[ValueError] = main()
        assert isinstance(val, ValueError)
        assert 'does not exist' in str(val)
        content: Any = result_storage.read_path('task1-result-A', _sync=True)
        record: ResultRecord = ResultRecord.deserialize(content)
        assert record.result == {'some': 'data'}

    def test_commit_isnt_called_on_rollback(self) -> None:
        data: Dict[str, Any] = {}

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
        main(return_state=True)
        assert data == {}

class TestLoadFlowArgumentFromEntrypoint:

    def test_load_flow_name_from_entrypoint(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow(name="My custom name")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'My custom name'

    def test_load_flow_name_from_entrypoint_no_name(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        version: str = "1.0"\n\n        @flow(name=f"flow-function-{version}")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_flow_name_from_entrypoint_dyanmic_name_function(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        def get_name() -> str:\n            return "from-a-function"\n\n        @flow(name=get_name())\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'from-a-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_depends_on_missing_import(self, tmp_path: Path, caplog: Any) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        from non_existent import get_name\n\n        @flow(name=get_name())\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Optional[Dict[str, Any]] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'
        assert 'Failed to parse @flow argument: `name=get_name()`' in caplog.text

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring_multiline(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        flow_base_name: str = "flow-function"\n        version: str = "1.0"\n\n        @flow(\n            name=(\n                f"{flow_base_name}-"\n                f"{version}"\n            )\n        )\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_async_flow_from_entrypoint_no_name(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n        from prefect import flow\n\n        @flow\n        async def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_description_from_entrypoint(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow(description="My custom description")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['description'] == 'My custom description'

    def test_load_flow_description_from_entrypoint_no_description(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert 'description' not in result

    def test_load_no_flow(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        with pytest.raises(ValueError, match='Could not find flow'):
            load_flow_arguments_from_entrypoint(entrypoint)

    def test_function_with_enum_argument(self, tmp_path: Path) -> None:

        class Color(enum.Enum):
            RED: str = 'RED'
            GREEN: str = 'GREEN'
            BLUE: str = 'BLUE'
        source_code: str = dedent('\n        from enum import Enum\n\n        from prefect import flow\n\n        class Color(Enum):\n            RED = "RED"\n            GREEN = "GREEN"\n            BLUE = "BLUE"\n\n        @flow\n        def f(x: Color = Color.RED) -> str:\n            return x.name\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'f'
        loaded_flow: Flow = load_flow_from_entrypoint(entrypoint)
        assert loaded_flow() == 'RED'

    def test_handles_dynamically_created_models(self, tmp_path: Path) -> None:
        source_code: str = dedent('\n            from typing import Optional\n            from prefect import flow\n            from pydantic import BaseModel, create_model, Field\n\n\n            def get_model() -> BaseModel:\n                return create_model(\n                    "MyModel",\n                    param=(\n                        int,\n                        Field(\n                            title="param",\n                            default=1,\n                        ),\n                    ),\n                )\n\n\n            MyModel = get_model()\n\n\n            @flow\n            def f(\n                param: Optional[MyModel] = None,\n            ) -> None:\n                return MyModel()\n            ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result: Flow = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert isinstance(result, Flow)
        assert result() == ParameterTestModel(data=1)

    def test_raises_name_error_when_loaded_flow_cannot_run(self, tmp_path: Path) -> None:
        source_code: str = dedent('\n        from not_a_module import not_a_function\n\n        from prefect import flow\n\n        @flow(description="Says woof!")\n        def dog():\n            return not_a_function(\'dog\')\n            ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:dog'
        with pytest.raises(NameError, match="name 'not_a_function' is not defined"):
            safe_load_flow_from_entrypoint(entrypoint)()

class TestTransactions:

    def test_grouped_rollback_behavior(self) -> None:
        data1: Dict[str, Any] = {}
        data2: Dict[str, Any] = {}

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
        main(return_state=True)
        assert data2['called'] is True
        assert data1['called'] is True

    def test_isolated_shared_state_on_txn_between_tasks(self) -> None:
        data1: Dict[str, Any] = {}
        data2: Dict[str, Any] = {}

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
        main(return_state=True)
        assert data2['hook'] == 2
        assert data1['hook'] == 1

    def test_task_failure_causes_previous_to_rollback(self) -> None:
        data1: Dict[str, Any] = {}
        data2: Dict[str, Any] = {}

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
        main(return_state=True)
        assert 'called' not in data2
        assert data1['called'] is True

    def test_task_doesnt_persist_prior_to_commit(self, tmp_path: Path) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('txn-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result')
        def task1() -> None:
            pass

        @task(result_storage=result_storage, result_storage_key='task2-result')
        def task2() -> None:
            raise RuntimeError('oopsie')

        @flow
        def main() -> Optional[ValueError]:
            with transaction():
                task1()
                task2()
            return None
        val: Optional[ValueError] = main()
        with pytest.raises(ValueError, match='does not exist'):
            result_storage.read_path('task1-result', _sync=True)

    def test_task_persists_only_at_commit(self, tmp_path: Path) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('moar-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result-A', persist_result=True)
        def task1() -> Dict[str, Any]:
            return dict(some='data')

        @task(result_storage=result_storage, result_storage_key='task2-result-B', persist_result=True)
        def task2() -> None:
            pass

        @flow
        def main() -> Optional[ValueError]:
            retval: Optional[ValueError] = None
            with transaction():
                task1()
                try:
                    result_storage.read_path('task1-result-A', _sync=True)
                except ValueError as exc:
                    retval = exc
                task2()
            return retval
        val: Optional[ValueError] = main()
        assert isinstance(val, ValueError)
        assert 'does not exist' in str(val)
        content: Any = result_storage.read_path('task1-result-A', _sync=True)
        record: ResultRecord = ResultRecord.deserialize(content)
        assert record.result == {'some': 'data'}

    def test_commit_isnt_called_on_rollback(self) -> None:
        data: Dict[str, Any] = {}

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
        main(return_state=True)
        assert data == {}

class TestLoadFlowArgumentFromEntrypoint:

    def test_load_flow_name_from_entrypoint(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow(name="My custom name")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'My custom name'

    def test_load_flow_name_from_entrypoint_no_name(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        version: str = "1.0"\n\n        @flow(name=f"flow-function-{version}")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_flow_name_from_entrypoint_dyanmic_name_function(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        def get_name() -> str:\n            return "from-a-function"\n\n        @flow(name=get_name())\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'from-a-function'

    def test_load_flow_name_from_entrypoint_dynamic_name_depends_on_missing_import(self, tmp_path: Path, caplog: Any) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        from non_existent import get_name\n\n        @flow(name=get_name())\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Optional[Dict[str, Any]] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'
        assert 'Failed to parse @flow argument: `name=get_name()`' in caplog.text

    def test_load_flow_name_from_entrypoint_dynamic_name_fstring_multiline(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        flow_base_name: str = "flow-function"\n        version: str = "1.0"\n\n        @flow(\n            name=(\n                f"{flow_base_name}-"\n                f"{version}"\n            )\n        )\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function-1.0'

    def test_load_async_flow_from_entrypoint_no_name(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n        from prefect import flow\n\n        @flow\n        async def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'flow-function'

    def test_load_flow_description_from_entrypoint(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow(description="My custom description")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['description'] == 'My custom description'

    def test_load_flow_description_from_entrypoint_no_description(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert 'description' not in result

    def test_load_no_flow(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        with pytest.raises(ValueError, match='Could not find flow'):
            load_flow_arguments_from_entrypoint(entrypoint)

    def test_function_with_enum_argument(self, tmp_path: Path) -> None:

        class Color(enum.Enum):
            RED: str = 'RED'
            GREEN: str = 'GREEN'
            BLUE: str = 'BLUE'

        source_code: str = dedent('\n        from enum import Enum\n\n        from prefect import flow\n\n        class Color(Enum):\n            RED = "RED"\n            GREEN = "GREEN"\n            BLUE = "BLUE"\n\n        @flow\n        def f(x: Color = Color.RED) -> str:\n            return x.name\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'f'
        loaded_flow: Flow = load_flow_from_entrypoint(entrypoint)
        assert loaded_flow() == 'RED'

    def test_handles_dynamically_created_models(self, tmp_path: Path) -> None:
        source_code: str = dedent('\n            from typing import Optional\n            from prefect import flow\n            from pydantic import BaseModel, create_model, Field\n\n\n            def get_model() -> BaseModel:\n                return create_model(\n                    "MyModel",\n                    param=(\n                        int,\n                        Field(\n                            title="param",\n                            default=1,\n                        ),\n                    ),\n                )\n\n\n            MyModel = get_model()\n\n\n            @flow\n            def f(\n                param: Optional[MyModel] = None,\n            ) -> None:\n                return MyModel()\n            ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:f'
        result: Flow = safe_load_flow_from_entrypoint(entrypoint)
        assert result is not None
        assert isinstance(result, Flow)
        assert result() == ParameterTestModel(data=1)

    def test_raises_name_error_when_loaded_flow_cannot_run(self, tmp_path: Path) -> None:
        source_code: str = dedent('\n        from not_a_module import not_a_function\n\n        from prefect import flow\n\n        @flow(description="Says woof!")\n        def dog() -> str:\n            return not_a_function(\'dog\')\n            ')
        tmp_path.joinpath('test.py').write_text(source_code)
        entrypoint: str = f'{tmp_path.joinpath("test.py")}:dog'
        with pytest.raises(NameError, match="name 'not_a_function' is not defined"):
            safe_load_flow_from_entrypoint(entrypoint)()

class TestTransactions:

    def test_grouped_rollback_behavior(self) -> None:
        data1: Dict[str, Any] = {}
        data2: Dict[str, Any] = {}

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
        main(return_state=True)
        assert data2['called'] is True
        assert data1['called'] is True

    def test_isolated_shared_state_on_txn_between_tasks(self) -> None:
        data1: Dict[str, Any] = {}
        data2: Dict[str, Any] = {}

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
        main(return_state=True)
        assert data2['hook'] == 2
        assert data1['hook'] == 1

    def test_task_failure_causes_previous_to_rollback(self) -> None:
        data1: Dict[str, Any] = {}
        data2: Dict[str, Any] = {}

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
        main(return_state=True)
        assert 'called' not in data2
        assert data1['called'] is True

    def test_task_doesnt_persist_prior_to_commit(self, tmp_path: Path) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('txn-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result')
        def task1() -> None:
            pass

        @task(result_storage=result_storage, result_storage_key='task2-result')
        def task2() -> None:
            raise RuntimeError('oopsie')

        @flow
        def main() -> Optional[ValueError]:
            with transaction():
                task1()
                task2()
            return None
        val: Optional[ValueError] = main()
        with pytest.raises(ValueError, match='does not exist'):
            result_storage.read_path('task1-result', _sync=True)

    def test_task_persists_only_at_commit(self, tmp_path: Path) -> None:
        result_storage: LocalFileSystem = LocalFileSystem(basepath=tmp_path)
        result_storage.save('moar-results', _sync=True)

        @task(result_storage=result_storage, result_storage_key='task1-result-A', persist_result=True)
        def task1() -> Dict[str, Any]:
            return dict(some='data')

        @task(result_storage=result_storage, result_storage_key='task2-result-B', persist_result=True)
        def task2() -> None:
            pass

        @flow
        def main() -> Optional[ValueError]:
            retval: Optional[ValueError] = None
            with transaction():
                task1()
                try:
                    result_storage.read_path('task1-result-A', _sync=True)
                except ValueError as exc:
                    retval = exc
                task2()
            return retval
        val: Optional[ValueError] = main()
        assert isinstance(val, ValueError)
        assert 'does not exist' in str(val)
        content: Any = result_storage.read_path('task1-result-A', _sync=True)
        record: ResultRecord = ResultRecord.deserialize(content)
        assert record.result == {'some': 'data'}

    def test_commit_isnt_called_on_rollback(self) -> None:
        data: Dict[str, Any] = {}

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
        main(return_state=True)
        assert data == {}

class TestLoadFlowArgumentFromEntrypoint:

    def test_load_flow_name_from_entrypoint(self, tmp_path: Path) -> None:
        flow_source: str = dedent('\n\n        from prefect import flow\n\n        @flow(name="My custom name")\n        def flow_function(name: str) -> str:\n            return name\n        ')
        tmp_path.joinpath('flow.py').write_text(flow_source)
        entrypoint: str = f'{tmp_path.joinpath("flow.py")}:flow_function'
        result: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
        assert result['name'] == 'My custom name'
