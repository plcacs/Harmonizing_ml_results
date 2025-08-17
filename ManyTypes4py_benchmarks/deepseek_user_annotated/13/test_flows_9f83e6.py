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
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
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
from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    Worker,
    WorkerStatus,
)
from prefect.client.schemas.schedules import (
    CronSchedule,
    IntervalSchedule,
    RRuleSchedule,
)
from prefect.context import FlowRunContext, get_run_context
from prefect.deployments.runner import RunnerDeployment, RunnerStorage
from prefect.docker.docker_image import DockerImage
from prefect.events import DeploymentEventTrigger, Posture
from prefect.exceptions import (
    CancelledRun,
    InvalidNameError,
    MissingFlowError,
    ParameterTypeError,
    ReservedArgumentError,
    ScriptError,
    UnfinishedRun,
)
from prefect.filesystems import LocalFileSystem
from prefect.flows import (
    Flow,
    load_flow_arguments_from_entrypoint,
    load_flow_from_entrypoint,
    load_flow_from_flow_run,
    load_function_and_convert_to_flow,
    safe_load_flow_from_entrypoint,
)
from prefect.logging import get_run_logger
from prefect.results import ResultRecord
from prefect.runtime import flow_run as flow_run_ctx
from prefect.schedules import Cron, Interval, RRule, Schedule
from prefect.server.schemas.core import TaskRunResult
from prefect.server.schemas.filters import FlowFilter, FlowRunFilter
from prefect.server.schemas.sorting import FlowRunSort
from prefect.settings import (
    PREFECT_FLOW_DEFAULT_RETRIES,
    temporary_settings,
)
from prefect.states import (
    Cancelled,
    Cancelling,
    Paused,
    PausedRun,
    State,
    StateType,
    raise_state_exception,
)
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.testing.utilities import (
    AsyncMock,
    exceptions_equal,
    get_most_recent_flow_run,
)
from prefect.transactions import get_transaction, transaction
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.annotations import allow_failure, quote
from prefect.utilities.callables import parameter_schema
from prefect.utilities.collections import flatdict_to_dict
from prefect.utilities.hashing import file_hash

T = TypeVar('T')

# Give an ample amount of sleep time in order to test flow timeouts
SLEEP_TIME: int = 10


@pytest.fixture
def mock_sigterm_handler() -> Iterable[Tuple[Callable[..., None], MagicMock]]:
    if threading.current_thread() != threading.main_thread():
        pytest.skip("Can't test signal handlers from a thread")
    mock = MagicMock()

    def handler(*args: Any, **kwargs: Any) -> None:
        mock(*args, **kwargs)

    prev_handler = signal.signal(signal.SIGTERM, handler)
    try:
        yield handler, mock
    finally:
        signal.signal(signal.SIGTERM, prev_handler)


class ParameterTestModel(pydantic.BaseModel):
    data: int


class ParameterTestClass:
    pass


class ParameterTestEnum(enum.Enum):
    X = 1
    Y = 2


class TestFlow:
    def test_initializes(self) -> None:
        f = Flow(
            name="test",
            fn=lambda **kwargs: 42,
            version="A",
            description="B",
            flow_run_name="hi",
        )
        assert f.name == "test"
        assert f.fn() == 42
        assert f.version == "A"
        assert f.description == "B"
        assert f.flow_run_name == "hi"

    def test_initializes_with_callable_flow_run_name(self) -> None:
        f = Flow(name="test", fn=lambda **kwargs: 42, flow_run_name=lambda: "hi")
        assert f.name == "test"
        assert f.fn() == 42
        assert f.flow_run_name() == "hi"

    def test_initializes_with_default_version(self) -> None:
        f = Flow(name="test", fn=lambda **kwargs: 42)
        assert isinstance(f.version, str)

    @pytest.mark.parametrize(
        "sourcefile", [None, "<stdin>", "<ipython-input-1-d31e8a6792d4>"]
    )
    def test_version_none_if_source_file_cannot_be_determined(
        self, monkeypatch: pytest.MonkeyPatch, sourcefile: Optional[str]
    ) -> None:
        monkeypatch.setattr(
            "prefect.flows.inspect.getsourcefile", MagicMock(return_value=sourcefile)
        )

        f = Flow(name="test", fn=lambda **kwargs: 42)
        assert f.version is None

    def test_raises_on_bad_funcs(self) -> None:
        with pytest.raises(TypeError):
            Flow(name="test", fn={})

    def test_default_description_is_from_docstring(self) -> None:
        def my_fn() -> None:
            """
            Hello
            """

        f = Flow(
            name="test",
            fn=my_fn,
        )
        assert f.description == "Hello"

    def test_default_name_is_from_function(self) -> None:
        def my_fn() -> None:
            pass

        f = Flow(
            fn=my_fn,
        )
        assert f.name == "my-fn"

    def test_raises_clear_error_when_not_compatible_with_validator(self) -> None:
        def my_fn(v__args: Any) -> None:
            pass

        with pytest.raises(
            ValueError,
            match="Flow function is not compatible with `validate_parameters`",
        ):
            Flow(fn=my_fn)

    @pytest.mark.parametrize(
        "name",
        [
            "my/flow",
            r"my%flow",
            "my<flow",
            "my>flow",
            "my&flow",
        ],
    )
    def test_invalid_name(self, name: str) -> None:
        with pytest.raises(InvalidNameError, match="contains an invalid character"):
            Flow(fn=lambda: 1, name=name)

    def test_lambda_name_coerced_to_legal_characters(self) -> None:
        f = Flow(fn=lambda: 42)
        assert f.name == "unknown-lambda"

    def test_invalid_run_name(self) -> None:
        class InvalidFlowRunNameArg:
            def format(*args: Any, **kwargs: Any) -> None:
                pass

        with pytest.raises(
            TypeError,
            match=(
                "Expected string or callable for 'flow_run_name'; got"
                " InvalidFlowRunNameArg instead."
            ),
        ):
            Flow(fn=lambda: 1, name="hello", flow_run_name=InvalidFlowRunNameArg())

    def test_using_return_state_in_flow_definition_raises_reserved(self) -> None:
        with pytest.raises(
            ReservedArgumentError, match="'return_state' is a reserved argument name"
        ):
            Flow(name="test", fn=lambda return_state: 42, version="A", description="B")

    def test_param_description_from_docstring(self) -> None:
        def my_fn(x: Any) -> None:
            """
            Hello

            Args:
                x: description
            """

        f = Flow(fn=my_fn)
        assert parameter_schema(f).properties["x"]["description"] == "description"


class TestDecorator:
    def test_flow_decorator_initializes(self) -> None:
        @flow(name="foo", version="B", flow_run_name="hi")
        def my_flow() -> str:
            return "bar"

        assert isinstance(my_flow, Flow)
        assert my_flow.name == "foo"
        assert my_flow.version == "B"
        assert my_flow.fn() == "bar"
        assert my_flow.flow_run_name == "hi"

    def test_flow_decorator_initializes_with_callable_flow_run_name(self) -> None:
        @flow(flow_run_name=lambda: "hi")
        def my_flow() -> str:
            return "bar"

        assert isinstance(my_flow, Flow)
        assert my_flow.fn() == "bar"
        assert my_flow.flow_run_name() == "hi"

    def test_flow_decorator_sets_default_version(self) -> None:
        my_flow = flow(flatdict_to_dict)

        assert my_flow.version == file_hash(flatdict_to_dict.__globals__["__file__"])

    def test_invalid_run_name(self) -> None:
        class InvalidFlowRunNameArg:
            def format(*args: Any, **kwargs: Any) -> None:
                pass

        with pytest.raises(
            TypeError,
            match=(
                "Expected string or callable for 'flow_run_name'; got"
                " InvalidFlowRunNameArg instead."
            ),
        ):

            @flow(flow_run_name=InvalidFlowRunNameArg())
            def flow_with_illegal_run_name() -> None:
                pass


class TestResultPersistence:
    @pytest.mark.parametrize("persist_result", [True, False])
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

    def test_setting_result_storage_sets_persist_result_to_true(self, tmpdir: Path) -> None:
        block = LocalFileSystem(basepath=str(tmpdir))
        block.save("foo-bar-flow", _sync=True)

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
        @flow(result_serializer="json")
        def my_flow() -> None:
            pass

        @flow
        def base() -> None:
            pass

        new_flow = base.with_options(result_serializer="json")

        assert my_flow.persist_result is True
        assert new_flow.persist_result is True


class TestFlowWithOptions:
    def test_with_options_allows_override_of_flow_settings(self) -> None:
        fooblock = LocalFileSystem(basepath="foo")
        barblock = LocalFileSystem(basepath="bar")

        fooblock.save("fooblock", _sync=True)
        barblock.save("barblock", _sync=True)

        @flow(
            name="Initial flow",
            description="Flow before with options",
            flow_run_name="OG",
            timeout_seconds=10,
            validate_parameters=True,
            persist_result=True,
            result_serializer="pickle",
            result_storage=fooblock,
            cache_result_in_memory=False,
            on_completion=None,
            on_failure=None,
            on_cancellation=None,
            on_crashed=None,
        )
        def initial_flow() -> None:
            pass

        def failure_hook(flow: Flow, flow_run: Any, state: State) -> None:
            return print("Woof!")

        def success_hook(flow: Flow, flow_run: Any, state: State) -> None:
            return print("Meow!")

        def cancellation_hook(flow: Flow, flow_run: Any, state: State) -> None:
            return print("Fizz Buzz!")

        def crash_hook(flow: Flow, flow_run: Any, state: State) -> None:
            return print("Crash!")

        flow_with_options = initial_flow.with_options(
            name="Copied flow",
            description="A copied flow",
            flow_run_name=lambda: "new-name",
            task_runner=ThreadPoolTaskRunner,
            retries=3,
            retry_delay_seconds=20,
            timeout_seconds=5,
            validate_parameters=False,
            persist_result=False,
            result_serializer="json",
            result_storage=barblock,
            cache_result_in_memory=True,
            on_completion=[success_hook],
            on_failure=[failure_hook],
            on_cancellation=[cancellation_hook],
            on_crashed=[crash_hook],
        )

        assert flow_with_options.name == "Copied flow"
        assert flow_with_options.description == "A copied flow"
        assert flow_with_options.flow_run_name() == "new-name"
        assert isinstance(flow_with_options.task_runner, ThreadPoolTaskRunner)
        assert flow_with_options.timeout_seconds == 5
        assert flow_with_options.retries == 3
        assert flow_with_options.retry_delay_seconds == 20
        assert flow_with_options.should_validate_parameters is False
        assert flow_with_options.persist_result is False
        assert flow_with_options.result_serializer == "json"
        assert flow_with_options.result_storage == barblock
        assert flow_with_options.cache_result_in_memory is True
        assert flow_with_options.on_completion_hooks == [success_hook]
        assert flow_with_options.on_failure_hooks == [failure_hook]
        assert flow_with_options.on_cancellation_hooks == [cancellation_hook]
        assert flow_with_options.on_crashed_hooks == [crash_hook]

    def test_with_options_uses_existing_settings_when_no_override(self, tmp_path: Path) -> None:
        storage = LocalFileSystem(basepath=tmp_path)
        storage.save("test-overrides", _sync=True)

        @flow(
            name="Initial flow",
            description="Flow before with options",
            task_runner=ThreadPoolTaskRunner,
            timeout_seconds=10,
            validate_parameters=True,
            retries=3,
            retry_delay_seconds=20,
            persist_result=False,
            result_serializer="json",
            result_storage=storage,
            cache_result_in_memory=False,
            log_prints=False,
        )
        def initial_flow() -> None:
            pass

        flow_with_options = initial_flow.with_options()

        assert flow_with_options is not initial_flow
        assert flow_with_options.name == "Initial flow"
        assert flow_with_options.description == "Flow before with options"
        assert isinstance(flow_with_options.task_runner, ThreadPoolTaskRunner)
        assert flow_with_options.timeout_seconds == 10
        assert flow_with_options.should_validate_parameters is True
        assert flow_with_options.retries == 3
        assert flow_with_options.retry_delay_seconds == 20
        assert flow_with_options.persist_result is False
        assert flow_with_options.result_serializer == "json"
        assert flow_with_options.result_storage == storage
        assert flow_with_options.cache_result_in_memory is False
        assert flow_with_options.log_prints is False

    def test_with_options_can_unset_timeout_seconds_with_zero(self) -> None:
        @flow(timeout_seconds=1)
        def initial_flow() -> None:
            pass

        flow_with_options = initial_flow.with_options(timeout_seconds=0)
        assert flow_with_options.timeout_seconds is None

    def test_with_options_can_unset_retries_with_zero(self) -> None:
        @flow(retries=3)
        def initial_flow() -> None:
            pass

        flow_with_options = initial_flow.with_options(retries=0)
        assert flow_with_options.retries == 0

    def test_with_options_can_unset_retry_delay_seconds_with_zero(self) -> None:
        @flow(retry_delay_seconds=3)
        def initial_flow() -> None:
            pass

        flow_with_options = initial_flow.with_options(retry_delay_seconds=0)
        assert flow_with_options.retry_delay_seconds == 0

    def test_with_options_uses_parent_flow_run_name_if_not_provided(self) -> None:
        def generate_flow_run_name() -> str:
            return "new-name"

        @flow(retry_delay_seconds=3, flow_run_name=generate_flow_run_name)
        def initial_flow() -> None:
            pass

        flow_with_options = initial_flow.with_options()
        assert flow_with_options.flow_run_name is generate_flow_run_name

    def test_with_options_can_unset_result_options_with_none(self, tmp_path: Path) -> None:
        storage = LocalFileSystem(basepath=tmp_path)
        storage.save("test-unset", _sync=True)

        @flow(
            result_serializer="json",
            result_storage=storage,
        )
        def initial_flow() -> None:
            pass

        flow_with_options = initial_flow.with_options(
            result_serializer=None,
            result_storage=None,
        )
        assert flow_with_options.result_serializer is None
        assert flow_with_options.result_storage is None

    def test_with_options_signature_aligns_with_flow_signature(self) -> None:
        flow_params = set(inspect.signature(flow).parameters.keys())
        with_options_params = set(
            inspect.signature(Flow.with_options).parameters.keys()
        )
        flow_params.remove("_FlowDecorator__fn")
        with_options_params.remove("self")

        assert flow_params == with_options_params

    def get_flow_run_name() -> str:
        name = "test"
        date = "todays_date