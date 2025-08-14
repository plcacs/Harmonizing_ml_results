from __future__ import annotations

import ast
import asyncio
import datetime
import importlib.util
import inspect
import os
import re
import sys
import tempfile
import warnings
from copy import copy
from functools import partial, update_wrapper
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    NoReturn,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from uuid import UUID

import pydantic
from pydantic.v1 import BaseModel as V1BaseModel
from pydantic.v1.decorator import ValidatedFunction as V1ValidatedFunction
from pydantic.v1.errors import ConfigError
from rich.console import Console
from typing_extensions import Literal, ParamSpec

from prefect._experimental.sla.objects import SlaTypes
from prefect._internal.concurrency.api import create_call, from_async
from prefect.blocks.core import Block
from prefect.client.schemas.filters import WorkerFilter, WorkerFilterStatus
from prefect.client.schemas.objects import ConcurrencyLimitConfig, FlowRun
from prefect.client.utilities import client_injector
from prefect.docker.docker_image import DockerImage
from prefect.events import DeploymentTriggerTypes, TriggerTypes
from prefect.exceptions import (
    InvalidNameError,
    MissingFlowError,
    ObjectNotFound,
    ParameterTypeError,
    ScriptError,
    TerminationSignal,
    UnspecifiedFlowError,
)
from prefect.filesystems import LocalFileSystem, ReadableDeploymentStorage
from prefect.futures import PrefectFuture
from prefect.logging import get_logger
from prefect.logging.loggers import flow_run_logger
from prefect.results import ResultSerializer, ResultStorage
from prefect.schedules import Schedule
from prefect.settings import (
    PREFECT_DEFAULT_WORK_POOL_NAME,
    PREFECT_FLOW_DEFAULT_RETRIES,
    PREFECT_FLOW_DEFAULT_RETRY_DELAY_SECONDS,
    PREFECT_TESTING_UNIT_TEST_MODE,
    PREFECT_UI_URL,
)
from prefect.states import State
from prefect.task_runners import TaskRunner, ThreadPoolTaskRunner
from prefect.types import BANNED_CHARACTERS, WITHOUT_BANNED_CHARACTERS
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.annotations import NotSet
from prefect.utilities.asyncutils import (
    run_coro_as_sync,
    run_sync_in_worker_thread,
    sync_compatible,
)
from prefect.utilities.callables import (
    ParameterSchema,
    get_call_parameters,
    parameter_schema,
    parameters_to_args_kwargs,
    raise_for_reserved_arguments,
)
from prefect.utilities.collections import listrepr, visit_collection
from prefect.utilities.filesystem import relative_path_to_current_platform
from prefect.utilities.hashing import file_hash
from prefect.utilities.importtools import import_object, safe_load_namespace

from ._internal.compatibility.async_dispatch import async_dispatch, is_in_async_context
from ._internal.pydantic.v2_schema import is_v2_type
from ._internal.pydantic.v2_validated_func import V2ValidatedFunction
from ._internal.pydantic.v2_validated_func import (
    V2ValidatedFunction as ValidatedFunction,
)

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")
F = TypeVar("F", bound="Flow[Any, Any]")

class FlowStateHook(Protocol, Generic[P, R]):
    __name__: str

    def __call__(
        self, flow: Flow[P, R], flow_run: FlowRun, state: State
    ) -> Awaitable[None] | None: ...

if TYPE_CHECKING:
    import logging
    from prefect.client.orchestration import PrefectClient
    from prefect.client.schemas.objects import FlowRun
    from prefect.client.types.flexible_schedule_list import FlexibleScheduleList
    from prefect.deployments.runner import RunnerDeployment
    from prefect.runner.storage import RunnerStorage

logger: logging.Logger = get_logger("flows")

class Flow(Generic[P, R]):
    def __init__(
        self,
        fn: Callable[P, R],
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[Callable[[], str], str]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[int, float]] = None,
        task_runner: Union[
            Type[TaskRunner[PrefectFuture[Any]]], TaskRunner[PrefectFuture[Any]], None
        ] = None,
        description: Optional[str] = None,
        timeout_seconds: Union[int, float, None] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[Union[ResultStorage, str]] = None,
        result_serializer: Optional[ResultSerializer] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[list[FlowStateHook[P, R]]] = None,
        on_failure: Optional[list[FlowStateHook[P, R]]] = None,
        on_cancellation: Optional[list[FlowStateHook[P, R]]] = None,
        on_crashed: Optional[list[FlowStateHook[P, R]]] = None,
        on_running: Optional[list[FlowStateHook[P, R]]] = None,
    ) -> None:
        ...

    @property
    def ismethod(self) -> bool:
        ...

    def __get__(self, instance: Any, owner: Any) -> Flow[P, R]:
        ...

    def with_options(
        self,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[int, float]] = None,
        description: Optional[str] = None,
        flow_run_name: Optional[Union[Callable[[], str], str]] = None,
        task_runner: Union[
            Type[TaskRunner[PrefectFuture[Any]]], TaskRunner[PrefectFuture[Any]], None
        ] = None,
        timeout_seconds: Union[int, float, None] = None,
        validate_parameters: Optional[bool] = None,
        persist_result: Optional[bool] = NotSet,
        result_storage: Optional[ResultStorage] = NotSet,
        result_serializer: Optional[ResultSerializer] = NotSet,
        cache_result_in_memory: Optional[bool] = None,
        log_prints: Optional[bool] = NotSet,
        on_completion: Optional[list[FlowStateHook[P, R]]] = None,
        on_failure: Optional[list[FlowStateHook[P, R]]] = None,
        on_cancellation: Optional[list[FlowStateHook[P, R]]] = None,
        on_crashed: Optional[list[FlowStateHook[P, R]]] = None,
        on_running: Optional[list[FlowStateHook[P, R]]] = None,
    ) -> Flow[P, R]:
        ...

    def validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        ...

    def serialize_parameters(
        self, parameters: dict[str, Any | PrefectFuture[Any] | State]
    ) -> dict[str, Any]:
        ...

    async def ato_deployment(
        self,
        name: str,
        interval: Optional[
            Union[
                Iterable[Union[int, float, datetime.timedelta]],
                int,
                float,
                datetime.timedelta,
            ]
        ] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[FlexibleScheduleList] = None,
        concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig, None]] = None,
        parameters: Optional[dict[str, Any]] = None,
        triggers: Optional[list[Union[DeploymentTriggerTypes, TriggerTypes]]] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[dict[str, Any]] = None,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        _sla: Optional[Union[SlaTypes, list[SlaTypes]]] = None,
    ) -> RunnerDeployment:
        ...

    @async_dispatch(ato_deployment)
    def to_deployment(
        self,
        name: str,
        interval: Optional[
            Union[
                Iterable[Union[int, float, datetime.timedelta]],
                int,
                float,
                datetime.timedelta,
            ]
        ] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[FlexibleScheduleList] = None,
        concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig, None]] = None,
        parameters: Optional[dict[str, Any]] = None,
        triggers: Optional[list[Union[DeploymentTriggerTypes, TriggerTypes]]] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[dict[str, Any]] = None,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        _sla: Optional[Union[SlaTypes, list[SlaTypes]]] = None,
    ) -> RunnerDeployment:
        ...

    def on_completion(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        ...

    def on_cancellation(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        ...

    def on_crashed(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        ...

    def on_running(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        ...

    def on_failure(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        ...

    def serve(
        self,
        name: Optional[str] = None,
        interval: Optional[
            Union[
                Iterable[Union[int, float, datetime.timedelta]],
                int,
                float,
                datetime.timedelta,
            ]
        ] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[FlexibleScheduleList] = None,
        global_limit: Optional[Union[int, ConcurrencyLimitConfig, None]] = None,
        triggers: Optional[list[Union[DeploymentTriggerTypes, TriggerTypes]]] = None,
        parameters: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        pause_on_shutdown: bool = True,
        print_starting_message: bool = True,
        limit: Optional[int] = None,
        webserver: bool = False,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
    ) -> None:
        ...

    @classmethod
    async def afrom_source(
        cls,
        source: Union[str, RunnerStorage, ReadableDeploymentStorage],
        entrypoint: str,
    ) -> Flow[..., Any]:
        ...

    @classmethod
    @async_dispatch(afrom_source)
    def from_source(
        cls,
        source: Union[str, RunnerStorage, ReadableDeploymentStorage],
        entrypoint: str,
    ) -> Flow[..., Any]:
        ...

    @sync_compatible
    async def deploy(
        self,
        name: str,
        work_pool_name: Optional[str] = None,
        image: Optional[Union[str, DockerImage]] = None,
        build: bool = True,
        push: bool = True,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[dict[str, Any]] = None,
        interval: Optional[Union[int, float, datetime.timedelta]] = None,
        cron: Optional[str] = None,
        rrule: Optional[str] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[list[Schedule]] = None,
        concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig, None]] = None,
        triggers: Optional[list[Union[DeploymentTriggerTypes, TriggerTypes]]] = None,
        parameters: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        print_next_steps: bool = True,
        ignore_warnings: bool = False,
        _sla: Optional[Union[SlaTypes, list[SlaTypes]]] = None,
    ) -> UUID:
        ...

    @overload
    def __call__(self: Flow[P, NoReturn], *args: P.args, **kwargs: P.kwargs) -> None:
        ...

    @overload
    def __call__(
        self: Flow[P, Coroutine[Any, Any, T]], *args: P.args, **kwargs: P.kwargs
    ) -> Coroutine[Any, Any, T]: ...

    @overload
    def __call__(
        self: Flow[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...

    @overload
    def __call__(
        self: Flow[P, Coroutine[Any, Any, T]],
        *args: P.args,
        return_state: Literal[True],
        **kwargs: P.kwargs,
    ) -> Awaitable[State[T]]: ...

    @overload
    def __call__(
        self: Flow[P, T],
        *args: P.args,
        return_state: Literal[True],
        **kwargs: P.kwargs,
    ) -> State[T]: ...

    def __call__(
        self,
        *args: P.args,
        return_state: bool = False,
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        **kwargs: P.kwargs,
    ) -> Any:
        ...

    @sync_compatible
    async def visualize(self, *args: P.args, **kwargs: P.kwargs) -> None:
        ...

class FlowDecorator:
    @overload
    def __call__(self, __fn: Callable[P, R]) -> Flow[P, R]: ...

    @overload
    def __call__(
        self,
        __fn: None = None,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[Callable[[], str], str]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[int, float]] = None,
        task_runner: None = None,
        description: Optional[str] = None,
        timeout_seconds: Union[int, float, None] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[ResultStorage] = None,
        result_serializer: Optional[ResultSerializer] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[list[FlowStateHook[..., Any]]] = None,
        on_failure: Optional[list[FlowStateHook[..., Any]]] = None,
        on_cancellation: Optional[list[FlowStateHook[..., Any]]] = None,
        on_crashed: Optional[list[FlowStateHook[..., Any]]] = None,
        on_running: Optional[list[FlowStateHook[..., Any]]] = None,
    ) -> Callable[[Callable[P, R]], Flow[P, R]]: ...

    @overload
    def __call__(
        self,
        __fn: None = None,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[Callable[[], str], str]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[int, float]] = None,
        task_runner: Optional[TaskRunner[PrefectFuture[R]]] = None,
        description: Optional[str] = None,
        timeout_seconds: Union[int, float, None] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[ResultStorage] = None,
        result_serializer: Optional[ResultSerializer] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[list[FlowStateHook[