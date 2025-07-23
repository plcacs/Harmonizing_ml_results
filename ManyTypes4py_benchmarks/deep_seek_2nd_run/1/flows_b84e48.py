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
    Dict,
    List,
    Set,
    Sequence,
    Mapping,
    MutableMapping,
    Iterator,
    Generator,
    AsyncGenerator,
    AsyncIterator,
    ContextManager,
    AsyncContextManager,
    TypeGuard,
    runtime_checkable,
)
from uuid import UUID
import pydantic
from pydantic.v1 import BaseModel as V1BaseModel
from pydantic.v1.decorator import ValidatedFunction as V1ValidatedFunction
from pydantic.v1.errors import ConfigError
from rich.console import Console
from typing_extensions import Literal, ParamSpec, Self
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
from prefect.utilities.asyncutils import run_coro_as_sync, run_sync_in_worker_thread, sync_compatible
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
from ._internal.pydantic.v2_validated_func import V2ValidatedFunction as ValidatedFunction

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")
F = TypeVar("F", bound="Flow[Any, Any]")

@runtime_checkable
class FlowStateHook(Protocol, Generic[P, R]):
    def __call__(self, flow: "Flow[P, R]", flow_run: "FlowRun", state: State) -> None:
        ...

if TYPE_CHECKING:
    import logging
    from prefect.client.orchestration import PrefectClient
    from prefect.client.schemas.objects import FlowRun
    from prefect.client.types.flexible_schedule_list import FlexibleScheduleList
    from prefect.deployments.runner import RunnerDeployment
    from prefect.runner.storage import RunnerStorage

logger = get_logger("flows")

class Flow(Generic[P, R]):
    def __init__(
        self,
        fn: Callable[P, R],
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
        task_runner: Optional[TaskRunner] = None,
        description: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[ReadableDeploymentStorage] = None,
        result_serializer: Optional[ResultSerializer] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_failure: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_cancellation: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_crashed: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_running: Optional[Sequence[FlowStateHook[P, R]]] = None,
    ) -> None:
        ...

    @property
    def ismethod(self) -> bool:
        ...

    def __get__(self, instance: Any, owner: Type[Any]) -> "Flow[P, R]":
        ...

    def with_options(
        self,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
        description: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        task_runner: Optional[TaskRunner] = None,
        timeout_seconds: Optional[float] = None,
        validate_parameters: Optional[bool] = None,
        persist_result: Union[bool, NotSet] = NotSet,
        result_storage: Union[ReadableDeploymentStorage, NotSet] = NotSet,
        result_serializer: Union[ResultSerializer, NotSet] = NotSet,
        cache_result_in_memory: Optional[bool] = None,
        log_prints: Union[bool, NotSet] = NotSet,
        on_completion: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_failure: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_cancellation: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_crashed: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_running: Optional[Sequence[FlowStateHook[P, R]]] = None,
    ) -> "Flow[P, R]":
        ...

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def serialize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        ...

    async def ato_deployment(
        self,
        name: str,
        interval: Optional[Union[float, datetime.timedelta, Sequence[Union[float, datetime.timedelta]]] = None,
        cron: Optional[Union[str, Sequence[str]]] = None,
        rrule: Optional[Union[str, Sequence[str]]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[Sequence[Schedule]] = None,
        concurrency_limit: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[Sequence[TriggerTypes]] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        _sla: Optional[SlaTypes] = None,
    ) -> "RunnerDeployment":
        ...

    @async_dispatch(ato_deployment)
    def to_deployment(
        self,
        name: str,
        interval: Optional[Union[float, datetime.timedelta, Sequence[Union[float, datetime.timedelta]]] = None,
        cron: Optional[Union[str, Sequence[str]]] = None,
        rrule: Optional[Union[str, Sequence[str]]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[Sequence[Schedule]] = None,
        concurrency_limit: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[Sequence[TriggerTypes]] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        _sla: Optional[SlaTypes] = None,
    ) -> "RunnerDeployment":
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
        interval: Optional[Union[float, datetime.timedelta, Sequence[Union[float, datetime.timedelta]]]] = None,
        cron: Optional[Union[str, Sequence[str]]] = None,
        rrule: Optional[Union[str, Sequence[str]]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[Sequence[Schedule]] = None,
        global_limit: Optional[int] = None,
        triggers: Optional[Sequence[TriggerTypes]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
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
        source: Union[str, Path, "RunnerStorage"],
        entrypoint: str,
    ) -> "Flow[P, R]":
        ...

    @classmethod
    @async_dispatch(afrom_source)
    def from_source(
        cls,
        source: Union[str, Path, "RunnerStorage"],
        entrypoint: str,
    ) -> "Flow[P, R]":
        ...

    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        ...

    @overload
    def __call__(self, *args: P.args, return_state: Literal[False] = False, **kwargs: P.kwargs) -> R:
        ...

    @overload
    def __call__(self, *args: P.args, return_state: Literal[True], **kwargs: P.kwargs) -> State[R]:
        ...

    def __call__(
        self,
        *args: P.args,
        return_state: bool = False,
        wait_for: Optional[Sequence[PrefectFuture]] = None,
        **kwargs: P.kwargs,
    ) -> Union[R, State[R]]:
        ...

    @sync_compatible
    async def visualize(self, *args: P.args, **kwargs: P.kwargs) -> None:
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
        job_variables: Optional[Dict[str, Any]] = None,
        interval: Optional[Union[float, datetime.timedelta, Sequence[Union[float, datetime.timedelta]]] = None,
        cron: Optional[Union[str, Sequence[str]]] = None,
        rrule: Optional[Union[str, Sequence[str]]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[Sequence[Schedule]] = None,
        concurrency_limit: Optional[int] = None,
        triggers: Optional[Sequence[TriggerTypes]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        print_next_steps: bool = True,
        ignore_warnings: bool = False,
        _sla: Optional[SlaTypes] = None,
    ) -> UUID:
        ...

class FlowDecorator:
    @overload
    def __call__(self, __fn: Callable[P, R]) -> Flow[P, R]:
        ...

    @overload
    def __call__(
        self,
        __fn: None = None,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
        task_runner: Optional[TaskRunner] = None,
        description: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[ReadableDeploymentStorage] = None,
        result_serializer: Optional[ResultSerializer] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_failure: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_cancellation: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_crashed: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_running: Optional[Sequence[FlowStateHook[P, R]]] = None,
    ) -> Callable[[Callable[P, R]], Flow[P, R]]:
        ...

    def __call__(
        self,
        __fn: Optional[Callable[P, R]] = None,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
        task_runner: Optional[TaskRunner] = None,
        description: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[ReadableDeploymentStorage] = None,
        result_serializer: Optional[ResultSerializer] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_failure: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_cancellation: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_crashed: Optional[Sequence[FlowStateHook[P, R]]] = None,
        on_running: Optional[Sequence[FlowStateHook[P, R]]] = None,
    ) -> Union[Flow[P, R], Callable[[Callable[P, R]], Flow[P, R]]]:
        ...

flow = FlowDecorator()

def _raise_on_name_with_banned_characters(name: Optional[str]) -> Optional[str]:
    ...

def select_flow(
    flows: Iterable[Flow[Any, Any]],
    flow_name: Optional[str] = None,
    from_message: Optional[str] = None,
) -> Flow[Any, Any]:
    ...

def load_flow_from_entrypoint(
    entrypoint: str,
    use_placeholder_flow: bool = True,
) -> Flow[Any, Any]:
    ...

def load_function_and_convert_to_flow(entrypoint: str) -> Flow[Any, Any]:
    ...

def serve(
    *args: