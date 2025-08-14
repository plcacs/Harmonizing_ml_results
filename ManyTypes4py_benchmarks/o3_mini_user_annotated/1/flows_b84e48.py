#!/usr/bin/env python3
"""
Module containing the base workflow class and decorator - for most use cases, using the [`@flow` decorator][prefect.flows.flow] is preferred.
"""

from __future__ import annotations

# This file requires type-checking with pyright because mypy does not yet support PEP612
# See https://github.com/python/mypy/issues/8645
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
from pydantic.v1.errors import ConfigError  # TODO
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

T = TypeVar("T")  # Generic type var for capturing the inner return type of async funcs
R = TypeVar("R")  # The return type of the user's function
P = ParamSpec("P")  # The parameters of the flow
F = TypeVar("F", bound="Flow[Any, Any]")  # The type of the flow


class FlowStateHook(Protocol, Generic[P, R]):
    """
    A callable that is invoked when a flow enters a given state.
    """

    __name__: str

    def __call__(
        self, flow: Flow[P, R], flow_run: FlowRun, state: State
    ) -> Awaitable[None] | None:
        ...


if TYPE_CHECKING:
    import logging

    from prefect.client.orchestration import PrefectClient
    from prefect.client.schemas.objects import FlowRun
    from prefect.client.types.flexible_schedule_list import FlexibleScheduleList
    from prefect.deployments.runner import RunnerDeployment
    from prefect.runner.storage import RunnerStorage

logger: "logging.Logger" = get_logger("flows")


class Flow(Generic[P, R]):
    """
    A Prefect workflow definition.

    !!! note
        We recommend using the [`@flow` decorator][prefect.flows.flow] for most use-cases.

    Wraps a function with an entrypoint to the Prefect engine. To preserve the input
    and output types, we use the generic type variables `P` and `R` for "Parameters" and
    "Returns" respectively.
    """

    # NOTE: These parameters (types, defaults, and docstrings) should be duplicated
    #       exactly in the @flow decorator
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
        if name is not None and not isinstance(name, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                "Expected string for flow parameter 'name'; got {} instead. {}".format(
                    type(name).__name__,
                    (
                        "Perhaps you meant to call it? e.g."
                        " '@flow(name=get_flow_run_name())'"
                        if callable(name)
                        else ""
                    ),
                )
            )

        # Validate if hook passed is list and contains callables
        hook_categories: list[Optional[Iterable[FlowStateHook[P, R]]]] = [
            on_completion,
            on_failure,
            on_cancellation,
            on_crashed,
            on_running,
        ]
        hook_names: list[str] = [
            "on_completion",
            "on_failure",
            "on_cancellation",
            "on_crashed",
            "on_running",
        ]
        for hooks, hook_name in zip(hook_categories, hook_names):
            if hooks is not None:
                try:
                    hooks = list(hooks)
                except TypeError:
                    raise TypeError(
                        f"Expected iterable for '{hook_name}'; got"
                        f" {type(hooks).__name__} instead. Please provide a list of"
                        f" hooks to '{hook_name}':\n\n"
                        f"@flow({hook_name}=[hook1, hook2])\ndef"
                        " my_flow():\n\tpass"
                    )

                for hook in hooks:
                    if not callable(hook):
                        raise TypeError(
                            f"Expected callables in '{hook_name}'; got"
                            f" {type(hook).__name__} instead. Please provide a list of"
                            f" hooks to '{hook_name}':\n\n"
                            f"@flow({hook_name}=[hook1, hook2])\ndef"
                            " my_flow():\n\tpass"
                        )

        if not callable(fn):
            raise TypeError("'fn' must be callable")

        self.name: str = name or fn.__name__.replace("_", "-").replace(
            "<lambda>",
            "unknown-lambda",  # prefect API will not accept "<" or ">" in flow names
        )
        _raise_on_name_with_banned_characters(self.name)

        if flow_run_name is not None:
            if not isinstance(flow_run_name, str) and not callable(flow_run_name):
                raise TypeError(
                    "Expected string or callable for 'flow_run_name'; got"
                    f" {type(flow_run_name).__name__} instead."
                )
        self.flow_run_name = flow_run_name

        if task_runner is None:
            self.task_runner: TaskRunner[PrefectFuture[Any]] = cast(
                TaskRunner[PrefectFuture[Any]], ThreadPoolTaskRunner()
            )
        else:
            self.task_runner = (
                task_runner() if isinstance(task_runner, type) else task_runner
            )

        self.log_prints = log_prints

        self.description: Optional[str] = description or inspect.getdoc(fn)
        update_wrapper(self, fn)
        self.fn = fn

        # the flow is considered async if its function is async or an async
        # generator
        self.isasync: bool = asyncio.iscoroutinefunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)

        # the flow is considered a generator if its function is a generator or
        # an async generator
        self.isgenerator: bool = inspect.isgeneratorfunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)

        raise_for_reserved_arguments(self.fn, ["return_state", "wait_for"])

        # Version defaults to a hash of the function's file
        if not version:
            try:
                flow_file: Optional[str] = inspect.getsourcefile(self.fn)
                if flow_file is None:
                    raise FileNotFoundError
                version = file_hash(flow_file)
            except (FileNotFoundError, TypeError, OSError):
                pass  # `getsourcefile` can return null values and "<stdin>" for objects in repls
        self.version = version

        self.timeout_seconds: Optional[float] = (
            float(timeout_seconds) if timeout_seconds else None
        )

        # FlowRunPolicy settings
        # TODO: We can instantiate a `FlowRunPolicy` and add Pydantic bound checks to
        #       validate that the user passes positive numbers here
        self.retries: int = (
            retries if retries is not None else PREFECT_FLOW_DEFAULT_RETRIES.value()
        )

        self.retry_delay_seconds: Union[int, float] = (
            retry_delay_seconds
            if retry_delay_seconds is not None
            else PREFECT_FLOW_DEFAULT_RETRY_DELAY_SECONDS.value()
        )

        self.parameters: ParameterSchema = parameter_schema(self.fn)
        self.should_validate_parameters = validate_parameters

        if self.should_validate_parameters:
            # Try to create the validated function now so that incompatibility can be
            # raised at declaration time rather than at runtime
            # We cannot, however, store the validated function on the flow because it
            # is not picklable in some environments
            try:
                ValidatedFunction(self.fn, config={"arbitrary_types_allowed": True})
            except ConfigError as exc:
                raise ValueError(
                    "Flow function is not compatible with `validate_parameters`. "
                    "Disable validation or change the argument names."
                ) from exc

        # result persistence settings
        if persist_result is None:
            if result_storage is not None or result_serializer is not None:
                persist_result = True

        self.persist_result = persist_result
        if result_storage and not isinstance(result_storage, str):
            if getattr(result_storage, "_block_document_id", None) is None:
                raise TypeError(
                    "Result storage configuration must be persisted server-side."
                    " Please call `.save()` on your block before passing it in."
                )
        self.result_storage = result_storage
        self.result_serializer = result_serializer
        self.cache_result_in_memory = cache_result_in_memory
        self.on_completion_hooks: list[FlowStateHook[P, R]] = on_completion or []
        self.on_failure_hooks: list[FlowStateHook[P, R]] = on_failure or []
        self.on_cancellation_hooks: list[FlowStateHook[P, R]] = on_cancellation or []
        self.on_crashed_hooks: list[FlowStateHook[P, R]] = on_crashed or []
        self.on_running_hooks: list[FlowStateHook[P, R]] = on_running or []

        # Used for flows loaded from remote storage
        self._storage: Optional["RunnerStorage"] = None
        self._entrypoint: Optional[str] = None

        module = fn.__module__
        if module and (module == "__main__" or module.startswith("__prefect_loader_")):
            module_name: str = inspect.getfile(fn)
            module = module_name if module_name != "__main__" else module

        self._entrypoint = f"{module}:{fn.__name__}"

    @property
    def ismethod(self) -> bool:
        return hasattr(self.fn, "__prefect_self__")

    def __get__(self, instance: Any, owner: Any) -> Flow[P, R]:
        """
        Implement the descriptor protocol so that the flow can be used as an instance method.
        When an instance method is loaded, this method is called with the "self" instance as
        an argument. We return a copy of the flow with that instance bound to the flow's function.
        """

        # if no instance is provided, it's being accessed on the class
        if instance is None:
            return self

        # if the flow is being accessed on an instance, bind the instance to the __prefect_self__ attribute
        # of the flow's function. This will allow it to be automatically added to the flow's parameters
        else:
            bound_flow: Flow[P, R] = copy(self)
            setattr(bound_flow.fn, "__prefect_self__", instance)
            return bound_flow

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
        persist_result: Optional[bool] = NotSet,  # type: ignore
        result_storage: Optional[ResultStorage] = NotSet,  # type: ignore
        result_serializer: Optional[ResultSerializer] = NotSet,  # type: ignore
        cache_result_in_memory: Optional[bool] = None,
        log_prints: Optional[bool] = NotSet,  # type: ignore
        on_completion: Optional[list[FlowStateHook[P, R]]] = None,
        on_failure: Optional[list[FlowStateHook[P, R]]] = None,
        on_cancellation: Optional[list[FlowStateHook[P, R]]] = None,
        on_crashed: Optional[list[FlowStateHook[P, R]]] = None,
        on_running: Optional[list[FlowStateHook[P, R]]] = None,
    ) -> Flow[P, R]:
        """
        Create a new flow from the current object, updating provided options.
        """
        new_task_runner: TaskRunner[PrefectFuture[Any]] = (
            task_runner() if isinstance(task_runner, type) else task_runner
        )
        if new_task_runner is None:
            new_task_runner = self.task_runner
        new_flow: Flow[P, R] = Flow(
            fn=self.fn,
            name=name or self.name,
            description=description or self.description,
            flow_run_name=flow_run_name or self.flow_run_name,
            version=version or self.version,
            task_runner=new_task_runner,
            retries=retries if retries is not None else self.retries,
            retry_delay_seconds=(
                retry_delay_seconds
                if retry_delay_seconds is not None
                else self.retry_delay_seconds
            ),
            timeout_seconds=(
                timeout_seconds if timeout_seconds is not None else self.timeout_seconds
            ),
            validate_parameters=(
                validate_parameters
                if validate_parameters is not None
                else self.should_validate_parameters
            ),
            persist_result=(
                persist_result if persist_result is not NotSet else self.persist_result
            ),
            result_storage=(
                result_storage if result_storage is not NotSet else self.result_storage
            ),
            result_serializer=(
                result_serializer
                if result_serializer is not NotSet
                else self.result_serializer
            ),
            cache_result_in_memory=(
                cache_result_in_memory
                if cache_result_in_memory is not None
                else self.cache_result_in_memory
            ),
            log_prints=log_prints if log_prints is not NotSet else self.log_prints,
            on_completion=on_completion or self.on_completion_hooks,
            on_failure=on_failure or self.on_failure_hooks,
            on_cancellation=on_cancellation or self.on_cancellation_hooks,
            on_crashed=on_crashed or self.on_crashed_hooks,
            on_running=on_running or self.on_running_hooks,
        )
        new_flow._storage = self._storage
        new_flow._entrypoint = self._entrypoint
        return new_flow

    def validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Validate parameters for compatibility with the flow by attempting to cast the inputs to the
        associated types specified by the function's type annotations.

        Returns:
            A new dict of parameters that have been cast to the appropriate types

        Raises:
            ParameterTypeError: if the provided parameters are not valid
        """

        def resolve_block_reference(data: Any | dict[str, Any]) -> Any:
            if isinstance(data, dict) and "$ref" in data:
                return Block.load_from_ref(data["$ref"], _sync=True)
            return data

        try:
            parameters = visit_collection(
                parameters, resolve_block_reference, return_data=True
            )
        except (ValueError, RuntimeError) as exc:
            raise ParameterTypeError(
                "Failed to resolve block references in parameters."
            ) from exc

        args, kwargs = parameters_to_args_kwargs(self.fn, parameters)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=pydantic.warnings.PydanticDeprecatedSince20
            )
            has_v1_models: bool = any(isinstance(o, V1BaseModel) for o in args) or any(
                isinstance(o, V1BaseModel) for o in kwargs.values()
            )

        has_v2_types: bool = any(is_v2_type(o) for o in args) or any(
            is_v2_type(o) for o in kwargs.values()
        )

        if has_v1_models and has_v2_types:
            raise ParameterTypeError(
                "Cannot mix Pydantic v1 and v2 types as arguments to a flow."
            )

        validated_fn_kwargs: dict[str, Any] = dict(arbitrary_types_allowed=True)

        if has_v1_models:
            validated_fn = V1ValidatedFunction(self.fn, config=validated_fn_kwargs)
        else:
            validated_fn = V2ValidatedFunction(self.fn, config=validated_fn_kwargs)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=pydantic.warnings.PydanticDeprecatedSince20
                )
                model = validated_fn.init_model_instance(*args, **kwargs)
        except pydantic.ValidationError as exc:
            logger.error(
                f"Parameter validation failed for flow {self.name!r}: {exc.errors()}"
                f"\nParameters: {parameters}"
            )
            raise ParameterTypeError.from_validation_error(exc) from None

        cast_parameters: dict[str, Any] = {
            k: v
            for k, v in dict(iter(model)).items()
            if k in model.model_fields_set or model.model_fields[k].default_factory
        }
        return cast_parameters

    def serialize_parameters(
        self, parameters: dict[str, Any | PrefectFuture[Any] | State]
    ) -> dict[str, Any]:
        """
        Convert parameters to a serializable form.
        """
        serialized_parameters: dict[str, Any] = {}
        for key, value in parameters.items():
            if self.ismethod and value is getattr(self.fn, "__prefect_self__", None):
                continue
            if isinstance(value, (PrefectFuture, State)):
                serialized_parameters[key] = f"<{type(value).__name__}>"
                continue
            try:
                from fastapi.encoders import jsonable_encoder

                serialized_parameters[key] = jsonable_encoder(value)
            except (TypeError, ValueError):
                logger.debug(
                    f"Parameter {key!r} for flow {self.name!r} is unserializable. "
                    f"Type {type(value).__name__!r} and will not be stored "
                    "in the backend."
                )
                serialized_parameters[key] = f"<{type(value).__name__}>"
        return serialized_parameters

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
        schedules: Optional["FlexibleScheduleList"] = None,
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
        _sla: Optional[Union[SlaTypes, list[SlaTypes]]] = None,  # experimental
    ) -> "RunnerDeployment":
        """
        Asynchronously creates a runner deployment object for this flow.
        """
        from prefect.deployments.runner import RunnerDeployment

        if not name.endswith(".py"):
            _raise_on_name_with_banned_characters(name)

        if self._storage and self._entrypoint:
            return await RunnerDeployment.afrom_storage(
                storage=self._storage,
                entrypoint=self._entrypoint,
                name=name,
                flow_name=self.name,
                interval=interval,
                cron=cron,
                rrule=rrule,
                paused=paused,
                schedule=schedule,
                schedules=schedules,
                concurrency_limit=concurrency_limit,
                tags=tags,
                triggers=triggers,
                parameters=parameters or {},
                description=description,
                version=version,
                enforce_parameter_schema=enforce_parameter_schema,
                work_pool_name=work_pool_name,
                work_queue_name=work_queue_name,
                job_variables=job_variables,
                _sla=_sla,
            )
        else:
            return RunnerDeployment.from_flow(
                flow=self,
                name=name,
                interval=interval,
                cron=cron,
                rrule=rrule,
                paused=paused,
                schedule=schedule,
                schedules=schedules,
                concurrency_limit=concurrency_limit,
                tags=tags,
                triggers=triggers,
                parameters=parameters or {},
                description=description,
                version=version,
                enforce_parameter_schema=enforce_parameter_schema,
                work_pool_name=work_pool_name,
                work_queue_name=work_queue_name,
                job_variables=job_variables,
                entrypoint_type=entrypoint_type,
                _sla=_sla,
            )

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
        schedules: Optional["FlexibleScheduleList"] = None,
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
        _sla: Optional[Union[SlaTypes, list[SlaTypes]]] = None,  # experimental
    ) -> "RunnerDeployment":
        from prefect.deployments.runner import RunnerDeployment

        if not name.endswith(".py"):
            _raise_on_name_with_banned_characters(name)

        if self._storage and self._entrypoint:
            return cast(
                RunnerDeployment,
                RunnerDeployment.from_storage(
                    storage=self._storage,
                    entrypoint=self._entrypoint,
                    name=name,
                    flow_name=self.name,
                    interval=interval,
                    cron=cron,
                    rrule=rrule,
                    paused=paused,
                    schedule=schedule,
                    schedules=schedules,
                    concurrency_limit=concurrency_limit,
                    tags=tags,
                    triggers=triggers,
                    parameters=parameters or {},
                    description=description,
                    version=version,
                    enforce_parameter_schema=enforce_parameter_schema,
                    work_pool_name=work_pool_name,
                    work_queue_name=work_queue_name,
                    job_variables=job_variables,
                    _sla=_sla,
                    _sync=True,  # _sync is valid because .from_storage is decorated with async_dispatch
                ),
            )
        else:
            return RunnerDeployment.from_flow(
                flow=self,
                name=name,
                interval=interval,
                cron=cron,
                rrule=rrule,
                paused=paused,
                schedule=schedule,
                schedules=schedules,
                concurrency_limit=concurrency_limit,
                tags=tags,
                triggers=triggers,
                parameters=parameters or {},
                description=description,
                version=version,
                enforce_parameter_schema=enforce_parameter_schema,
                work_pool_name=work_pool_name,
                work_queue_name=work_queue_name,
                job_variables=job_variables,
                entrypoint_type=entrypoint_type,
                _sla=_sla,
            )

    def on_completion(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        self.on_completion_hooks.append(fn)
        return fn

    def on_cancellation(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        self.on_cancellation_hooks.append(fn)
        return fn

    def on_crashed(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        self.on_crashed_hooks.append(fn)
        return fn

    def on_running(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        self.on_running_hooks.append(fn)
        return fn

    def on_failure(self, fn: FlowStateHook[P, R]) -> FlowStateHook[P, R]:
        self.on_failure_hooks.append(fn)
        return fn

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
        schedules: Optional["FlexibleScheduleList"] = None,
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
        from prefect.runner import Runner

        if not name:
            name = self.name
        else:
            name = Path(name).stem

        runner: Runner = Runner(name=name, pause_on_shutdown=pause_on_shutdown, limit=limit)
        deployment_id: Any = runner.add_flow(
            self,
            name=name,
            triggers=triggers,
            interval=interval,
            cron=cron,
            rrule=rrule,
            paused=paused,
            schedule=schedule,
            schedules=schedules,
            concurrency_limit=global_limit,
            parameters=parameters,
            description=description,
            tags=tags,
            version=version,
            enforce_parameter_schema=enforce_parameter_schema,
            entrypoint_type=entrypoint_type,
        )
        if print_starting_message:
            _display_serve_start_message(self)
        try:
            loop: Optional[asyncio.AbstractEventLoop] = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                if "no running event loop" in str(exc):
                    loop = None
                else:
                    raise
            try:
                if loop is not None:
                    loop.run_until_complete(runner.start(webserver=webserver))
                else:
                    asyncio.run(runner.start(webserver=webserver))
            except (KeyboardInterrupt, TerminationSignal) as exc:
                logger.info(f"Received {type(exc).__name__}, shutting down...")
                if loop is not None:
                    loop.stop()
        except Exception:
            raise

    @classmethod
    async def afrom_source(
        cls,
        source: Union[str, "RunnerStorage", ReadableDeploymentStorage],
        entrypoint: str,
    ) -> Flow[..., Any]:
        from prefect.runner.storage import (
            BlockStorageAdapter,
            LocalStorage,
            RunnerStorage,
            create_storage_from_source,
        )

        if isinstance(source, (Path, str)):
            if isinstance(source, Path):
                source = str(source)
            storage: RunnerStorage = create_storage_from_source(source)
        elif isinstance(source, RunnerStorage):
            storage = source
        elif hasattr(source, "get_directory"):
            storage = BlockStorageAdapter(source)
        else:
            raise TypeError(
                f"Unsupported source type {type(source).__name__!r}. Please provide a"
                " URL to remote storage or a storage object."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            if not isinstance(storage, LocalStorage):
                storage.set_base_path(Path(tmpdir))
                await storage.pull_code()
            full_entrypoint: str = str(storage.destination / entrypoint)
            flow: Flow[..., Any] = cast(
                Flow[..., Any],
                await from_async.wait_for_call_in_new_thread(
                    create_call(load_flow_from_entrypoint, full_entrypoint)
                ),
            )
            flow._storage = storage
            flow._entrypoint = entrypoint
        return flow

    @classmethod
    @async_dispatch(afrom_source)
    def from_source(
        cls,
        source: Union[str, "RunnerStorage", ReadableDeploymentStorage],
        entrypoint: str,
    ) -> Flow[..., Any]:
        from prefect.runner.storage import (
            BlockStorageAdapter,
            LocalStorage,
            RunnerStorage,
            create_storage_from_source,
        )

        if isinstance(source, (Path, str)):
            if isinstance(source, Path):
                source = str(source)
            storage = create_storage_from_source(source)
        elif isinstance(source, RunnerStorage):
            storage = source
        elif hasattr(source, "get_directory"):
            storage = BlockStorageAdapter(source)
        else:
            raise TypeError(
                f"Unsupported source type {type(source).__name__!r}. Please provide a"
                " URL to remote storage or a storage object."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            if not isinstance(storage, LocalStorage):
                storage.set_base_path(Path(tmpdir))
                run_coro_as_sync(storage.pull_code())
            full_entrypoint = str(storage.destination / entrypoint)
            flow: Flow[..., Any] = load_flow_from_entrypoint(full_entrypoint)
            flow._storage = storage
            flow._entrypoint = entrypoint
        return flow

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
        if not (
            work_pool_name := work_pool_name or PREFECT_DEFAULT_WORK_POOL_NAME.value()
        ):
            raise ValueError(
                "No work pool name provided. Please provide a `work_pool_name` or set the"
                " `PREFECT_DEFAULT_WORK_POOL_NAME` environment variable."
            )

        from prefect.client.orchestration import get_client

        try:
            async with get_client() as client:
                work_pool = await client.read_work_pool(work_pool_name)
                active_workers = await client.read_workers_for_work_pool(
                    work_pool_name,
                    worker_filter=WorkerFilter(
                        status=WorkerFilterStatus(any_=["ONLINE"])
                    ),
                )
        except ObjectNotFound as exc:
            raise ValueError(
                f"Could not find work pool {work_pool_name!r}. Please create it before"
                " deploying this flow."
            ) from exc

        to_deployment_coro = self.to_deployment(
            name=name,
            interval=interval,
            cron=cron,
            rrule=rrule,
            schedule=schedule,
            schedules=schedules,
            concurrency_limit=concurrency_limit,
            paused=paused,
            triggers=triggers,
            parameters=parameters,
            description=description,
            tags=tags,
            version=version,
            enforce_parameter_schema=enforce_parameter_schema,
            work_queue_name=work_queue_name,
            job_variables=job_variables,
            entrypoint_type=entrypoint_type,
            _sla=_sla,
        )
        if TYPE_CHECKING:
            assert inspect.isawaitable(to_deployment_coro)
        deployment = await to_deployment_coro

        from prefect.deployments.runner import deploy
        deploy_coro = deploy(
            deployment,
            work_pool_name=work_pool_name,
            image=image,
            build=build,
            push=push,
            print_next_steps_message=False,
            ignore_warnings=ignore_warnings,
        )
        if TYPE_CHECKING:
            assert inspect.isawaitable(deploy_coro)
        deployment_ids = await deploy_coro

        if print_next_steps:
            console = Console()
            if (
                not work_pool.is_push_pool
                and not work_pool.is_managed_pool
                and not active_workers
            ):
                console.print(
                    "\nTo execute flow runs from this deployment, start a worker in a"
                    " separate terminal that pulls work from the"
                    f" {work_pool_name!r} work pool:"
                )
                console.print(
                    f"\n\t$ prefect worker start --pool {work_pool_name!r}",
                    style="blue",
                )
            console.print(
                "\nTo schedule a run for this deployment, use the following command:"
            )
            console.print(
                f"\n\t$ prefect deployment run '{self.name}/{name}'\n",
                style="blue",
            )
            if PREFECT_UI_URL:
                message = (
                    "\nYou can also run your flow via the Prefect UI:"
                    f" [blue]{PREFECT_UI_URL.value()}/deployments/deployment/{deployment_ids[0]}[/]\n"
                )
                console.print(message, soft_wrap=True)
        return deployment_ids[0]

    @overload
    def __call__(self: Flow[P, NoReturn], *args: P.args, **kwargs: P.kwargs) -> None:
        ...

    @overload
    def __call__(
        self: Flow[P, Coroutine[Any, Any, T]], *args: P.args, **kwargs: P.kwargs
    ) -> Coroutine[Any, Any, T]:
        ...

    @overload
    def __call__(
        self: Flow[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        ...

    @overload
    def __call__(
        self: Flow[P, Coroutine[Any, Any, T]],
        *args: P.args,
        return_state: Literal[True],
        **kwargs: P.kwargs,
    ) -> Awaitable[State[T]]:
        ...

    @overload
    def __call__(
        self: Flow[P, T],
        *args: P.args,
        return_state: Literal[True],
        **kwargs: P.kwargs,
    ) -> State[T]:
        ...

    def __call__(
        self,
        *args: P.args,
        return_state: bool = False,
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        **kwargs: P.kwargs,
    ):
        from prefect.utilities.visualization import (
            get_task_viz_tracker,
            track_viz_task,
        )
        parameters: dict[str, Any] = get_call_parameters(self.fn, args, kwargs)
        return_type: str = "state" if return_state else "result"
        task_viz_tracker = get_task_viz_tracker()
        if task_viz_tracker:
            return track_viz_task(self.isasync, self.name, parameters)
        from prefect.flow_engine import run_flow
        return run_flow(
            flow=self,
            parameters=parameters,
            wait_for=wait_for,
            return_type=return_type,
        )

    @sync_compatible
    async def visualize(self, *args: P.args, **kwargs: P.kwargs) -> None:
        from prefect.utilities.visualization import (
            FlowVisualizationError,
            GraphvizExecutableNotFoundError,
            GraphvizImportError,
            TaskVizTracker,
            VisualizationUnsupportedError,
            build_task_dependencies,
            visualize_task_dependencies,
        )
        if not PREFECT_TESTING_UNIT_TEST_MODE:
            warnings.warn(
                "`flow.visualize()` will execute code inside of your flow that is not"
                " decorated with `@task` or `@flow`."
            )
        try:
            with TaskVizTracker() as tracker:
                if self.isasync:
                    await self.fn(*args, **kwargs)  # type: ignore
                else:
                    self.fn(*args, **kwargs)
                graph = build_task_dependencies(tracker)
                visualize_task_dependencies(graph, self.name)
        except GraphvizImportError:
            raise
        except GraphvizExecutableNotFoundError:
            raise
        except VisualizationUnsupportedError:
            raise
        except FlowVisualizationError:
            raise
        except Exception as e:
            msg = (
                "It's possible you are trying to visualize a flow that contains "
                "code that directly interacts with the result of a task"
                " inside of the flow. \nTry passing a `viz_return_value` "
                "to the task decorator, e.g. `@task(viz_return_value=[1, 2, 3]).`"
            )
            new_exception = type(e)(str(e) + "\n" + msg)
            new_exception.__traceback__ = e.__traceback__
            raise new_exception


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
    ) -> Callable[[Callable[P, R]], Flow[P, R]]:
        ...

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
        on_completion: Optional[list[FlowStateHook[..., Any]]] = None,
        on_failure: Optional[list[FlowStateHook[..., Any]]] = None,
        on_cancellation: Optional[list[FlowStateHook[..., Any]]] = None,
        on_crashed: Optional[list[FlowStateHook[..., Any]]] = None,
        on_running: Optional[list[FlowStateHook[..., Any]]] = None,
    ) -> Callable[[Callable[P, R]], Flow[P, R]]:
        ...

    def __call__(
        self,
        __fn: Optional[Callable[P, R]] = None,
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
        on_completion: Optional[list[FlowStateHook[..., Any]]] = None,
        on_failure: Optional[list[FlowStateHook[..., Any]]] = None,
        on_cancellation: Optional[list[FlowStateHook[..., Any]]] = None,
        on_crashed: Optional[list[FlowStateHook[..., Any]]] = None,
        on_running: Optional[list[FlowStateHook[..., Any]]] = None,
    ) -> Union[Flow[P, R], Callable[[Callable[P, R]], Flow[P, R]]]:
        if __fn:
            if isinstance(__fn, (classmethod, staticmethod)):
                method_decorator = type(__fn).__name__
                raise TypeError(
                    f"@{method_decorator} should be applied on top of @flow"
                )
            return Flow(
                fn=__fn,
                name=name,
                version=version,
                flow_run_name=flow_run_name,
                task_runner=task_runner,
                description=description,
                timeout_seconds=timeout_seconds,
                validate_parameters=validate_parameters,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                persist_result=persist_result,
                result_storage=result_storage,
                result_serializer=result_serializer,
                cache_result_in_memory=cache_result_in_memory,
                log_prints=log_prints,
                on_completion=on_completion,
                on_failure=on_failure,
                on_cancellation=on_cancellation,
                on_crashed=on_crashed,
                on_running=on_running,
            )
        else:
            return cast(
                Callable[[Callable[P, R]], Flow[P, R]],
                partial(
                    flow,
                    name=name,
                    version=version,
                    flow_run_name=flow_run_name,
                    task_runner=task_runner,
                    description=description,
                    timeout_seconds=timeout_seconds,
                    validate_parameters=validate_parameters,
                    retries=retries,
                    retry_delay_seconds=retry_delay_seconds,
                    persist_result=persist_result,
                    result_storage=result_storage,
                    result_serializer=result_serializer,
                    cache_result_in_memory=cache_result_in_memory,
                    log_prints=log_prints,
                    on_completion=on_completion,
                    on_failure=on_failure,
                    on_cancellation=on_cancellation,
                    on_crashed=on_crashed,
                    on_running=on_running,
                ),
            )

    if not TYPE_CHECKING:
        from_source = staticmethod(Flow.from_source)
    else:
        @staticmethod
        def from_source(
            source: Union[str, "RunnerStorage", ReadableDeploymentStorage],
            entrypoint: str,
        ) -> Union[Flow[..., Any], Coroutine[Any, Any, Flow[..., Any]]]:
            ...


flow: FlowDecorator = FlowDecorator()


def _raise_on_name_with_banned_characters(name: Optional[str]) -> Optional[str]:
    if name is None:
        return name
    if not re.match(WITHOUT_BANNED_CHARACTERS, name):
        raise InvalidNameError(
            f"Name {name!r} contains an invalid character. "
            f"Must not contain any of: {BANNED_CHARACTERS}."
        )
    return name


def select_flow(
    flows: Iterable[Flow[P, R]],
    flow_name: Optional[str] = None,
    from_message: Optional[str] = None,
) -> Flow[P, R]:
    flows_dict: dict[str, Flow[P, R]] = {f.name: f for f in flows}
    from_message = (" " + from_message) if from_message else ""
    if not flows_dict:
        raise MissingFlowError(f"No flows found{from_message}.")
    elif flow_name and flow_name not in flows_dict:
        raise MissingFlowError(
            f"Flow {flow_name!r} not found{from_message}. "
            f"Found the following flows: {listrepr(flows_dict.keys())}. "
            "Check to make sure that your flow function is decorated with `@flow`."
        )
    elif not flow_name and len(flows_dict) > 1:
        raise UnspecifiedFlowError(
            (
                f"Found {len(flows_dict)} flows{from_message}:"
                f" {listrepr(sorted(flows_dict.keys()))}. Specify a flow name to select a"
                " flow."
            ),
        )
    if flow_name:
        return flows_dict[flow_name]
    else:
        return list(flows_dict.values())[0]


def load_flow_from_entrypoint(
    entrypoint: str,
    use_placeholder_flow: bool = True,
) -> Flow[P, Any]:
    if ":" in entrypoint:
        path, func_name = entrypoint.rsplit(":", maxsplit=1)
    else:
        path, func_name = entrypoint.rsplit(".", maxsplit=1)
    try:
        flow: Flow[P, Any] = import_object(entrypoint)  # pyright: ignore[reportRedeclaration]
    except AttributeError as exc:
        raise MissingFlowError(
            f"Flow function with name {func_name!r} not found in {path!r}. "
        ) from exc
    except ScriptError:
        if use_placeholder_flow:
            flow: Optional[Flow[P, Any]] = safe_load_flow_from_entrypoint(entrypoint)
            if flow is None:
                raise
        else:
            raise
    if not isinstance(flow, Flow):
        raise MissingFlowError(
            f"Function with name {func_name!r} is not a flow. Make sure that it is "
            "decorated with '@flow'."
        )
    return flow


def load_function_and_convert_to_flow(entrypoint: str) -> Flow[P, Any]:
    if ":" in entrypoint:
        path, func_name = entrypoint.rsplit(":", maxsplit=1)
    else:
        path, func_name = entrypoint.rsplit(".", maxsplit=1)
    try:
        func = import_object(entrypoint)  # pyright: ignore[reportRedeclaration]
    except AttributeError as exc:
        raise RuntimeError(
            f"Function with name {func_name!r} not found in {path!r}."
        ) from exc
    if isinstance(func, Flow):
        return func
    else:
        return Flow(func, log_prints=True)


def serve(
    *args: RunnerDeployment,
    pause_on_shutdown: bool = True,
    print_starting_message: bool = True,
    limit: Optional[int] = None,
    **kwargs: Any,
) -> None:
    from prefect.runner import Runner
    if is_in_async_context():
        raise RuntimeError(
            "Cannot call `serve` in an asynchronous context. Use `aserve` instead."
        )
    runner: Runner = Runner(pause_on_shutdown=pause_on_shutdown, limit=limit, **kwargs)
    for deployment in args:
        runner.add_deployment(deployment)
    if print_starting_message:
        _display_serve_start_message(*args)
    try:
        asyncio.run(runner.start())
    except (KeyboardInterrupt, TerminationSignal) as exc:
        logger.info(f"Received {type(exc).__name__}, shutting down...")


async def aserve(
    *args: RunnerDeployment,
    pause_on_shutdown: bool = True,
    print_starting_message: bool = True,
    limit: Optional[int] = None,
    **kwargs: Any,
) -> None:
    from prefect.runner import Runner
    runner: Runner = Runner(pause_on_shutdown=pause_on_shutdown, limit=limit, **kwargs)
    for deployment in args:
        await runner.add_deployment(deployment)
    if print_starting_message:
        _display_serve_start_message(*args)
    await runner.start()


def _display_serve_start_message(*args: RunnerDeployment) -> None:
    from rich.console import Console, Group
    from rich.table import Table
    help_message_top: str = (
        "[green]Your deployments are being served and polling for scheduled runs!\n[/]"
    )
    table: Table = Table(title="Deployments", show_header=False)
    table.add_column(style="blue", no_wrap=True)
    for deployment in args:
        table.add_row(f"{deployment.flow_name}/{deployment.name}")
    help_message_bottom: str = (
        "\nTo trigger any of these deployments, use the"
        " following command:\n[blue]\n\t$ prefect deployment run"
        " [DEPLOYMENT_NAME]\n[/]"
    )
    if PREFECT_UI_URL:
        help_message_bottom += (
            "\nYou can also trigger your deployments via the Prefect UI:"
            f" [blue]{PREFECT_UI_URL.value()}/deployments[/]\n"
        )
    console: Console = Console()
    console.print(Group(help_message_top, table, help_message_bottom), soft_wrap=True)


@client_injector
async def load_flow_from_flow_run(
    client: PrefectClient,
    flow_run: FlowRun,
    ignore_storage: bool = False,
    storage_base_path: Optional[str] = None,
    use_placeholder_flow: bool = True,
) -> Flow[..., Any]:
    if flow_run.deployment_id is None:
        raise ValueError("Flow run does not have an associated deployment")
    deployment = await client.read_deployment(flow_run.deployment_id)
    if deployment.entrypoint is None:
        raise ValueError(
            f"Deployment {deployment.id} does not have an entrypoint and can not be run."
        )
    run_logger = flow_run_logger(flow_run)
    runner_storage_base_path: Optional[str] = storage_base_path or os.environ.get(
        "PREFECT__STORAGE_BASE_PATH"
    )
    if ":" not in deployment.entrypoint:
        run_logger.debug(
            f"Importing flow code from module path {deployment.entrypoint}"
        )
        flow: Flow[..., Any] = await run_sync_in_worker_thread(
            load_flow_from_entrypoint,
            deployment.entrypoint,
            use_placeholder_flow=use_placeholder_flow,
        )
        return flow
    if not ignore_storage and not deployment.pull_steps:
        sys.path.insert(0, ".")
        if deployment.storage_document_id:
            storage_document = await client.read_block_document(
                deployment.storage_document_id
            )
            storage_block = Block._from_block_document(storage_document)
        else:
            basepath = deployment.path
            if runner_storage_base_path:
                basepath = str(basepath).replace(
                    "$STORAGE_BASE_PATH", runner_storage_base_path
                )
            storage_block = LocalFileSystem(basepath=basepath)
        from_path: str = (
            str(deployment.path).replace("$STORAGE_BASE_PATH", runner_storage_base_path)
            if runner_storage_base_path and deployment.path
            else deployment.path
        )
        run_logger.info(f"Downloading flow code from storage at {from_path!r}")
        await storage_block.get_directory(from_path=from_path, local_path=".")
    if deployment.pull_steps:
        run_logger.debug(
            f"Running {len(deployment.pull_steps)} deployment pull step(s)"
        )
        from prefect.deployments.steps.core import run_steps
        output: dict[str, Any] = await run_steps(deployment.pull_steps)
        if output.get("directory"):
            run_logger.debug(f"Changing working directory to {output['directory']!r}")
            os.chdir(output["directory"])
    import_path: str = relative_path_to_current_platform(deployment.entrypoint)
    run_logger.debug(f"Importing flow code from '{import_path}'")
    try:
        flow = await run_sync_in_worker_thread(
            load_flow_from_entrypoint,
            str(import_path),
            use_placeholder_flow=use_placeholder_flow,
        )
    except MissingFlowError:
        flow = await run_sync_in_worker_thread(
            load_function_and_convert_to_flow,
            str(import_path),
        )
    return flow


def load_placeholder_flow(entrypoint: str, raises: Exception) -> Flow[P, Any]:
    def _base_placeholder() -> None:
        raise raises

    def sync_placeholder_flow(*args: P.args, **kwargs: P.kwargs) -> NoReturn:
        _base_placeholder()

    async def async_placeholder_flow(*args: P.args, **kwargs: P.kwargs) -> NoReturn:
        _base_placeholder()

    placeholder_flow: Callable[..., Any] = (
        async_placeholder_flow
        if is_entrypoint_async(entrypoint)
        else sync_placeholder_flow
    )
    arguments: dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
    arguments["fn"] = placeholder_flow
    return Flow(**arguments)


def safe_load_flow_from_entrypoint(entrypoint: str) -> Optional[Flow[P, Any]]:
    func_def, source_code = _entrypoint_definition_and_source(entrypoint)
    path: Optional[str] = None
    if ":" in entrypoint:
        path = entrypoint.rsplit(":", maxsplit=1)[0]
    namespace: dict[str, Any] = safe_load_namespace(source_code, filepath=path)
    if func_def.name in namespace:
        return namespace[func_def.name]
    else:
        return _sanitize_and_load_flow(func_def, namespace)


def _sanitize_and_load_flow(
    func_def: Union[ast.FunctionDef, ast.AsyncFunctionDef], namespace: dict[str, Any]
) -> Optional[Flow[P, Any]]:
    args = func_def.args.posonlyargs + func_def.args.args + func_def.args.kwonlyargs
    if func_def.args.vararg:
        args.append(func_def.args.vararg)
    if func_def.args.kwarg:
        args.append(func_def.args.kwarg)
    for arg in args:
        if arg.annotation is not None:
            try:
                code = compile(
                    ast.Expression(arg.annotation),
                    filename="<ast>",
                    mode="eval",
                )
                exec(code, namespace)
            except Exception as e:
                logger.debug(
                    "Failed to evaluate annotation for argument %s due to the following error. Ignoring annotation.",
                    arg.arg,
                    exc_info=e,
                )
                arg.annotation = None
    new_defaults: list[Any] = []
    for default in func_def.args.defaults:
        try:
            code = compile(ast.Expression(default), "<ast>", "eval")
            exec(code, namespace)
            new_defaults.append(default)
        except Exception as e:
            logger.debug(
                "Failed to evaluate default value %s due to the following error. Ignoring default.",
                default,
                exc_info=e,
            )
            new_defaults.append(
                ast.Constant(
                    value=None, lineno=default.lineno, col_offset=default.col_offset
                )
            )
    func_def.args.defaults = new_defaults
    new_kw_defaults: list[Any] = []
    for default in func_def.args.kw_defaults:
        if default is not None:
            try:
                code = compile(ast.Expression(default), "<ast>", "eval")
                exec(code, namespace)
                new_kw_defaults.append(default)
            except Exception as e:
                logger.debug(
                    "Failed to evaluate default value %s due to the following error. Ignoring default.",
                    default,
                    exc_info=e,
                )
                new_kw_defaults.append(
                    ast.Constant(
                        value=None,
                        lineno=default.lineno,
                        col_offset=default.col_offset,
                    )
                )
        else:
            new_kw_defaults.append(
                ast.Constant(
                    value=None,
                    lineno=func_def.lineno,
                    col_offset=func_def.col_offset,
                )
            )
    func_def.args.kw_defaults = new_kw_defaults
    if func_def.returns is not None:
        try:
            code = compile(
                ast.Expression(func_def.returns), filename="<ast>", mode="eval"
            )
            exec(code, namespace)
        except Exception as e:
            logger.debug(
                "Failed to evaluate return annotation due to the following error. Ignoring annotation.",
                exc_info=e,
            )
            func_def.returns = None
    try:
        code = compile(
            ast.Module(body=[func_def], type_ignores=[]),
            filename="<ast>",
            mode="exec",
        )
        exec(code, namespace)
    except Exception as e:
        logger.debug("Failed to compile: %s", e)
    else:
        return namespace.get(func_def.name)


def load_flow_arguments_from_entrypoint(
    entrypoint: str, arguments: Optional[Union[list[str], set[str]]] = None
) -> dict[str, Any]:
    func_def, source_code = _entrypoint_definition_and_source(entrypoint)
    path: Optional[str] = None
    if ":" in entrypoint:
        path = entrypoint.rsplit(":", maxsplit=1)[0]
    if arguments is None:
        arguments = {
            "name",
            "version",
            "retries",
            "retry_delay_seconds",
            "description",
            "timeout_seconds",
            "validate_parameters",
            "persist_result",
            "cache_result_in_memory",
            "log_prints",
        }
    result: dict[str, Any] = {}
    for decorator in func_def.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and getattr(decorator.func, "id", "") == "flow"
        ):
            for keyword in decorator.keywords:
                if keyword.arg not in arguments:
                    continue
                if isinstance(keyword.value, ast.Constant):
                    result[cast(str, keyword.arg)] = str(keyword.value.value)
                    continue
                namespace: dict[str, Any] = safe_load_namespace(source_code, filepath=path)
                literal_arg_value = ast.get_source_segment(source_code, keyword.value)
                cleaned_value: str = literal_arg_value.replace("\n", "") if literal_arg_value else ""
                try:
                    evaluated_value = eval(cleaned_value, namespace)  # type: ignore
                    result[cast(str, keyword.arg)] = str(evaluated_value)
                except Exception as e:
                    logger.info(
                        "Failed to parse @flow argument: `%s=%s` due to the following error. Ignoring and falling back to default behavior.",
                        keyword.arg,
                        literal_arg_value,
                        exc_info=e,
                    )
                    continue
    if "name" in arguments and "name" not in result:
        result["name"] = func_def.name.replace("_", "-")
    return result


def is_entrypoint_async(entrypoint: str) -> bool:
    func_def, _ = _entrypoint_definition_and_source(entrypoint)
    return isinstance(func_def, ast.AsyncFunctionDef)


def _entrypoint_definition_and_source(
    entrypoint: str,
) -> Tuple[Union[ast.FunctionDef, ast.AsyncFunctionDef], str]:
    if ":" in entrypoint:
        path, func_name = entrypoint.rsplit(":", maxsplit=1)
        source_code: str = Path(path).read_text()
    else:
        path, func_name = entrypoint.rsplit(".", maxsplit=1)
        spec = importlib.util.find_spec(path)
        if not spec or not spec.origin:
            raise ValueError(f"Could not find module {path!r}")
        source_code = Path(spec.origin).read_text()
    parsed_code = ast.parse(source_code)
    func_def = next(
        (
            node
            for node in ast.walk(parsed_code)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ),
        None,
    )
    if not func_def:
        raise ValueError(f"Could not find flow {func_name!r} in {path!r}")
    return func_def, source_code