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
    ) -> Awaitable[None] | None: ...


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

    Args:
        fn: The function defining the workflow.
        name: An optional name for the flow; if not provided, the name will be inferred
            from the given function.
        version: An optional version string for the flow; if not provided, we will
            attempt to create a version string as a hash of the file containing the
            wrapped function; if the file cannot be located, the version will be null.
        flow_run_name: An optional name to distinguish runs of this flow; this name can
            be provided as a string template with the flow's parameters as variables,
            or a function that returns a string.
        task_runner: An optional task runner to use for task execution within the flow;
            if not provided, a `ThreadPoolTaskRunner` will be used.
        description: An optional string description for the flow; if not provided, the
            description will be pulled from the docstring for the decorated function.
        timeout_seconds: An optional number of seconds indicating a maximum runtime for
            the flow. If the flow exceeds this runtime, it will be marked as failed.
            Flow execution may continue until the next task is called.
        validate_parameters: By default, parameters passed to flows are validated by
            Pydantic. This will check that input values conform to the annotated types
            on the function. Where possible, values will be coerced into the correct
            type; for example, if a parameter is defined as `x: int` and "5" is passed,
            it will be resolved to `5`. If set to `False`, no validation will be
            performed on flow parameters.
        retries: An optional number of times to retry on flow run failure.
        retry_delay_seconds: An optional number of seconds to wait before retrying the
            flow after failure. This is only applicable if `retries` is nonzero.
        persist_result: An optional toggle indicating whether the result of this flow
            should be persisted to result storage. Defaults to `None`, which indicates
            that Prefect should choose whether the result should be persisted depending on
            the features being used.
        result_storage: An optional block to use to persist the result of this flow.
            This value will be used as the default for any tasks in this flow.
            If not provided, the local file system will be used unless called as
            a subflow, at which point the default will be loaded from the parent flow.
        result_serializer: An optional serializer to use to serialize the result of this
            flow for persistence. This value will be used as the default for any tasks
            in this flow. If not provided, the value of `PREFECT_RESULTS_DEFAULT_SERIALIZER`
            will be used unless called as a subflow, at which point the default will be
            loaded from the parent flow.
        on_failure: An optional list of callables to run when the flow enters a failed state.
        on_completion: An optional list of callables to run when the flow enters a completed state.
        on_cancellation: An optional list of callables to run when the flow enters a cancelling state.
        on_crashed: An optional list of callables to run when the flow enters a crashed state.
        on_running: An optional list of callables to run when the flow enters a running state.
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
    ):
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
        hook_categories = [
            on_completion,
            on_failure,
            on_cancellation,
            on_crashed,
            on_running,
        ]
        hook_names = [
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
            self.task_runner: TaskRunner[PrefectFuture[Any]] = (
                task_runner() if isinstance(task_runner, type) else task_runner
            )

        self.log_prints = log_prints

        self.description: str | None = description or inspect.getdoc(fn)
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
                flow_file = inspect.getsourcefile(self.fn)
                if flow_file is None:
                    raise FileNotFoundError
                version = file_hash(flow_file)
            except (FileNotFoundError, TypeError, OSError):
                pass  # `getsourcefile` can return null values and "<stdin>" for objects in repls
        self.version = version

        self.timeout_seconds: float | None = (
            float(timeout_seconds) if timeout_seconds else None
        )

        # FlowRunPolicy settings
        # TODO: We can instantiate a `FlowRunPolicy` and add Pydantic bound checks to
        #       validate that the user passes positive numbers here
        self.retries: int = (
            retries if retries is not None else PREFECT_FLOW_DEFAULT_RETRIES.value()
        )

        self.retry_delay_seconds: float | int = (
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
            module_name = inspect.getfile(fn)
            module = module_name if module_name != "__main__" else module

        self._entrypoint = f"{module}:{fn.__name__}"

    @property
    def ismethod(self) -> bool:
        return hasattr(self.fn, "__prefect_self__")

    def __get__(self, instance: Any, owner: Any) -> "Flow[P, R]":
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
            bound_flow = copy(self)
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
    ) -> "Flow[P, R]":
        """
        Create a new flow from the current object, updating provided options.

        Args:
            name: A new name for the flow.
            version: A new version for the flow.
            description: A new description for the flow.
            flow_run_name: An optional name to distinguish runs of this flow; this name
                can be provided as a string template with the flow's parameters as variables,
                or a function that returns a string.
            task_runner: A new task runner for the flow.
            timeout_seconds: A new number of seconds to fail the flow after if still
                running.
            validate_parameters: A new value indicating if flow calls should validate
                given parameters.
            retries: A new number of times to retry on flow run failure.
            retry_delay_seconds: A new number of seconds to wait before retrying the
                flow after failure. This is only applicable if `retries` is nonzero.
            persist_result: A new option for enabling or disabling result persistence.
            result_storage: A new storage type to use for results.
            result_serializer: A new serializer to use for results.
            cache_result_in_memory: A new value indicating if the flow's result should
                be cached in memory.
            on_failure: A new list of callables to run when the flow enters a failed state.
            on_completion: A new list of callables to run when the flow enters a completed state.
            on_cancellation: A new list of callables to run when the flow enters a cancelling state.
            on_crashed: A new list of callables to run when the flow enters a crashed state.
            on_running: A new list of callables to run when the flow enters a running state.

        Returns:
            A new `Flow` instance.

        Examples:

            Create a new flow from an existing flow and update the name:

            >>> @flow(name="My flow")
            >>> def my_flow():
            >>>     return 1
            >>>
            >>> new_flow = my_flow.with_options(name="My new flow")

            Create a new flow from an existing flow, update the task runner, and call
            it without an intermediate variable:

            >>> from prefect.task_runners import ThreadPoolTaskRunner
            >>>
            >>> @flow
            >>> def my_flow(x, y):
            >>>     return x + y
            >>>
            >>> state = my_flow.with_options(task_runner=ThreadPoolTaskRunner)(1, 3)
            >>> assert state.result() == 4
        """
        new_task_runner = (
            task_runner() if isinstance(task_runner, type) else task_runner
        )
        if new_task_runner is None:
            new_task_runner = self.task_runner
        new_flow = Flow(
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
            has_v1_models = any(isinstance(o, V1BaseModel) for o in args) or any(
                isinstance(o, V1BaseModel) for o in kwargs.values()
            )

        has_v2_types = any(is_v2_type(o) for o in args) or any(
            is_v2_type(o) for o in kwargs.values()
        )

        if has_v1_models and has_v2_types:
            raise ParameterTypeError(
                "Cannot mix Pydantic v1 and v2 types as arguments to a flow."
            )

        validated_fn_kwargs = dict(arbitrary_types_allowed=True)

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
            # We capture the pydantic exception and raise our own because the pydantic
            # exception is not picklable when using a cythonized pydantic installation
            logger.error(
                f"Parameter validation failed for flow {self.name!r}: {exc.errors()}"
                f"\nParameters: {parameters}"
            )
            raise ParameterTypeError.from_validation_error(exc) from None

        # Get the updated parameter dict with cast values from the model
        cast_parameters = {
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

        Uses FastAPI's `jsonable_encoder` to convert to JSON compatible objects without
        converting everything directly to a string. This maintains basic types like
        integers during API roundtrips.
        """
        serialized_parameters: dict[str, Any] = {}
        for key, value in parameters.items():
            # do not serialize the bound self object
            if self.ismethod and value is getattr(self.fn, "__prefect_self__", None):
                continue
            if isinstance(value, (PrefectFuture, State)):
                # Don't call jsonable_encoder() on a PrefectFuture or State to
                # avoid triggering a __getitem__ call
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

        Args:
            name: The name to give the created deployment.
            interval: An interval on which to execute the new deployment. Accepts either a number
                or a timedelta object. If a number is given, it will be interpreted as seconds.
            cron: A cron schedule of when to execute runs of this deployment.
            rrule: An rrule schedule of when to execute runs of this deployment.
            paused: Whether or not to set this deployment as paused.
            schedule: A schedule object defining when to execute runs of this deployment.
                Used to provide additional scheduling options like `timezone` or `parameters`.
            schedules: A list of schedule objects defining when to execute runs of this deployment.
                Used to define multiple schedules or additional scheduling options such as `timezone`.
            concurrency_limit: The maximum number of runs of this deployment that can run at the same time.
            parameters: A dictionary of default parameter values to pass to runs of this deployment.
            triggers: A list of triggers that will kick off runs of this deployment.
            description: A description for the created deployment. Defaults to the flow's
                description if not provided.
            tags: A list of tags to associate with the created deployment for organizational
                purposes.
            version: A version for the created deployment. Defaults to the flow's version.
            enforce_parameter_schema: Whether or not the Prefect API should enforce the
                parameter schema for the created deployment.
            work_pool_name: The name of the work pool to use for this deployment.
            work_queue_name: The name of the work queue to use for this deployment's scheduled runs.
                If not provided the default work queue for the work pool will be used.
            job_variables: Settings used to override the values specified default base job template
                of the chosen work pool. Refer to the base job template of the chosen work pool for
            entrypoint_type: Type of entrypoint to use for the deployment. When using a module path
                entrypoint, ensure that the module will be importable in the execution environment.
            _sla: (Experimental) SLA configuration for the deployment. May be removed or modified at any time. Currently only supported on Prefect Cloud.

        Examples:
            Prepare two deployments and serve them:

            