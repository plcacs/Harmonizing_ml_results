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
    Dict,
    Generic,
    Iterable,
    List,
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


class FlowStateHook(Protocol, Generic[P, R]):
    def __call__(self, flow: "Flow[P, R]", flow_run: FlowRun, state: State) -> None:
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
        result_storage: Optional[Union[Block, str]] = None,
        result_serializer: Optional[ResultSerializer] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[Iterable[FlowStateHook]] = None,
        on_failure: Optional[Iterable[FlowStateHook]] = None,
        on_cancellation: Optional[Iterable[FlowStateHook]] = None,
        on_crashed: Optional[Iterable[FlowStateHook]] = None,
        on_running: Optional[Iterable[FlowStateHook]] = None,
    ) -> None:
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f"Expected string for flow parameter 'name'; got {type(name).__name__} instead. "
                f"{'Perhaps you meant to call it? e.g. \'@flow(name=get_flow_run_name())\'' if callable(name) else ''}"
            )
        hook_categories = [on_completion, on_failure, on_cancellation, on_crashed, on_running]
        hook_names = ["on_completion", "on_failure", "on_cancellation", "on_crashed", "on_running"]
        for hooks, hook_name in zip(hook_categories, hook_names):
            if hooks is not None:
                try:
                    hooks = list(hooks)
                except TypeError:
                    raise TypeError(
                        f"Expected iterable for '{hook_name}'; got {type(hooks).__name__} instead. "
                        f"Please provide a list of hooks to '{hook_name}':\n\n"
                        f"@flow({hook_name}=[hook1, hook2])\n"
                        f"def my_flow():\n\tpass"
                    )
                for hook in hooks:
                    if not callable(hook):
                        raise TypeError(
                            f"Expected callables in '{hook_name}'; got {type(hook).__name__} instead. "
                            f"Please provide a list of hooks to '{hook_name}':\n\n"
                            f"@flow({hook_name}=[hook1, hook2])\n"
                            f"def my_flow():\n\tpass"
                        )
        if not callable(fn):
            raise TypeError("'fn' must be callable")
        self.name = name or fn.__name__.replace("_", "-").replace("<lambda>", "unknown-lambda")
        _raise_on_name_with_banned_characters(self.name)
        if flow_run_name is not None:
            if not isinstance(flow_run_name, str) and not callable(flow_run_name):
                raise TypeError(
                    f"Expected string or callable for 'flow_run_name'; got {type(flow_run_name).__name__} instead."
                )
        self.flow_run_name = flow_run_name
        if task_runner is None:
            self.task_runner = cast(
                TaskRunner[PrefectFuture[Any]], ThreadPoolTaskRunner()
            )
        else:
            self.task_runner = (
                task_runner() if isinstance(task_runner, type) else task_runner
            )
        self.log_prints = log_prints
        self.description = description or inspect.getdoc(fn)
        update_wrapper(self, fn)
        self.fn = fn
        self.isasync = asyncio.iscoroutinefunction(self.fn) or inspect.isasyncgenfunction(
            self.fn
        )
        self.isgenerator = inspect.isgeneratorfunction(self.fn) or inspect.isasyncgenfunction(
            self.fn
        )
        raise_for_reserved_arguments(self.fn, ["return_state", "wait_for"])
        if not version:
            try:
                flow_file = inspect.getsourcefile(self.fn)
                if flow_file is None:
                    raise FileNotFoundError
                version = file_hash(flow_file)
            except (FileNotFoundError, TypeError, OSError):
                pass
        self.version = version
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds else None
        self.retries = (
            retries if retries is not None else PREFECT_FLOW_DEFAULT_RETRIES.value()
        )
        self.retry_delay_seconds = (
            retry_delay_seconds
            if retry_delay_seconds is not None
            else PREFECT_FLOW_DEFAULT_RETRY_DELAY_SECONDS.value()
        )
        self.parameters = parameter_schema(self.fn)
        self.should_validate_parameters = validate_parameters
        if self.should_validate_parameters:
            try:
                ValidatedFunction(self.fn, config={"arbitrary_types_allowed": True})
            except ConfigError as exc:
                raise ValueError(
                    "Flow function is not compatible with `validate_parameters`. "
                    "Disable validation or change the argument names."
                ) from exc
        if persist_result is None:
            if result_storage is not None or result_serializer is not None:
                persist_result = True
        self.persist_result = persist_result
        if result_storage and not isinstance(result_storage, str):
            if getattr(result_storage, "_block_document_id", None) is None:
                raise TypeError(
                    "Result storage configuration must be persisted server-side. "
                    "Please call `.save()` on your block before passing it in."
                )
        self.result_storage = result_storage
        self.result_serializer = result_serializer
        self.cache_result_in_memory = cache_result_in_memory
        self.on_completion_hooks = on_completion or []
        self.on_failure_hooks = on_failure or []
        self.on_cancellation_hooks = on_cancellation or []
        self.on_crashed_hooks = on_crashed or []
        self.on_running_hooks = on_running or []
        self._storage = None
        self._entrypoint = None
        module = fn.__module__
        if module and (module == "__main__" or module.startswith("__prefect_loader_")):
            module_name = inspect.getfile(fn)
            module = module_name if module_name != "__main__" else module
        self._entrypoint = f"{module}:{fn.__name__}"

    @property
    def ismethod(self) -> bool:
        return hasattr(self.fn, "__prefect_self__")

    def __get__(self, instance: Any, owner: Any) -> "Flow[P, R]":
        if instance is None:
            return self
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
        retry_delay_seconds: Optional[float] = None,
        description: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        task_runner: Optional[TaskRunner] = None,
        timeout_seconds: Optional[float] = None,
        validate_parameters: Optional[bool] = None,
        persist_result: Any = NotSet,
        result_storage: Any = NotSet,
        result_serializer: Any = NotSet,
        cache_result_in_memory: Optional[bool] = None,
        log_prints: Any = NotSet,
        on_completion: Optional[Iterable[FlowStateHook]] = None,
        on_failure: Optional[Iterable[FlowStateHook]] = None,
        on_cancellation: Optional[Iterable[FlowStateHook]] = None,
        on_crashed: Optional[Iterable[FlowStateHook]] = None,
        on_running: Optional[Iterable[FlowStateHook]] = None,
    ) -> "Flow[P, R]":
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
            persist_result=persist_result if persist_result is not NotSet else self.persist_result,
            result_storage=result_storage if result_storage is not NotSet else self.result_storage,
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

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        def resolve_block_reference(data: Any) -> Any:
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
            logger.error(
                f"Parameter validation failed for flow {self.name!r}: {exc.errors()}\n"
                f"Parameters: {parameters}"
            )
            raise ParameterTypeError.from_validation_error(exc) from None
        cast_parameters = {
            k: v
            for k, v in dict(iter(model)).items()
            if k in model.model_fields_set or model.model_fields[k].default_factory
        }
        return cast_parameters

    def serialize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        serialized_parameters = {}
        for key, value in parameters.items():
            if self.ismethod and value is getattr(
                self.fn, "__prefect_self__", None
            ):
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
                    f"Type {type(value).__name__!r} and will not be stored in the backend."
                )
                serialized_parameters[key] = f"<{type(value).__name__}>"
        return serialized_parameters

    async def ato_deployment(
        self,
        name: str,
        interval: Optional[Union[float, datetime