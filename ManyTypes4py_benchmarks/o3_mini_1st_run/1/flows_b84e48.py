#!/usr/bin/env python3
"""
Module containing the base workflow class and decorator - for most use cases, using the
[`@flow` decorator][prefect.flows.flow] is preferred.
"""

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
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterable,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
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
from prefect.utilities.callables import ParameterSchema, get_call_parameters, parameter_schema, parameters_to_args_kwargs, raise_for_reserved_arguments
from prefect.utilities.collections import listrepr, visit_collection
from prefect.utilities.filesystem import relative_path_to_current_platform
from prefect.utilities.hashing import file_hash
from prefect.utilities.importtools import import_object, safe_load_namespace
from ._internal.compatibility.async_dispatch import async_dispatch, is_in_async_context
from ._internal.pydantic.v2_schema import is_v2_type
from ._internal.pydantic.v2_validated_func import V2ValidatedFunction

if TYPE_CHECKING:
    import logging
    from prefect.client.orchestration import PrefectClient
    from prefect.client.schemas.objects import FlowRun
    from prefect.client.types.flexible_schedule_list import FlexibleScheduleList
    from prefect.deployments.runner import RunnerDeployment
    from prefect.runner.storage import RunnerStorage

logger = get_logger('flows')

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')
F = TypeVar('F', bound='Flow[Any, Any]')

class FlowStateHook(Protocol, Generic[P, R]):
    """
    A callable that is invoked when a flow enters a given state.
    """

    def __call__(self, flow: Flow[Any, Any], flow_run: Any, state: Any) -> Any:
        ...


class Flow(Generic[P, R]):
    """
    A Prefect workflow definition.

    !!! note
        We recommend using the [`@flow` decorator][prefect.flows.flow] for most use-cases.

    Wraps a function with an entrypoint to the Prefect engine. To preserve the input
    and output types, we use the generic type variables `P` and `R` for "Parameters" and
    "Returns" respectively.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[int, float]] = None,
        task_runner: Optional[Union[TaskRunner[Any], Type[TaskRunner[Any]]]] = None,
        description: Optional[str] = None,
        timeout_seconds: Optional[Union[int, float]] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[Any] = None,
        result_serializer: Optional[Any] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[Sequence[Callable[..., Any]]] = None,
        on_failure: Optional[Sequence[Callable[..., Any]]] = None,
        on_cancellation: Optional[Sequence[Callable[..., Any]]] = None,
        on_crashed: Optional[Sequence[Callable[..., Any]]] = None,
        on_running: Optional[Sequence[Callable[..., Any]]] = None,
    ) -> None:
        if name is not None and (not isinstance(name, str)):
            raise TypeError(
                "Expected string for flow parameter 'name'; got {} instead. {}".format(
                    type(name).__name__,
                    "Perhaps you meant to call it? e.g. '@flow(name=get_flow_run_name())'" if callable(name) else ''
                )
            )
        hook_categories = [on_completion, on_failure, on_cancellation, on_crashed, on_running]
        hook_names = ['on_completion', 'on_failure', 'on_cancellation', 'on_crashed', 'on_running']
        for hooks, hook_name in zip(hook_categories, hook_names):
            if hooks is not None:
                try:
                    hooks = list(hooks)
                except TypeError:
                    raise TypeError(
                        f"Expected iterable for '{hook_name}'; got {type(hooks).__name__} instead. Please provide a list of hooks to '{hook_name}':\n\n@flow({hook_name}=[hook1, hook2])\ndef my_flow():\n\tpass"
                    )
                for hook in hooks:
                    if not callable(hook):
                        raise TypeError(
                            f"Expected callables in '{hook_name}'; got {type(hook).__name__} instead. Please provide a list of hooks to '{hook_name}':\n\n@flow({hook_name}=[hook1, hook2])\ndef my_flow():\n\tpass"
                        )
        if not callable(fn):
            raise TypeError("'fn' must be callable")
        self.name: str = name or fn.__name__.replace('_', '-').replace('<lambda>', 'unknown-lambda')
        _raise_on_name_with_banned_characters(self.name)
        if flow_run_name is not None:
            if not isinstance(flow_run_name, str) and (not callable(flow_run_name)):
                raise TypeError(f"Expected string or callable for 'flow_run_name'; got {type(flow_run_name).__name__} instead.")
        self.flow_run_name = flow_run_name
        if task_runner is None:
            self.task_runner = cast(TaskRunner[PrefectFuture[Any]], ThreadPoolTaskRunner())
        else:
            self.task_runner = task_runner() if isinstance(task_runner, type) else task_runner
        self.log_prints = log_prints
        self.description: Optional[str] = description or inspect.getdoc(fn)
        update_wrapper(self, fn)
        self.fn: Callable[..., Any] = fn
        self.isasync: bool = asyncio.iscoroutinefunction(self.fn) or inspect.isasyncgenfunction(self.fn)
        self.isgenerator: bool = inspect.isgeneratorfunction(self.fn) or inspect.isasyncgenfunction(self.fn)
        raise_for_reserved_arguments(self.fn, ['return_state', 'wait_for'])
        if not version:
            try:
                flow_file = inspect.getsourcefile(self.fn)
                if flow_file is None:
                    raise FileNotFoundError
                version = file_hash(flow_file)
            except (FileNotFoundError, TypeError, OSError):
                pass
        self.version: Optional[str] = version
        self.timeout_seconds: Optional[float] = float(timeout_seconds) if timeout_seconds else None
        self.retries: int = retries if retries is not None else PREFECT_FLOW_DEFAULT_RETRIES.value()
        self.retry_delay_seconds: Union[int, float] = (
            retry_delay_seconds if retry_delay_seconds is not None else PREFECT_FLOW_DEFAULT_RETRY_DELAY_SECONDS.value()
        )
        self.parameters: ParameterSchema = parameter_schema(self.fn)
        self.should_validate_parameters: bool = validate_parameters
        if self.should_validate_parameters:
            try:
                # This may raise a ConfigError if the function is not compatible.
                V2ValidatedFunction(self.fn, config={'arbitrary_types_allowed': True})
            except ConfigError as exc:
                raise ValueError(
                    'Flow function is not compatible with `validate_parameters`. Disable validation or change the argument names.'
                ) from exc
        if persist_result is None:
            if result_storage is not None or result_serializer is not None:
                persist_result = True
        self.persist_result: Optional[bool] = persist_result
        if result_storage and (not isinstance(result_storage, str)):
            if getattr(result_storage, '_block_document_id', None) is None:
                raise TypeError('Result storage configuration must be persisted server-side. Please call `.save()` on your block before passing it in.')
        self.result_storage = result_storage
        self.result_serializer = result_serializer
        self.cache_result_in_memory: bool = cache_result_in_memory
        self.on_completion_hooks: Sequence[Callable[..., Any]] = on_completion or []
        self.on_failure_hooks: Sequence[Callable[..., Any]] = on_failure or []
        self.on_cancellation_hooks: Sequence[Callable[..., Any]] = on_cancellation or []
        self.on_crashed_hooks: Sequence[Callable[..., Any]] = on_crashed or []
        self.on_running_hooks: Sequence[Callable[..., Any]] = on_running or []
        self._storage: Optional[Any] = None
        self._entrypoint: Optional[str] = None
        module = fn.__module__
        if module and (module == '__main__' or module.startswith('__prefect_loader_')):
            module_name = inspect.getfile(fn)
            module = module_name if module_name != '__main__' else module
        self._entrypoint = f'{module}:{fn.__name__}'

    @property
    def ismethod(self) -> bool:
        return hasattr(self.fn, '__prefect_self__')

    def __get__(self, instance: Any, owner: Any) -> Any:
        """
        Implement the descriptor protocol so that the flow can be used as an instance method.
        When an instance method is loaded, this method is called with the "self" instance as
        an argument. We return a copy of the flow with that instance bound to the flow's function.
        """
        if instance is None:
            return self
        else:
            bound_flow = copy(self)
            setattr(bound_flow.fn, '__prefect_self__', instance)
            return bound_flow

    def with_options(
        self,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[int, float]] = None,
        description: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        task_runner: Optional[Union[TaskRunner[Any], Type[TaskRunner[Any]]]] = None,
        timeout_seconds: Optional[Union[int, float]] = None,
        validate_parameters: Optional[bool] = None,
        persist_result: Any = NotSet,
        result_storage: Any = NotSet,
        result_serializer: Any = NotSet,
        cache_result_in_memory: Optional[bool] = None,
        log_prints: Any = NotSet,
        on_completion: Optional[Sequence[Callable[..., Any]]] = None,
        on_failure: Optional[Sequence[Callable[..., Any]]] = None,
        on_cancellation: Optional[Sequence[Callable[..., Any]]] = None,
        on_crashed: Optional[Sequence[Callable[..., Any]]] = None,
        on_running: Optional[Sequence[Callable[..., Any]]] = None,
    ) -> Flow[Any, Any]:
        new_task_runner: Union[TaskRunner[Any], Type[TaskRunner[Any]]] = task_runner() if isinstance(task_runner, type) else task_runner
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
            retry_delay_seconds=retry_delay_seconds if retry_delay_seconds is not None else self.retry_delay_seconds,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            validate_parameters=validate_parameters if validate_parameters is not None else self.should_validate_parameters,
            persist_result=persist_result if persist_result is not NotSet else self.persist_result,
            result_storage=result_storage if result_storage is not NotSet else self.result_storage,
            result_serializer=result_serializer if result_serializer is not NotSet else self.result_serializer,
            cache_result_in_memory=cache_result_in_memory if cache_result_in_memory is not None else self.cache_result_in_memory,
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
        """
        Validate parameters for compatibility with the flow by attempting to cast the inputs to the
        associated types specified by the function's type annotations.

        Returns:
            A new dict of parameters that have been cast to the appropriate types

        Raises:
            ParameterTypeError: if the provided parameters are not valid
        """

        def resolve_block_reference(data: Any) -> Any:
            if isinstance(data, dict) and '$ref' in data:
                return Block.load_from_ref(data['$ref'], _sync=True)
            return data

        try:
            parameters = visit_collection(parameters, resolve_block_reference, return_data=True)
        except (ValueError, RuntimeError) as exc:
            raise ParameterTypeError('Failed to resolve block references in parameters.') from exc
        args, kwargs = parameters_to_args_kwargs(self.fn, parameters)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pydantic.warnings.PydanticDeprecatedSince20)
            has_v1_models = any((isinstance(o, V1BaseModel) for o in args)) or any((isinstance(o, V1BaseModel) for o in kwargs.values()))
        has_v2_types = any((is_v2_type(o) for o in args)) or any((is_v2_type(o) for o in kwargs.values()))
        if has_v1_models and has_v2_types:
            raise ParameterTypeError('Cannot mix Pydantic v1 and v2 types as arguments to a flow.')
        validated_fn_kwargs: Dict[str, Any] = dict(arbitrary_types_allowed=True)
        if has_v1_models:
            validated_fn = V1ValidatedFunction(self.fn, config=validated_fn_kwargs)
        else:
            validated_fn = V2ValidatedFunction(self.fn, config=validated_fn_kwargs)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=pydantic.warnings.PydanticDeprecatedSince20)
                model = validated_fn.init_model_instance(*args, **kwargs)
        except pydantic.ValidationError as exc:
            logger.error(f'Parameter validation failed for flow {self.name!r}: {exc.errors()}\nParameters: {parameters}')
            raise ParameterTypeError.from_validation_error(exc) from None
        cast_parameters = {k: v for k, v in dict(iter(model)).items() if k in model.model_fields_set or model.model_fields[k].default_factory}
        return cast_parameters

    def serialize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parameters to a serializable form.

        Uses FastAPI's `jsonable_encoder` to convert to JSON compatible objects without
        converting everything directly to a string. This maintains basic types like
        integers during API roundtrips.
        """
        serialized_parameters: Dict[str, Any] = {}
        for key, value in parameters.items():
            if self.ismethod and value is getattr(self.fn, '__prefect_self__', None):
                continue
            if isinstance(value, (PrefectFuture, State)):
                serialized_parameters[key] = f'<{type(value).__name__}>'
                continue
            try:
                from fastapi.encoders import jsonable_encoder
                serialized_parameters[key] = jsonable_encoder(value)
            except (TypeError, ValueError):
                logger.debug(f'Parameter {key!r} for flow {self.name!r} is unserializable. Type {type(value).__name__!r} and will not be stored in the backend.')
                serialized_parameters[key] = f'<{type(value).__name__}>'
        return serialized_parameters

    async def ato_deployment(
        self,
        name: str,
        interval: Optional[Union[int, datetime.timedelta]] = None,
        cron: Optional[Union[str, Sequence[str]]] = None,
        rrule: Optional[Union[str, Sequence[str]]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[Sequence[Schedule]] = None,
        concurrency_limit: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[Sequence[Any]] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        _sla: Optional[Any] = None,
    ) -> Any:
        from prefect.deployments.runner import RunnerDeployment
        if not name.endswith('.py'):
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
        interval: Optional[Union[int, datetime.timedelta]] = None,
        cron: Optional[Union[str, Sequence[str]]] = None,
        rrule: Optional[Union[str, Sequence[str]]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[Sequence[Schedule]] = None,
        concurrency_limit: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[Sequence[Any]] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        _sla: Optional[Any] = None,
    ) -> Any:
        from prefect.deployments.runner import RunnerDeployment
        if not name.endswith('.py'):
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
                    _sync=True,
                )
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

    def on_completion(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.on_completion_hooks.append(fn)
        return fn

    def on_cancellation(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.on_cancellation_hooks.append(fn)
        return fn

    def on_crashed(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.on_crashed_hooks.append(fn)
        return fn

    def on_running(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.on_running_hooks.append(fn)
        return fn

    def on_failure(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.on_failure_hooks.append(fn)
        return fn

    def serve(
        self,
        name: Optional[str] = None,
        interval: Optional[Union[int, datetime.timedelta]] = None,
        cron: Optional[Union[str, Sequence[str]]] = None,
        rrule: Optional[Union[str, Sequence[str]]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[Sequence[Schedule]] = None,
        global_limit: Optional[int] = None,
        triggers: Optional[Sequence[Any]] = None,
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
        from prefect.runner import Runner
        if not name:
            name = self.name
        else:
            name = Path(name).stem
        runner = Runner(name=name, pause_on_shutdown=pause_on_shutdown, limit=limit)
        deployment_id = runner.add_flow(
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
            help_message = f"[green]Your flow {self.name!r} is being served and polling for scheduled runs!\n[/]\nTo trigger a run for this flow, use the following command:\n[blue]\n\t$ prefect deployment run '{self.name}/{name}'\n[/]"
            if PREFECT_UI_URL:
                help_message += f'\nYou can also run your flow via the Prefect UI: [blue]{PREFECT_UI_URL.value()}/deployments/deployment/{deployment_id}[/]\n'
            console = Console()
            console.print(help_message, soft_wrap=True)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            if 'no running event loop' in str(exc):
                loop = None
            else:
                raise
        try:
            if loop is not None:
                loop.run_until_complete(runner.start(webserver=webserver))
            else:
                asyncio.run(runner.start(webserver=webserver))
        except (KeyboardInterrupt, TerminationSignal) as exc:
            logger.info(f'Received {type(exc).__name__}, shutting down...')
            if loop is not None:
                loop.stop()

    @classmethod
    async def afrom_source(cls, source: Union[str, Path, Any], entrypoint: str) -> Flow[Any, Any]:
        from prefect.runner.storage import BlockStorageAdapter, LocalStorage, RunnerStorage, create_storage_from_source
        if isinstance(source, (Path, str)):
            if isinstance(source, Path):
                source = str(source)
            storage = create_storage_from_source(source)
        elif isinstance(source, RunnerStorage):
            storage = source
        elif hasattr(source, 'get_directory'):
            storage = BlockStorageAdapter(source)
        else:
            raise TypeError(f'Unsupported source type {type(source).__name__!r}. Please provide a URL to remote storage or a storage object.')
        with tempfile.TemporaryDirectory() as tmpdir:
            if not isinstance(storage, LocalStorage):
                storage.set_base_path(Path(tmpdir))
                await storage.pull_code()
            full_entrypoint: str = str(storage.destination / entrypoint)
            flow = cast('Flow[..., Any]', await from_async.wait_for_call_in_new_thread(create_call(load_flow_from_entrypoint, full_entrypoint)))
            flow._storage = storage
            flow._entrypoint = entrypoint
        return flow

    @classmethod
    @async_dispatch(afrom_source)
    def from_source(cls, source: Union[str, Path, Any], entrypoint: str) -> Flow[Any, Any]:
        from prefect.runner.storage import BlockStorageAdapter, LocalStorage, RunnerStorage, create_storage_from_source
        if isinstance(source, (Path, str)):
            if isinstance(source, Path):
                source = str(source)
            storage = create_storage_from_source(source)
        elif isinstance(source, RunnerStorage):
            storage = source
        elif hasattr(source, 'get_directory'):
            storage = BlockStorageAdapter(source)
        else:
            raise TypeError(f'Unsupported source type {type(source).__name__!r}. Please provide a URL to remote storage or a storage object.')
        with tempfile.TemporaryDirectory() as tmpdir:
            if not isinstance(storage, LocalStorage):
                storage.set_base_path(Path(tmpdir))
                run_coro_as_sync(storage.pull_code())
            full_entrypoint: str = str(storage.destination / entrypoint)
            flow = load_flow_from_entrypoint(full_entrypoint)
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
        job_variables: Optional[Dict[str, Any]] = None,
        interval: Optional[Union[int, datetime.timedelta]] = None,
        cron: Optional[Union[str, Sequence[str]]] = None,
        rrule: Optional[Union[str, Sequence[str]]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[Sequence[Schedule]] = None,
        concurrency_limit: Optional[int] = None,
        triggers: Optional[Sequence[Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        print_next_steps: bool = True,
        ignore_warnings: bool = False,
        _sla: Optional[Any] = None,
    ) -> Any:
        if not (work_pool_name := (work_pool_name or PREFECT_DEFAULT_WORK_POOL_NAME.value())):
            raise ValueError('No work pool name provided. Please provide a `work_pool_name` or set the `PREFECT_DEFAULT_WORK_POOL_NAME` environment variable.')
        from prefect.client.orchestration import get_client
        try:
            async with get_client() as client:
                work_pool = await client.read_work_pool(work_pool_name)
                active_workers = await client.read_workers_for_work_pool(
                    work_pool_name,
                    worker_filter=WorkerFilter(status=WorkerFilterStatus(any_=['ONLINE']))
                )
        except ObjectNotFound as exc:
            raise ValueError(f'Could not find work pool {work_pool_name!r}. Please create it before deploying this flow.') from exc
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
            if not work_pool.is_push_pool and (not work_pool.is_managed_pool) and (not active_workers):
                console.print(f'\nTo execute flow runs from this deployment, start a worker in a separate terminal that pulls work from the {work_pool_name!r} work pool:')
                console.print(f'\n\t$ prefect worker start --pool {work_pool_name!r}', style='blue')
            console.print('\nTo schedule a run for this deployment, use the following command:')
            console.print(f"\n\t$ prefect deployment run '{self.name}/{name}'\n", style='blue')
            if PREFECT_UI_URL:
                message = f'\nYou can also run your flow via the Prefect UI: [blue]{PREFECT_UI_URL.value()}/deployments/deployment/{deployment_ids[0]}[/]\n'
                console.print(message, soft_wrap=True)
        return deployment_ids[0]

    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any: ...
    @overload
    def __call__(self, *args: P.args, return_state: bool, **kwargs: P.kwargs) -> Any: ...

    def __call__(self, *args: Any, return_state: bool = False, wait_for: Optional[Any] = None, **kwargs: Any) -> Any:
        """
        Run the flow and return its result.

        Flow parameter values must be serializable by Pydantic.

        If writing an async flow, this call must be awaited.

        This will create a new flow run in the API.

        Args:
            *args: Arguments to run the flow with.
            return_state: Return a Prefect State containing the result of the
                flow run.
            wait_for: Upstream task futures to wait for before starting the flow if called as a subflow
            **kwargs: Keyword arguments to run the flow with.

        Returns:
            If `return_state` is False, returns the result of the flow run.
            If `return_state` is True, returns the result of the flow run
                wrapped in a Prefect State which provides error handling.
        """
        from prefect.utilities.visualization import get_task_viz_tracker, track_viz_task
        parameters = get_call_parameters(self.fn, args, kwargs)
        return_type = 'state' if return_state else 'result'
        task_viz_tracker = get_task_viz_tracker()
        if task_viz_tracker:
            return track_viz_task(self.isasync, self.name, parameters)
        from prefect.flow_engine import run_flow
        return run_flow(flow=self, parameters=parameters, wait_for=wait_for, return_type=return_type)

    @sync_compatible
    async def visualize(self, *args: Any, **kwargs: Any) -> None:
        """
        Generates a graphviz object representing the current flow. In IPython notebooks,
        it's rendered inline, otherwise in a new window as a PNG.

        Raises:
            - ImportError: If `graphviz` isn't installed.
            - GraphvizExecutableNotFoundError: If the `dot` executable isn't found.
            - FlowVisualizationError: If the flow can't be visualized for any other reason.
        """
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
            warnings.warn('`flow.visualize()` will execute code inside of your flow that is not decorated with `@task` or `@flow`.')
        try:
            with TaskVizTracker() as tracker:
                if self.isasync:
                    await self.fn(*args, **kwargs)
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
            msg = "It's possible you are trying to visualize a flow that contains code that directly interacts with the result of a task inside of the flow. \nTry passing a `viz_return_value` to the task decorator, e.g. `@task(viz_return_value=[1, 2, 3]).`"
            new_exception = type(e)(str(e) + '\n' + msg)
            new_exception.__traceback__ = e.__traceback__
            raise new_exception


class FlowDecorator:
    @overload
    def __call__(self, __fn: Callable[..., Any]) -> Flow[Any, Any]: ...
    @overload
    def __call__(
        self,
        __fn: None = None,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[int, float]] = None,
        task_runner: Optional[Union[TaskRunner[Any], Type[TaskRunner[Any]]]] = None,
        description: Optional[str] = None,
        timeout_seconds: Optional[Union[int, float]] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[Any] = None,
        result_serializer: Optional[Any] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[Sequence[Callable[..., Any]]] = None,
        on_failure: Optional[Sequence[Callable[..., Any]]] = None,
        on_cancellation: Optional[Sequence[Callable[..., Any]]] = None,
        on_crashed: Optional[Sequence[Callable[..., Any]]] = None,
        on_running: Optional[Sequence[Callable[..., Any]]] = None,
    ) -> Callable[[Callable[..., Any]], Flow[Any, Any]]: ...

    def __call__(
        self,
        __fn: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        flow_run_name: Optional[Union[str, Callable[..., str]]] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[int, float]] = None,
        task_runner: Optional[Union[TaskRunner[Any], Type[TaskRunner[Any]]]] = None,
        description: Optional[str] = None,
        timeout_seconds: Optional[Union[int, float]] = None,
        validate_parameters: bool = True,
        persist_result: Optional[bool] = None,
        result_storage: Optional[Any] = None,
        result_serializer: Optional[Any] = None,
        cache_result_in_memory: bool = True,
        log_prints: Optional[bool] = None,
        on_completion: Optional[Sequence[Callable[..., Any]]] = None,
        on_failure: Optional[Sequence[Callable[..., Any]]] = None,
        on_cancellation: Optional[Sequence[Callable[..., Any]]] = None,
        on_crashed: Optional[Sequence[Callable[..., Any]]] = None,
        on_running: Optional[Sequence[Callable[..., Any]]] = None,
    ) -> Union[Flow[Any, Any], Callable[[Callable[..., Any]], Flow[Any, Any]]]:
        """
        Decorator to designate a function as a Prefect workflow.
        """
        if __fn:
            if isinstance(__fn, (classmethod, staticmethod)):
                method_decorator = type(__fn).__name__
                raise TypeError(f'@{method_decorator} should be applied on top of @flow')
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
                Callable[[Callable[..., Any]], Flow[Any, Any]],
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
                )
            )

if not TYPE_CHECKING:
    from_source = staticmethod(Flow.from_source)
else:
    @staticmethod
    def from_source(source: Union[str, Path, Any], entrypoint: str) -> Any:
        ...

flow: FlowDecorator = FlowDecorator()


def _raise_on_name_with_banned_characters(name: Optional[str]) -> Optional[str]:
    """
    Raise an InvalidNameError if the given name contains any invalid characters.
    """
    if name is None:
        return name
    if not re.match(WITHOUT_BANNED_CHARACTERS, name):
        raise InvalidNameError(f'Name {name!r} contains an invalid character. Must not contain any of: {BANNED_CHARACTERS}.')
    return name


def select_flow(
    flows: Iterable[Flow[Any, Any]],
    flow_name: Optional[str] = None,
    from_message: Optional[str] = None
) -> Flow[Any, Any]:
    """
    Select the only flow in an iterable or a flow specified by name.

    Returns:
        A single flow object

    Raises:
        MissingFlowError: If no flows exist in the iterable
        MissingFlowError: If a flow name is provided and that flow does not exist
        UnspecifiedFlowError: If multiple flows exist but no flow name was provided
    """
    flows_dict: Dict[str, Flow[Any, Any]] = {f.name: f for f in flows}
    from_message = ' ' + from_message if from_message else ''
    if not flows_dict:
        raise MissingFlowError(f'No flows found{from_message}.')
    elif flow_name and flow_name not in flows_dict:
        raise MissingFlowError(
            f'Flow {flow_name!r} not found{from_message}. Found the following flows: {listrepr(list(flows_dict.keys()))}. Check to make sure that your flow function is decorated with `@flow`.'
        )
    elif not flow_name and len(flows_dict) > 1:
        raise UnspecifiedFlowError(
            f'Found {len(flows_dict)} flows{from_message}: {listrepr(sorted(list(flows_dict.keys())))}. Specify a flow name to select a flow.'
        )
    if flow_name:
        return flows_dict[flow_name]
    else:
        return list(flows_dict.values())[0]


def load_flow_from_entrypoint(entrypoint: str, use_placeholder_flow: bool = True) -> Flow[Any, Any]:
    """
    Extract a flow object from a script at an entrypoint by running all of the code in the file.

    Args:
        entrypoint: a string in the format `<path_to_script>:<flow_func_name>` or a module path
            to a flow function
        use_placeholder_flow: if True, use a placeholder Flow object if the actual flow object
            cannot be loaded from the entrypoint (e.g. dependencies are missing)

    Returns:
        The flow object from the script

    Raises:
        ScriptError: If an exception is encountered while running the script
        MissingFlowError: If the flow function specified in the entrypoint does not exist
    """
    if ':' in entrypoint:
        path, func_name = entrypoint.rsplit(':', maxsplit=1)
    else:
        path, func_name = entrypoint.rsplit('.', maxsplit=1)
    try:
        flow_obj = import_object(entrypoint)
    except AttributeError as exc:
        raise MissingFlowError(f'Flow function with name {func_name!r} not found in {path!r}.') from exc
    except ScriptError:
        if use_placeholder_flow:
            flow_obj = safe_load_flow_from_entrypoint(entrypoint)
            if flow_obj is None:
                raise
        else:
            raise
    if not isinstance(flow_obj, Flow):
        raise MissingFlowError(f"Function with name {func_name!r} is not a flow. Make sure that it is decorated with '@flow'.")
    return flow_obj


def load_function_and_convert_to_flow(entrypoint: str) -> Flow[Any, Any]:
    """
    Loads a function from an entrypoint and converts it to a flow if it is not already a flow.
    """
    if ':' in entrypoint:
        path, func_name = entrypoint.rsplit(':', maxsplit=1)
    else:
        path, func_name = entrypoint.rsplit('.', maxsplit=1)
    try:
        func = import_object(entrypoint)
    except AttributeError as exc:
        raise RuntimeError(f'Function with name {func_name!r} not found in {path!r}.') from exc
    if isinstance(func, Flow):
        return func
    else:
        return Flow(func, log_prints=True)


def serve(*args: Any, pause_on_shutdown: bool = True, print_starting_message: bool = True, limit: Optional[int] = None, **kwargs: Any) -> None:
    """
    Serve the provided list of deployments.

    Args:
        *args: A list of deployments to serve.
        pause_on_shutdown: A boolean for whether or not to automatically pause deployment schedules on shutdown.
        print_starting_message: Whether or not to print message to the console on startup.
        limit: The maximum number of runs that can be executed concurrently.
        **kwargs: Additional keyword arguments to pass to the runner.
    """
    from prefect.runner import Runner
    if is_in_async_context():
        raise RuntimeError('Cannot call `serve` in an asynchronous context. Use `aserve` instead.')
    runner = Runner(pause_on_shutdown=pause_on_shutdown, limit=limit, **kwargs)
    for deployment in args:
        runner.add_deployment(deployment)
    if print_starting_message:
        _display_serve_start_message(*args)
    try:
        asyncio.run(runner.start())
    except (KeyboardInterrupt, TerminationSignal) as exc:
        logger.info(f'Received {type(exc).__name__}, shutting down...')


async def aserve(*args: Any, pause_on_shutdown: bool = True, print_starting_message: bool = True, limit: Optional[int] = None, **kwargs: Any) -> None:
    """
    Asynchronously serve the provided list of deployments.

    Use `serve` instead if calling from a synchronous context.
    """
    from prefect.runner import Runner
    runner = Runner(pause_on_shutdown=pause_on_shutdown, limit=limit, **kwargs)
    for deployment in args:
        add_deployment_coro = runner.add_deployment(deployment)
        if TYPE_CHECKING:
            assert inspect.isawaitable(add_deployment_coro)
        await add_deployment_coro
    if print_starting_message:
        _display_serve_start_message(*args)
    await runner.start()


def _display_serve_start_message(*args: Any) -> None:
    from rich.console import Console, Group
    from rich.table import Table
    help_message_top = '[green]Your deployments are being served and polling for scheduled runs!\n[/]'
    table = Table(title='Deployments', show_header=False)
    table.add_column(style='blue', no_wrap=True)
    for deployment in args:
        table.add_row(f'{deployment.flow_name}/{deployment.name}')
    help_message_bottom = '\nTo trigger any of these deployments, use the following command:\n[blue]\n\t$ prefect deployment run [DEPLOYMENT_NAME]\n[/]'
    if PREFECT_UI_URL:
        help_message_bottom += f'\nYou can also trigger your deployments via the Prefect UI: [blue]{PREFECT_UI_URL.value()}/deployments[/]\n'
    console = Console()
    console.print(Group(help_message_top, table, help_message_bottom), soft_wrap=True)


@client_injector
async def load_flow_from_flow_run(
    client: Any,
    flow_run: FlowRun,
    ignore_storage: bool = False,
    storage_base_path: Optional[str] = None,
    use_placeholder_flow: bool = True,
) -> Flow[Any, Any]:
    """
    Load a flow from the location/script provided in a deployment's storage document.
    """
    if flow_run.deployment_id is None:
        raise ValueError('Flow run does not have an associated deployment')
    deployment = await client.read_deployment(flow_run.deployment_id)
    if deployment.entrypoint is None:
        raise ValueError(f'Deployment {deployment.id} does not have an entrypoint and can not be run.')
    run_logger = flow_run_logger(flow_run)
    runner_storage_base_path: Optional[str] = storage_base_path or os.environ.get('PREFECT__STORAGE_BASE_PATH')
    if ':' not in deployment.entrypoint:
        run_logger.debug(f'Importing flow code from module path {deployment.entrypoint}')
        flow = await run_sync_in_worker_thread(load_flow_from_entrypoint, deployment.entrypoint, use_placeholder_flow=use_placeholder_flow)
        return flow
    if not ignore_storage and (not deployment.pull_steps):
        sys.path.insert(0, '.')
        if deployment.storage_document_id:
            storage_document = await client.read_block_document(deployment.storage_document_id)
            storage_block = Block._from_block_document(storage_document)
        else:
            basepath = deployment.path
            if runner_storage_base_path:
                basepath = str(basepath).replace('$STORAGE_BASE_PATH', runner_storage_base_path)
            storage_block = LocalFileSystem(basepath=basepath)
        from_path = str(deployment.path).replace('$STORAGE_BASE_PATH', runner_storage_base_path) if runner_storage_base_path and deployment.path else deployment.path
        run_logger.info(f'Downloading flow code from storage at {from_path!r}')
        await storage_block.get_directory(from_path=from_path, local_path='.')
    if deployment.pull_steps:
        run_logger.debug(f'Running {len(deployment.pull_steps)} deployment pull step(s)')
        from prefect.deployments.steps.core import run_steps
        output = await run_steps(deployment.pull_steps)
        if output.get('directory'):
            run_logger.debug(f"Changing working directory to {output['directory']!r}")
            os.chdir(output['directory'])
    import_path = relative_path_to_current_platform(deployment.entrypoint)
    run_logger.debug(f"Importing flow code from '{import_path}'")
    try:
        flow = await run_sync_in_worker_thread(load_flow_from_entrypoint, str(import_path), use_placeholder_flow=use_placeholder_flow)
    except MissingFlowError:
        flow = await run_sync_in_worker_thread(load_function_and_convert_to_flow, str(import_path))
    return flow


def load_placeholder_flow(entrypoint: str, raises: Exception) -> Flow[Any, Any]:
    """
    Load a placeholder flow that is initialized with the same arguments as the
    flow specified in the entrypoint. If called the flow will raise `raises`.
    """
    def _base_placeholder() -> NoReturn:
        raise raises

    def sync_placeholder_flow(*args: Any, **kwargs: Any) -> Any:
        _base_placeholder()

    async def async_placeholder_flow(*args: Any, **kwargs: Any) -> Any:
        _base_placeholder()

    placeholder_flow: Callable[..., Any] = async_placeholder_flow if is_entrypoint_async(entrypoint) else sync_placeholder_flow
    arguments: Dict[str, Any] = load_flow_arguments_from_entrypoint(entrypoint)
    arguments['fn'] = placeholder_flow
    return Flow(**arguments)


def safe_load_flow_from_entrypoint(entrypoint: str) -> Optional[Flow[Any, Any]]:
    """
    Load a flow from an entrypoint and return None if an exception is raised.
    """
    func_def, source_code = _entrypoint_definition_and_source(entrypoint)
    path: Optional[str] = None
    if ':' in entrypoint:
        path = entrypoint.rsplit(':')[0]
    namespace: Dict[str, Any] = safe_load_namespace(source_code, filepath=path)
    if func_def.name in namespace:
        return namespace[func_def.name]
    else:
        return _sanitize_and_load_flow(func_def, namespace)


def _sanitize_and_load_flow(func_def: ast.FunctionDef, namespace: Dict[str, Any]) -> Optional[Flow[Any, Any]]:
    """
    Attempt to load a flow from the function definition after sanitizing the annotations
    and defaults that can't be compiled.
    """
    args = func_def.args.posonlyargs + func_def.args.args + func_def.args.kwonlyargs
    if func_def.args.vararg:
        args.append(func_def.args.vararg)
    if func_def.args.kwarg:
        args.append(func_def.args.kwarg)
    for arg in args:
        if arg.annotation is not None:
            try:
                code = compile(ast.Expression(arg.annotation), filename='<ast>', mode='eval')
                exec(code, namespace)
            except Exception as e:
                logger.debug('Failed to evaluate annotation for argument %s due to the following error. Ignoring annotation.', arg.arg, exc_info=e)
                arg.annotation = None
    new_defaults = []
    for default in func_def.args.defaults:
        try:
            code = compile(ast.Expression(default), '<ast>', 'eval')
            exec(code, namespace)
            new_defaults.append(default)
        except Exception as e:
            logger.debug('Failed to evaluate default value %s due to the following error. Ignoring default.', default, exc_info=e)
            new_defaults.append(ast.Constant(value=None, lineno=default.lineno, col_offset=default.col_offset))
    func_def.args.defaults = new_defaults
    new_kw_defaults = []
    for default in func_def.args.kw_defaults:
        if default is not None:
            try:
                code = compile(ast.Expression(default), '<ast>', 'eval')
                exec(code, namespace)
                new_kw_defaults.append(default)
            except Exception as e:
                logger.debug('Failed to evaluate default value %s due to the following error. Ignoring default.', default, exc_info=e)
                new_kw_defaults.append(ast.Constant(value=None, lineno=default.lineno, col_offset=default.col_offset))
        else:
            new_kw_defaults.append(ast.Constant(value=None, lineno=func_def.lineno, col_offset=func_def.col_offset))
    func_def.args.kw_defaults = new_kw_defaults
    if func_def.returns is not None:
        try:
            code = compile(ast.Expression(func_def.returns), filename='<ast>', mode='eval')
            exec(code, namespace)
        except Exception as e:
            logger.debug('Failed to evaluate return annotation due to the following error. Ignoring annotation.', exc_info=e)
            func_def.returns = None
    try:
        code = compile(ast.Module(body=[func_def], type_ignores=[]), filename='<ast>', mode='exec')
        exec(code, namespace)
    except Exception as e:
        logger.debug('Failed to compile: %s', e)
    else:
        return namespace.get(func_def.name)
    return None


def load_flow_arguments_from_entrypoint(entrypoint: str, arguments: Optional[Set[str]] = None) -> Dict[str, str]:
    """
    Extract flow arguments from an entrypoint string.
    """
    func_def, source_code = _entrypoint_definition_and_source(entrypoint)
    path: Optional[str] = None
    if ':' in entrypoint:
        path = entrypoint.rsplit(':')[0]
    if arguments is None:
        arguments = {'name', 'version', 'retries', 'retry_delay_seconds', 'description', 'timeout_seconds', 'validate_parameters', 'persist_result', 'cache_result_in_memory', 'log_prints'}
    result: Dict[str, str] = {}
    for decorator in func_def.decorator_list:
        if isinstance(decorator, ast.Call) and getattr(decorator.func, 'id', '') == 'flow':
            for keyword in decorator.keywords:
                if keyword.arg not in arguments:
                    continue
                if isinstance(keyword.value, ast.Constant):
                    result[str(keyword.arg)] = str(keyword.value.value)
                    continue
                namespace = safe_load_namespace(source_code, filepath=path)
                literal_arg_value = ast.get_source_segment(source_code, keyword.value)
                cleaned_value = literal_arg_value.replace('\n', '') if literal_arg_value else ''
                try:
                    evaluated_value = eval(cleaned_value, namespace)
                    result[str(keyword.arg)] = str(evaluated_value)
                except Exception as e:
                    logger.info('Failed to parse @flow argument: `%s=%s` due to the following error. Ignoring and falling back to default behavior.', keyword.arg, literal_arg_value, exc_info=e)
                    continue
    if 'name' in arguments and 'name' not in result:
        result['name'] = func_def.name.replace('_', '-')
    return result


def is_entrypoint_async(entrypoint: str) -> bool:
    """
    Determine if the function specified in the entrypoint is asynchronous.
    """
    func_def, _ = _entrypoint_definition_and_source(entrypoint)
    return isinstance(func_def, ast.AsyncFunctionDef)


def _entrypoint_definition_and_source(entrypoint: str) -> Tuple[Union[ast.FunctionDef, ast.AsyncFunctionDef], str]:
    if ':' in entrypoint:
        path, func_name = entrypoint.rsplit(':', maxsplit=1)
        source_code = Path(path).read_text()
    else:
        path, func_name = entrypoint.rsplit('.', maxsplit=1)
        spec = importlib.util.find_spec(path)
        if not spec or not spec.origin:
            raise ValueError(f'Could not find module {path!r}')
        source_code = Path(spec.origin).read_text()
    parsed_code = ast.parse(source_code)
    func_def = next(
        (node for node in ast.walk(parsed_code)
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name),
        None
    )
    if not func_def:
        raise ValueError(f'Could not find flow {func_name!r} in {path!r}')
    return func_def, source_code