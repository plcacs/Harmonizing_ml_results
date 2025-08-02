from __future__ import annotations
import io
import logging
import sys
from builtins import print
from contextlib import contextmanager
from functools import lru_cache
from logging import LogRecord
from typing import TYPE_CHECKING, Any, List, Mapping, MutableMapping, Optional, Union, Iterator, Dict, Callable, TypeVar, cast
from typing_extensions import Self
from prefect.exceptions import MissingContextError
from prefect.logging.filters import ObfuscateApiKeyFilter
from prefect.telemetry.logging import add_telemetry_log_handler
if sys.version_info >= (3, 12):
    LoggingAdapter = logging.LoggerAdapter[logging.Logger]
elif TYPE_CHECKING:
    LoggingAdapter = logging.LoggerAdapter[logging.Logger]
else:
    LoggingAdapter = logging.LoggerAdapter
if TYPE_CHECKING:
    from prefect.client.schemas.objects import FlowRun, TaskRun
    from prefect.context import RunContext
    from prefect.flows import Flow
    from prefect.tasks import Task
    from prefect.workers.base import BaseWorker

class PrefectLogAdapter(LoggingAdapter):
    """
    Adapter that ensures extra kwargs are passed through correctly; without this
    the `extra` fields set on the adapter would overshadow any provided on a
    log-by-log basis.

    See https://bugs.python.org/issue32732 — the Python team has declared that this is
    not a bug in the LoggingAdapter and subclassing is the intended workaround.
    """

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        kwargs['extra'] = {**(self.extra or {}), **(kwargs.get('extra') or {})}
        return (msg, kwargs)

    def getChild(self, suffix: str, extra: Optional[Dict[str, Any]] = None) -> PrefectLogAdapter:
        _extra = extra or {}
        return PrefectLogAdapter(self.logger.getChild(suffix), extra={**(self.extra or {}), **_extra})

@lru_cache()
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a `prefect` logger. These loggers are intended for internal use within the
    `prefect` package.

    See `get_run_logger` for retrieving loggers for use within task or flow runs.
    By default, only run-related loggers are connected to the `APILogHandler`.
    """
    parent_logger = logging.getLogger('prefect')
    if name:
        if not name.startswith(parent_logger.name + '.'):
            logger = parent_logger.getChild(name)
        else:
            logger = logging.getLogger(name)
    else:
        logger = parent_logger
    obfuscate_api_key_filter = ObfuscateApiKeyFilter()
    logger.addFilter(obfuscate_api_key_filter)
    add_telemetry_log_handler(logger=logger)
    return logger

def get_run_logger(context: Optional[Any] = None, **kwargs: Any) -> Union[logging.Logger, PrefectLogAdapter]:
    """
    Get a Prefect logger for the current task run or flow run.

    The logger will be named either `prefect.task_runs` or `prefect.flow_runs`.
    Contextual data about the run will be attached to the log records.

    These loggers are connected to the `APILogHandler` by default to send log records to
    the API.

    Arguments:
        context: A specific context may be provided as an override. By default, the
            context is inferred from global state and this should not be needed.
        **kwargs: Additional keyword arguments will be attached to the log records in
            addition to the run metadata

    Raises:
        MissingContextError: If no context can be found
    """
    from prefect.context import FlowRunContext, TaskRunContext
    task_run_context = TaskRunContext.get()
    flow_run_context = FlowRunContext.get()
    if context:
        if isinstance(context, FlowRunContext):
            flow_run_context = context
        elif isinstance(context, TaskRunContext):
            task_run_context = context
        else:
            raise TypeError(f"Received unexpected type {type(context).__name__!r} for context. Expected one of 'None', 'FlowRunContext', or 'TaskRunContext'.")
    if task_run_context:
        logger = task_run_logger(task_run=task_run_context.task_run, task=task_run_context.task, flow_run=flow_run_context.flow_run if flow_run_context else None, flow=flow_run_context.flow if flow_run_context else None, **kwargs)
    elif flow_run_context:
        logger = flow_run_logger(flow_run=flow_run_context.flow_run, flow=flow_run_context.flow, **kwargs)
    elif get_logger('prefect.flow_run').disabled and get_logger('prefect.task_run').disabled:
        logger = logging.getLogger('null')
    else:
        raise MissingContextError('There is no active flow or task run context.')
    if isinstance(logger, logging.LoggerAdapter):
        assert isinstance(logger.logger, logging.Logger)
        add_telemetry_log_handler(logger.logger)
    else:
        add_telemetry_log_handler(logger)
    return logger

def flow_run_logger(flow_run: Optional['FlowRun'], flow: Optional['Flow'] = None, **kwargs: Any) -> PrefectLogAdapter:
    """
    Create a flow run logger with the run's metadata attached.

    Additional keyword arguments can be provided to attach custom data to the log
    records.

    If the flow run context is available, see `get_run_logger` instead.
    """
    return PrefectLogAdapter(get_logger('prefect.flow_runs'), extra={**{'flow_run_name': flow_run.name if flow_run else '<unknown>', 'flow_run_id': str(flow_run.id) if flow_run else '<unknown>', 'flow_name': flow.name if flow else '<unknown>'}, **kwargs})

def task_run_logger(task_run: 'TaskRun', task: Optional['Task'] = None, flow_run: Optional['FlowRun'] = None, flow: Optional['Flow'] = None, **kwargs: Any) -> PrefectLogAdapter:
    """
    Create a task run logger with the run's metadata attached.

    Additional keyword arguments can be provided to attach custom data to the log
    records.

    If the task run context is available, see `get_run_logger` instead.

    If only the flow run context is available, it will be used for default values
    of `flow_run` and `flow`.
    """
    from prefect.context import FlowRunContext
    if not flow_run or not flow:
        flow_run_context = FlowRunContext.get()
        if flow_run_context:
            flow_run = flow_run or flow_run_context.flow_run
            flow = flow or flow_run_context.flow
    return PrefectLogAdapter(get_logger('prefect.task_runs'), extra={**{'task_run_id': str(task_run.id), 'flow_run_id': str(task_run.flow_run_id), 'task_run_name': task_run.name, 'task_name': task.name if task else '<unknown>', 'flow_run_name': flow_run.name if flow_run else '<unknown>', 'flow_name': flow.name if flow else '<unknown>'}, **kwargs})

def get_worker_logger(worker: 'BaseWorker', name: Optional[str] = None) -> Union[logging.Logger, PrefectLogAdapter]:
    """
    Create a worker logger with the worker's metadata attached.

    If the worker has a backend_id, it will be attached to the log records.
    If the worker does not have a backend_id a basic logger will be returned.
    If the worker does not have a backend_id attribute, a basic logger will be returned.
    """
    worker_log_name = name or f'workers.{worker.__class__.type}.{worker.name.lower()}'
    worker_id = getattr(worker, 'backend_id', None)
    if worker_id:
        return PrefectLogAdapter(get_logger(worker_log_name), extra={'worker_id': str(worker.backend_id)})
    else:
        return get_logger(worker_log_name)

@contextmanager
def disable_logger(name: str) -> Iterator[None]:
    """
    Get a logger by name and disables it within the context manager.
    Upon exiting the context manager, the logger is returned to its
    original state.
    """
    logger = logging.getLogger(name=name)
    base_state = logger.disabled
    try:
        logger.disabled = True
        yield
    finally:
        logger.disabled = base_state

@contextmanager
def disable_run_logger() -> Iterator[None]:
    """
    Gets both `prefect.flow_run` and `prefect.task_run` and disables them
    within the context manager. Upon exiting the context manager, both loggers
    are returned to its original state.
    """
    with disable_logger('prefect.flow_run'), disable_logger('prefect.task_run'):
        yield

def print_as_log(*args: Any, **kwargs: Any) -> None:
    """
    A patch for `print` to send printed messages to the Prefect run logger.

    If no run is active, `print` will behave as if it were not patched.

    If `print` sends data to a file other than `sys.stdout` or `sys.stderr`, it will
    not be forwarded to the Prefect logger either.
    """
    from prefect.context import FlowRunContext, TaskRunContext
    context = TaskRunContext.get() or FlowRunContext.get()
    if not context or not context.log_prints or kwargs.get('file') not in {None, sys.stdout, sys.stderr}:
        return print(*args, **kwargs)
    logger = get_run_logger()
    buffer = io.StringIO()
    kwargs['file'] = buffer
    print(*args, **kwargs)
    logger.info(buffer.getvalue().rstrip())

@contextmanager
def patch_print() -> Iterator[None]:
    """
    Patches the Python builtin `print` method to use `print_as_log`
    """
    import builtins
    original = builtins.print
    try:
        builtins.print = print_as_log  # type: ignore
        yield
    finally:
        builtins.print = original

class LogEavesdropper(logging.Handler):
    """A context manager that collects logs for the duration of the context

    Example:

        