from __future__ import annotations
import io
import logging
import sys
from builtins import print
from contextlib import contextmanager
from functools import lru_cache
from logging import LogRecord
from typing import TYPE_CHECKING, Any, List, Mapping, MutableMapping, Optional, Union
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
    def process(self, msg, kwargs) -> tuple[str, dict]:
        kwargs['extra'] = {**(self.extra or {}), **(kwargs.get('extra') or {})}
        return (msg, kwargs)

    def getChild(self, suffix, extra=None) -> PrefectLogAdapter:
        _extra = extra or {}
        return PrefectLogAdapter(self.logger.getChild(suffix), extra={**(self.extra or {}), **_extra})

@lru_cache()
def get_logger(name=None) -> logging.Logger:
    ...

def get_run_logger(context=None, **kwargs) -> Union[PrefectLogAdapter, logging.Logger]:
    ...

def flow_run_logger(flow_run, flow=None, **kwargs) -> PrefectLogAdapter:
    ...

def task_run_logger(task_run, task=None, flow_run=None, flow=None, **kwargs) -> PrefectLogAdapter:
    ...

def get_worker_logger(worker, name=None) -> Union[PrefectLogAdapter, logging.Logger]:
    ...

@contextmanager
def disable_logger(name) -> None:
    ...

@contextmanager
def disable_run_logger() -> None:
    ...

def print_as_log(*args, **kwargs) -> None:
    ...

@contextmanager
def patch_print() -> None:
    ...

class LogEavesdropper(logging.Handler):
    def __init__(self, eavesdrop_on, level=logging.NOTSET) -> None:
        ...

    def __enter__(self) -> LogEavesdropper:
        ...

    def __exit__(self, *_):
        ...

    def emit(self, record) -> None:
        ...

    def text(self) -> str:
        ...
