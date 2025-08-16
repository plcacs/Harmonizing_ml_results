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
    LoggingAdapter: type[logging.LoggerAdapter[logging.Logger]]
elif TYPE_CHECKING:
    LoggingAdapter: type[logging.LoggerAdapter[logging.Logger]]
else:
    LoggingAdapter: type[logging.LoggerAdapter]

class PrefectLogAdapter(LoggingAdapter):
    def process(self, msg: Any, kwargs: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    def getChild(self, suffix: str, extra: Optional[Mapping[str, Any]] = None) -> PrefectLogAdapter

@lru_cache()
def get_logger(name: Optional[str] = None) -> logging.Logger:
def get_run_logger(context: Optional[Union[FlowRunContext, TaskRunContext]] = None, **kwargs: Any) -> logging.Logger:
def flow_run_logger(flow_run: FlowRun, flow: Optional[Flow] = None, **kwargs: Any) -> PrefectLogAdapter:
def task_run_logger(task_run: TaskRun, task: Optional[Task] = None, flow_run: Optional[FlowRun] = None, flow: Optional[Flow] = None, **kwargs: Any) -> PrefectLogAdapter:
def get_worker_logger(worker: BaseWorker, name: Optional[str] = None) -> Union[PrefectLogAdapter, logging.Logger]:
@contextmanager
def disable_logger(name: str) -> None:
@contextmanager
def disable_run_logger() -> None:
def print_as_log(*args: Any, **kwargs: Any) -> None:
@contextmanager
def patch_print() -> None:
class LogEavesdropper(logging.Handler):
    def __init__(self, eavesdrop_on: str, level: int = logging.NOTSET) -> None:
    def __enter__(self) -> LogEavesdropper:
    def __exit__(self, *_: Any) -> None:
    def emit(self, record: LogRecord) -> None:
    def text(self) -> str:
