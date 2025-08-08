from __future__ import annotations
import functools
import inspect
import logging
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, cast, Literal, TYPE_CHECKING
from flask import g, request
from flask_appbuilder.const import API_URI_RIS_KEY
from sqlalchemy.exc import SQLAlchemyError
from superset.extensions import stats_logger_manager
from superset.utils import json
from superset.utils.core import get_user_id, LoggerLevel, to_int
if TYPE_CHECKING:
    pass
logger: logging.Logger = logging.getLogger(__name__)

def collect_request_payload() -> dict:
    ...

def get_logger_from_status(status: int) -> tuple[Callable, LoggerLevel]:
    ...

class AbstractEventLogger(ABC):
    curated_payload_params: set[str] = {'force', 'standalone', 'runAsync', 'json', 'csv', 'queryLimit', 'select_as_cta'}
    curated_form_data_params: set[str] = {'dashboardId', 'sliceId', 'viz_type', 'force', 'compare_lag', 'forecastPeriods', 'granularity_sqla', 'legendType', 'legendOrientation', 'show_legend', 'time_grain_sqla'}

    def __call__(self, action: str, object_ref: Any = None, log_to_statsd: bool = True, duration: timedelta = None, **payload_override: Any) -> AbstractEventLogger:
        ...

    def __enter__(self) -> None:
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    @classmethod
    def curate_payload(cls, payload: dict) -> dict:
        ...

    @classmethod
    def curate_form_data(cls, payload: dict) -> dict:
        ...

    @abstractmethod
    def log(self, user_id: int, action: str, dashboard_id: int, duration_ms: int, slice_id: int, referrer: str, curated_payload: dict, curated_form_data: dict, *args: Any, **kwargs: Any) -> None:
        ...

    def log_with_context(self, action: str, duration: timedelta = None, object_ref: Any = None, log_to_statsd: bool = True, database: Any = None, **payload_override: Any) -> None:
        ...

    @contextmanager
    def log_context(self, action: str, object_ref: Any = None, log_to_statsd: bool = True, **kwargs: Any) -> None:
        ...

    def _wrapper(self, f: Callable, action: str = None, object_ref: Any = None, allow_extra_payload: bool = False, **wrapper_kwargs: Any) -> Callable:
        ...

    def log_this(self, f: Callable) -> Callable:
        ...

    def log_this_with_context(self, **kwargs: Any) -> Callable:
        ...

    def log_this_with_extra_payload(self, f: Callable) -> Callable:
        ...

def get_event_logger_from_cfg_value(cfg_value: Any) -> AbstractEventLogger:
    ...

class DBEventLogger(AbstractEventLogger):
    def log(self, user_id: int, action: str, dashboard_id: int, duration_ms: int, slice_id: int, referrer: str, *args: Any, **kwargs: Any) -> None:
        ...

class StdOutEventLogger(AbstractEventLogger):
    def log(self, user_id: int, action: str, dashboard_id: int, duration_ms: int, slice_id: int, referrer: str, curated_payload: dict, curated_form_data: dict, *args: Any, **kwargs: Any) -> None:
        ...
