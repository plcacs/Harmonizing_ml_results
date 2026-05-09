from __future__ import annotations
import datetime
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)
from flask import g, request
from flask_appbuilder.const import API_URI_RIS_KEY
from sqlalchemy.exc import SQLAlchemyError
from superset.utils.core import LoggerLevel
from superset.models.core import Log
from superset.utils.log import AbstractEventLogger

logger: logging.Logger = ...

def collect_request_payload() -> Dict[str, Any]:
    ...

def get_logger_from_status(status: int) -> Tuple[Callable[..., Any], LoggerLevel]:
    ...

class AbstractEventLogger:
    curated_payload_params: set[str] = ...
    curated_form_data_params: set[str] = ...

    def __call__(
        self,
        action: str,
        object_ref: Optional[str] = None,
        log_to_statsd: bool = True,
        duration: Optional[datetime.timedelta] = None,
        **payload_override: Any
    ) -> "AbstractEventLogger":
        ...

    def __enter__(self) -> "AbstractEventLogger":
        ...

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Iterator[BaseException]],
    ) -> None:
        ...

    @classmethod
    def curate_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @classmethod
    def curate_form_data(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def log(
        self,
        user_id: int,
        action: str,
        dashboard_id: int,
        duration_ms: int,
        slice_id: int,
        referrer: str,
        curated_payload: Dict[str, Any],
        curated_form_data: Dict[str, Any],
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...

    def log_with_context(
        self,
        action: str,
        duration: Optional[datetime.timedelta] = None,
        object_ref: Optional[str] = None,
        log_to_statsd: bool = True,
        database: Optional[Any] = None,
        **payload_override: Any
    ) -> None:
        ...

    @contextmanager
    def log_context(
        self,
        action: str,
        object_ref: Optional[str] = None,
        log_to_statsd: bool = True,
        **kwargs: Any
    ) -> Generator[Callable[..., None], None, None]:
        ...

    def _wrapper(
        self,
        f: Callable[..., Any],
        action: Optional[Union[str, Callable[..., str]]] = None,
        object_ref: Optional[Union[str, Callable[..., str]]] = None,
        allow_extra_payload: bool = False,
        **wrapper_kwargs: Any
    ) -> Callable[..., Any]:
        ...

    def log_this(self, f: Callable[..., Any]) -> Callable[..., Any]:
        ...

    def log_this_with_context(self, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        ...

    def log_this_with_extra_payload(self, f: Callable[..., Any]) -> Callable[..., Any]:
        ...

def get_event_logger_from_cfg_value(cfg_value: Any) -> AbstractEventLogger:
    ...

class DBEventLogger(AbstractEventLogger):
    def log(
        self,
        user_id: int,
        action: str,
        dashboard_id: int,
        duration_ms: int,
        slice_id: int,
        referrer: str,
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...

class StdOutEventLogger(AbstractEventLogger):
    def log(
        self,
        user_id: int,
        action: str,
        dashboard_id: int,
        duration_ms: int,
        slice_id: int,
        referrer: str,
        curated_payload: Dict[str, Any],
        curated_form_data: Dict[str, Any],
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...