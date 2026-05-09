from __future__ import annotations
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
import flask
from flask import Request
from sqlalchemy.orm import Session
from superset.utils.core import LoggerLevel

logger: logging.Logger = ...

def collect_request_payload() -> dict[str, Any]:
    ...

def get_logger_from_status(status: int) -> tuple[Callable[..., Any], LoggerLevel]:
    ...

class AbstractEventLogger:
    curated_payload_params: ClassVar[set[str]] = ...
    curated_form_data_params: ClassVar[set[str]] = ...

    def __call__(self, action: str, object_ref: Optional[str] = ..., log_to_statsd: bool = ..., duration: Optional[timedelta] = ..., **payload_override: Any) -> AbstractEventLogger:
        ...

    def __enter__(self) -> AbstractEventLogger:
        ...

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        ...

    @classmethod
    def curate_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    @classmethod
    def curate_form_data(cls, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    @abstractmethod
    def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], curated_payload: dict[str, Any], curated_form_data: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ...

    def log_with_context(self, action: str, duration: Optional[timedelta] = ..., object_ref: Optional[str] = ..., log_to_statsd: bool = ..., database: Optional[Any] = ..., **payload_override: Any) -> None:
        ...

    @contextmanager
    def log_context(self, action: str, object_ref: Optional[str] = ..., log_to_statsd: bool = ..., **kwargs: Any) -> Generator[Callable[[**Any], None], None, None]:
        ...

    def _wrapper(self, f: Callable[..., Any], action: Optional[Union[str, Callable[..., str]]] = ..., object_ref: Optional[Union[str, Callable[..., str]]] = ..., allow_extra_payload: bool = ..., **wrapper_kwargs: Any) -> Callable[..., Any]:
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
    def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], *args: Any, **kwargs: Any) -> None:
        ...

class StdOutEventLogger(AbstractEventLogger):
    def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], curated_payload: dict[str, Any], curated_form_data: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ...