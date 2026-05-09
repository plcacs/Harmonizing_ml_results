from __future__ import annotations
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Literal,
    ContextManager,
    Generator,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
    Literal,
)

from flask import request
from sqlalchemy.exc import SQLAlchemyError
from superset.utils.core import LoggerLevel

logger: Any = ...

def collect_request_payload() -> Dict[str, Any]:
    ...

def get_logger_from_status(status: int) -> Tuple[Callable[..., None], LoggerLevel]:
    ...

class AbstractEventLogger:
    curated_payload_params: Set[str]
    curated_form_data_params: Set[str]

    def __call__(self, action: str, object_ref: Optional[str] = None, log_to_statsd: bool = True, duration: Optional[timedelta] = None, **payload_override: Any) -> ContextManager:
        ...

    def __enter__(self) -> None:
        ...

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        ...

    @classmethod
    def curate_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @classmethod
    def curate_form_data(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], curated_payload: Dict[str, Any], curated_form_data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ...

    def log_with_context(self, action: str, duration: Optional[timedelta] = None, object_ref: Optional[str] = None, log_to_statsd: bool = True, database: Optional[Any] = None, **payload_override: Any) -> None:
        ...

    @contextmanager
    def log_context(self, action: str, object_ref: Optional[str] = None, log_to_statsd: bool = True, **kwargs: Any) -> Generator[Callable[..., None], None, None]:
        ...

    def _wrapper(self, f: Callable, action: Optional[Union[str, Callable]] = None, object_ref: Optional[Union[str, Callable]] = None, allow_extra_payload: bool = False, **wrapper_kwargs: Any) -> Callable:
        ...

    def log_this(self, f: Callable) -> Callable:
        ...

    def log_this_with_context(self, **kwargs: Any) -> Callable[[Callable], Callable]:
        ...

    def log_this_with_extra_payload(self, f: Callable) -> Callable:
        ...

def get_event_logger_from_cfg_value(cfg_value: Any) -> AbstractEventLogger:
    ...

class DBEventLogger(AbstractEventLogger):
    def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], *args: Any, **kwargs: Any) -> None:
        ...

class StdOutEventLogger(AbstractEventLogger):
    def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], curated_payload: Dict[str, Any], curated_form_data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ...