from __future__ import annotations
import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from typing import Any, Optional, Union, overload, cast, Type, TypeVar

# Type variable for the decorator's wrapped function
F = TypeVar("F", bound=Callable[..., Any])

logger: logging.Logger ...

def collect_request_payload() -> dict[str, Any]: ...

def get_logger_from_status(status: Union[int, str]) -> tuple[Callable[..., Any], Any]: ...

class AbstractEventLogger(ABC):
    curated_payload_params: set[str]
    curated_form_data_params: set[str]
    action: str
    object_ref: Optional[str]
    log_to_statsd: bool
    payload_override: dict[str, Any]
    start: datetime

    def __call__(
        self, 
        action: str, 
        object_ref: Optional[str] = ..., 
        log_to_statsd: bool = ..., 
        duration: Optional[timedelta] = ..., 
        **payload_override: Any
    ) -> AbstractEventLogger: ...

    def __enter__(self) -> AbstractEventLogger: ...

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any) -> None: ...

    @classmethod
    def curate_payload(cls, payload: dict[str, Any]) -> dict[str, Any]: ...

    @classmethod
    def curate_form_data(cls, payload: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def log(
        self, 
        user_id: Optional[int], 
        action: str, 
        dashboard_id: Optional[int], 
        duration_ms: Optional[int], 
        slice_id: Optional[int], 
        referrer: Optional[str], 
        curated_payload: dict[str, Any], 
        curated_form_data: dict[str, Any], 
        *args: Any, 
        **kwargs: Any
    ) -> None: ...

    def log_with_context(
        self, 
        action: str, 
        duration: Optional[timedelta] = ..., 
        object_ref: Optional[str] = ..., 
        log_to_statsd: bool = ..., 
        database: Any = ..., 
        **payload_override: Any
    ) -> None: ...

    @contextmanager
    def log_context(
        self, 
        action: str, 
        object_ref: Optional[str] = ..., 
        log_to_statsd: bool = ..., 
        **kwargs: Any
    ) -> Iterator[Callable[..., None]]: ...

    def _wrapper(
        self, 
        f: F, 
        action: Optional[Union[str, Callable[..., str]]] = ..., 
        object_ref: Optional[Union[str, Callable[..., str]]] = ..., 
        allow_extra_payload: bool = ..., 
        **wrapper_kwargs: Any
    ) -> F: ...

    def log_this(self, f: F) -> F: ...

    def log_this_with_context(self, **kwargs: Any) -> Callable[[F], F]: ...

    def log_this_with_extra_payload(self, f: F) -> F: ...

def get_event_logger_from_cfg_value(cfg_value: Union[Type[AbstractEventLogger], AbstractEventLogger]) -> AbstractEventLogger: ...

class DBEventLogger(AbstractEventLogger):
    """Event logger that commits logs to Superset DB"""
    def log(
        self, 
        user_id: Optional[int], 
        action: str, 
        dashboard_id: Optional[int], 
        duration_ms: Optional[int], 
        slice_id: Optional[int], 
        referrer: Optional[str], 
        *args: Any, 
        **kwargs: Any
    ) -> None: ...

class StdOutEventLogger(AbstractEventLogger):
    """Event logger that prints to stdout for debugging purposes"""
    def log(
        self, 
        user_id: Optional[int], 
        action: str, 
        dashboard_id: Optional[int], 
        duration_ms: Optional[int], 
        slice_id: Optional[int], 
        referrer: Optional[str], 
        curated_payload: dict[str, Any], 
        curated_form_data: dict[str, Any], 
        *args: Any, 
        **kwargs: Any
    ) -> None: ...