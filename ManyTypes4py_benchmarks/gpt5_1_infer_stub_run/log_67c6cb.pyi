from abc import ABC, abstractmethod
import logging
from datetime import timedelta
from types import TracebackType
from typing import Any, Callable, ContextManager, Optional, Tuple

from superset.utils.core import LoggerLevel

logger: logging.Logger

def collect_request_payload() -> dict[str, Any]: ...
def get_logger_from_status(status: int | str) -> tuple[Callable[..., None], LoggerLevel]: ...

class AbstractEventLogger(ABC):
    curated_payload_params: set[str]
    curated_form_data_params: set[str]

    def __call__(
        self,
        action: str,
        object_ref: str | None = ...,
        log_to_statsd: bool = ...,
        duration: timedelta | None = ...,
        **payload_override: Any,
    ) -> AbstractEventLogger: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    @classmethod
    def curate_payload(cls, payload: dict[str, Any]) -> dict[str, Any]: ...
    @classmethod
    def curate_form_data(cls, payload: dict[str, Any]) -> dict[str, Any]: ...
    @abstractmethod
    def log(
        self,
        user_id: int | None,
        action: str,
        dashboard_id: int | None,
        duration_ms: int | None,
        slice_id: int | None,
        referrer: str | None,
        curated_payload: dict[str, Any],
        curated_form_data: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def log_with_context(
        self,
        action: str,
        duration: timedelta | None = ...,
        object_ref: str | None = ...,
        log_to_statsd: bool = ...,
        database: object | None = ...,
        **payload_override: Any,
    ) -> None: ...
    def log_context(
        self,
        action: str,
        object_ref: str | None = ...,
        log_to_statsd: bool = ...,
        **kwargs: Any,
    ) -> ContextManager[Callable[..., None]]: ...
    def _wrapper(
        self,
        f: Callable[..., Any],
        action: Callable[..., str] | str | None = ...,
        object_ref: Callable[..., str | None] | str | bool | None = ...,
        allow_extra_payload: bool = ...,
        **wrapper_kwargs: Any,
    ) -> Callable[..., Any]: ...
    def log_this(self, f: Callable[..., Any]) -> Callable[..., Any]: ...
    def log_this_with_context(self, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    def log_this_with_extra_payload(self, f: Callable[..., Any]) -> Callable[..., Any]: ...

def get_event_logger_from_cfg_value(cfg_value: Any) -> AbstractEventLogger: ...

class DBEventLogger(AbstractEventLogger):
    def log(
        self,
        user_id: int | None,
        action: str,
        dashboard_id: int | None,
        duration_ms: int | None,
        slice_id: int | None,
        referrer: str | None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

class StdOutEventLogger(AbstractEventLogger):
    def log(
        self,
        user_id: int | None,
        action: str,
        dashboard_id: int | None,
        duration_ms: int | None,
        slice_id: int | None,
        referrer: str | None,
        curated_payload: dict[str, Any],
        curated_form_data: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...