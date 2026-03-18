```pyi
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Literal, TypeVar, overload

_T = TypeVar('_T')

logger: logging.Logger

def collect_request_payload() -> dict[str, Any]: ...
def get_logger_from_status(status: Any) -> tuple[Callable[..., Any], str]: ...
def get_event_logger_from_cfg_value(cfg_value: Any) -> AbstractEventLogger: ...

class AbstractEventLogger(ABC):
    curated_payload_params: set[str]
    curated_form_data_params: set[str]
    action: str
    object_ref: Any
    log_to_statsd: bool
    payload_override: dict[str, Any]
    start: datetime

    def __call__(
        self,
        action: str,
        object_ref: Any = None,
        log_to_statsd: bool = True,
        duration: timedelta | None = None,
        **payload_override: Any,
    ) -> AbstractEventLogger: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    @classmethod
    def curate_payload(cls, payload: dict[str, Any]) -> dict[str, Any]: ...
    @classmethod
    def curate_form_data(cls, payload: dict[str, Any]) -> dict[str, Any]: ...
    @abstractmethod
    def log(
        self,
        user_id: Any,
        action: str,
        dashboard_id: Any,
        duration_ms: Any,
        slice_id: Any,
        referrer: Any,
        curated_payload: dict[str, Any],
        curated_form_data: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def log_with_context(
        self,
        action: str,
        duration: timedelta | None = None,
        object_ref: Any = None,
        log_to_statsd: bool = True,
        database: Any = None,
        **payload_override: Any,
    ) -> None: ...
    @contextmanager
    def log_context(
        self,
        action: str,
        object_ref: Any = None,
        log_to_statsd: bool = True,
        **kwargs: Any,
    ) -> Iterator[Callable[..., None]]: ...
    def _wrapper(
        self,
        f: Callable[..., _T],
        action: str | Callable[..., str] | None = None,
        object_ref: Any = None,
        allow_extra_payload: bool = False,
        **wrapper_kwargs: Any,
    ) -> Callable[..., _T]: ...
    def log_this(self, f: Callable[..., _T]) -> Callable[..., _T]: ...
    def log_this_with_context(self, **kwargs: Any) -> Callable[[Callable[..., _T]], Callable[..., _T]]: ...
    def log_this_with_extra_payload(self, f: Callable[..., _T]) -> Callable[..., _T]: ...

class DBEventLogger(AbstractEventLogger):
    def log(
        self,
        user_id: Any,
        action: str,
        dashboard_id: Any,
        duration_ms: Any,
        slice_id: Any,
        referrer: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

class StdOutEventLogger(AbstractEventLogger):
    def log(
        self,
        user_id: Any,
        action: str,
        dashboard_id: Any,
        duration_ms: Any,
        slice_id: Any,
        referrer: Any,
        curated_payload: dict[str, Any],
        curated_form_data: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
```