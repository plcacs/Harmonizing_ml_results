from __future__ import annotations

import functools
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, Literal, TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from flask import Flask, Request
    from sqlalchemy.orm import Session
    from superset.models.core import Database, Log

logger: logging.Logger

def collect_request_payload() -> dict[str, Any]: ...

def get_logger_from_status(status: int | str) -> tuple[Callable[..., Any], str]: ...

class AbstractEventLogger(ABC):
    curated_payload_params: set[str]
    curated_form_data_params: set[str]
    action: str
    object_ref: str | None
    log_to_statsd: bool
    payload_override: dict[str, Any]
    start: datetime

    def __call__(
        self,
        action: str,
        object_ref: str | None = None,
        log_to_statsd: bool = True,
        duration: timedelta | None = None,
        **payload_override: Any,
    ) -> AbstractEventLogger: ...

    def __enter__(self) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
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
        duration: timedelta | None = None,
        object_ref: str | None = None,
        log_to_statsd: bool = True,
        database: Database | None = None,
        **payload_override: Any,
    ) -> None: ...

    @contextmanager
    def log_context(
        self,
        action: str,
        object_ref: str | None = None,
        log_to_statsd: bool = True,
        **kwargs: Any,
    ) -> Iterator[Callable[..., None]]: ...

    def _wrapper(
        self,
        f: Callable[..., Any],
        action: str | Callable[..., str] | None = None,
        object_ref: str | Callable[..., str] | None | bool = None,
        allow_extra_payload: bool = False,
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