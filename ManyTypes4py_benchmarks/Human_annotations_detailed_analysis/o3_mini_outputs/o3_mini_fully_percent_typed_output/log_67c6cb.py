from __future__ import annotations

import functools
import inspect
import logging
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, cast, Literal, Optional

from flask import g, request
from flask_appbuilder.const import API_URI_RIS_KEY
from sqlalchemy.exc import SQLAlchemyError

from superset.extensions import stats_logger_manager
from superset.utils import json
from superset.utils.core import get_user_id, LoggerLevel, to_int

logger = logging.getLogger(__name__)


def collect_request_payload() -> dict[str, Any]:
    """Collect log payload identifiable from request context"""
    if not request:
        return {}

    payload: dict[str, Any] = {
        "path": request.path,
        **request.form.to_dict(),
        **request.args.to_dict(),
    }

    if request.is_json:
        json_payload: dict[str, Any] = request.get_json(cache=True, silent=True) or {}
        payload.update(json_payload)

    url_rule: str = str(request.url_rule)
    if url_rule != request.path:
        payload["url_rule"] = url_rule

    if "rison" in payload and API_URI_RIS_KEY in payload:
        del payload[API_URI_RIS_KEY]
    if "rison" in payload and not payload["rison"]:
        del payload["rison"]

    return payload


def get_logger_from_status(status: int) -> tuple[Callable[..., None], str]:
    """
    Return logger method by status of exception.
    Maps logger level to status code level
    """
    log_map: dict[str, str] = {
        "2": LoggerLevel.INFO,
        "3": LoggerLevel.INFO,
        "4": LoggerLevel.WARNING,
        "5": LoggerLevel.EXCEPTION,
    }
    log_level: str = log_map[str(status)[0]]
    return (getattr(logger, log_level), log_level)


class AbstractEventLogger(ABC):
    curated_payload_params: set[str] = {
        "force",
        "standalone",
        "runAsync",
        "json",
        "csv",
        "queryLimit",
        "select_as_cta",
    }
    curated_form_data_params: set[str] = {
        "dashboardId",
        "sliceId",
        "viz_type",
        "force",
        "compare_lag",
        "forecastPeriods",
        "granularity_sqla",
        "legendType",
        "legendOrientation",
        "show_legend",
        "time_grain_sqla",
    }

    def __call__(
        self,
        action: str,
        object_ref: Optional[str] = None,
        log_to_statsd: bool = True,
        duration: Optional[timedelta] = None,
        **payload_override: Any,
    ) -> AbstractEventLogger:
        self.action = action
        self.object_ref = object_ref
        self.log_to_statsd = log_to_statsd
        self.payload_override = payload_override
        return self

    def __enter__(self) -> None:
        self.start: datetime = datetime.now()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.log_with_context(
            action=self.action,
            object_ref=self.object_ref,
            log_to_statsd=self.log_to_statsd,
            duration=datetime.now() - self.start,
            **self.payload_override,
        )

    @classmethod
    def curate_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Curate payload to only include relevant keys/safe keys"""
        return {k: v for k, v in payload.items() if k in cls.curated_payload_params}

    @classmethod
    def curate_form_data(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Curate form_data to only include relevant keys/safe keys"""
        return {k: v for k, v in payload.items() if k in cls.curated_form_data_params}

    @abstractmethod
    def log(
        self,
        user_id: Optional[int],
        action: str,
        dashboard_id: Optional[int],
        duration_ms: Optional[int],
        slice_id: Optional[int],
        referrer: Optional[str],
        curated_payload: Optional[dict[str, Any]],
        curated_form_data: Optional[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def log_with_context(
        self,
        action: str,
        duration: Optional[timedelta] = None,
        object_ref: Optional[str] = None,
        log_to_statsd: bool = True,
        database: Optional[Any] = None,
        **payload_override: Any,
    ) -> None:
        from superset import db
        from superset.views.core import get_form_data

        referrer: Optional[str] = request.referrer[:1000] if request and request.referrer else None
        duration_ms: Optional[int] = int(duration.total_seconds() * 1000) if duration else None

        user_id: Optional[int] = get_user_id()

        if user_id is None:
            try:
                db.session.add(g.user)
                user_id = get_user_id()
            except Exception as ex:
                logging.warning(ex)
                user_id = None

        payload: dict[str, Any] = collect_request_payload()
        if object_ref:
            payload["object_ref"] = object_ref
        if payload_override:
            payload.update(payload_override)

        dashboard_id: Optional[int] = to_int(payload.get("dashboard_id"))

        database_params: dict[str, Any] = {"database_id": payload.get("database_id")}
        if database and type(database).__name__ == "Database":
            database_params = {
                "database_id": database.id,
                "engine": database.backend,
                "database_driver": database.driver,
            }

        form_data: dict[str, Any] = {}
        if "form_data" in payload:
            form_data, _ = get_form_data()
            payload["form_data"] = form_data
            slice_id_raw: Any = form_data.get("slice_id")
        else:
            slice_id_raw = payload.get("slice_id")

        slice_id: Optional[int] = to_int(slice_id_raw)

        if log_to_statsd:
            stats_logger_manager.instance.incr(action)

        try:
            explode_by: Any = payload.get("explode")
            records: list[Any] = json.loads(payload.get(explode_by))  # type: ignore
        except Exception:
            records = [payload]

        self.log(
            user_id,
            action,
            dashboard_id,
            slice_id=slice_id,
            duration_ms=duration_ms,
            referrer=referrer,
            curated_payload=self.curate_payload(payload),
            curated_form_data=self.curate_form_data(form_data),
            records=records,
            **database_params,
        )

    @contextmanager
    def log_context(
        self,
        action: str,
        object_ref: Optional[str] = None,
        log_to_statsd: bool = True,
        **kwargs: Any,
    ) -> Iterator[Callable[..., None]]:
        payload_override: dict[str, Any] = kwargs.copy()
        start: datetime = datetime.now()
        yield lambda **kwargs: payload_override.update(kwargs)
        duration: timedelta = datetime.now() - start
        action_str: str = payload_override.pop("action", action)
        self.log_with_context(
            action_str, duration, object_ref, log_to_statsd, **payload_override
        )

    def _wrapper(
        self,
        f: Callable[..., Any],
        action: Optional[str | Callable[..., str]] = None,
        object_ref: Optional[str | Callable[..., str] | Literal[False]] = None,
        allow_extra_payload: Optional[bool] = False,
        **wrapper_kwargs: Any,
    ) -> Callable[..., Any]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            action_str: str = (
                action(*args, **kwargs) if callable(action) else action
            ) or f.__name__
            object_ref_str: Optional[str] = (
                object_ref(*args, **kwargs) if callable(object_ref) else object_ref
            ) or (f.__qualname__ if object_ref is not False else None)
            with self.log_context(action=action_str, object_ref=object_ref_str, **wrapper_kwargs) as log:
                log(**kwargs)
                if allow_extra_payload:
                    value: Any = f(*args, add_extra_log_payload=log, **kwargs)
                else:
                    value = f(*args, **kwargs)
            return value
        return wrapper

    def log_this(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return self._wrapper(f)

    def log_this_with_context(self, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def func(f: Callable[..., Any]) -> Callable[..., Any]:
            return self._wrapper(f, **kwargs)
        return func

    def log_this_with_extra_payload(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return self._wrapper(f, allow_extra_payload=True)


def get_event_logger_from_cfg_value(cfg_value: Any) -> AbstractEventLogger:
    result: Any = cfg_value
    if inspect.isclass(cfg_value):
        logging.warning(
            textwrap.dedent(
                """
                In superset private config, EVENT_LOGGER has been assigned a class
                object. In order to accomodate pre-configured instances without a
                default constructor, assignment of a class is deprecated and may no
                longer work at some point in the future. Please assign an object
                instance of a type that implements
                superset.utils.log.AbstractEventLogger.
                """
            )
        )
        event_logger_type: type[Any] = cast(type[Any], cfg_value)
        result = event_logger_type()

    if not isinstance(result, AbstractEventLogger):
        raise TypeError(
            "EVENT_LOGGER must be configured with a concrete instance"
            "of superset.utils.log.AbstractEventLogger."
        )

    logging.debug("Configured event logger of type %s", type(result))
    return cast(AbstractEventLogger, result)


class DBEventLogger(AbstractEventLogger):
    def log(
        self,
        user_id: Optional[int],
        action: str,
        dashboard_id: Optional[int],
        duration_ms: Optional[int],
        slice_id: Optional[int],
        referrer: Optional[str],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from superset import db
        from superset.models.core import Log

        records: list[Any] = kwargs.get("records", [])
        logs: list[Log] = []
        for record in records:
            json_string: Optional[str]
            try:
                json_string = json.dumps(record)
            except Exception:
                json_string = None
            log_obj = Log(
                action=action,
                json=json_string,
                dashboard_id=dashboard_id,
                slice_id=slice_id,
                duration_ms=duration_ms,
                referrer=referrer,
                user_id=user_id,
            )
            logs.append(log_obj)
        try:
            db.session.bulk_save_objects(logs)
            db.session.commit()
        except SQLAlchemyError as ex:
            logging.error("DBEventLogger failed to log event(s)")
            logging.exception(ex)


class StdOutEventLogger(AbstractEventLogger):
    def log(
        self,
        user_id: Optional[int],
        action: str,
        dashboard_id: Optional[int],
        duration_ms: Optional[int],
        slice_id: Optional[int],
        referrer: Optional[str],
        curated_payload: Optional[dict[str, Any]],
        curated_form_data: Optional[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        data: dict[str, Any] = dict(
            user_id=user_id,
            action=action,
            dashboard_id=dashboard_id,
            duration_ms=duration_ms,
            slice_id=slice_id,
            referrer=referrer,
            curated_payload=curated_payload,
            curated_form_data=curated_form_data,
            **kwargs,
        )
        print("StdOutEventLogger: ", data)