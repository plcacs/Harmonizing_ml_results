import asyncio
import logging
import sys
import typing
import traceback
from functools import partial
from typing import Any, Iterable, Optional, Type, Dict
from faust.exceptions import ImproperlyConfigured
from faust.types import AppT

try:
    import raven
except ImportError:
    raven = None
try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None
    _sdk_aiohttp = None
else:
    import sentry_sdk.integrations.aiohttp as _sdk_aiohttp
try:
    import raven_aiohttp
except ImportError:
    raven_aiohttp = None
if typing.TYPE_CHECKING:
    from raven.handlers.logging import SentryHandler as _SentryHandler
else:

    class _SentryHandler:
        pass

__all__ = ['handler_from_dsn', 'setup']
DEFAULT_LEVEL: int = logging.WARNING

def _build_sentry_handler() -> Type[logging.Handler]:
    from raven.handlers import logging as _logging

    class FaustSentryHandler(_logging.SentryHandler):

        def can_record(self, record: logging.LogRecord) -> bool:
            return super().can_record(record) and (not self._is_expected_cancel(record))

        def _is_expected_cancel(self, record: logging.LogRecord) -> bool:
            if record.exc_info and record.exc_info[0] is not None:
                return bool(
                    issubclass(record.exc_info[0], asyncio.CancelledError)
                    and getattr(record.exc_info[1], 'is_expected', True)
                )
            return False

        def emit(self, record: logging.LogRecord) -> None:
            try:
                self.format(record)
                if self.can_record(record):
                    self._emit(record)
                else:
                    self.carp(record.message)
            except Exception:
                if self.client.raise_send_errors:
                    raise
                self.carp('Top level Sentry exception caught - failed creating log record')
                self.carp(record.msg)
                self.carp(traceback.format_exc())

        def carp(self, obj: Any) -> None:
            print(_logging.to_string(obj), file=sys.__stderr__)

    return FaustSentryHandler

def handler_from_dsn(
    dsn: Optional[str] = None,
    workers: int = 5,
    include_paths: Optional[Iterable[str]] = None,
    loglevel: Optional[int] = None,
    qsize: int = 1000,
    **kwargs: Any
) -> Optional[logging.Handler]:
    if raven is None:
        raise ImproperlyConfigured('faust.contrib.sentry requires the `raven` library.')
    if raven_aiohttp is None:
        raise ImproperlyConfigured('faust.contrib.sentry requires the `raven_aiohttp` library.')
    level: int = loglevel if loglevel is not None else DEFAULT_LEVEL
    if dsn:
        client = raven.Client(
            dsn=dsn,
            include_paths=include_paths,
            transport=partial(raven_aiohttp.QueuedAioHttpTransport, workers=workers, qsize=qsize),
            disable_existing_loggers=False,
            **kwargs
        )
        handler = _build_sentry_handler()(client)
        handler.setLevel(level)
        return handler
    return None

def setup(
    app: AppT,
    *,
    dsn: Optional[str] = None,
    workers: int = 4,
    max_queue_size: int = 1000,
    loglevel: Optional[int] = None
) -> None:
    sentry_handler: Optional[logging.Handler] = handler_from_dsn(
        dsn=dsn,
        workers=workers,
        qsize=max_queue_size,
        loglevel=loglevel
    )
    if sentry_handler is not None:
        if sentry_sdk is None or _sdk_aiohttp is None:
            raise ImproperlyConfigured('faust.contrib.sentry requires the `sentry_sdk` library.')
        sentry_sdk.init(dsn=dsn, integrations=[_sdk_aiohttp.AioHttpIntegration()])
        app.conf.loghandlers.append(sentry_handler)