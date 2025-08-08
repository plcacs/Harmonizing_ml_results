from __future__ import annotations
import json
import logging
import sys
import time
import traceback
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, TextIO
from rich.console import Console
from rich.highlighter import Highlighter, NullHighlighter
from rich.theme import Theme
from typing_extensions import Self
import prefect.context
from prefect._internal.concurrency.api import create_call, from_sync
from prefect._internal.concurrency.event_loop import get_running_loop
from prefect._internal.concurrency.services import BatchedQueueService
from prefect._internal.concurrency.threads import in_global_loop
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import LogCreate
from prefect.exceptions import MissingContextError
from prefect.logging.highlighters import PrefectConsoleHighlighter
from prefect.settings import PREFECT_API_URL, PREFECT_LOGGING_COLORS, PREFECT_LOGGING_INTERNAL_LEVEL, PREFECT_LOGGING_MARKUP, PREFECT_LOGGING_TO_API_BATCH_INTERVAL, PREFECT_LOGGING_TO_API_BATCH_SIZE, PREFECT_LOGGING_TO_API_MAX_LOG_SIZE, PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW
from prefect.types._datetime import from_timestamp
if sys.version_info >= (3, 12):
    StreamHandler: Type[logging.StreamHandler[TextIO]]
elif TYPE_CHECKING:
    StreamHandler: Type[logging.StreamHandler[TextIO]]
else:
    StreamHandler: Type[logging.StreamHandler]

class APILogWorker(BatchedQueueService[Dict[str, Any]):

    @property
    def _max_batch_size(self) -> int:
        return max(PREFECT_LOGGING_TO_API_BATCH_SIZE.value() - PREFECT_LOGGING_TO_API_MAX_LOG_SIZE.value(), PREFECT_LOGGING_TO_API_MAX_LOG_SIZE.value())

    @property
    def _min_interval(self) -> int:
        return PREFECT_LOGGING_TO_API_BATCH_INTERVAL.value()

    async def _handle_batch(self, items: List[Dict[str, Any]]) -> None:
        try:
            await self._client.create_logs(items)
        except Exception as e:
            if logging.raiseExceptions and sys.stderr:
                sys.stderr.write('--- Error logging to API ---\n')
                if PREFECT_LOGGING_INTERNAL_LEVEL.value() == 'DEBUG':
                    traceback.print_exc(file=sys.stderr)
                else:
                    sys.stderr.write(str(e))

    @asynccontextmanager
    async def _lifespan(self) -> AsyncGenerator[None, None]:
        async with get_client() as self._client:
            yield

    @classmethod
    def instance(cls) -> APILogWorker:
        settings = (PREFECT_LOGGING_TO_API_BATCH_SIZE.value(), PREFECT_API_URL.value(), PREFECT_LOGGING_TO_API_MAX_LOG_SIZE.value())
        return super().instance(*settings)

    def _get_size(self, item: Dict[str, Any]) -> int:
        return item.pop('__payload_size__', None) or len(json.dumps(item).encode())

class APILogHandler(logging.Handler):

    @classmethod
    def flush(cls) -> None:
        ...

    @classmethod
    async def aflush(cls) -> None:
        ...

    def emit(self, record: logging.LogRecord) -> None:
        ...

    def handleError(self, record: logging.LogRecord) -> None:
        ...

    def prepare(self, record: logging.LogRecord) -> Dict[str, Any]:
        ...

    def _get_payload_size(self, log: Dict[str, Any]) -> int:
        ...

class WorkerAPILogHandler(APILogHandler):

    def emit(self, record: logging.LogRecord) -> None:
        ...

    def prepare(self, record: logging.LogRecord) -> Dict[str, Any]:
        ...

class PrefectConsoleHandler(StreamHandler):

    def __init__(self, stream: TextIO = None, highlighter: Type[Highlighter] = PrefectConsoleHighlighter, styles: Any = None, level: int = logging.NOTSET) -> None:
        ...

    def emit(self, record: logging.LogRecord) -> None:
        ...
