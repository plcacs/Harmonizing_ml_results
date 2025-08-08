from typing import List, Tuple
import logging
import re

class AJAXHandler(logging.Handler):
    def __init__(self, url: str, method: str = 'GET', headers: List[Tuple[str, str]] = []):
        ...

    def mapLogRecord(self, record: logging.LogRecord) -> dict:
        ...

    def urlencode(self, msg: str) -> str:
        ...

    def emit(self, record: logging.LogRecord) -> None:
        ...

class BufferingHandler(logging.Handler):
    def __init__(self, capacity: int):
        ...

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        ...

    def emit(self, record: logging.LogRecord) -> None:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...

class MemoryHandler(BufferingHandler):
    def __init__(self, capacity: int, flushLevel: int = logging.ERROR, target: logging.Handler = None, flushOnClose: bool = True):
        ...

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        ...

    def setTarget(self, target: logging.Handler) -> None:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...

class QueueHandler(logging.Handler):
    def __init__(self, queue):
        ...

    def enqueue(self, record: logging.LogRecord) -> None:
        ...

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        ...

    def emit(self, record: logging.LogRecord) -> None:
        ...
