import time
from functools import wraps
from alerta.app import db
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

F = TypeVar('F', bound=Callable[..., Any])

class Gauge:
    __init__: Callable[..., None]
    serialize: Callable[..., Union[str, Dict[str, Any]]]
    __repr__: Callable[..., str]
    from_document: classmethod[Callable[..., 'Gauge']]
    from_record: classmethod[Callable[..., 'Gauge']]
    from_db: classmethod[Callable[..., Optional['Gauge']]]
    set: Callable[..., Optional['Gauge']]
    find_all: classmethod[Callable[..., List[Optional['Gauge']]]]

class Counter:
    __init__: Callable[..., None]
    serialize: Callable[..., Union[str, Dict[str, Any]]]
    __repr__: Callable[..., str]
    from_document: classmethod[Callable[..., 'Counter']]
    from_record: classmethod[Callable[..., 'Counter']]
    from_db: classmethod[Callable[..., Optional['Counter']]]
    inc: Callable[..., None]
    find_all: classmethod[Callable[..., List[Optional['Counter']]]]

class Timer:
    __init__: Callable[..., None]
    serialize: Callable[..., Union[str, Dict[str, Any]]]
    __repr__: Callable[..., str]
    from_document: classmethod[Callable[..., 'Timer']]
    from_record: classmethod[Callable[..., 'Timer']]
    from_db: classmethod[Callable[..., Optional['Timer']]]
    _time_in_millis: Callable[..., int]
    start_timer: Callable[..., int]
    stop_timer: Callable[..., None]
    find_all: classmethod[Callable[..., List[Optional['Timer']]]]

def timer(metric: Timer) -> Callable[[F], F]: ...