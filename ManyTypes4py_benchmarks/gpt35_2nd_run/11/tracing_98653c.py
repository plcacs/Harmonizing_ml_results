from opentracing import Tracer
from contextvars import ContextVar
from typing import Any, Callable, Optional, Tuple

_current_span: ContextVar[Optional[Any]]

def current_span() -> Optional[Any]:
    ...

def set_current_span(span: Any) -> None:
    ...

def noop_span() -> Any:
    ...

def finish_span(span: Any, *, error: Optional[BaseException] = None) -> None:
    ...

def operation_name_from_fun(fun: Callable) -> str:
    ...

def traced_from_parent_span(parent_span: Optional[Any] = None, callback: Optional[Callable] = None, **extra_context: Any) -> Callable:
    ...

def _restore_span(span: Any, expected_current_span: Any) -> None:
    ...

def call_with_trace(span: Any, fun: Callable, callback: Optional[Tuple[Callable, Tuple[Any, ...]]], *args: Any, **kwargs: Any) -> Any:
    ...
