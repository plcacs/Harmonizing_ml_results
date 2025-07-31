#!/usr/bin/env python3
"""OpenTracing utilities."""
import asyncio
import sys
import typing
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import opentracing
from opentracing import Span, Tracer
from mode import shortlabel

__all__ = [
    'current_span',
    'noop_span',
    'set_current_span',
    'finish_span',
    'operation_name_from_fun',
    'traced_from_parent_span',
    'call_with_trace',
]

# Define a ContextVar holding an optional Span.
_current_span: ContextVar[Optional[Span]] = ContextVar('current_span')


def current_span() -> Optional[Span]:
    """Get the current span for this context (if any)."""
    return _current_span.get(None)


def set_current_span(span: Span) -> None:
    """Set the current span for the current context."""
    _current_span.set(span)


def noop_span() -> Span:
    """Return a span that does nothing when traced."""
    return opentracing.Tracer()._noop_span  # type: ignore


def finish_span(span: Optional[Span], *, error: Optional[BaseException] = None) -> None:
    """Finish span, and optionally set error tag."""
    if span is not None:
        if error:
            span.__exit__(type(error), error, error.__traceback__)
        else:
            span.finish()


def operation_name_from_fun(fun: Callable[..., Any]) -> str:
    """Generate opentracing name from function."""
    obj = getattr(fun, '__self__', None)
    if obj is not None:
        objlabel: str = shortlabel(obj)
        funlabel: str = shortlabel(fun)
        if funlabel.startswith(objlabel):
            funlabel = funlabel[len(objlabel):]
        return f'{objlabel}-{funlabel}'
    else:
        return f'{shortlabel(fun)}'


F = TypeVar('F', bound=Callable[..., Any])


def traced_from_parent_span(
    parent_span: Optional[Span] = None,
    callback: Optional[Callable[[Span], None]] = None,
    **extra_context: Any
) -> Callable[[F], F]:
    """Decorate function to be traced from parent span."""

    def _wrapper(fun: F, **more_context: Any) -> F:
        operation_name: str = operation_name_from_fun(fun)

        @wraps(fun)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            parent: Optional[Span] = parent_span
            if parent is None:
                parent = current_span()
            if parent is not None:
                # Merge extra context from decorator and the call site.
                tags: Dict[str, Any] = {**extra_context, **more_context}
                child: Span = parent.tracer.start_span(
                    operation_name=operation_name, child_of=parent, tags=tags
                )
                if callback is not None:
                    callback(child)
                on_exit: Optional[Tuple[Callable[..., Any], Tuple[Any, ...]]] = (
                    _restore_span, (parent, child)
                )
                set_current_span(child)
                return call_with_trace(child, fun, on_exit, *args, **kwargs)
            return fun(*args, **kwargs)

        return typing.cast(F, _inner)

    return _wrapper


def _restore_span(span: Span, expected_current_span: Span) -> None:
    current: Optional[Span] = current_span()
    assert current is expected_current_span
    set_current_span(span)


def call_with_trace(
    span: Span,
    fun: Callable[..., Any],
    callback: Optional[Tuple[Callable[..., Any], Tuple[Any, ...]]],
    *args: Any,
    **kwargs: Any
) -> Any:
    """Call function and trace it from parent span."""
    cb: Optional[Callable[..., Any]] = None
    cb_args: Tuple[Any, ...] = ()
    if callback:
        cb, cb_args = callback
    span.__enter__()
    try:
        ret: Any = fun(*args, **kwargs)
    except BaseException:
        span.__exit__(*sys.exc_info())
        raise
    else:
        if asyncio.iscoroutine(ret):

            async def corowrapped() -> Any:
                await_ret: Any = None
                try:
                    await_ret = await ret
                except BaseException:
                    span.__exit__(*sys.exc_info())
                    if cb:
                        cb(*cb_args)
                    raise
                else:
                    span.__exit__(None, None, None)
                    if cb:
                        cb(*cb_args)
                    return await_ret

            return corowrapped()
        else:
            span.__exit__(None, None, None)
            if cb:
                cb(*cb_args)
            return ret
