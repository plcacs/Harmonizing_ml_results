"""OpenTracing utilities."""
import asyncio
import sys
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Optional, Tuple, TypeVar, Union, Coroutine, List

import opentracing
from mode import shortlabel

__all__: List[str] = [
    'current_span',
    'noop_span',
    'set_current_span',
    'finish_span',
    'operation_name_from_fun',
    'traced_from_parent_span',
    'call_with_trace',
]

R = TypeVar('R')
A = TypeVar('A')

_current_span: ContextVar[Optional[opentracing.Span]] = ContextVar('current_span')


def current_span() -> Optional[opentracing.Span]:
    """Get the current span for this context (if any)."""
    return _current_span.get(None)


def set_current_span(span: Optional[opentracing.Span]) -> None:
    """Set the current span for the current context."""
    _current_span.set(span)


def noop_span() -> opentracing.Span:
    """Return a span that does nothing when traced."""
    return opentracing.Tracer()._noop_span


def finish_span(span: Optional[opentracing.Span], *, error: Optional[BaseException] = None) -> None:
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
        objlabel = shortlabel(obj)
        funlabel = shortlabel(fun)
        if funlabel.startswith(objlabel):
            funlabel = funlabel[len(objlabel):]
        return f'{objlabel}-{funlabel}'
    else:
        return f'{shortlabel(fun)}'


def traced_from_parent_span(
    parent_span: Optional[opentracing.Span] = None,
    callback: Optional[Callable[[opentracing.Span], Any]] = None,
    **extra_context: Any,
) -> Callable[
    [Callable[..., Union[R, Coroutine[Any, Any, A]]]],
    Callable[..., Union[R, Coroutine[Any, Any, A]]]
]:
    """Decorate function to be traced from parent span."""

    def _wrapper(
        fun: Callable[..., Union[R, Coroutine[Any, Any, A]]],
        **more_context: Any,
    ) -> Callable[..., Union[R, Coroutine[Any, Any, A]]]:
        operation_name = operation_name_from_fun(fun)

        @wraps(fun)
        def _inner(*args: Any, **kwargs: Any) -> Union[R, Coroutine[Any, Any, A]]:
            parent = parent_span
            if parent is None:
                parent = current_span()
            if parent is not None:
                child = parent.tracer.start_span(
                    operation_name=operation_name,
                    child_of=parent,
                    tags={**extra_context, **more_context},
                )
                if callback is not None:
                    callback(child)
                on_exit = (_restore_span, (parent, child))
                set_current_span(child)
                return call_with_trace(child, fun, on_exit, *args, **kwargs)
            return fun(*args, **kwargs)

        return _inner

    return _wrapper


def _restore_span(span: opentracing.Span, expected_current_span: opentracing.Span) -> None:
    current = current_span()
    assert current is expected_current_span
    set_current_span(span)


def call_with_trace(
    span: opentracing.Span,
    fun: Callable[..., Union[R, Coroutine[Any, Any, A]]],
    callback: Optional[Tuple[Callable[..., Any], Tuple[Any, ...]]],
    *args: Any,
    **kwargs: Any,
) -> Union[R, Coroutine[Any, Any, A]]:
    """Call function and trace it from parent span."""
    cb: Optional[Callable[..., Any]] = None
    cb_args: Tuple[Any, ...] = ()
    if callback:
        cb, cb_args = callback
    span.__enter__()
    try:
        ret: Union[R, Coroutine[Any, Any, A]] = fun(*args, **kwargs)
    except BaseException:
        span.__exit__(*sys.exc_info())
        raise
    else:
        if asyncio.iscoroutine(ret):

            async def corowrapped() -> A:
                try:
                    await_ret = await ret  # type: ignore[func-returns-value]
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