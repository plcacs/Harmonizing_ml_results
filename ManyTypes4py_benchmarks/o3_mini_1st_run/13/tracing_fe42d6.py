import inspect
import logging
import random
import sys
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from types import CodeType, FrameType
from typing import Any, Callable, Dict, Iterator, Optional, Union, cast
import opcode
from monkeytype.compat import cached_property
from monkeytype.typing import get_type
from monkeytype.util import get_func_fqname

logger: logging.Logger = logging.getLogger(__name__)

class CallTrace:
    """CallTrace contains the types observed during a single invocation of a function"""

    def __init__(
        self,
        func: Callable[..., Any],
        arg_types: Dict[str, Any],
        return_type: Optional[Any] = None,
        yield_type: Optional[Any] = None
    ) -> None:
        """
        Args:
            func: The function where the trace occurred
            arg_types: The collected argument types
            return_type: The collected return type. This will be None if the called function returns
                due to an unhandled exception. It will be NoneType if the function returns the value None.
            yield_type: The collected yield type. This will be None if the called function never
                yields. It will be NoneType if the function yields the value None.
        """
        self.func: Callable[..., Any] = func
        self.arg_types: Dict[str, Any] = arg_types
        self.return_type: Optional[Any] = return_type
        self.yield_type: Optional[Any] = yield_type

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __repr__(self) -> str:
        return 'CallTrace(%s, %s, %s, %s)' % (self.func, self.arg_types, self.return_type, self.yield_type)

    def __hash__(self) -> int:
        return hash((self.func, frozenset(self.arg_types.items()), self.return_type, self.yield_type))

    def add_yield_type(self, typ: Any) -> None:
        if self.yield_type is None:
            self.yield_type = typ
        else:
            self.yield_type = cast(type, Union[self.yield_type, typ])

    @property
    def funcname(self) -> str:
        return get_func_fqname(self.func)


class CallTraceLogger(metaclass=ABCMeta):
    """Log and store/print records collected by a CallTracer."""

    @abstractmethod
    def log(self, trace: CallTrace) -> None:
        """Log a single call trace."""
        pass

    def flush(self) -> None:
        """Flush all logged traces to output / database.

        Not an abstractmethod because it's OK to leave it as a no-op; for very
        simple loggers it may not be necessary to batch-flush traces, and `log`
        can handle everything.
        """
        pass


def get_func_in_mro(obj: Any, code: CodeType) -> Optional[Callable[..., Any]]:
    """Attempt to find a function in a side-effect free way.

    This looks in obj's mro manually and does not invoke any descriptors.
    """
    val: Any = inspect.getattr_static(obj, code.co_name, None)
    if val is None:
        return None
    if isinstance(val, (classmethod, staticmethod)):
        cand = val.__func__  # type: ignore
    elif isinstance(val, property) and val.fset is None and (val.fdel is None):
        cand = cast(Callable[..., Any], val.fget)
    elif cached_property and isinstance(val, cached_property):
        cand = val.func
    else:
        cand = cast(Callable[..., Any], val)
    return _has_code(cand, code)


def _has_code(func: Optional[Callable[..., Any]], code: CodeType) -> Optional[Callable[..., Any]]:
    while func is not None:
        func_code: Optional[CodeType] = getattr(func, '__code__', None)
        if func_code is code:
            return func
        func = getattr(func, '__wrapped__', None)
    return None


def get_previous_frames(frame: FrameType) -> Iterator[FrameType]:
    while frame is not None:
        yield frame
        frame = frame.f_back


def get_locals_from_previous_frames(frame: FrameType) -> Iterator[Any]:
    for previous_frame in get_previous_frames(frame):
        yield from previous_frame.f_locals.values()


def get_func(frame: FrameType) -> Optional[Callable[..., Any]]:
    """Return the function whose code object corresponds to the supplied stack frame."""
    code: CodeType = frame.f_code
    if code.co_name is None:
        return None
    cand: Any = frame.f_globals.get(code.co_name, None)
    func: Optional[Callable[..., Any]] = _has_code(cand, code)
    if func is None and code.co_argcount >= 1:
        first_arg = frame.f_locals.get(code.co_varnames[0])
        func = get_func_in_mro(first_arg, code)
    if func is None:
        for v in frame.f_globals.values():
            if not isinstance(v, type):
                continue
            func = get_func_in_mro(v, code)
            if func is not None:
                break
    if func is None:
        for v in get_locals_from_previous_frames(frame):
            if not callable(v):
                continue
            func = _has_code(v, code)
            if func is not None:
                break
    return func


RETURN_VALUE_OPCODE: int = opcode.opmap['RETURN_VALUE']
YIELD_VALUE_OPCODE: int = opcode.opmap['YIELD_VALUE']
CodeFilter = Callable[[CodeType], bool]
EVENT_CALL: str = 'call'
EVENT_RETURN: str = 'return'
SUPPORTED_EVENTS = {EVENT_CALL, EVENT_RETURN}


class CallTracer:
    """CallTracer captures the concrete types involved in a function invocation.

    On a per function call basis, CallTracer will record the types of arguments
    supplied, the type of the function's return value (if any), and the types
    of values yielded by the function (if any). It emits a CallTrace object
    that contains the captured types when the function returns.

    Use it like so:

        sys.setprofile(CallTracer(MyCallLogger()))

    """

    def __init__(
        self,
        logger: CallTraceLogger,
        max_typed_dict_size: int,
        code_filter: Optional[CodeFilter] = None,
        sample_rate: Optional[int] = None
    ) -> None:
        self.logger: CallTraceLogger = logger
        self.traces: Dict[FrameType, CallTrace] = {}
        self.sample_rate: Optional[int] = sample_rate
        self.cache: Dict[CodeType, Optional[Callable[..., Any]]] = {}
        self.should_trace: Optional[CodeFilter] = code_filter
        self.max_typed_dict_size: int = max_typed_dict_size

    def _get_func(self, frame: FrameType) -> Optional[Callable[..., Any]]:
        code: CodeType = frame.f_code
        if code not in self.cache:
            self.cache[code] = get_func(frame)
        return self.cache[code]

    def handle_call(self, frame: FrameType) -> None:
        if self.sample_rate and random.randrange(self.sample_rate) != 0:
            return
        func: Optional[Callable[..., Any]] = self._get_func(frame)
        if func is None:
            return
        code: CodeType = frame.f_code
        if frame in self.traces:
            return
        arg_names = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
        arg_types: Dict[str, Any] = {}
        for name in arg_names:
            if name in frame.f_locals:
                arg_types[name] = get_type(frame.f_locals[name], max_typed_dict_size=self.max_typed_dict_size)
        self.traces[frame] = CallTrace(func, arg_types)

    def handle_return(self, frame: FrameType, arg: Any) -> None:
        typ: Any = get_type(arg, max_typed_dict_size=self.max_typed_dict_size)
        last_opcode: int = frame.f_code.co_code[frame.f_lasti]
        trace: Optional[CallTrace] = self.traces.get(frame)
        if trace is None:
            return
        elif last_opcode == YIELD_VALUE_OPCODE:
            trace.add_yield_type(typ)
        else:
            if last_opcode == RETURN_VALUE_OPCODE:
                trace.return_type = typ
            del self.traces[frame]
            self.logger.log(trace)

    def __call__(self, frame: FrameType, event: str, arg: Any) -> "CallTracer":
        code: CodeType = frame.f_code
        if (
            event not in SUPPORTED_EVENTS
            or code.co_name == 'trace_types'
            or (self.should_trace and (not self.should_trace(code)))
        ):
            return self
        try:
            if event == EVENT_CALL:
                self.handle_call(frame)
            elif event == EVENT_RETURN:
                self.handle_return(frame, arg)
            else:
                logger.error('Cannot handle event %s', event)
        except Exception:
            logger.exception('Failed collecting trace')
        return self


@contextmanager
def trace_calls(
    logger: CallTraceLogger,
    max_typed_dict_size: int,
    code_filter: Optional[CodeFilter] = None,
    sample_rate: Optional[int] = None
) -> Iterator[None]:
    """Enable call tracing for a block of code"""
    old_trace: Optional[Callable[..., Any]] = sys.getprofile()
    sys.setprofile(CallTracer(logger, max_typed_dict_size, code_filter, sample_rate))
    try:
        yield
    finally:
        sys.setprofile(old_trace)
        logger.flush()