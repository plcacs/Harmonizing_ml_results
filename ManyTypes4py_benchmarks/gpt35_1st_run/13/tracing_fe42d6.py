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
    def __init__(self, func: Callable, arg_types: Dict[str, Any], return_type: Optional[type] = None, yield_type: Optional[type] = None) -> None:
        self.func = func
        self.arg_types = arg_types
        self.return_type = return_type
        self.yield_type = yield_type

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __repr__(self) -> str:
        return 'CallTrace(%s, %s, %s, %s)' % (self.func, self.arg_types, self.return_type, self.yield_type)

    def __hash__(self) -> int:
        return hash((self.func, frozenset(self.arg_types.items()), self.return_type, self.yield_type))

    def add_yield_type(self, typ: type) -> None:
        if self.yield_type is None:
            self.yield_type = typ
        else:
            self.yield_type = cast(type, Union[self.yield_type, typ])

    @property
    def funcname(self) -> str:
        return get_func_fqname(self.func)

class CallTraceLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, trace: CallTrace) -> None:
        pass

    def flush(self) -> None:
        pass

def get_func_in_mro(obj: Any, code: CodeType) -> Optional[Callable]:
    ...

def _has_code(func: Callable, code: CodeType) -> Optional[Callable]:
    ...

def get_previous_frames(frame: FrameType) -> Iterator[FrameType]:
    ...

def get_locals_from_previous_frames(frame: FrameType) -> Iterator[Any]:
    ...

def get_func(frame: FrameType) -> Optional[Callable]:
    ...

RETURN_VALUE_OPCODE: int = opcode.opmap['RETURN_VALUE']
YIELD_VALUE_OPCODE: int = opcode.opmap['YIELD_VALUE']
CodeFilter = Callable[[CodeType], bool]
EVENT_CALL: str = 'call'
EVENT_RETURN: str = 'return'
SUPPORTED_EVENTS: set = {EVENT_CALL, EVENT_RETURN}

class CallTracer:
    def __init__(self, logger: CallTraceLogger, max_typed_dict_size: int, code_filter: Optional[CodeFilter] = None, sample_rate: Optional[int] = None) -> None:
        ...

    def _get_func(self, frame: FrameType) -> Optional[Callable]:
        ...

    def handle_call(self, frame: FrameType) -> None:
        ...

    def handle_return(self, frame: FrameType, arg: Any) -> None:
        ...

    def __call__(self, frame: FrameType, event: str, arg: Any) -> 'CallTracer':
        ...

@contextmanager
def trace_calls(logger: CallTraceLogger, max_typed_dict_size: int, code_filter: Optional[CodeFilter] = None, sample_rate: Optional[int] = None) -> Iterator[None]:
    ...
