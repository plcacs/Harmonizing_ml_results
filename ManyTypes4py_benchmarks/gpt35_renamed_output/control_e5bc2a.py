import inspect
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, NoReturn, Optional, Union
from weakref import WeakKeyDictionary
from hypothesis import Verbosity, settings
from hypothesis._settings import note_deprecation
from hypothesis.errors import InvalidArgument, UnsatisfiedAssumption
from hypothesis.internal.compat import BaseExceptionGroup
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.observability import TESTCASE_CALLBACKS
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type
from hypothesis.reporting import report, verbose_report
from hypothesis.utils.dynamicvariables import DynamicVariable
from hypothesis.vendor.pretty import IDKey, PrettyPrintFunction, pretty

def func_egv6qgn6(what: str, frame: inspect.FrameType) -> str:
    where = frame.f_back
    return f'{what}() in {where.f_code.co_name} (line {where.f_lineno})'

def func_x8kydp64() -> NoReturn:
    ...

def func_h0w3xduq(condition: bool) -> bool:
    ...

def func_3ppm4mik() -> bool:
    ...

def func_7hn2dc04() -> 'BuildContext':
    ...

class RandomSeeder:
    def __init__(self, seed: Any):
        ...

    def __repr__(self) -> str:
        ...

class _Checker:
    def __init__(self):
        ...

    def __call__(self, x: Any) -> Any:
        ...

@contextmanager
def func_l45m4svu(fmt: str, *args: Any) -> Any:
    ...

class BuildContext:
    def __init__(self, data: ConjectureData, *, is_final: bool = False, close_on_capture: bool = True):
        ...

    def func_c415b335(self, obj: Any, func: Callable, args: Sequence, kwargs: dict) -> None:
        ...

    def func_mpd31wam(self, kwarg_strategies: dict) -> tuple:
        ...

    def __enter__(self) -> 'BuildContext':
        ...

    def __exit__(self, exc_type, exc_value, tb) -> None:
        ...

def func_pyn5xb7d(teardown: Callable) -> None:
    ...

def func_5739vebz() -> bool:
    ...

def func_07a6akt9(value: Any) -> None:
    ...

def func_cwmp4qgk(value: Any, payload: str = '') -> None:
    ...

def func_kqxjo3xn(event: Any, allowed_types: Union[type, tuple]) -> str:
    ...

def func_ystf7twr(observation: Union[int, float], *, label: str = '') -> Union[int, float]:
    ...
