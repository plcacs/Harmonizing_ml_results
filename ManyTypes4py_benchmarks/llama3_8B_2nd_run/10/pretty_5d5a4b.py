from typing import TypeVar, Callable, Optional, Type, Any, Union
from collections import Counter, OrderedDict, defaultdict, deque
from enum import Enum, Flag
from functools import partial
from io import StringIO, TextIOBase
from math import copysign, isnan
from contextlib import contextmanager, suppress
from hypothesis.control import BuildContext
from hypothesis.internal.reflection import get_pretty_function_description

T: TypeVar('T')

PrettyPrintFunction: Callable[[Any, 'RepresentationPrinter', bool], None]
RepresentationPrinter: Type['RepresentationPrinter']

class RepresentationPrinter:
    # ... (rest of the class remains the same)

def _safe_getattr(obj: Any, attr: str, default: Optional[Any] = None) -> Any:
    """Safe version of getattr."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

def _repr_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    """A pprint that just redirects to the normal repr function."""
    output = repr(obj)
    for idx, output_line in enumerate(output.splitlines()):
        if idx:
            p.break_()
        p.text(output_line)

def pprint_fields(obj: Any, p: 'RepresentationPrinter', cycle: bool, fields: Union[Sequence[str], str]) -> None:
    """Pretty print the fields of an object."""
    name = get_class_name(obj.__class__)
    if cycle:
        return p.text(f'{name}(...)')
    with p.group(1, name + '(', ')'):
        for idx, field in enumerate(fields):
            if idx:
                p.text(',')
                p.breakable()
            p.text(field)
            p.text('=')
            p.pretty(getattr(obj, field))

def _function_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    """Base pprint for all functions and builtin functions."""
    p.text(get_pretty_function_description(obj))

def _exception_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    """Base pprint for all exceptions."""
    name = getattr(obj.__class__, '__qualname__', obj.__class__.__name__)
    if obj.__class__.__module__ not in ('exceptions', 'builtins'):
        name = f'{obj.__class__.__module__}.{name}'
    step = len(name) + 1
    with p.group(step, name + '(', ')'):
        for idx, arg in enumerate(getattr(obj, 'args', ())):
            if idx:
                p.text(',')
                p.breakable()
            p.pretty(arg)

def _repr_integer(obj: int, p: 'RepresentationPrinter', cycle: bool) -> None:
    """Pretty print an integer."""
    if abs(obj) < 1000000000:
        p.text(repr(obj))
    elif abs(obj) < 10 ** 640:
        p.text(f'{obj:#_d}')
    else:
        p.text(f'{obj:#_x}')

def _repr_float_counting_nans(obj: float, p: 'RepresentationPrinter', cycle: bool) -> None:
    """Pretty print a float."""
    if isnan(obj):
        if struct.pack('!d', abs(obj)) != struct.pack('!d', float('nan')):
            show = hex(*struct.unpack('Q', struct.pack('d', obj)))
            return p.text(f"struct.unpack('d', struct.pack('Q', {show}))[0]")
        elif copysign(1.0, obj) == -1.0:
            return p.text('-nan')
    p.text(repr(obj))

_type_pprinters: dict[Type[Any], Callable[[Any, 'RepresentationPrinter', bool], None]]
_deferred_type_pprinters: dict[tuple[str, str], Callable[[Any, 'RepresentationPrinter', bool], None]]
_singleton_pprinters: dict[object, Callable[[Any, 'RepresentationPrinter', bool], None]]
__all__: list[str] = ['IDKey', 'RepresentationPrinter', 'pretty']
