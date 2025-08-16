from typing import TYPE_CHECKING, Any, Callable, TypeVar

T = TypeVar('T')
PrettyPrintFunction = Callable[[Any, 'RepresentationPrinter', bool], None]
__all__: list[str] = ['IDKey', 'RepresentationPrinter', 'pretty']

def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    ...

def pretty(obj: Any) -> str:
    ...

class IDKey:
    ...

class RepresentationPrinter:
    ...

class Printable:
    ...

class Text(Printable):
    ...

class Breakable(Printable):
    ...

class Group(Printable):
    ...

class GroupQueue:
    ...

def _seq_pprinter_factory(start: str, end: str, basetype: type) -> PrettyPrintFunction:
    ...

def get_class_name(cls: type) -> str:
    ...

def _set_pprinter_factory(start: str, end: str, basetype: type) -> PrettyPrintFunction:
    ...

def _dict_pprinter_factory(start: str, end: str, basetype: type = None) -> PrettyPrintFunction:
    ...

def _super_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _re_pattern_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _type_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _repr_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def pprint_fields(obj: Any, p: 'RepresentationPrinter', cycle: bool, fields: list[str]) -> None:
    ...

def _function_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _exception_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _repr_integer(obj: int, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _repr_float_counting_nans(obj: float, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

_type_pprinters: dict[type, PrettyPrintFunction] = {int: _repr_integer, float: _repr_float_counting_nans, str: _repr_pprint, tuple: _seq_pprinter_factory('(', ')', tuple), list: _seq_pprinter_factory('[', ']', list), dict: _dict_pprinter_factory('{', '}', dict), set: _set_pprinter_factory('{', '}', set), frozenset: _set_pprinter_factory('frozenset({', '})', frozenset), super: _super_pprint, re.Pattern: _re_pattern_pprint, type: _type_pprint, types.FunctionType: _function_pprint, types.BuiltinFunctionType: _function_pprint, types.MethodType: _function_pprint, datetime.datetime: _repr_pprint, datetime.timedelta: _repr_pprint, BaseException: _exception_pprint, slice: _repr_pprint, range: _repr_pprint, bytes: _repr_pprint}

_deferred_type_pprinters: dict[tuple[str, str], PrettyPrintFunction] = {}

def for_type_by_name(type_module: str, type_name: str, func: PrettyPrintFunction) -> PrettyPrintFunction:
    ...

_singleton_pprinters: dict[int, PrettyPrintFunction] = dict.fromkeys(map(id, [None, True, False, Ellipsis, NotImplemented]), _repr_pprint)

def _defaultdict_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _ordereddict_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _deque_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _counter_pprint(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _repr_dataframe(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

def _repr_enum(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

class _ReprDots:
    ...

def _repr_partial(obj: Any, p: 'RepresentationPrinter', cycle: bool) -> None:
    ...

for_type_by_name('collections', 'defaultdict', _defaultdict_pprint)
for_type_by_name('collections', 'OrderedDict', _ordereddict_pprint)
for_type_by_name('ordereddict', 'OrderedDict', _ordereddict_pprint)
for_type_by_name('collections', 'deque', _deque_pprint)
for_type_by_name('collections', 'Counter', _counter_pprint)
for_type_by_name('pandas.core.frame', 'DataFrame', _repr_dataframe)
for_type_by_name('enum', 'Enum', _repr_enum)
for_type_by_name('functools', 'partial', _repr_partial)
