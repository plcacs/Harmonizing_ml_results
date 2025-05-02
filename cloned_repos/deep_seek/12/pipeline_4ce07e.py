"""Experimental pipeline API functionality. Be careful with this API, it's subject to change."""
from __future__ import annotations
import datetime
import operator
import re
import sys
from collections import deque
from collections.abc import Container
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property, partial
from re import Pattern
from typing import TYPE_CHECKING, Annotated, Any, Callable, Generic, Protocol, TypeVar, Union, overload, Optional, Type, cast
import annotated_types
if TYPE_CHECKING:
    from pydantic_core import core_schema as cs
    from pydantic import GetCoreSchemaHandler
from pydantic._internal._internal_dataclass import slots_true as _slots_true
if sys.version_info < (3, 10):
    EllipsisType = type(Ellipsis)
else:
    from types import EllipsisType
__all__ = ['validate_as', 'validate_as_deferred', 'transform']
_slots_frozen = {**_slots_true, 'frozen': True}

@dataclass(**_slots_frozen)
class _ValidateAs:
    tp: Any
    strict: bool = False

@dataclass
class _ValidateAsDefer:
    func: Callable[[], Any]

    @cached_property
    def tp(self) -> Any:
        return self.func()

@dataclass(**_slots_frozen)
class _Transform:
    func: Callable[[Any], Any]

@dataclass(**_slots_frozen)
class _PipelineOr:
    left: _Pipeline[Any, Any]
    right: _Pipeline[Any, Any]

@dataclass(**_slots_frozen)
class _PipelineAnd:
    left: _Pipeline[Any, Any]
    right: _Pipeline[Any, Any]

@dataclass(**_slots_frozen)
class _Eq:
    value: Any

@dataclass(**_slots_frozen)
class _NotEq:
    value: Any

@dataclass(**_slots_frozen)
class _In:
    values: Container[Any]

@dataclass(**_slots_frozen)
class _NotIn:
    values: Container[Any]

_ConstraintAnnotation = Union[
    annotated_types.Le, annotated_types.Ge, annotated_types.Lt, annotated_types.Gt, 
    annotated_types.Len, annotated_types.MultipleOf, annotated_types.Timezone, 
    annotated_types.Interval, annotated_types.Predicate, _Eq, _NotEq, _In, _NotIn, 
    Pattern[str]
]

@dataclass(**_slots_frozen)
class _Constraint:
    constraint: _ConstraintAnnotation

_Step = Union[_ValidateAs, _ValidateAsDefer, _Transform, _PipelineOr, _PipelineAnd, _Constraint]
_InT = TypeVar('_InT')
_OutT = TypeVar('_OutT')
_NewOutT = TypeVar('_NewOutT')

class _FieldTypeMarker:
    pass

@dataclass(**_slots_true)
class _Pipeline(Generic[_InT, _OutT]):
    """Abstract representation of a chain of validation, transformation, and parsing steps."""
    _steps: tuple[_Step, ...] = ()

    def transform(self, func: Callable[[_OutT], _NewOutT]) -> _Pipeline[_InT, _NewOutT]:
        """Transform the output of the previous step.

        If used as the first step in a pipeline, the type of the field is used.
        That is, the transformation is applied to after the value is parsed to the field's type.
        """
        return _Pipeline[_InT, _NewOutT](self._steps + (_Transform(func),))

    @overload
    def validate_as(self, tp: Type[_NewOutT], *, strict: bool = ...) -> _Pipeline[_InT, _NewOutT]: ...

    @overload
    def validate_as(self, tp: EllipsisType, *, strict: bool = ...) -> _Pipeline[_InT, Any]: ...

    def validate_as(self, tp: Union[Type[_NewOutT], EllipsisType], *, strict: bool = False) -> Union[_Pipeline[_InT, _NewOutT], _Pipeline[_InT, Any]]:
        """Validate / parse the input into a new type.

        If no type is provided, the type of the field is used.

        Types are parsed in Pydantic's `lax` mode by default,
        but you can enable `strict` mode by passing `strict=True`.
        """
        if isinstance(tp, EllipsisType):
            return _Pipeline[_InT, Any](self._steps + (_ValidateAs(_FieldTypeMarker, strict=strict),))
        return _Pipeline[_InT, _NewOutT](self._steps + (_ValidateAs(tp, strict=strict),))

    def validate_as_deferred(self, func: Callable[[], Type[_NewOutT]]) -> _Pipeline[_InT, _NewOutT]:
        """Parse the input into a new type, deferring resolution of the type until the current class
        is fully defined.

        This is useful when you need to reference the class in it's own type annotations.
        """
        return _Pipeline[_InT, _NewOutT](self._steps + (_ValidateAsDefer(func),))

    @overload
    def constrain(self, constraint: annotated_types.Gt) -> _Pipeline[_InT, _NewOutGt]: ...

    @overload
    def constrain(self, constraint: annotated_types.Ge) -> _Pipeline[_InT, _NewOutGe]: ...

    @overload
    def constrain(self, constraint: annotated_types.Lt) -> _Pipeline[_InT, _NewOutLt]: ...

    @overload
    def constrain(self, constraint: annotated_types.Le) -> _Pipeline[_InT, _NewOutLe]: ...

    @overload
    def constrain(self, constraint: annotated_types.Len) -> _Pipeline[_InT, _NewOutLen]: ...

    @overload
    def constrain(self, constraint: annotated_types.MultipleOf) -> _Pipeline[_InT, _NewOutDiv]: ...

    @overload
    def constrain(self, constraint: annotated_types.Timezone) -> _Pipeline[_InT, _NewOutDatetime]: ...

    @overload
    def constrain(self, constraint: annotated_types.Interval) -> _Pipeline[_InT, _NewOutInterval]: ...

    @overload
    def constrain(self, constraint: annotated_types.Predicate) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self, constraint: _Eq) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self, constraint: _NotEq) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self, constraint: _In) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self, constraint: _NotIn) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self, constraint: Pattern[str]) -> _Pipeline[_InT, str]: ...

    def constrain(self, constraint: _ConstraintAnnotation) -> _Pipeline[_InT, Any]:
        """Constrain a value to meet a certain condition.

        We support most conditions from `annotated_types`, as well as regular expressions.

        Most of the time you'll be calling a shortcut method like `gt`, `lt`, `len`, etc
        so you don't need to call this directly.
        """
        return _Pipeline[_InT, Any](self._steps + (_Constraint(constraint),))

    def predicate(self, func: Callable[[_OutT], bool]) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to meet a certain predicate."""
        return self.constrain(annotated_types.Predicate(func))

    def gt(self, gt: _NewOutGt) -> _Pipeline[_InT, _NewOutGt]:
        """Constrain a value to be greater than a certain value."""
        return self.constrain(annotated_types.Gt(gt))

    def lt(self, lt: _NewOutLt) -> _Pipeline[_InT, _NewOutLt]:
        """Constrain a value to be less than a certain value."""
        return self.constrain(annotated_types.Lt(lt))

    def ge(self, ge: _NewOutGe) -> _Pipeline[_InT, _NewOutGe]:
        """Constrain a value to be greater than or equal to a certain value."""
        return self.constrain(annotated_types.Ge(ge))

    def le(self, le: _NewOutLe) -> _Pipeline[_InT, _NewOutLe]:
        """Constrain a value to be less than or equal to a certain value."""
        return self.constrain(annotated_types.Le(le))

    def len(self, min_len: int, max_len: Optional[int] = None) -> _Pipeline[_InT, _NewOutLen]:
        """Constrain a value to have a certain length."""
        return self.constrain(annotated_types.Len(min_len, max_len))

    @overload
    def multiple_of(self, multiple_of: int) -> _Pipeline[_InT, int]: ...

    @overload
    def multiple_of(self, multiple_of: float) -> _Pipeline[_InT, float]: ...

    def multiple_of(self, multiple_of: Union[int, float]) -> _Pipeline[_InT, Union[int, float]]:
        """Constrain a value to be a multiple of a certain number."""
        return self.constrain(annotated_types.MultipleOf(multiple_of))

    def eq(self, value: Any) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to be equal to a certain value."""
        return self.constrain(_Eq(value))

    def not_eq(self, value: Any) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to not be equal to a certain value."""
        return self.constrain(_NotEq(value))

    def in_(self, values: Container[Any]) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to be in a certain set."""
        return self.constrain(_In(values))

    def not_in(self, values: Container[Any]) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to not be in a certain set."""
        return self.constrain(_NotIn(values))

    def datetime_tz_naive(self) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(None))

    def datetime_tz_aware(self) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(...))

    def datetime_tz(self, tz: Any) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(tz))

    def datetime_with_tz(self, tz: Any) -> _Pipeline[_InT, datetime.datetime]:
        return self.transform(partial(datetime.datetime.replace, tzinfo=tz))

    def str_lower(self) -> _Pipeline[_InT, str]:
        return self.transform(str.lower)

    def str_upper(self) -> _Pipeline[_InT, str]:
        return self.transform(str.upper)

    def str_title(self) -> _Pipeline[_InT, str]:
        return self.transform(str.title)

    def str_strip(self) -> _Pipeline[_InT, str]:
        return self.transform(str.strip)

    def str_pattern(self, pattern: str) -> _Pipeline[_InT, str]:
        return self.constrain(re.compile(pattern))

    def str_contains(self, substring: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: substring in v)

    def str_starts_with(self, prefix: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: v.startswith(prefix))

    def str_ends_with(self, suffix: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: v.endswith(suffix))

    def otherwise(self, other: _Pipeline[_InT, _OutT]) -> _Pipeline[_InT, _OutT]:
        """Combine two validation chains, returning the result of the first chain if it succeeds, and the second chain if it fails."""
        return _Pipeline((_PipelineOr(self, other),))
    __or__ = otherwise

    def then(self, other: _Pipeline[_OutT, _NewOutT]) -> _Pipeline[_InT, _NewOutT]:
        """Pipe the result of one validation chain into another."""
        return _Pipeline((_PipelineAnd(self, other),))
    __and__ = then

    def __get_pydantic_core_schema__(self, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema as cs
        queue = deque(self._steps)
        s = None
        while queue:
            step = queue.popleft()
            s = _apply_step(step, s, handler, source_type)
        s = s or cs.any_schema()
        return s

    def __supports_type__(self, _: Any) -> bool:
        raise NotImplementedError

validate_as: Callable[..., _Pipeline[Any, Any]] = _Pipeline[Any, Any](()).validate_as
validate_as_deferred: Callable[..., _Pipeline[Any, Any]] = _Pipeline[Any, Any](()).validate_as_deferred
transform: Callable[..., _Pipeline[Any, Any]] = _Pipeline[Any, Any]((_ValidateAs(_FieldTypeMarker),)).transform

def _check_func(func: Callable[[Any], bool], predicate_err: Union[str, Callable[[], str]], s: Optional[Any]) -> Any:
    from pydantic_core import core_schema as cs

    def handler(v: Any) -> Any:
        if func(v):
            return v
        raise ValueError(f'Expected {(predicate_err if isinstance(predicate_err, str) else predicate_err())}')
    if s is None:
        return cs.no_info_plain_validator_function(handler)
    else:
        return cs.no_info_after_validator_function(handler, s)

def _apply_step(step: _Step, s: Optional[Any], handler: Any, source_type: Any) -> Any:
    from pydantic_core import core_schema as cs
    if isinstance(step, _ValidateAs):
        s = _apply_parse(s, step.tp, step.strict, handler, source_type)
    elif isinstance(step, _ValidateAsDefer):
        s = _apply_parse(s, step.tp, False, handler, source_type)
    elif isinstance(step, _Transform):
        s = _apply_transform(s, step.func, handler)
    elif isinstance(step, _Constraint):
        s = _apply_constraint(s, step.constraint)
    elif isinstance(step, _PipelineOr):
        s = cs.union_schema([handler(step.left), handler(step.right)])
    else:
        assert isinstance(step, _PipelineAnd)
        s = cs.chain_schema([handler(step.left), handler(step.right)])
    return s

def _apply_parse(s: Optional[Any], tp: Any, strict: bool, handler: Any, source_type: Any) -> Any:
    from pydantic_core import core_schema as cs
    from pydantic import Strict
    if tp is _FieldTypeMarker:
        return handler(source_type)
    if strict:
        tp = Annotated[tp, Strict()]
    if s and s['type'] == 'any':
        return handler(tp)
    else:
        return cs.chain_schema([s, handler(tp)]) if s else handler(tp)

def _apply_transform(s: Optional[Any], func: Callable[[Any], Any], handler: Any) -> Any:
    from pydantic_core import core_schema as cs
    if s is None:
        return cs.no_info_plain_validator_function(func)
    if s['type'] == 'str':
        if func is str.strip:
            s = s.copy()
            s['strip_whitespace'] = True
            return s
        elif func is str.lower:
            s = s.copy()
            s['to_lower'] = True
            return s
        elif func is str.upper:
            s = s.copy()
            s['to_upper'] = True
            return s
    return cs.no_info_after_validator_function(func, s)

def _apply_constraint(s: Optional[Any], constraint: _ConstraintAnnotation) -> Any:
    """Apply a single constraint to a schema."""
    if isinstance(constraint, annotated_types.Gt):
        gt = constraint.gt
        if s and s['type'] in {'int', 'float', 'decimal'}:
            s = s.copy()
            if s['type'] == 'int' and isinstance(gt, int):
                s['gt'] = gt
            elif s['type'] == 'float' and isinstance(gt, float):
                s['gt'] = gt
            elif s['type'] == 'decimal' and isinstance(gt, Decimal):
                s['gt'] = gt
        else:
            def check_gt(v: Any) -> bool:
                return v > gt
            s = _check_func(check_gt, f'> {gt}', s)
    elif isinstance(constraint, annotated_types.Ge):
        ge = constraint.ge
        if s and s['type'] in {'int', 'float', 'decimal'}:
            s = s.copy()
            if s['type'] == 'int' and isinstance(ge, int):
                s['ge'] = ge
            elif s['type'] == 'float' and isinstance(ge, float):
                s['ge'] = ge
            elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                s['ge'] = ge

        def check_ge(v: Any) -> bool:
            return v >= ge
        s = _check_func(check_ge, f'>= {ge}', s)
    elif isinstance(constraint, annotated_types.Lt):
        lt = constraint.lt
        if s and s['type'] in {'int', 'float', 'decimal'}:
            s = s.copy()
            if s['type'] == 'int' and isinstance(lt, int):
                s['lt'] = lt
            elif s['type'] == 'float' and isinstance(lt, float):
                s['lt'] = lt
