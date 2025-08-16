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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    Union,
    overload,
    cast,
)

import annotated_types
from typing_extensions import Annotated

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
    tp: type[Any]
    strict: bool = False


@dataclass
class _ValidateAsDefer:
    func: Callable[[], type[Any]]

    @cached_property
    def tp(self) -> type[Any]:
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
    annotated_types.Le,
    annotated_types.Ge,
    annotated_types.Lt,
    annotated_types.Gt,
    annotated_types.Len,
    annotated_types.MultipleOf,
    annotated_types.Timezone,
    annotated_types.Interval,
    annotated_types.Predicate,
    _Eq,
    _NotEq,
    _In,
    _NotIn,
    Pattern[str],
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
    _steps: tuple[_Step, ...]

    def transform(
        self,
        func: Callable[[_OutT], _NewOutT],
    ) -> _Pipeline[_InT, _NewOutT]:
        return _Pipeline[_InT, _NewOutT](self._steps + (_Transform(func),))

    @overload
    def validate_as(self, tp: type[_NewOutT], *, strict: bool = False) -> _Pipeline[_InT, _NewOutT]: ...

    @overload
    def validate_as(self, tp: EllipsisType, *, strict: bool = False) -> _Pipeline[_InT, Any]: ...

    def validate_as(
        self, tp: type[_NewOutT] | EllipsisType, *, strict: bool = False
    ) -> _Pipeline[_InT, _NewOutT] | _Pipeline[_InT, Any]:
        if isinstance(tp, EllipsisType):
            return _Pipeline[_InT, Any](self._steps + (_ValidateAs(_FieldTypeMarker, strict=strict),)
        return _Pipeline[_InT, _NewOutT](self._steps + (_ValidateAs(tp, strict=strict),))

    def validate_as_deferred(self, func: Callable[[], type[_NewOutT]]) -> _Pipeline[_InT, _NewOutT]:
        return _Pipeline[_InT, _NewOutT](self._steps + (_ValidateAsDefer(func),))

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutGe], constraint: annotated_types.Ge) -> _Pipeline[_InT, _NewOutGe]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutGt], constraint: annotated_types.Gt) -> _Pipeline[_InT, _NewOutGt]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutLe], constraint: annotated_types.Le) -> _Pipeline[_InT, _NewOutLe]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutLt], constraint: annotated_types.Lt) -> _Pipeline[_InT, _NewOutLt]: ...

    @overload
    def constrain(
        self: _Pipeline[_InT, _NewOutLen], constraint: annotated_types.Len
    ) -> _Pipeline[_InT, _NewOutLen]: ...

    @overload
    def constrain(
        self: _Pipeline[_InT, _NewOutT], constraint: annotated_types.MultipleOf
    ) -> _Pipeline[_InT, _NewOutT]: ...

    @overload
    def constrain(
        self: _Pipeline[_InT, _NewOutDatetime], constraint: annotated_types.Timezone
    ) -> _Pipeline[_InT, _NewOutDatetime]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: annotated_types.Predicate) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(
        self: _Pipeline[_InT, _NewOutInterval], constraint: annotated_types.Interval
    ) -> _Pipeline[_InT, _NewOutInterval]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _Eq) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _NotEq) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _In) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _NotIn) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutT], constraint: Pattern[str]) -> _Pipeline[_InT, _NewOutT]: ...

    def constrain(self, constraint: _ConstraintAnnotation) -> Any:
        return _Pipeline[_InT, _OutT](self._steps + (_Constraint(constraint),))

    def predicate(self: _Pipeline[_InT, _NewOutT], func: Callable[[_NewOutT], bool]) -> _Pipeline[_InT, _NewOutT]:
        return self.constrain(annotated_types.Predicate(func))

    def gt(self: _Pipeline[_InT, _NewOutGt], gt: _NewOutGt) -> _Pipeline[_InT, _NewOutGt]:
        return self.constrain(annotated_types.Gt(gt))

    def lt(self: _Pipeline[_InT, _NewOutLt], lt: _NewOutLt) -> _Pipeline[_InT, _NewOutLt]:
        return self.constrain(annotated_types.Lt(lt))

    def ge(self: _Pipeline[_InT, _NewOutGe], ge: _NewOutGe) -> _Pipeline[_InT, _NewOutGe]:
        return self.constrain(annotated_types.Ge(ge))

    def le(self: _Pipeline[_InT, _NewOutLe], le: _NewOutLe) -> _Pipeline[_InT, _NewOutLe]:
        return self.constrain(annotated_types.Le(le))

    def len(self: _Pipeline[_InT, _NewOutLen], min_len: int, max_len: int | None = None) -> _Pipeline[_InT, _NewOutLen]:
        return self.constrain(annotated_types.Len(min_len, max_len))

    @overload
    def multiple_of(self: _Pipeline[_InT, _NewOutDiv], multiple_of: _NewOutDiv) -> _Pipeline[_InT, _NewOutDiv]: ...

    @overload
    def multiple_of(self: _Pipeline[_InT, _NewOutMod], multiple_of: _NewOutMod) -> _Pipeline[_InT, _NewOutMod]: ...

    def multiple_of(self: _Pipeline[_InT, Any], multiple_of: Any) -> _Pipeline[_InT, Any]:
        return self.constrain(annotated_types.MultipleOf(multiple_of))

    def eq(self: _Pipeline[_InT, _OutT], value: _OutT) -> _Pipeline[_InT, _OutT]:
        return self.constrain(_Eq(value))

    def not_eq(self: _Pipeline[_InT, _OutT], value: _OutT) -> _Pipeline[_InT, _OutT]:
        return self.constrain(_NotEq(value))

    def in_(self: _Pipeline[_InT, _OutT], values: Container[_OutT]) -> _Pipeline[_InT, _OutT]:
        return self.constrain(_In(values))

    def not_in(self: _Pipeline[_InT, _OutT], values: Container[_OutT]) -> _Pipeline[_InT, _OutT]:
        return self.constrain(_NotIn(values))

    def datetime_tz_naive(self: _Pipeline[_InT, datetime.datetime]) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(None))

    def datetime_tz_aware(self: _Pipeline[_InT, datetime.datetime]) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(...))

    def datetime_tz(
        self: _Pipeline[_InT, datetime.datetime], tz: datetime.tzinfo
    ) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(tz))

    def datetime_with_tz(
        self: _Pipeline[_InT, datetime.datetime], tz: datetime.tzinfo | None
    ) -> _Pipeline[_InT, datetime.datetime]:
        return self.transform(partial(datetime.datetime.replace, tzinfo=tz))

    def str_lower(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
        return self.transform(str.lower)

    def str_upper(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
        return self.transform(str.upper)

    def str_title(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
        return self.transform(str.title)

    def str_strip(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
        return self.transform(str.strip)

    def str_pattern(self: _Pipeline[_InT, str], pattern: str) -> _Pipeline[_InT, str]:
        return self.constrain(re.compile(pattern))

    def str_contains(self: _Pipeline[_InT, str], substring: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: substring in v)

    def str_starts_with(self: _Pipeline[_InT, str], prefix: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: v.startswith(prefix))

    def str_ends_with(self: _Pipeline[_InT, str], suffix: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: v.endswith(suffix))

    def otherwise(self, other: _Pipeline[_OtherIn, _OtherOut]) -> _Pipeline[_InT | _OtherIn, _OutT | _OtherOut]:
        return _Pipeline((_PipelineOr(self, other),))

    __or__ = otherwise

    def then(self, other: _Pipeline[_OutT, _OtherOut]) -> _Pipeline[_InT, _OtherOut]:
        return _Pipeline((_PipelineAnd(self, other),))

    __and__ = then

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> cs.CoreSchema:
        from pydantic_core import core_schema as cs

        queue = deque(self._steps)

        s = None

        while queue:
            step = queue.popleft()
            s = _apply_step(step, s, handler, source_type)

        s = s or cs.any_schema()
        return s

    def __supports_type__(self, _: _OutT) -> bool:
        raise NotImplementedError


validate_as = _Pipeline[Any, Any](()).validate_as
validate_as_deferred = _Pipeline[Any, Any](()).validate_as_deferred
transform = _Pipeline[Any, Any]((_ValidateAs(_FieldTypeMarker),)).transform


def _check_func(
    func: Callable[[Any], bool], predicate_err: str | Callable[[], str], s: cs.CoreSchema | None
) -> cs.CoreSchema:
    from pydantic_core import core_schema as cs

    def handler(v: Any) -> Any:
        if func(v):
            return v
        raise ValueError(f'Expected {predicate_err if isinstance(predicate_err, str) else predicate_err()}')

    if s is None:
        return cs.no_info_plain_validator_function(handler)
    else:
        return cs.no_info_after_validator_function(handler, s)


def _apply_step(step: _Step, s: cs.CoreSchema | None, handler: GetCoreSchemaHandler, source_type: Any) -> cs.CoreSchema:
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


def _apply_parse(
    s: cs.CoreSchema | None,
    tp: type[Any],
    strict: bool,
    handler: GetCoreSchemaHandler,
    source_type: Any,
) -> cs.CoreSchema:
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


def _apply_transform(
    s: cs.CoreSchema | None, func: Callable[[Any], Any], handler: GetCoreSchemaHandler
) -> cs.CoreSchema:
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


def _apply_constraint(s: cs.CoreSchema | None, constraint: _ConstraintAnnotation) -> cs.CoreSchema:
    from pydantic_core import core_schema as cs

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
        else:
            def check_ge(v: Any) -> bool:
                return v >= ge
            s = _check_func(check_ge, f'>= {ge}', s)
    elif isinstance(constraint, annotated_types.Lt):
        lt = constraint.lt
        if s and s['