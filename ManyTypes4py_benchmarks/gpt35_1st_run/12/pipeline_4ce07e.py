from typing import Any, Union, Annotated, TypeVar, Protocol, Pattern

class _FieldTypeMarker:
    pass

_ConstraintAnnotation = Union[annotated_types.Le, annotated_types.Ge, annotated_types.Lt, annotated_types.Gt, annotated_types.Len, annotated_types.MultipleOf, annotated_types.Timezone, annotated_types.Interval, annotated_types.Predicate, _Eq, _NotEq, _In, _NotIn, Pattern[str]]

_InT = TypeVar('_InT')
_OutT = TypeVar('_OutT')
_NewOutT = TypeVar('_NewOutT')

class _Pipeline(Generic[_InT, _OutT]):
    def transform(self, func: Callable) -> _Pipeline[_InT, _NewOutT]:
        ...

    def validate_as(self, tp: Any, *, strict: bool = False) -> Union[_Pipeline[_InT, Any], _Pipeline[_InT, _NewOutT]]:
        ...

    def validate_as_deferred(self, func: Callable) -> _Pipeline[_InT, _NewOutT]:
        ...

    def constrain(self, constraint: _ConstraintAnnotation) -> _Pipeline[_InT, _OutT]:
        ...

    def predicate(self, func: Callable) -> _Pipeline[_InT, _OutT]:
        ...

    def gt(self, gt: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def lt(self, lt: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def ge(self, ge: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def le(self, le: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def len(self, min_len: int, max_len: int = None) -> _Pipeline[_InT, _OutT]:
        ...

    def multiple_of(self, multiple_of: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def eq(self, value: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def not_eq(self, value: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def in_(self, values: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def not_in(self, values: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def datetime_tz_naive(self) -> _Pipeline[_InT, _OutT]:
        ...

    def datetime_tz_aware(self) -> _Pipeline[_InT, _OutT]:
        ...

    def datetime_tz(self, tz: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def datetime_with_tz(self, tz: Any) -> _Pipeline[_InT, _OutT]:
        ...

    def str_lower(self) -> _Pipeline[_InT, _OutT]:
        ...

    def str_upper(self) -> _Pipeline[_InT, _OutT]:
        ...

    def str_title(self) -> _Pipeline[_InT, _OutT]:
        ...

    def str_strip(self) -> _Pipeline[_InT, _OutT]:
        ...

    def str_pattern(self, pattern: str) -> _Pipeline[_InT, _OutT]:
        ...

    def str_contains(self, substring: str) -> _Pipeline[_InT, _OutT]:
        ...

    def str_starts_with(self, prefix: str) -> _Pipeline[_InT, _OutT]:
        ...

    def str_ends_with(self, suffix: str) -> _Pipeline[_InT, _OutT]:
        ...

    def otherwise(self, other: _Pipeline[_InT, _OutT]) -> _Pipeline[_InT, _OutT]:
        ...

    def then(self, other: _Pipeline[_InT, _OutT]) -> _Pipeline[_InT, _OutT]:
        ...

    def __get_pydantic_core_schema__(self, source_type: Any, handler: Any) -> Any:
        ...

    def __supports_type__(self, _: Any) -> None:
        ...
