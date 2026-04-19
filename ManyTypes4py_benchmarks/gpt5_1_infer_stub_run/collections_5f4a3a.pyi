from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional, overload

from hypothesis.strategies._internal.strategies import (
    Ex,
    MappedStrategy,
    SearchStrategy,
    T,
    T3,
    T4,
    T5,
)


class TupleStrategy(SearchStrategy[tuple[Any, ...]]):
    element_strategies: tuple[SearchStrategy[Any], ...]

    def __init__(self, strategies: Iterable[SearchStrategy[Any]]) -> None: ...
    def do_validate(self) -> None: ...
    def calc_label(self) -> Any: ...
    def __repr__(self) -> str: ...
    def calc_has_reusable_values(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def do_draw(self, data: Any) -> tuple[Any, ...]: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...


@overload
def tuples() -> SearchStrategy[tuple[()]]: ...
@overload
def tuples(__a1: SearchStrategy[T]) -> SearchStrategy[tuple[T]]: ...
@overload
def tuples(__a1: SearchStrategy[T], __a2: SearchStrategy[T3]) -> SearchStrategy[tuple[T, T3]]: ...
@overload
def tuples(
    __a1: SearchStrategy[T], __a2: SearchStrategy[T3], __a3: SearchStrategy[T4]
) -> SearchStrategy[tuple[T, T3, T4]]: ...
@overload
def tuples(
    __a1: SearchStrategy[T],
    __a2: SearchStrategy[T3],
    __a3: SearchStrategy[T4],
    __a4: SearchStrategy[T5],
) -> SearchStrategy[tuple[T, T3, T4, T5]]: ...
@overload
def tuples(
    __a1: SearchStrategy[T],
    __a2: SearchStrategy[T3],
    __a3: SearchStrategy[T4],
    __a4: SearchStrategy[T5],
    __a5: SearchStrategy[Ex],
) -> SearchStrategy[tuple[T, T3, T4, T5, Ex]]: ...
@overload
def tuples(*args: SearchStrategy[Any]) -> SearchStrategy[tuple[Any, ...]]: ...
def tuples(*args: SearchStrategy[Any]) -> SearchStrategy[tuple[Any, ...]]: ...


class ListStrategy(SearchStrategy[list[Any]]):
    _nonempty_filters: tuple[Any, ...]
    min_size: int
    max_size: float
    average_size: float
    element_strategy: SearchStrategy[Any]

    def __init__(
        self,
        elements: SearchStrategy[Any],
        min_size: Optional[int] = 0,
        max_size: Optional[int] = None,
    ) -> None: ...
    def calc_label(self) -> Any: ...
    def do_validate(self) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def do_draw(self, data: Any) -> list[Any]: ...
    def __repr__(self) -> str: ...
    def filter(self, condition: Any) -> SearchStrategy[list[Any]]: ...


class UniqueListStrategy(ListStrategy):
    keys: Iterable[Callable[[Any], Any]]
    tuple_suffixes: Optional[SearchStrategy[tuple[Any, ...]]]

    def __init__(
        self,
        elements: SearchStrategy[Any],
        min_size: Optional[int],
        max_size: Optional[int],
        keys: Iterable[Callable[[Any], Any]],
        tuple_suffixes: Optional[SearchStrategy[tuple[Any, ...]]],
    ) -> None: ...
    def do_draw(self, data: Any) -> list[Any]: ...


class UniqueSampledListStrategy(UniqueListStrategy):
    def do_draw(self, data: Any) -> list[Any]: ...


class FixedKeysDictStrategy(MappedStrategy[dict[Any, Any]]):
    keys: tuple[Any, ...]

    def __init__(self, strategy_dict: Mapping[Any, SearchStrategy[Any]]) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def __repr__(self) -> str: ...


class FixedAndOptionalKeysDictStrategy(SearchStrategy[dict[Any, Any]]):
    required: Mapping[Any, SearchStrategy[Any]]
    fixed: FixedKeysDictStrategy
    optional: Mapping[Any, SearchStrategy[Any]]

    def __init__(
        self,
        strategy_dict: Mapping[Any, SearchStrategy[Any]],
        optional: Mapping[Any, SearchStrategy[Any]],
    ) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def __repr__(self) -> str: ...
    def do_draw(self, data: Any) -> dict[Any, Any]: ...