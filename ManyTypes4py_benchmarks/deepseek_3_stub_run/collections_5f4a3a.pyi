import copy
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    overload,
)

from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.junkdrawer import LazySequenceCopy
from hypothesis.internal.conjecture.utils import combine_labels
from hypothesis.internal.filtering import get_integer_predicate_bounds
from hypothesis.internal.reflection import is_identity_function
from hypothesis.strategies._internal.strategies import (
    Ex,
    MappedStrategy,
    SearchStrategy,
    T,
    T3,
    T4,
    T5,
    check_strategy,
    filter_not_satisfied,
)
from hypothesis.strategies._internal.utils import cacheable, defines_strategy

_T = TypeVar("_T")
_Ex = TypeVar("_Ex", bound=Ex)
_T_co = TypeVar("_T_co", covariant=True)

class TupleStrategy(SearchStrategy[tuple[Any, ...]]):
    def __init__(self, strategies: Iterable[SearchStrategy[Any]]) -> None: ...
    def do_validate(self) -> None: ...
    def calc_label(self) -> int: ...
    def __repr__(self) -> str: ...
    def calc_has_reusable_values(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def do_draw(self, data: Any) -> tuple[Any, ...]: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    element_strategies: tuple[SearchStrategy[Any], ...]

@overload
def tuples() -> SearchStrategy[tuple[()]]: ...
@overload
def tuples(__a1: SearchStrategy[_T]) -> SearchStrategy[tuple[_T]]: ...
@overload
def tuples(__a1: SearchStrategy[_T], __a2: SearchStrategy[_T]) -> SearchStrategy[tuple[_T, _T]]: ...
@overload
def tuples(
    __a1: SearchStrategy[_T], __a2: SearchStrategy[_T], __a3: SearchStrategy[_T]
) -> SearchStrategy[tuple[_T, _T, _T]]: ...
@overload
def tuples(
    __a1: SearchStrategy[_T],
    __a2: SearchStrategy[_T],
    __a3: SearchStrategy[_T],
    __a4: SearchStrategy[_T],
) -> SearchStrategy[tuple[_T, _T, _T, _T]]: ...
@overload
def tuples(
    __a1: SearchStrategy[_T],
    __a2: SearchStrategy[_T],
    __a3: SearchStrategy[_T],
    __a4: SearchStrategy[_T],
    __a5: SearchStrategy[_T],
) -> SearchStrategy[tuple[_T, _T, _T, _T, _T]]: ...
@overload
def tuples(*args: SearchStrategy[Any]) -> SearchStrategy[tuple[Any, ...]]: ...
@cacheable
@defines_strategy()
def tuples(*args: SearchStrategy[Any]) -> SearchStrategy[tuple[Any, ...]]: ...

class ListStrategy(SearchStrategy[list[_T]]):
    _nonempty_filters: tuple[Callable[..., Any], ...]
    def __init__(
        self,
        elements: SearchStrategy[_T],
        min_size: int = 0,
        max_size: Union[int, float] = float("inf"),
    ) -> None: ...
    def calc_label(self) -> int: ...
    def do_validate(self) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def do_draw(self, data: Any) -> list[_T]: ...
    def __repr__(self) -> str: ...
    def filter(
        self, condition: Callable[[list[_T]], Any]
    ) -> SearchStrategy[list[_T]]: ...
    min_size: int
    max_size: Union[int, float]
    average_size: float
    element_strategy: SearchStrategy[_T]

class UniqueListStrategy(ListStrategy[_T]):
    def __init__(
        self,
        elements: SearchStrategy[_T],
        min_size: int,
        max_size: Union[int, float],
        keys: tuple[Callable[[_T], Any], ...],
        tuple_suffixes: Optional[SearchStrategy[tuple[Any, ...]]],
    ) -> None: ...
    def do_draw(self, data: Any) -> list[_T]: ...
    keys: tuple[Callable[[_T], Any], ...]
    tuple_suffixes: Optional[SearchStrategy[tuple[Any, ...]]]

class UniqueSampledListStrategy(UniqueListStrategy[_T]):
    def do_draw(self, data: Any) -> list[_T]: ...

class FixedKeysDictStrategy(MappedStrategy[dict[str, Any]]):
    def __init__(self, strategy_dict: dict[str, SearchStrategy[Any]]) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def __repr__(self) -> str: ...
    keys: tuple[str, ...]

class FixedAndOptionalKeysDictStrategy(SearchStrategy[dict[str, Any]]):
    def __init__(
        self,
        strategy_dict: dict[str, SearchStrategy[Any]],
        optional: dict[str, SearchStrategy[Any]],
    ) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def __repr__(self) -> str: ...
    def do_draw(self, data: Any) -> dict[str, Any]: ...
    required: dict[str, SearchStrategy[Any]]
    fixed: FixedKeysDictStrategy
    optional: dict[str, SearchStrategy[Any]]