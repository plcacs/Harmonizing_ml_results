```python
from collections.abc import Callable, Iterable
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    overload,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from hypothesis.internal.conjecture import utils as cu
    from hypothesis.internal.conjecture.junkdrawer import LazySequenceCopy
    from hypothesis.strategies._internal.strategies import (
        T,
        T3,
        T4,
        T5,
        Ex,
        SearchStrategy,
        check_strategy,
        filter_not_satisfied,
    )

T = TypeVar("T")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
Ex = TypeVar("Ex")

class TupleStrategy(SearchStrategy[tuple[Any, ...]]):
    element_strategies: tuple[SearchStrategy[Any], ...]
    
    def __init__(self, strategies: Iterable[SearchStrategy[Any]]) -> None: ...
    def do_validate(self) -> None: ...
    def calc_label(self) -> int: ...
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
def tuples(__a1: SearchStrategy[T], __a2: SearchStrategy[T3], __a3: SearchStrategy[T4]) -> SearchStrategy[tuple[T, T3, T4]]: ...
@overload
def tuples(__a1: SearchStrategy[T], __a2: SearchStrategy[T3], __a3: SearchStrategy[T4], __a4: SearchStrategy[T5]) -> SearchStrategy[tuple[T, T3, T4, T5]]: ...
@overload
def tuples(*args: SearchStrategy[Any]) -> SearchStrategy[tuple[Any, ...]]: ...
def tuples(*args: SearchStrategy[Any]) -> SearchStrategy[tuple[Any, ...]]: ...

class ListStrategy(SearchStrategy[list[T]]):
    min_size: int
    max_size: float
    average_size: float
    element_strategy: SearchStrategy[T]
    _nonempty_filters: tuple[Any, ...]
    
    def __init__(
        self,
        elements: SearchStrategy[T],
        min_size: int = 0,
        max_size: Optional[float] = None,
    ) -> None: ...
    def calc_label(self) -> int: ...
    def do_validate(self) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def do_draw(self, data: Any) -> list[T]: ...
    def __repr__(self) -> str: ...
    def filter(self, condition: Callable[[Any], Any]) -> SearchStrategy[list[T]]: ...

class UniqueListStrategy(ListStrategy[T]):
    keys: tuple[Callable[[Any], Any], ...]
    tuple_suffixes: Optional[SearchStrategy[Any]]
    
    def __init__(
        self,
        elements: SearchStrategy[T],
        min_size: int,
        max_size: float,
        keys: tuple[Callable[[Any], Any], ...],
        tuple_suffixes: Optional[SearchStrategy[Any]],
    ) -> None: ...
    def do_draw(self, data: Any) -> list[T]: ...

class UniqueSampledListStrategy(UniqueListStrategy[T]):
    def do_draw(self, data: Any) -> list[T]: ...

class FixedKeysDictStrategy(MappedStrategy[dict[str, Any]]):
    keys: tuple[str, ...]
    
    def __init__(self, strategy_dict: dict[str, SearchStrategy[Any]]) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def __repr__(self) -> str: ...

class FixedAndOptionalKeysDictStrategy(SearchStrategy[dict[str, Any]]):
    required: dict[str, SearchStrategy[Any]]
    fixed: FixedKeysDictStrategy
    optional: dict[str, SearchStrategy[Any]]
    
    def __init__(
        self,
        strategy_dict: dict[str, SearchStrategy[Any]],
        optional: dict[str, SearchStrategy[Any]],
    ) -> None: ...
    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool: ...
    def __repr__(self) -> str: ...
    def do_draw(self, data: Any) -> dict[str, Any]: ...
```