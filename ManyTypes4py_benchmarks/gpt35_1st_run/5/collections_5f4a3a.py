from typing import Any, Tuple, overload

class TupleStrategy(SearchStrategy):
    def __init__(self, strategies: Tuple[SearchStrategy, ...]) -> None:
    def calc_label(self) -> Any:
    def __repr__(self) -> str:
    def calc_has_reusable_values(self, recur: Any) -> bool:
    def do_draw(self, data: Any) -> Tuple:
    def calc_is_empty(self, recur: Any) -> bool

@overload
def tuples() -> Any:
@overload
def tuples(__a1: Any) -> Any:
@overload
def tuples(__a1: Any, __a2: Any) -> Any:
@overload
def tuples(__a1: Any, __a2: Any, __a3: Any) -> Any:
@overload
def tuples(__a1: Any, __a2: Any, __a3: Any, __a4: Any) -> Any:
@overload
def tuples(__a1: Any, __a2: Any, __a3: Any, __a4: Any, __a5: Any) -> Any:
@overload
def tuples(*args: Any) -> Any:

@cacheable
@defines_strategy()
def tuples(*args: SearchStrategy) -> TupleStrategy:

class ListStrategy(SearchStrategy):
    def __init__(self, elements: SearchStrategy, min_size: int = 0, max_size: float = float('inf')) -> None:
    def calc_label(self) -> Any:
    def do_validate(self) -> None:
    def calc_is_empty(self, recur: Any) -> bool:
    def do_draw(self, data: Any) -> list:
    def __repr__(self) -> str:
    def filter(self, condition: Any) -> Any:

class UniqueListStrategy(ListStrategy):
    def __init__(self, elements: SearchStrategy, min_size: int, max_size: float, keys: Tuple, tuple_suffixes: Any) -> None:
    def do_draw(self, data: Any) -> list:

class UniqueSampledListStrategy(UniqueListStrategy):
    def do_draw(self, data: Any) -> list:

class FixedKeysDictStrategy(MappedStrategy):
    def __init__(self, strategy_dict: dict) -> None:
    def calc_is_empty(self, recur: Any) -> bool:
    def __repr__(self) -> str:

class FixedAndOptionalKeysDictStrategy(SearchStrategy):
    def __init__(self, strategy_dict: dict, optional: dict) -> None:
    def calc_is_empty(self, recur: Any) -> bool:
    def __repr__(self) -> str:
    def do_draw(self, data: Any) -> dict:
