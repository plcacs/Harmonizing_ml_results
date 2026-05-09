from collections.abc import Iterable
from typing import Any, overload, Tuple, List, Dict, Callable, Optional

class TupleStrategy(SearchStrategy):
    """A strategy responsible for fixed length tuples based on heterogeneous
    strategies for each of their elements."""

    def __init__(self, strategies: Tuple[SearchStrategy, ...]) -> None:
        super().__init__()
        self.element_strategies: Tuple[SearchStrategy, ...] = strategies

    # ...

@overload
def tuples(*args: SearchStrategy) -> SearchStrategy:
    ...

# ...

@cacheable
@defines_strategy()
def tuples(*args: SearchStrategy) -> SearchStrategy:
    """Return a strategy which generates a tuple of the same length as args by
    generating the value at index i from args[i].

    e.g. tuples(integers(), integers()) would generate a tuple of length
    two with both values an integer.

    Examples from this strategy shrink by shrinking their component parts.
    """
    for arg in args:
        check_strategy(arg)
    return TupleStrategy(args)

class ListStrategy(SearchStrategy):
    """A strategy for lists which takes a strategy for its elements and the
    allowed lengths, and generates lists with the correct size and contents."""

    _nonempty_filters: Tuple[Callable[[Any], bool], ...] = (bool, len, tuple, list)

    def __init__(self, elements: SearchStrategy, min_size: int = 0, max_size: int = float('inf')) -> None:
        super().__init__()
        self.min_size: int = min_size or 0
        self.max_size: int = max_size if max_size is not None else float('inf')
        assert 0 <= self.min_size <= self.max_size
        self.average_size: float = min(max(self.min_size * 2, self.min_size + 5), 0.5 * (self.min_size + self.max_size))
        self.element_strategy: SearchStrategy = elements
        if min_size > BUFFER_SIZE:
            raise InvalidArgument(f'{self!r} can never generate an example, because min_size is larger than Hypothesis supports.  Including it is at best slowing down your tests for no benefit; at worst making them fail (maybe flakily) with a HealthCheck error.')

    # ...

class UniqueListStrategy(ListStrategy):
    # ...

    def do_draw(self, data: Any) -> List[Any]:
        # ...

class UniqueSampledListStrategy(UniqueListStrategy):
    # ...

class FixedKeysDictStrategy(MappedStrategy):
    """A strategy which produces dicts with a fixed set of keys, given a
    strategy for each of their equivalent values.

    e.g. {'foo' : some_int_strategy} would generate dicts with the single
    key 'foo' mapping to some integer.
    """

    def __init__(self, strategy_dict: Dict[Any, SearchStrategy]) -> None:
        dict_type: type = type(strategy_dict)
        self.keys: Tuple[Any, ...] = tuple(strategy_dict.keys())
        super().__init__(strategy=TupleStrategy((strategy_dict[k] for k in self.keys)), pack=lambda value: dict_type(zip(self.keys, value)))

    # ...

class FixedAndOptionalKeysDictStrategy(SearchStrategy):
    """A strategy which produces dicts with a fixed set of keys, given a
    strategy for each of their equivalent values.

    e.g. {'foo' : some_int_strategy} would generate dicts with the single
    key 'foo' mapping to some integer.
    """

    def __init__(self, strategy_dict: Dict[Any, SearchStrategy], optional: Dict[Any, SearchStrategy]) -> None:
        self.required: Dict[Any, SearchStrategy] = strategy_dict
        self.fixed: FixedKeysDictStrategy = FixedKeysDictStrategy(strategy_dict)
        self.optional: Dict[Any, SearchStrategy] = optional

    # ...
