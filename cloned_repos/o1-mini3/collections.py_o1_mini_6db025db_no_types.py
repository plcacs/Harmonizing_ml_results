import copy
from collections.abc import Iterable, Callable, MutableMapping
from typing import Any, overload, Tuple, Dict, Optional, TypeVar, Generic
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.engine import BUFFER_SIZE
from hypothesis.internal.conjecture.junkdrawer import LazySequenceCopy
from hypothesis.internal.conjecture.utils import combine_labels
from hypothesis.internal.filtering import get_integer_predicate_bounds
from hypothesis.internal.reflection import is_identity_function
from hypothesis.strategies._internal.strategies import T3, T4, T5, Ex, MappedStrategy, SearchStrategy, T, check_strategy, filter_not_satisfied
from hypothesis.strategies._internal.utils import cacheable, defines_strategy

class TupleStrategy(SearchStrategy, Generic[Any]):
    """A strategy responsible for fixed length tuples based on heterogeneous
    strategies for each of their elements."""

    def __init__(self, strategies):
        super().__init__()
        self.element_strategies: Tuple[SearchStrategy[Any], ...] = tuple(strategies)

    def do_validate(self):
        for s in self.element_strategies:
            s.validate()

    def calc_label(self):
        return combine_labels(self.class_label, *(s.label for s in self.element_strategies))

    def __repr__(self):
        tuple_string = ', '.join(map(repr, self.element_strategies))
        return f'TupleStrategy(({tuple_string}))'

    def calc_has_reusable_values(self, recur):
        return all((recur(e) for e in self.element_strategies))

    def do_draw(self, data):
        return tuple((data.draw(e) for e in self.element_strategies))

    def calc_is_empty(self, recur):
        return any((recur(e) for e in self.element_strategies))

@overload
def tuples():
    ...

@overload
def tuples(__a1):
    ...

@overload
def tuples(__a1, __a2):
    ...

@overload
def tuples(__a1, __a2, __a3):
    ...

@overload
def tuples(__a1, __a2, __a3, __a4):
    ...

@overload
def tuples(__a1, __a2, __a3, __a4, __a5):
    ...

@overload
def tuples(*args: SearchStrategy[Any]):
    ...

@cacheable
@defines_strategy()
def tuples(*args: SearchStrategy[Any]):
    """Return a strategy which generates a tuple of the same length as args by
    generating the value at index i from args[i].

    e.g. tuples(integers(), integers()) would generate a tuple of length
    two with both values an integer.

    Examples from this strategy shrink by shrinking their component parts.
    """
    for arg in args:
        check_strategy(arg)
    return TupleStrategy(args)

class ListStrategy(SearchStrategy, Generic[Any]):
    """A strategy for lists which takes a strategy for its elements and the
    allowed lengths, and generates lists with the correct size and contents."""
    _nonempty_filters: Tuple[type, ...] = (bool, len, tuple, list)

    def __init__(self, elements, min_size=0, max_size=None):
        super().__init__()
        self.min_size: int = min_size or 0
        self.max_size: float = float(max_size) if max_size is not None else float('inf')
        assert 0 <= self.min_size <= self.max_size
        self.average_size: float = min(max(self.min_size * 2, self.min_size + 5), 0.5 * (self.min_size + self.max_size))
        self.element_strategy: SearchStrategy[Any] = elements
        if self.min_size > BUFFER_SIZE:
            raise InvalidArgument(f'{self!r} can never generate an example, because min_size is larger than Hypothesis supports.  Including it is at best slowing down your tests for no benefit; at worst making them fail (maybe flakily) with a HealthCheck error.')

    def calc_label(self):
        return combine_labels(self.class_label, self.element_strategy.label)

    def do_validate(self):
        self.element_strategy.validate()
        if self.is_empty:
            raise InvalidArgument(f'Cannot create non-empty lists with elements drawn from strategy {self.element_strategy!r} because it has no values.')
        if self.element_strategy.is_empty and 0 < self.max_size < float('inf'):
            raise InvalidArgument(f'Cannot create a collection of max_size={self.max_size!r}, because no elements can be drawn from the element strategy {self.element_strategy!r}')

    def calc_is_empty(self, recur):
        if self.min_size == 0:
            return False
        else:
            return recur(self.element_strategy)

    def do_draw(self, data):
        if self.element_strategy.is_empty:
            assert self.min_size == 0
            return []
        elements = cu.many(data, min_size=self.min_size, max_size=self.max_size, average_size=self.average_size)
        result: list[Any] = []
        while elements.more():
            result.append(data.draw(self.element_strategy))
        return result

    def __repr__(self):
        return f'{self.__class__.__name__}({self.element_strategy!r}, min_size={self.min_size:_}, max_size={self.max_size:_})'

    def filter(self, condition):
        if condition in self._nonempty_filters or is_identity_function(condition):
            assert self.max_size >= 1, 'Always-empty is special cased in st.lists()'
            if self.min_size >= 1:
                return self
            new = copy.copy(self)
            new.min_size = 1
            return new
        kwargs, pred = get_integer_predicate_bounds(condition)
        if kwargs.get('len') and ('min_value' in kwargs or 'max_value' in kwargs):
            new = copy.copy(self)
            new.min_size = max(self.min_size, kwargs.get('min_value', self.min_size))
            new.max_size = min(self.max_size, kwargs.get('max_value', self.max_size))
            if new.min_size > new.max_size:
                return SearchStrategy.filter(self, condition)
            new.average_size = min(max(new.min_size * 2, new.min_size + 5), 0.5 * (new.min_size + new.max_size))
            if pred is None:
                return new
            return SearchStrategy.filter(new, condition)
        return SearchStrategy.filter(self, condition)

class UniqueListStrategy(ListStrategy, Generic[Any]):

    def __init__(self, elements, min_size, max_size, keys, tuple_suffixes):
        super().__init__(elements, min_size, max_size)
        self.keys: Tuple[Callable[[Any], Any], ...] = tuple(keys)
        self.tuple_suffixes: Optional[SearchStrategy[Any]] = tuple_suffixes

    def do_draw(self, data):
        if self.element_strategy.is_empty:
            assert self.min_size == 0
            return []
        elements = cu.many(data, min_size=self.min_size, max_size=self.max_size, average_size=self.average_size)
        seen_sets: Tuple[set[Any], ...] = tuple((set() for _ in self.keys))
        result: list[Any] = []

        def not_yet_in_unique_list(val):
            return all((key(val) not in seen for key, seen in zip(self.keys, seen_sets)))
        filtered = self.element_strategy._filter_for_filtered_draw(not_yet_in_unique_list)
        while elements.more():
            value = filtered.do_filtered_draw(data)
            if value is filter_not_satisfied:
                elements.reject(f'Aborted test because unable to satisfy {filtered!r}')
            else:
                for key, seen in zip(self.keys, seen_sets):
                    seen.add(key(value))
                if self.tuple_suffixes is not None:
                    suffix = data.draw(self.tuple_suffixes)
                    value = (value, suffix)
                result.append(value)
        assert self.max_size >= len(result) >= self.min_size
        return result

class UniqueSampledListStrategy(UniqueListStrategy, Generic[Any]):

    def do_draw(self, data):
        should_draw = cu.many(data, min_size=self.min_size, max_size=self.max_size, average_size=self.average_size)
        seen_sets: Tuple[set[Any], ...] = tuple((set() for _ in self.keys))
        result: list[Any] = []
        remaining = LazySequenceCopy(self.element_strategy.elements)
        while remaining and should_draw.more():
            j = data.draw_integer(0, len(remaining) - 1)
            transformed = self.element_strategy._transform(remaining.pop(j))
            value = transformed
            if value is not filter_not_satisfied and all((key(value) not in seen for key, seen in zip(self.keys, seen_sets))):
                for key, seen in zip(self.keys, seen_sets):
                    seen.add(key(value))
                if self.tuple_suffixes is not None:
                    suffix = data.draw(self.tuple_suffixes)
                    value = (value, suffix)
                result.append(value)
            else:
                should_draw.reject('UniqueSampledListStrategy filter not satisfied or value already seen')
        assert self.max_size >= len(result) >= self.min_size
        return result

class FixedKeysDictStrategy(MappedStrategy, Generic[Any, Any]):
    """A strategy which produces dicts with a fixed set of keys, given a
    strategy for each of their equivalent values.

    e.g. {'foo' : some_int_strategy} would generate dicts with the single
    key 'foo' mapping to some integer.
    """

    def __init__(self, strategy_dict):
        self.keys: Tuple[Any, ...] = tuple(strategy_dict.keys())
        super().__init__(strategy=TupleStrategy((strategy_dict[k] for k in self.keys)), pack=lambda value: dict(zip(self.keys, value)))

    def calc_is_empty(self, recur):
        return recur(self.mapped_strategy)

    def __repr__(self):
        return f'FixedKeysDictStrategy({self.keys!r}, {self.mapped_strategy!r})'

class FixedAndOptionalKeysDictStrategy(SearchStrategy, Generic[Any, Any]):
    """A strategy which produces dicts with a fixed set of keys, given a
    strategy for each of their equivalent values.

    e.g. {'foo' : some_int_strategy} would generate dicts with the single
    key 'foo' mapping to some integer.
    """

    def __init__(self, strategy_dict, optional):
        self.required: Dict[Any, SearchStrategy[Any]] = strategy_dict
        self.fixed: FixedKeysDictStrategy[Any, Any] = FixedKeysDictStrategy(strategy_dict)
        self.optional: Dict[Any, SearchStrategy[Any]] = optional

    def calc_is_empty(self, recur):
        return recur(self.fixed)

    def __repr__(self):
        return f'FixedAndOptionalKeysDictStrategy({self.required!r}, {self.optional!r})'

    def do_draw(self, data):
        result: Dict[Any, Any] = data.draw(self.fixed)
        remaining: list[Any] = [k for k, v in self.optional.items() if not v.is_empty]
        should_draw = cu.many(data, min_size=0, max_size=len(remaining), average_size=len(remaining) / 2)
        while should_draw.more():
            j = data.draw_integer(0, len(remaining) - 1)
            remaining[-1], remaining[j] = (remaining[j], remaining[-1])
            key = remaining.pop()
            result[key] = data.draw(self.optional[key])
        return result