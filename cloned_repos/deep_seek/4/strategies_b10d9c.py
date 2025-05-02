import sys
import warnings
from collections import abc, defaultdict
from collections.abc import Sequence
from functools import lru_cache
from random import shuffle
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Dict, FrozenSet, Generic, 
    Iterable, List, Literal, Optional, Set, Tuple, Type, TypeVar, 
    Union, cast, overload
)
from hypothesis._settings import HealthCheck, Phase, Verbosity, settings
from hypothesis.control import _current_build_context, current_build_context
from hypothesis.errors import (
    HypothesisException, HypothesisWarning, InvalidArgument, 
    NonInteractiveExampleWarning, UnsatisfiedAssumption
)
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.utils import (
    calc_label_from_cls, calc_label_from_name, combine_labels
)
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import (
    get_pretty_function_description, is_identity_function
)
from hypothesis.strategies._internal.utils import defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier

if TYPE_CHECKING:
    from typing import TypeAlias
    Ex = TypeVar('Ex', covariant=True, default=Any)
else:
    Ex = TypeVar('Ex', covariant=True)

T = TypeVar('T')
T3 = TypeVar('T3')
T4 = TypeVar('T4')
T5 = TypeVar('T5')
MappedFrom = TypeVar('MappedFrom')
MappedTo = TypeVar('MappedTo')
RecurT = Callable[['SearchStrategy[Any]'], Any]
PackT = Callable[[T], T3]
PredicateT = Callable[[T], object]
TransformationsT = Tuple[
    Union[Tuple[Literal['filter'], PredicateT[Any]], 
    Tuple[Literal['map'], PackT[Any, Any]], ...
]

calculating = UniqueIdentifier('calculating')
MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    'another attempted draw in MappedStrategy'
)
FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    'single loop iteration in FilteredStrategy'
)

def recursive_property(
    strategy: 'SearchStrategy[Any]', 
    name: str, 
    default: Any
) -> Any:
    cache_key = 'cached_' + name
    calculation = 'calc_' + name
    force_key = 'force_' + name

    def forced_value(target: 'SearchStrategy[Any]') -> Any:
        try:
            return getattr(target, force_key)
        except AttributeError:
            return getattr(target, cache_key)

    try:
        return forced_value(strategy)
    except AttributeError:
        pass

    mapping: Dict['SearchStrategy[Any]', Any] = {}
    sentinel = object()
    hit_recursion = False

    def recur(strat: 'SearchStrategy[Any]') -> Any:
        nonlocal hit_recursion
        try:
            return forced_value(strat)
        except AttributeError:
            pass
        result = mapping.get(strat, sentinel)
        if result is calculating:
            hit_recursion = True
            return default
        elif result is sentinel:
            mapping[strat] = calculating
            mapping[strat] = getattr(strat, calculation)(recur)
            return mapping[strat]
        return result

    recur(strategy)
    needs_update: Optional[Set['SearchStrategy[Any]']] = None
    if hit_recursion:
        needs_update = set(mapping)
    listeners: Dict['SearchStrategy[Any]', Set['SearchStrategy[Any]']] = defaultdict(set)

    def recur2(strat: 'SearchStrategy[Any]') -> Callable[['SearchStrategy[Any]'], Any]:
        def recur_inner(other: 'SearchStrategy[Any]') -> Any:
            try:
                return forced_value(other)
            except AttributeError:
                pass
            listeners[other].add(strat)
            result = mapping.get(other, sentinel)
            if result is sentinel:
                assert needs_update is not None
                needs_update.add(other)
                mapping[other] = default
                return default
            return result
        return recur_inner

    count = 0
    seen: Set[FrozenSet[Tuple['SearchStrategy[Any]', Any]]] = set()
    while needs_update:
        count += 1
        if count > 50:
            key = frozenset(mapping.items())
            assert key not in seen, (key, name)
            seen.add(key)
        to_update = needs_update
        needs_update = set()
        for strat in to_update:
            new_value = getattr(strat, calculation)(recur2(strat))
            if new_value != mapping[strat]:
                needs_update.update(listeners[strat])
                mapping[strat] = new_value

    for k, v in mapping.items():
        setattr(k, cache_key, v)
    return getattr(strategy, cache_key)

class SearchStrategy(Generic[Ex]):
    supports_find: ClassVar[bool] = True
    validate_called: bool = False
    __label: Optional[int] = None
    __module__: ClassVar[str] = 'hypothesis.strategies'
    LABELS: ClassVar[Dict[Type['SearchStrategy[Any]'], int]] = {}

    def available(self, data: ConjectureData) -> bool:
        return not self.is_empty

    @property
    def is_empty(self) -> bool:
        return recursive_property(self, 'is_empty', True)

    @property
    def has_reusable_values(self) -> bool:
        return recursive_property(self, 'has_reusable_values', True)

    @property
    def is_cacheable(self) -> bool:
        return recursive_property(self, 'is_cacheable', True)

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return True

    def calc_is_empty(self, recur: RecurT) -> bool:
        return False

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        return False

    def example(self) -> Ex:
        if getattr(sys, 'ps1', None) is None:
            warnings.warn(
                'The `.example()` method is good for exploring strategies, but should only be used interactively.',
                NonInteractiveExampleWarning,
                stacklevel=2
            )
        context = _current_build_context.value
        if context is not None:
            if context.data is not None and context.data.depth > 0:
                raise HypothesisException('Using example() inside a strategy definition is a bad idea.')
            else:
                raise HypothesisException('Using example() inside a test function is a bad idea.')
        try:
            return self.__examples.pop()
        except (AttributeError, IndexError):
            self.__examples: List[Ex] = []
        
        from hypothesis.core import given

        @given(self)
        @settings(
            database=None, 
            max_examples=10, 
            deadline=None, 
            verbosity=Verbosity.quiet, 
            phases=(Phase.generate,), 
            suppress_health_check=list(HealthCheck)
        def example_generating_inner_function(ex: Ex) -> None:
            self.__examples.append(ex)
        
        example_generating_inner_function()
        shuffle(self.__examples)
        return self.__examples.pop()

    def map(self, pack: Callable[[Ex], T]) -> 'SearchStrategy[T]':
        if is_identity_function(pack):
            return self  # type: ignore
        return MappedStrategy(self, pack=pack)

    def flatmap(self, expand: Callable[[Ex], 'SearchStrategy[T]']) -> 'SearchStrategy[T]':
        from hypothesis.strategies._internal.flatmapped import FlatMapStrategy
        return FlatMapStrategy(expand=expand, strategy=self)

    def filter(self, condition: Callable[[Ex], object]) -> 'SearchStrategy[Ex]':
        return FilteredStrategy(conditions=(condition,), strategy=self)

    def _filter_for_filtered_draw(self, condition: Callable[[Ex], object]) -> 'SearchStrategy[Ex]':
        return FilteredStrategy(conditions=(condition,), strategy=self)

    @property
    def branches(self) -> List['SearchStrategy[Ex]']:
        return [self]

    def __or__(self, other: 'SearchStrategy[T]') -> 'SearchStrategy[Union[Ex, T]]':
        if not isinstance(other, SearchStrategy):
            raise ValueError(f'Cannot | a SearchStrategy with {other!r}')
        return OneOfStrategy((self, other))

    def __bool__(self) -> Literal[True]:
        warnings.warn(
            f'bool({self!r}) is always True, did you mean to draw a value?',
            HypothesisWarning,
            stacklevel=2
        )
        return True

    def validate(self) -> None:
        if self.validate_called:
            return
        try:
            self.validate_called = True
            self.do_validate()
            self.is_empty
            self.has_reusable_values
        except Exception:
            self.validate_called = False
            raise

    @property
    def class_label(self) -> int:
        cls = self.__class__
        try:
            return cls.LABELS[cls]
        except KeyError:
            pass
        result = calc_label_from_cls(cls)
        cls.LABELS[cls] = result
        return result

    @property
    def label(self) -> int:
        if self.__label is calculating:
            return 0
        if self.__label is None:
            self.__label = calculating
            self.__label = self.calc_label()
        return cast(int, self.__label)

    def calc_label(self) -> int:
        return self.class_label

    def do_validate(self) -> None:
        pass

    def do_draw(self, data: ConjectureData) -> Ex:
        raise NotImplementedError(f'{type(self).__name__}.do_draw')

def is_simple_data(value: Any) -> bool:
    try:
        hash(value)
        return True
    except TypeError:
        return False

class SampledFromStrategy(SearchStrategy[Ex]):
    _MAX_FILTER_CALLS: ClassVar[int] = 10000

    def __init__(
        self, 
        elements: Iterable[Ex], 
        repr_: Optional[str] = None, 
        transformations: TransformationsT = ()
    ) -> None:
        super().__init__()
        self.elements = cu.check_sample(elements, 'sampled_from')
        assert self.elements
        self.repr_ = repr_
        self._transformations = transformations

    def map(self, pack: Callable[[Ex], T]) -> 'SearchStrategy[T]':
        s = type(self)(
            self.elements, 
            repr_=self.repr_, 
            transformations=(*self._transformations, ('map', pack))
        )
        return cast(SearchStrategy[T], s)

    def filter(self, condition: Callable[[Ex], object]) -> 'SearchStrategy[Ex]':
        return type(self)(
            self.elements, 
            repr_=self.repr_, 
            transformations=(*self._transformations, ('filter', condition))
        )

    def __repr__(self) -> str:
        return (self.repr_ or 'sampled_from([' + ', '.join(map(get_pretty_function_description, self.elements)) + '])') + ''.join(
            (f'.{name}({get_pretty_function_description(f)})' for name, f in self._transformations)
        )

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        return not self._transformations

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return is_simple_data(self.elements)

    def _transform(self, element: Ex) -> Union[Ex, UniqueIdentifier]:
        for name, f in self._transformations:
            if name == 'map':
                f = cast(PackT[Any, Any], f)
                result = f(element)
                if (build_context := _current_build_context.value):
                    build_context.record_call(result, f, [element], {})
                element = cast(Ex, result)
            else:
                assert name == 'filter'
                f = cast(PredicateT[Any], f)
                if not f(element):
                    return filter_not_satisfied
        return element

    def do_draw(self, data: ConjectureData) -> Ex:
        result = self.do_filtered_draw(data)
        if isinstance(result, SearchStrategy) and all(
            isinstance(x, SearchStrategy) for x in self.elements
        ):
            data._sampled_from_all_strategies_elements_message = (
                'sample_from was given a collection of strategies: {!r}. Was one_of intended?',
                self.elements
            )
        if result is filter_not_satisfied:
            data.mark_invalid(f'Aborted test because unable to satisfy {self!r}')
        assert not isinstance(result, UniqueIdentifier)
        return cast(Ex, result)

    def get_element(self, i: int) -> Union[Ex, UniqueIdentifier]:
        return self._transform(self.elements[i])

    def do_filtered_draw(self, data: ConjectureData) -> Union[Ex, UniqueIdentifier]:
        known_bad_indices: Set[int] = set()
        for _ in range(3):
            i = data.draw_integer(0, len(self.elements) - 1)
            if i not in known_bad_indices:
                element = self.get_element(i)
                if element is not filter_not_satisfied:
                    return element
                if not known_bad_indices:
                    data.events[f'Retried draw from {self!r} to satisfy filter'] = ''
                known_bad_indices.add(i)
        
        max_good_indices = len(self.elements) - len(known_bad_indices)
        if not max_good_indices:
            return filter_not_satisfied
        max_good_indices = min(max_good_indices, self._MAX_FILTER_CALLS - 3)
        speculative_index = data.draw_integer(0, max_good_indices - 1)
        allowed: List[Tuple[int, Ex]] = []
        
        for i in range(min(len(self.elements), self._MAX_FILTER_CALLS - 3)):
            if i not in known_bad_indices:
                element = self.get_element(i)
                if element is not filter_not_satisfied:
                    assert not isinstance(element, UniqueIdentifier)
                    allowed.append((i, cast(Ex, element)))
                    if len(allowed) > speculative_index:
                        data.draw_integer(0, len(self.elements) - 1, forced=i)
                        return element
        if allowed:
            i, element = data.choice(allowed)
            data.draw_integer(0, len(self.elements) - 1, forced=i)
            return element
        return filter_not_satisfied

class OneOfStrategy(SearchStrategy[Ex]):
    def __init__(self, strategies: Iterable['SearchStrategy[Ex]']) -> None:
        super().__init__()
        strategies = tuple(strategies)
        self.original_strategies = list(strategies)
        self.__element_strategies: Optional[List['SearchStrategy[Ex]']] = None
        self.__in_branches = False

    def calc_is_empty(self, recur: RecurT) -> bool:
        return all(recur(e) for e in self.original_strategies)

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        return all(recur(e) for e in self.original_strategies)

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return all(recur(e) for e in self.original_strategies)

    @property
    def element_strategies(self) -> List['SearchStrategy[Ex]']:
        if self.__element_strategies is None:
            seen: Set['SearchStrategy[Any]'] = {self}
            strategies: List['SearchStrategy[Ex]'] = []
            for arg in self.original_strategies:
                check_strategy(arg)
                if not arg.is_empty:
                    for s in arg.branches:
                        if s not in seen and not s.is_empty:
                            seen.add(s)
                            strategies.append(s)
            self.__element_strategies = strategies
        return self.__element_strategies

    def calc_label(self) -> int:
        return combine_labels(
            self.class_label, 
            *(p.label for p in self.original_strategies)
        )

    def do_draw(self, data: ConjectureData) -> Ex:
        strategy = data.draw(
            SampledFromStrategy(self.element_strategies).filter(
                lambda s: s.available(data)
            )
        )
        return data.draw(strategy)

    def __repr__(self) -> str:
        return 'one_of(%s)' % ', '.join(map(repr, self.original_strategies))

    def do_validate(self) -> None:
        for e in self.element_strategies:
            e.validate()

    @property
    def branches(self) -> List['SearchStrategy[Ex]']:
        if not self.__in_branches:
            try:
                self.__in_branches = True
                return self.element_strategies
            finally:
                self.__in_branches = False
        else:
            return [self]

    def filter(self, condition: Callable[[Ex], object]) -> 'SearchStrategy[Ex]':
        return FilteredStrategy(
            OneOfStrategy([s.filter(condition) for s in self.original_strategies]), 
            conditions=()
        )

@overload
def one_of(__args: Iterable['SearchStrategy[T]']) -> 'SearchStrategy[T]': ...

@overload
def one_of(__a1: 'SearchStrategy[T]') -> 'SearchStrategy[T]': ...

@overload
def one_of(__a1: 'SearchStrategy[T1]', __a2: 'SearchStrategy[T2]') -> 'SearchStrategy[Union[T1, T2]]': ...

@overload
def one_of(__a1: 'SearchStrategy[T1]', __a2: 'SearchStrategy[T2]', __a3: 'SearchStrategy[T3]') -> 'SearchStrategy[Union[T1, T2, T3]]': ...

@overload
def one_of(__a1: 'SearchStrategy[T1]', __a2: '