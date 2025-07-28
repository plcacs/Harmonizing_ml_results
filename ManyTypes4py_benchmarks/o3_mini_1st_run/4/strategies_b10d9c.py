#!/usr/bin/env python3
"""
Annotated version of the provided Python program.
"""

import sys
import warnings
from collections import abc, defaultdict
from collections.abc import Sequence
from functools import lru_cache
from random import shuffle
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    Sequence as TypingSequence,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    Dict,
)
from hypothesis._settings import HealthCheck, Phase, Verbosity, settings
from hypothesis.control import _current_build_context, current_build_context
from hypothesis.errors import (
    HypothesisException,
    HypothesisWarning,
    InvalidArgument,
    NonInteractiveExampleWarning,
    UnsatisfiedAssumption,
)
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.utils import calc_label_from_cls, calc_label_from_name, combine_labels
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import get_pretty_function_description, is_identity_function
from hypothesis.strategies._internal.utils import defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier

if __import__("typing").TYPE_CHECKING:
    from typing import TypeAlias
    ExType: TypeVar = TypeVar('Ex', covariant=True, default=Any)
else:
    ExType = TypeVar('Ex', covariant=True)

T = TypeVar('T')
T3 = TypeVar('T3')
T4 = TypeVar('T4')
T5 = TypeVar('T5')
MappedFrom = TypeVar('MappedFrom')
MappedTo = TypeVar('MappedTo')


TransformationsT = tuple[
    Union[tuple[Literal['filter'], Callable[[Any], object]],
          tuple[Literal['map'], Callable[[Any], Any]]],
    ...
]

calculating: UniqueIdentifier = UniqueIdentifier('calculating')
MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL: int = calc_label_from_name('another attempted draw in MappedStrategy')
FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL: int = calc_label_from_name('single loop iteration in FilteredStrategy')


def recursive_property(strategy: "SearchStrategy[Any]", name: str, default: bool) -> bool:
    cache_key: str = 'cached_' + name
    calculation: str = 'calc_' + name
    force_key: str = 'force_' + name

    def forced_value(target: "SearchStrategy[Any]") -> Any:
        try:
            return getattr(target, force_key)
        except AttributeError:
            return getattr(target, cache_key)

    try:
        return forced_value(strategy)
    except AttributeError:
        pass

    mapping: Dict["SearchStrategy[Any]", Any] = {}
    sentinel: object = object()
    hit_recursion: bool = False

    def recur(strat: "SearchStrategy[Any]") -> Any:
        nonlocal hit_recursion
        try:
            return forced_value(strat)
        except AttributeError:
            pass
        result: Any = mapping.get(strat, sentinel)
        if result is calculating:
            hit_recursion = True
            return default
        elif result is sentinel:
            mapping[strat] = calculating
            mapping[strat] = getattr(strat, calculation)(recur)
            return mapping[strat]
        return result

    recur(strategy)
    if hit_recursion:
        needs_update: set["SearchStrategy[Any]"] = set(mapping)
        listeners: defaultdict["SearchStrategy[Any]", set["SearchStrategy[Any]"]] = defaultdict(set)
    else:
        needs_update = None

    def recur2(strat: "SearchStrategy[Any]") -> Callable[["SearchStrategy[Any]"], Any]:
        def recur_inner(other: "SearchStrategy[Any]") -> Any:
            try:
                return forced_value(other)
            except AttributeError:
                pass
            listeners[other].add(strat)
            result: Any = mapping.get(other, sentinel)
            if result is sentinel:
                assert needs_update is not None
                needs_update.add(other)
                mapping[other] = default
                return default
            return result
        return recur_inner

    count: int = 0
    seen: set[Any] = set()
    while needs_update:
        count += 1
        if count > 50:
            key = frozenset(mapping.items())
            assert key not in seen, (key, name)
            seen.add(key)
        to_update: set["SearchStrategy[Any]"] = needs_update
        needs_update = set()
        for strat in to_update:
            new_value: Any = getattr(strat, calculation)(recur2(strat))
            if new_value != mapping[strat]:
                needs_update.update(list(listeners[strat]))
                mapping[strat] = new_value
    for k, v in mapping.items():
        setattr(k, cache_key, v)
    return getattr(strategy, cache_key)


class SearchStrategy(Generic[ExType]):
    """A SearchStrategy is an object that knows how to explore data of a given type.
    """

    supports_find: bool = True
    validate_called: bool = False
    __label: Union[int, UniqueIdentifier, None] = None
    __module__ = 'hypothesis.strategies'

    def available(self, data: ConjectureData) -> bool:
        """Returns whether this strategy can currently draw any values."""
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

    def calc_is_cacheable(self, recur: Callable[["SearchStrategy[Any]"], bool]) -> bool:
        return True

    def calc_is_empty(self, recur: Callable[["SearchStrategy[Any]"], bool]) -> bool:
        return False

    def calc_has_reusable_values(self, recur: Callable[["SearchStrategy[Any]"], bool]) -> bool:
        return False

    def example(self) -> ExType:
        """Provide an example of the sort of value that this strategy generates."""
        if getattr(sys, 'ps1', None) is None:
            warnings.warn(
                'The `.example()` method is good for exploring strategies, but should only be used interactively.  We recommend using `@given` for tests - it performs better, saves and replays failures to avoid flakiness, and reports minimal examples. (strategy: %r)'
                % (self,), NonInteractiveExampleWarning, stacklevel=2)
        context = _current_build_context.value
        if context is not None:
            if context.data is not None and context.data.depth > 0:
                raise HypothesisException('Using example() inside a strategy definition is a bad idea. Instead consider using hypothesis.strategies.builds() or @hypothesis.strategies.composite to define your strategy. See https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.builds or https://hypothesis.readthedocs.io/en/latest/data.html#composite-strategies for more details.')
            else:
                raise HypothesisException('Using example() inside a test function is a bad idea. Instead consider using hypothesis.strategies.data() to draw more examples during testing. See https://hypothesis.readthedocs.io/en/latest/data.html#drawing-interactively-in-tests for more details.')
        try:
            return self.__examples.pop()  # type: ignore
        except (AttributeError, IndexError):
            self.__examples = []  # type: ignore
        from hypothesis.core import given

        @given(self)
        @settings(database=None, max_examples=10, deadline=None, verbosity=Verbosity.quiet,
                  phases=(Phase.generate,), suppress_health_check=list(HealthCheck))
        def example_generating_inner_function(ex: ExType) -> None:
            self.__examples.append(ex)  # type: ignore

        example_generating_inner_function()
        shuffle(self.__examples)
        return self.__examples.pop()  # type: ignore

    def map(self, pack: Callable[[ExType], MappedTo]) -> "SearchStrategy[MappedTo]":
        """Returns a new strategy that generates values by mapping pack over values from this strategy."""
        if is_identity_function(pack):
            return self  # type: ignore
        return MappedStrategy(self, pack=pack)

    def flatmap(self, expand: Callable[[ExType], "SearchStrategy[T]"]) -> "SearchStrategy[T]":
        """Returns a new strategy that applies expand to values drawn from this strategy."""
        from hypothesis.strategies._internal.flatmapped import FlatMapStrategy
        return FlatMapStrategy(expand=expand, strategy=self)

    def filter(self, condition: Callable[[ExType], object]) -> "SearchStrategy[ExType]":
        """Returns a new strategy that only generates values satisfying the condition."""
        return FilteredStrategy(self, conditions=(condition,))

    def _filter_for_filtered_draw(self, condition: Callable[[ExType], object]) -> "SearchStrategy[ExType]":
        return FilteredStrategy(self, conditions=(condition,))

    @property
    def branches(self) -> list["SearchStrategy[ExType]"]:
        return [self]

    def __or__(self, other: "SearchStrategy[Any]") -> "SearchStrategy[Any]":
        """Return a strategy which produces values by randomly drawing from one of this strategy or the other."""
        if not isinstance(other, SearchStrategy):
            raise ValueError(f'Cannot | a SearchStrategy with {other!r}')
        return OneOfStrategy((self, other))

    def __bool__(self) -> bool:
        warnings.warn(f'bool({self!r}) is always True, did you mean to draw a value?', HypothesisWarning, stacklevel=2)
        return True

    def validate(self) -> None:
        """Throw an exception if the strategy is not valid."""
        if self.validate_called:
            return
        try:
            self.validate_called = True
            self.do_validate()
            _ = self.is_empty
            _ = self.has_reusable_values
        except Exception:
            self.validate_called = False
            raise

    LABELS: ClassVar[Dict[Type, int]] = {}

    @property
    def class_label(self) -> int:
        cls: Type[Any] = self.__class__
        try:
            return cls.LABELS[cls]
        except KeyError:
            pass
        result: int = calc_label_from_cls(cls)
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

    def do_draw(self, data: ConjectureData) -> ExType:
        raise NotImplementedError(f'{type(self).__name__}.do_draw')


def is_simple_data(value: Any) -> bool:
    try:
        hash(value)
        return True
    except TypeError:
        return False


class SampledFromStrategy(SearchStrategy[ExType]):
    """A strategy which samples from a set of elements."""
    _MAX_FILTER_CALLS: int = 10000

    def __init__(self, elements: TypingSequence[ExType], repr_: Optional[str] = None, transformations: TransformationsT = ()) -> None:
        super().__init__()
        self.elements: TypingSequence[ExType] = cu.check_sample(elements, 'sampled_from')
        assert self.elements
        self.repr_: Optional[str] = repr_
        self._transformations: TransformationsT = transformations

    def map(self, pack: Callable[[ExType], T]) -> "SearchStrategy[T]":
        s = type(self)(self.elements, repr_=self.repr_, transformations=(*self._transformations, ('map', pack)))
        return cast(SearchStrategy[T], s)

    def filter(self, condition: Callable[[ExType], object]) -> "SearchStrategy[ExType]":
        return type(self)(self.elements, repr_=self.repr_, transformations=(*self._transformations, ('filter', condition)))

    def __repr__(self) -> str:
        base: str = self.repr_ or ('sampled_from([' +
                                      ', '.join(map(get_pretty_function_description, self.elements)) +
                                      '])')
        transforms: str = ''.join(
            (f'.{name}({get_pretty_function_description(f)})' for name, f in self._transformations)
        )
        return base + transforms

    def calc_has_reusable_values(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return not self._transformations

    def calc_is_cacheable(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return is_simple_data(self.elements)

    def _transform(self, element: ExType) -> Any:
        for name, f in self._transformations:
            if name == 'map':
                f_cast = cast(Callable[[ExType], Any], f)
                result: Any = f_cast(element)
                if (build_context := _current_build_context.value) is not None:
                    build_context.record_call(result, f_cast, [element], {})
                element = result
            else:
                # name == 'filter'
                f_cast = cast(Callable[[ExType], object], f)
                if not f_cast(element):
                    return filter_not_satisfied
        return element

    def do_draw(self, data: ConjectureData) -> ExType:
        result: Any = self.do_filtered_draw(data)
        if isinstance(result, SearchStrategy) and all((isinstance(x, SearchStrategy) for x in self.elements)):
            data._sampled_from_all_strategies_elements_message = (
                'sample_from was given a collection of strategies: {!r}. Was one_of intended?',
                self.elements
            )
        if result is filter_not_satisfied:
            data.mark_invalid(f'Aborted test because unable to satisfy {self!r}')
        assert not isinstance(result, UniqueIdentifier)
        return result

    def get_element(self, i: int) -> Any:
        return self._transform(self.elements[i])

    def do_filtered_draw(self, data: ConjectureData) -> Any:
        known_bad_indices: set[int] = set()
        for _ in range(3):
            i: int = data.draw_integer(0, len(self.elements) - 1)
            if i not in known_bad_indices:
                element: Any = self.get_element(i)
                if element is not filter_not_satisfied:
                    return element
                if not known_bad_indices:
                    data.events[f'Retried draw from {self!r} to satisfy filter'] = ''
                known_bad_indices.add(i)
        max_good_indices: int = len(self.elements) - len(known_bad_indices)
        if not max_good_indices:
            return filter_not_satisfied
        max_good_indices = min(max_good_indices, self._MAX_FILTER_CALLS - 3)
        speculative_index: int = data.draw_integer(0, max_good_indices - 1)
        allowed: list[tuple[int, Any]] = []
        for i in range(min(len(self.elements), self._MAX_FILTER_CALLS - 3)):
            if i not in known_bad_indices:
                element = self.get_element(i)
                if element is not filter_not_satisfied:
                    allowed.append((i, element))
                    if len(allowed) > speculative_index:
                        data.draw_integer(0, len(self.elements) - 1, forced=i)
                        return element
        if allowed:
            i, element = data.choice(allowed)
            data.draw_integer(0, len(self.elements) - 1, forced=i)
            return element
        return filter_not_satisfied


class OneOfStrategy(SearchStrategy[Any]):
    """Implements a union of strategies."""
    def __init__(self, strategies: TypingSequence[SearchStrategy[Any]]) -> None:
        super().__init__()
        strategies_tuple = tuple(strategies)
        self.original_strategies: list[SearchStrategy[Any]] = list(strategies_tuple)
        self.__element_strategies: Optional[list[SearchStrategy[Any]]] = None
        self.__in_branches: bool = False

    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return all((recur(e) for e in self.original_strategies))

    def calc_has_reusable_values(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return all((recur(e) for e in self.original_strategies))

    def calc_is_cacheable(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return all((recur(e) for e in self.original_strategies))

    @property
    def element_strategies(self) -> list[SearchStrategy[Any]]:
        if self.__element_strategies is None:
            seen: set[SearchStrategy[Any]] = {self}
            strategies: list[SearchStrategy[Any]] = []
            for arg in self.original_strategies:
                check_strategy(arg)
                if not arg.is_empty:
                    for s in arg.branches:
                        if s not in seen and (not s.is_empty):
                            seen.add(s)
                            strategies.append(s)
            self.__element_strategies = strategies
        return self.__element_strategies

    def calc_label(self) -> int:
        return combine_labels(self.class_label, *(p.label for p in self.original_strategies))

    def do_draw(self, data: ConjectureData) -> Any:
        strategy: SearchStrategy[Any] = data.draw(
            SampledFromStrategy(self.element_strategies).filter(lambda s: s.available(data))
        )
        return data.draw(strategy)

    def __repr__(self) -> str:
        return 'one_of(%s)' % ', '.join(map(repr, self.original_strategies))

    def do_validate(self) -> None:
        for e in self.element_strategies:
            e.validate()

    @property
    def branches(self) -> list[SearchStrategy[Any]]:
        if not self.__in_branches:
            try:
                self.__in_branches = True
                return self.element_strategies
            finally:
                self.__in_branches = False
        else:
            return [self]

    def filter(self, condition: Callable[[Any], object]) -> "SearchStrategy[Any]":
        return FilteredStrategy(OneOfStrategy([s.filter(condition) for s in self.original_strategies]), conditions=())


@overload
def one_of(__args: object) -> Any: ...


@overload
def one_of(__a1: object) -> Any: ...


@overload
def one_of(__a1: object, __a2: object) -> Any: ...


@overload
def one_of(__a1: object, __a2: object, __a3: object) -> Any: ...


@overload
def one_of(__a1: object, __a2: object, __a3: object, __a4: object) -> Any: ...


@overload
def one_of(__a1: object, __a2: object, __a3: object, __a4: object, __a5: object) -> Any: ...


@overload
def one_of(*args: object) -> Any: ...


@defines_strategy(never_lazy=True)
def one_of(*args: object) -> SearchStrategy[Any]:
    """Return a strategy which generates values from any of the argument strategies."""
    if len(args) == 1 and (not isinstance(args[0], SearchStrategy)):
        try:
            args = tuple(args[0])  # type: ignore
        except TypeError:
            pass
    if len(args) == 1 and isinstance(args[0], SearchStrategy):
        return args[0]
    if args and (not any((isinstance(a, SearchStrategy) for a in args))):
        raise InvalidArgument(
            f'Did you mean st.sampled_from({list(args)!r})?  st.one_of() is used to combine strategies, but all of the arguments were of other types.'
        )
    args_seq: TypingSequence[SearchStrategy[Any]] = cast(TypingSequence[SearchStrategy[Any]], args)
    return OneOfStrategy(args_seq)


class MappedStrategy(SearchStrategy[MappedTo], Generic[MappedFrom, MappedTo]):
    """A strategy defined by applying a function to another strategy."""
    def __init__(self, strategy: SearchStrategy[MappedFrom], pack: Callable[[MappedFrom], MappedTo]) -> None:
        super().__init__()
        self.mapped_strategy: SearchStrategy[MappedFrom] = strategy
        self.pack: Callable[[MappedFrom], MappedTo] = pack

    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return recur(self.mapped_strategy)

    def calc_is_cacheable(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return recur(self.mapped_strategy)

    def __repr__(self) -> str:
        if not hasattr(self, '_cached_repr'):
            self._cached_repr = f'{self.mapped_strategy!r}.map({get_pretty_function_description(self.pack)})'
        return self._cached_repr

    def do_validate(self) -> None:
        self.mapped_strategy.validate()

    def do_draw(self, data: ConjectureData) -> MappedTo:
        with warnings.catch_warnings():
            if isinstance(self.pack, type) and issubclass(self.pack, (abc.Mapping, abc.Set)):
                warnings.simplefilter('ignore', BytesWarning)
            for _ in range(3):
                try:
                    data.start_example(MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL)
                    x: MappedFrom = data.draw(self.mapped_strategy)
                    result: MappedTo = self.pack(x)
                    data.stop_example()
                    current_build_context().record_call(result, self.pack, [x], {})
                    return result
                except UnsatisfiedAssumption:
                    data.stop_example(discard=True)
        raise UnsatisfiedAssumption

    @property
    def branches(self) -> list[SearchStrategy[MappedTo]]:
        return [MappedStrategy(strategy, pack=self.pack) for strategy in self.mapped_strategy.branches]

    def filter(self, condition: Callable[[MappedTo], object]) -> "SearchStrategy[MappedTo]":
        ListStrategyType = _list_strategy_type()
        if not isinstance(self.mapped_strategy, ListStrategyType) or not (
            isinstance(self.pack, type) and issubclass(self.pack, abc.Collection) or self.pack in _collection_ish_functions()
        ):
            return super().filter(condition)
        new = ListStrategyType.filter(self.mapped_strategy, condition)  # type: ignore
        if getattr(new, 'filtered_strategy', None) is self.mapped_strategy:
            return super().filter(condition)
        return FilteredStrategy(type(self)(new, self.pack), conditions=(condition,))


@lru_cache(maxsize=None)
def _list_strategy_type() -> Type[Any]:
    from hypothesis.strategies._internal.collections import ListStrategy
    return ListStrategy


def _collection_ish_functions() -> list[Callable[..., Any]]:
    funcs: list[Callable[..., Any]] = [sorted]
    if (np := sys.modules.get('numpy')):
        funcs += [
            np.empty_like, np.eye, np.identity, np.ones_like, np.zeros_like, np.array,
            np.asarray, np.asanyarray, np.ascontiguousarray, np.asmatrix, np.copy,
            np.rec.array, np.rec.fromarrays, np.rec.fromrecords, np.diag,
            np.asarray_chkfinite, np.asfortranarray
        ]
    return funcs


filter_not_satisfied: UniqueIdentifier = UniqueIdentifier('filter not satisfied')


class FilteredStrategy(SearchStrategy[ExType]):
    def __init__(self, strategy: SearchStrategy[ExType], conditions: tuple[Callable[[ExType], object], ...]) -> None:
        super().__init__()
        if isinstance(strategy, FilteredStrategy):
            self.flat_conditions: tuple[Callable[[ExType], object], ...] = strategy.flat_conditions + conditions  # type: ignore
            self.filtered_strategy: SearchStrategy[ExType] = strategy.filtered_strategy  # type: ignore
        else:
            self.flat_conditions = conditions
            self.filtered_strategy = strategy
        assert isinstance(self.flat_conditions, tuple)
        assert not isinstance(self.filtered_strategy, FilteredStrategy)
        self.__condition: Optional[Callable[[ExType], bool]] = None

    def calc_is_empty(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return recur(self.filtered_strategy)

    def calc_is_cacheable(self, recur: Callable[[SearchStrategy[Any]], bool]) -> bool:
        return recur(self.filtered_strategy)

    def __repr__(self) -> str:
        if not hasattr(self, '_cached_repr'):
            self._cached_repr = '{!r}{}'.format(
                self.filtered_strategy,
                ''.join((f'.filter({get_pretty_function_description(cond)})' for cond in self.flat_conditions))
            )
        return self._cached_repr

    def do_validate(self) -> None:
        self.filtered_strategy.validate()
        fresh: SearchStrategy[ExType] = self.filtered_strategy
        for cond in self.flat_conditions:
            fresh = fresh.filter(cond)  # type: ignore
        if isinstance(fresh, FilteredStrategy):
            FilteredStrategy.__init__(self, fresh.filtered_strategy, fresh.flat_conditions)  # type: ignore
        else:
            FilteredStrategy.__init__(self, fresh, ())

    def filter(self, condition: Callable[[ExType], object]) -> "FilteredStrategy[ExType]":
        out: SearchStrategy[ExType] = self.filtered_strategy.filter(condition)
        if isinstance(out, FilteredStrategy):
            return FilteredStrategy(out.filtered_strategy, self.flat_conditions + out.flat_conditions)  # type: ignore
        return FilteredStrategy(out, self.flat_conditions)

    @property
    def condition(self) -> Callable[[ExType], bool]:
        if self.__condition is None:
            if len(self.flat_conditions) == 1:
                self.__condition = self.flat_conditions[0]  # type: ignore
            elif len(self.flat_conditions) == 0:
                self.__condition = lambda _: True
            else:
                self.__condition = lambda x: all((cond(x) for cond in self.flat_conditions))
        return self.__condition

    def do_draw(self, data: ConjectureData) -> ExType:
        result: Any = self.do_filtered_draw(data)
        if result is not filter_not_satisfied:
            return cast(ExType, result)
        data.mark_invalid(f'Aborted test because unable to satisfy {self!r}')

    def do_filtered_draw(self, data: ConjectureData) -> Any:
        for i in range(3):
            data.start_example(FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL)
            value: Any = data.draw(self.filtered_strategy)
            if self.condition(value):
                data.stop_example()
                return value
            else:
                data.stop_example(discard=True)
                if i == 0:
                    data.events[f'Retried draw from {self!r} to satisfy filter'] = ''
        return filter_not_satisfied

    @property
    def branches(self) -> list[SearchStrategy[ExType]]:
        return [FilteredStrategy(strategy=strategy, conditions=self.flat_conditions) for strategy in self.filtered_strategy.branches]
