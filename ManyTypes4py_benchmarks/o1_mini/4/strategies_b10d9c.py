import sys
import warnings
from collections import abc, defaultdict
from collections.abc import Sequence
from functools import lru_cache
from random import shuffle
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
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
from hypothesis.internal.conjecture.utils import (
    calc_label_from_cls,
    calc_label_from_name,
    combine_labels,
)
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import (
    get_pretty_function_description,
    is_identity_function,
)
from hypothesis.strategies._internal.utils import defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier

if TYPE_CHECKING:
    ExAlias: TypeAlias = TypeVar("Ex", covariant=True, bound=Any)
    RecurT = Callable[["SearchStrategy[Any]"], Any]
    PackT = Callable[[Any], Any]
    PredicateT = Callable[[Any], object]
    TransformationsT = Tuple[
        Union[Tuple[Literal["filter"], PredicateT], Tuple[Literal["map"], PackT]],
        ...
    ]
else:
    ExAlias = TypeVar("Ex", covariant=True)

Ex: TypeVar = ExAlias
T = TypeVar("T")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
MappedFrom = TypeVar("MappedFrom")
MappedTo = TypeVar("MappedTo")
RecurT = Callable[["SearchStrategy[Any]"], Any]
PackT = Callable[[T], T3]
PredicateT = Callable[[T], object]
TransformationsT = Tuple[
    Union[Tuple[Literal["filter"], PredicateT], Tuple[Literal["map"], PackT]],
    ...
]

calculating: UniqueIdentifier = UniqueIdentifier("calculating")
MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL: Any = calc_label_from_name(
    "another attempted draw in MappedStrategy"
)
FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL: Any = calc_label_from_name(
    "single loop iteration in FilteredStrategy"
)


def recursive_property(
    strategy: "SearchStrategy[Any]", name: str, default: Any
) -> Any:
    """Handle properties which may be mutually recursive among a set of
    strategies.

    These are essentially lazily cached properties, with the ability to set
    an override: If the property has not been explicitly set, we calculate
    it on first access and memoize the result for later.

    The problem is that for properties that depend on each other, a naive
    calculation strategy may hit infinite recursion. Consider for example
    the property is_empty. A strategy defined as x = st.deferred(lambda: x)
    is certainly empty (in order to draw a value from x we would have to
    draw a value from x, for which we would have to draw a value from x,
    ...), but in order to calculate it the naive approach would end up
    calling x.is_empty in order to calculate x.is_empty in order to etc.

    The solution is one of fixed point calculation. We start with a default
    value that is the value of the property in the absence of evidence to
    the contrary, and then update the values of the property for all
    dependent strategies until we reach a fixed point.

    The approach taken roughly follows that in section 4.2 of Adams,
    Michael D., Celeste Hollenbeck, and Matthew Might. "On the complexity
    and performance of parsing with derivatives." ACM SIGPLAN Notices 51.6
    (2016): 224-236.
    """
    cache_key = "cached_" + name
    calculation = "calc_" + name
    force_key = "force_" + name

    def forced_value(target: Any) -> Any:
        try:
            return getattr(target, force_key)
        except AttributeError:
            return getattr(target, cache_key)

    try:
        return forced_value(strategy)
    except AttributeError:
        pass
    mapping: Dict["SearchStrategy[Any]", Any] = {}
    sentinel = object()
    hit_recursion = False

    def recur(strat: "SearchStrategy[Any]") -> Any:
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
    if hit_recursion:
        needs_update: Optional[Set["SearchStrategy[Any]"]] = set(mapping)
        listeners: DefaultDict["SearchStrategy[Any]", Set["SearchStrategy[Any]"]] = defaultdict(set)
    else:
        needs_update = None

    def recur2(strat: "SearchStrategy[Any]") -> Callable[[ "SearchStrategy[Any]"], Any]:
        def recur_inner(other: "SearchStrategy[Any]") -> Any:
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
    seen: Set[FrozenSet[Tuple["SearchStrategy[Any]", Any]]] = set()
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
                needs_update.update(list(listeners[strat]))
                mapping[strat] = new_value
    for k, v in mapping.items():
        setattr(k, cache_key, v)
    return getattr(strategy, cache_key)


class SearchStrategy(Generic[Ex]):
    """A SearchStrategy is an object that knows how to explore data of a given
    type.

    Except where noted otherwise, methods on this class are not part of
    the public API and their behaviour may change significantly between
    minor version releases. They will generally be stable between patch
    releases.
    """

    supports_find: ClassVar[bool] = True
    validate_called: bool = False
    __label: Optional[int] = None
    __module__: str = "hypothesis.strategies"

    def available(self, data: ConjectureData) -> bool:
        """Returns whether this strategy can *currently* draw any
        values. This typically useful for stateful testing where ``Bundle``
        grows over time a list of value to choose from.

        Unlike ``empty`` property, this method's return value may change
        over time.
        Note: ``data`` parameter will only be used for introspection and no
        value drawn from it.
        """
        return not self.is_empty

    @property
    def is_empty(self) -> bool:
        return recursive_property(self, "is_empty", True)

    @property
    def has_reusable_values(self) -> bool:
        return recursive_property(self, "has_reusable_values", True)

    @property
    def is_cacheable(self) -> bool:
        return recursive_property(self, "is_cacheable", True)

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return True

    def calc_is_empty(self, recur: RecurT) -> bool:
        return False

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        return False

    def example(self) -> Ex:
        """Provide an example of the sort of value that this strategy
        generates. This is biased to be slightly simpler than is typical for
        values from this strategy, for clarity purposes.

        This method shouldn't be taken too seriously. It's here for interactive
        exploration of the API, not for any sort of real testing.

        This method is part of the public API.
        """
        if getattr(sys, "ps1", None) is None:
            warnings.warn(
                'The `.example()` method is good for exploring strategies, but should only be used interactively.  We recommend using `@given` for tests - it performs better, saves and replays failures to avoid flakiness, and reports minimal examples. (strategy: %r)'
                % (self,),
                NonInteractiveExampleWarning,
                stacklevel=2,
            )
        context = _current_build_context.value
        if context is not None:
            if context.data is not None and context.data.depth > 0:
                raise HypothesisException(
                    "Using example() inside a strategy definition is a bad idea. Instead consider using hypothesis.strategies.builds() or @hypothesis.strategies.composite to define your strategy. See https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.builds or https://hypothesis.readthedocs.io/en/latest/data.html#composite-strategies for more details."
                )
            else:
                raise HypothesisException(
                    "Using example() inside a test function is a bad idea. Instead consider using hypothesis.strategies.data() to draw more examples during testing. See https://hypothesis.readthedocs.io/en/latest/data.html#drawing-interactively-in-tests for more details."
                )
        try:
            return self.__examples.pop()
        except (AttributeError, IndexError):
            self.__examples = []
        from hypothesis.core import given

        @given(
            self,
            settings=settings(
                database=None,
                max_examples=10,
                deadline=None,
                verbosity=Verbosity.quiet,
                phases=(Phase.generate,),
                suppress_health_check=list(HealthCheck),
            ),
        )
        def example_generating_inner_function(ex: Ex) -> None:
            self.__examples.append(ex)

        example_generating_inner_function()
        shuffle(self.__examples)
        return cast(Ex, self.__examples.pop())

    def map(self, pack: PackT[T, T3]) -> "SearchStrategy[T3]":
        """Returns a new strategy that generates values by generating a value
        from this strategy and then calling pack() on the result, giving that.

        This method is part of the public API.
        """
        if is_identity_function(pack):
            return self
        return cast("SearchStrategy[T3]", MappedStrategy[self, T3](self, pack=pack))

    def flatmap(
        self, expand: Callable[[Ex], "SearchStrategy[Any]"]
    ) -> "SearchStrategy[Any]":
        """Returns a new strategy that generates values by generating a value
        from this strategy, say x, then generating a value from
        strategy(expand(x))

        This method is part of the public API.
        """
        from hypothesis.strategies._internal.flatmapped import FlatMapStrategy

        return FlatMapStrategy(expand=expand, strategy=self)

    def filter(self, condition: PredicateT[Ex]) -> "SearchStrategy[Ex]":
        """Returns a new strategy that generates values from this strategy
        which satisfy the provided condition. Note that if the condition is too
        hard to satisfy this might result in your tests failing with
        Unsatisfiable.

        This method is part of the public API.
        """
        return FilteredStrategy(conditions=(condition,), strategy=self)

    def _filter_for_filtered_draw(
        self, condition: PredicateT[Ex]
    ) -> "SearchStrategy[Ex]":
        return FilteredStrategy(conditions=(condition,), strategy=self)

    @property
    def branches(self) -> Sequence["SearchStrategy[Any]"]:
        return [self]

    def __or__(self, other: "SearchStrategy[Any]") -> "SearchStrategy[Any]":
        """Return a strategy which produces values by randomly drawing from one
        of this strategy or the other strategy.

        This method is part of the public API.
        """
        if not isinstance(other, SearchStrategy):
            raise ValueError(f"Cannot | a SearchStrategy with {other!r}")
        return OneOfStrategy((self, other))

    def __bool__(self) -> bool:
        warnings.warn(
            f"bool({self!r}) is always True, did you mean to draw a value?",
            HypothesisWarning,
            stacklevel=2,
        )
        return True

    def validate(self) -> None:
        """Throw an exception if the strategy is not valid.

        This can happen due to lazy construction
        """
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

    LABELS: ClassVar[Dict[Type["SearchStrategy[Any]"], Any]] = {}

    @property
    def class_label(self) -> Any:
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
        raise NotImplementedError(f"{type(self).__name__}.do_draw")


def is_simple_data(value: Any) -> bool:
    try:
        hash(value)
        return True
    except TypeError:
        return False


class SampledFromStrategy(SearchStrategy[Ex]):
    """A strategy which samples from a set of elements. This is essentially
    equivalent to using a OneOfStrategy over Just strategies but may be more
    efficient and convenient.
    """

    _MAX_FILTER_CALLS: ClassVar[int] = 10000

    elements: Tuple[Any, ...]
    repr_: Optional[str]
    _transformations: TransformationsT

    def __init__(
        self,
        elements: Iterable[Any],
        repr_: Optional[str] = None,
        transformations: TransformationsT = (),
    ) -> None:
        super().__init__()
        self.elements = cu.check_sample(elements, "sampled_from")
        assert self.elements
        self.repr_ = repr_
        self._transformations = transformations

    def map(self, pack: PackT[Any, Any]) -> "SearchStrategy[Any]":
        s = type(self)(
            self.elements,
            repr_=self.repr_,
            transformations=(*self._transformations, ("map", pack)),
        )
        return cast(SearchStrategy[Any], s)

    def filter(self, condition: PredicateT[Any]) -> "SampledFromStrategy":
        return type(self)(
            self.elements,
            repr_=self.repr_,
            transformations=(*self._transformations, ("filter", condition)),
        )

    def __repr__(self) -> str:
        base = (
            self.repr_
            or "sampled_from([" + ", ".join(map(get_pretty_function_description, self.elements)) + "])"
        )
        transformations_repr = "".join(
            f".{name}({get_pretty_function_description(f)})" for name, f in self._transformations
        )
        return base + transformations_repr

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        return not self._transformations

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return is_simple_data(self.elements)

    def _transform(self, element: Any) -> Any:
        for name, f in self._transformations:
            if name == "map":
                f_func = cast(PackT[Any, Any], f)
                result = f_func(element)
                build_context = _current_build_context.value
                if build_context is not None:
                    build_context.record_call(result, f_func, [element], {})
                element = result
            else:
                assert name == "filter"
                f_func = cast(PredicateT[Any], f)
                if not f_func(element):
                    return filter_not_satisfied
        return element

    def do_draw(self, data: ConjectureData) -> Ex:
        result = self.do_filtered_draw(data)
        if (
            isinstance(result, SearchStrategy)
            and all(isinstance(x, SearchStrategy) for x in self.elements)
        ):
            data._sampled_from_all_strategies_elements_message = (
                "sample_from was given a collection of strategies: {!r}. Was one_of intended?".format(
                    self.elements
                )
            )
        if result is filter_not_satisfied:
            data.mark_invalid(f"Aborted test because unable to satisfy {self!r}")
        assert not isinstance(result, UniqueIdentifier)
        return cast(Ex, result)

    def get_element(self, i: int) -> Any:
        return self._transform(self.elements[i])

    def do_filtered_draw(self, data: ConjectureData) -> Any:
        known_bad_indices: Set[int] = set()
        for _ in range(3):
            i = data.draw_integer(0, len(self.elements) - 1)
            if i not in known_bad_indices:
                element = self.get_element(i)
                if element is not filter_not_satisfied:
                    return element
                if not known_bad_indices:
                    data.events[f"Retried draw from {self!r} to satisfy filter"] = ""
                known_bad_indices.add(i)
        max_good_indices = len(self.elements) - len(known_bad_indices)
        if not max_good_indices:
            return filter_not_satisfied
        max_good_indices = min(max_good_indices, self._MAX_FILTER_CALLS - 3)
        speculative_index = data.draw_integer(0, max_good_indices - 1)
        allowed: List[Tuple[int, Any]] = []
        for i in range(min(len(self.elements), self._MAX_FILTER_CALLS - 3)):
            if i not in known_bad_indices:
                element = self.get_element(i)
                if element is not filter_not_satisfied:
                    assert not isinstance(element, UniqueIdentifier)
                    allowed.append((i, element))
                    if len(allowed) > speculative_index:
                        data.draw_integer(0, len(self.elements) - 1, forced=i)
                        return element
        if allowed:
            i, element = data.choice(allowed)
            data.draw_integer(0, len(self.elements) - 1, forced=i)
            return element
        return filter_not_satisfied


class OneOfStrategy(SearchStrategy[Ex]):
    """Implements a union of strategies. Given a number of strategies this
    generates values which could have come from any of them.

    The conditional distribution draws uniformly at random from some
    non-empty subset of these strategies and then draws from the
    conditional distribution of that strategy.
    """

    original_strategies: List["SearchStrategy[Any]"]
    __element_strategies: Optional[List["SearchStrategy[Any]"]] = None
    __in_branches: bool = False

    def __init__(self, strategies: Iterable["SearchStrategy[Any]"]) -> None:
        super().__init__()
        self.original_strategies = list(strategies)
        self.__element_strategies = None
        self.__in_branches = False

    def calc_is_empty(self, recur: RecurT) -> bool:
        return all(recur(e) for e in self.original_strategies)

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        return all(recur(e) for e in self.original_strategies)

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return all(recur(e) for e in self.original_strategies)

    @property
    def element_strategies(self) -> List["SearchStrategy[Any]"]:
        if self.__element_strategies is None:
            seen: Set["SearchStrategy[Any]"] = {self}
            strategies: List["SearchStrategy[Any]"] = []
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
            self.class_label, *(p.label for p in self.original_strategies)
        )

    def do_draw(self, data: ConjectureData) -> Ex:
        strategy = data.draw(
            SampledFromStrategy(self.element_strategies).filter(
                lambda s: s.available(data)
            )
        )
        return data.draw(strategy)

    def __repr__(self) -> str:
        return "one_of(%s)" % ", ".join(map(repr, self.original_strategies))

    def do_validate(self) -> None:
        for e in self.element_strategies:
            e.validate()

    @property
    def branches(self) -> List["SearchStrategy[Any]"]:
        if not self.__in_branches:
            try:
                self.__in_branches = True
                return self.element_strategies
            finally:
                self.__in_branches = False
        else:
            return [self]

    def filter(self, condition: PredicateT[Ex]) -> "SearchStrategy[Ex]":
        return FilteredStrategy(
            OneOfStrategy([s.filter(condition) for s in self.original_strategies]),
            conditions=(),
        )


@overload
def one_of(__args: Iterable["SearchStrategy[Any]"]) -> "SearchStrategy[Any]":
    ...


@overload
def one_of(__a1: "SearchStrategy[Any]") -> "SearchStrategy[Any]":
    ...


@overload
def one_of(
    __a1: "SearchStrategy[Any]", __a2: "SearchStrategy[Any]"
) -> "SearchStrategy[Any]":
    ...


@overload
def one_of(
    __a1: "SearchStrategy[Any]",
    __a2: "SearchStrategy[Any]",
    __a3: "SearchStrategy[Any]",
) -> "SearchStrategy[Any]":
    ...


@overload
def one_of(
    __a1: "SearchStrategy[Any]",
    __a2: "SearchStrategy[Any]",
    __a3: "SearchStrategy[Any]",
    __a4: "SearchStrategy[Any]",
) -> "SearchStrategy[Any]":
    ...


@overload
def one_of(
    __a1: "SearchStrategy[Any]",
    __a2: "SearchStrategy[Any]",
    __a3: "SearchStrategy[Any]",
    __a4: "SearchStrategy[Any]",
    __a5: "SearchStrategy[Any]",
) -> "SearchStrategy[Any]":
    ...


@overload
def one_of(*args: "SearchStrategy[Any]") -> "SearchStrategy[Any]":
    ...


@defines_strategy(never_lazy=True)
def one_of(*args: Any) -> "SearchStrategy[Any]":
    """Return a strategy which generates values from any of the argument
    strategies.

    This may be called with one iterable argument instead of multiple
    strategy arguments, in which case ``one_of(x)`` and ``one_of(*x)`` are
    equivalent.

    Examples from this strategy will generally shrink to ones that come from
    strategies earlier in the list, then shrink according to behaviour of the
    strategy that produced them. In order to get good shrinking behaviour,
    try to put simpler strategies first. e.g. ``one_of(none(), text())`` is
    better than ``one_of(text(), none())``.

    This is especially important when using recursive strategies. e.g.
    ``x = st.deferred(lambda: st.none() | st.tuples(x, x))`` will shrink well,
    but ``x = st.deferred(lambda: st.tuples(x, x) | st.none())`` will shrink
    very badly indeed.
    """
    if len(args) == 1 and not isinstance(args[0], SearchStrategy):
        try:
            args = tuple(args[0])
        except TypeError:
            pass
    if len(args) == 1 and isinstance(args[0], SearchStrategy):
        return cast("SearchStrategy[Any]", args[0])
    if args and not any(isinstance(a, SearchStrategy) for a in args):
        raise InvalidArgument(
            f"Did you mean st.sampled_from({list(args)!r})?  st.one_of() is used to combine strategies, but all of the arguments were of other types."
        )
    args_casted = cast(Sequence[SearchStrategy[Any]], args)
    return OneOfStrategy(args_casted)


class MappedStrategy(SearchStrategy[MappedTo], Generic[MappedFrom, MappedTo]):
    """A strategy which is defined purely by conversion to and from another
    strategy.

    Its parameter and distribution come from that other strategy.
    """

    mapped_strategy: "SearchStrategy[MappedFrom]"
    pack: PackT[MappedFrom, MappedTo]

    def __init__(
        self, strategy: "SearchStrategy[MappedFrom]", pack: PackT[MappedFrom, MappedTo]
    ) -> None:
        super().__init__()
        self.mapped_strategy = strategy
        self.pack = pack

    def calc_is_empty(self, recur: RecurT) -> bool:
        return recur(self.mapped_strategy)

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return recur(self.mapped_strategy)

    def __repr__(self) -> str:
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = f"{self.mapped_strategy!r}.map({get_pretty_function_description(self.pack)})"
        return self._cached_repr

    def do_validate(self) -> None:
        self.mapped_strategy.validate()

    def do_draw(self, data: ConjectureData) -> MappedTo:
        with warnings.catch_warnings():
            if isinstance(self.pack, type) and issubclass(
                self.pack, (abc.Mapping, abc.Set)
            ):
                warnings.simplefilter("ignore", BytesWarning)
            for _ in range(3):
                try:
                    data.start_example(MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL)
                    x = data.draw(self.mapped_strategy)
                    result = self.pack(x)
                    data.stop_example()
                    current_build_context().record_call(result, self.pack, [x], {})
                    return result
                except UnsatisfiedAssumption:
                    data.stop_example(discard=True)
        raise UnsatisfiedAssumption

    @property
    def branches(self) -> List["MappedStrategy[Any, Any]"]:
        return [
            MappedStrategy(strategy=s, pack=self.pack)
            for s in self.mapped_strategy.branches
        ]

    def filter(self, condition: PredicateT[MappedTo]) -> "SearchStrategy[MappedTo]":
        ListStrategyType = _list_strategy_type()
        if (
            not isinstance(self.mapped_strategy, ListStrategyType)
            or not (
                isinstance(self.pack, type)
                and issubclass(self.pack, abc.Collection)
                or self.pack in _collection_ish_functions()
            )
        ):
            return super().filter(condition)
        new = ListStrategyType.filter(self.mapped_strategy, condition)
        if getattr(new, "filtered_strategy", None) is self.mapped_strategy:
            return super().filter(condition)
        return FilteredStrategy(
            type(self)(new, self.pack), conditions=(condition,)
        )


@lru_cache
def _list_strategy_type() -> Type["ListStrategy"]:
    from hypothesis.strategies._internal.collections import ListStrategy

    return ListStrategy


def _collection_ish_functions() -> List[Callable[..., Any]]:
    funcs: List[Callable[..., Any]] = [sorted]
    if (np := sys.modules.get("numpy")):
        funcs += [
            np.empty_like,
            np.eye,
            np.identity,
            np.ones_like,
            np.zeros_like,
            np.array,
            np.asarray,
            np.asanyarray,
            np.ascontiguousarray,
            np.asmatrix,
            np.copy,
            np.rec.array,
            np.rec.fromarrays,
            np.rec.fromrecords,
            np.diag,
            np.asarray_chkfinite,
            np.asfortranarray,
        ]
    return funcs


filter_not_satisfied: UniqueIdentifier = UniqueIdentifier("filter not satisfied")


class FilteredStrategy(SearchStrategy[Ex]):

    flat_conditions: Tuple[PredicateT[Ex], ...]
    filtered_strategy: "SearchStrategy[Any]"
    __condition: Optional[Callable[[Ex], bool]] = None

    def __init__(
        self, strategy: "SearchStrategy[Any]", conditions: Tuple[PredicateT[Ex], ...]
    ) -> None:
        super().__init__()
        if isinstance(strategy, FilteredStrategy):
            self.flat_conditions = strategy.flat_conditions + conditions
            self.filtered_strategy = strategy.filtered_strategy
        else:
            self.flat_conditions = conditions
            self.filtered_strategy = strategy
        assert isinstance(self.flat_conditions, tuple)
        assert not isinstance(self.filtered_strategy, FilteredStrategy)

    def calc_is_empty(self, recur: RecurT) -> bool:
        return recur(self.filtered_strategy)

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return recur(self.filtered_strategy)

    def __repr__(self) -> str:
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = "{!r}{}".format(
                self.filtered_strategy,
                "".join(
                    f".filter({get_pretty_function_description(cond)})"
                    for cond in self.flat_conditions
                ),
            )
        return self._cached_repr

    def do_validate(self) -> None:
        self.filtered_strategy.validate()
        fresh = self.filtered_strategy
        for cond in self.flat_conditions:
            fresh = fresh.filter(cond)
        if isinstance(fresh, FilteredStrategy):
            FilteredStrategy.__init__(
                self,
                fresh.filtered_strategy,
                fresh.flat_conditions,
            )
        else:
            FilteredStrategy.__init__(self, fresh, ())

    def filter(self, condition: PredicateT[Ex]) -> "FilteredStrategy":
        out = self.filtered_strategy.filter(condition)
        if isinstance(out, FilteredStrategy):
            return FilteredStrategy(
                out.filtered_strategy, self.flat_conditions + out.flat_conditions
            )
        return FilteredStrategy(
            out, self.flat_conditions + (condition,)
        )

    @property
    def condition(self) -> Callable[[Ex], bool]:
        if self.__condition is None:
            if len(self.flat_conditions) == 1:
                self.__condition = self.flat_conditions[0]
            elif len(self.flat_conditions) == 0:
                self.__condition = lambda _: True
            else:
                self.__condition = lambda x: all(cond(x) for cond in self.flat_conditions)
        return self.__condition

    def do_draw(self, data: ConjectureData) -> Ex:
        result = self.do_filtered_draw(data)
        if result is not filter_not_satisfied:
            return cast(Ex, result)
        data.mark_invalid(f"Aborted test because unable to satisfy {self!r}")

    def do_filtered_draw(self, data: ConjectureData) -> Any:
        for i in range(3):
            data.start_example(FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL)
            value = data.draw(self.filtered_strategy)
            if self.condition(value):
                data.stop_example()
                return value
            else:
                data.stop_example(discard=True)
                if i == 0:
                    data.events[f"Retried draw from {self!r} to satisfy filter"] = ""
        return filter_not_satisfied

    @property
    def branches(self) -> List["FilteredStrategy[Any]"]:
        return [
            FilteredStrategy(strategy=strategy, conditions=self.flat_conditions)
            for strategy in self.filtered_strategy.branches
        ]


@check_function
def check_strategy(arg: Any, name: str = "") -> None:
    assert isinstance(name, str)
    if not isinstance(arg, SearchStrategy):
        hint = ""
        if isinstance(arg, (list, tuple)):
            hint = f", such as st.sampled_from({name or '...'!r}),"
        if name:
            name += "="
        raise InvalidArgument(
            f"Expected a SearchStrategy{name} but got {arg!r} (type={type(arg).__name__}){hint}"
        )
