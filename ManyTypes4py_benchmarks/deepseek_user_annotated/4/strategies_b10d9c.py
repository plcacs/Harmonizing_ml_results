# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
    Dict,
    FrozenSet,
    Generic,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
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
    from typing import TypeAlias

    Ex = TypeVar("Ex", covariant=True, default=Any)
else:
    Ex = TypeVar("Ex", covariant=True)

T = TypeVar("T")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
MappedFrom = TypeVar("MappedFrom")
MappedTo = TypeVar("MappedTo")
RecurT: "TypeAlias" = Callable[["SearchStrategy"], Any]
# These PackT and PredicateT aliases can only be used when you don't want to
# specify a relationship between the generic Ts and some other function param
# / return value. If you do - like the actual map definition in SearchStrategy -
# you'll need to write Callable[[Ex], T] (replacing Ex/T as appropriate) instead.
# TypeAlias is *not* simply a macro that inserts the text. it has different semantics.
PackT: "TypeAlias" = Callable[[T], T3]
PredicateT: "TypeAlias" = Callable[[T], object]
TransformationsT: "TypeAlias" = Tuple[
    Union[Tuple[Literal["filter"], PredicateT], Tuple[Literal["map"], PackT]], ...
]

calculating = UniqueIdentifier("calculating")

MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    "another attempted draw in MappedStrategy"
)

FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    "single loop iteration in FilteredStrategy"
)


def recursive_property(strategy: "SearchStrategy", name: str, default: object) -> Any:
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

    def forced_value(target: "SearchStrategy") -> Any:
        try:
            return getattr(target, force_key)
        except AttributeError:
            return getattr(target, cache_key)

    try:
        return forced_value(strategy)
    except AttributeError:
        pass

    mapping: Dict["SearchStrategy", Any] = {}
    sentinel = object()
    hit_recursion = False

    # For a first pass we do a direct recursive calculation of the
    # property, but we block recursively visiting a value in the
    # computation of its property: When that happens, we simply
    # note that it happened and return the default value.
    def recur(strat: "SearchStrategy") -> Any:
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

    # If we hit self-recursion in the computation of any strategy
    # value, our mapping at the end is imprecise - it may or may
    # not have the right values in it. We now need to proceed with
    # a more careful fixed point calculation to get the exact
    # values. Hopefully our mapping is still pretty good and it
    # won't take a large number of updates to reach a fixed point.
    if hit_recursion:
        needs_update = set(mapping)

        # We track which strategies use which in the course of
        # calculating their property value. If A ever uses B in
        # the course of calculating its value, then whenever the
        # value of B changes we might need to update the value of
        # A.
        listeners: Dict["SearchStrategy", Set["SearchStrategy"]] = defaultdict(set)
    else:
        needs_update = None

    def recur2(strat: "SearchStrategy") -> Any:
        def recur_inner(other: "SearchStrategy") -> Any:
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
    seen: Set[FrozenSet[Tuple["SearchStrategy", Any]] = set()
    while needs_update:
        count += 1
        # If we seem to be taking a really long time to stabilize we
        # start tracking seen values to attempt to detect an infinite
        # loop. This should be impossible, and most code will never
        # hit the count, but having an assertion for it means that
        # testing is easier to debug and we don't just have a hung
        # test.
        # Note: This is actually covered, by test_very_deep_deferral
        # in tests/cover/test_deferred_strategies.py. Unfortunately it
        # runs into a coverage bug. See
        # https://github.com/nedbat/coveragepy/issues/605
        # for details.
        if count > 50:  # pragma: no cover
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

    # We now have a complete and accurate calculation of the
    # property values for everything we have seen in the course of
    # running this calculation. We simultaneously update all of
    # them (not just the strategy we started out with).
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
    __label: Union[int, UniqueIdentifier, None] = None
    __module__: ClassVar[str] = "hypothesis.strategies"

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
    def is_empty(self) -> Any:
        # Returns True if this strategy can never draw a value and will always
        # result in the data being marked invalid.
        # The fact that this returns False does not guarantee that a valid value
        # can be drawn - this is not intended to be perfect, and is primarily
        # intended to be an optimisation for some cases.
        return recursive_property(self, "is_empty", True)

    # Returns True if values from this strategy can safely be reused without
    # this causing unexpected behaviour.

    # True if values from this strategy can be implicitly reused (e.g. as
    # background values in a numpy array) without causing surprising
    # user-visible behaviour. Should be false for built-in strategies that
    # produce mutable values, and for strategies that have been mapped/filtered
    # by arbitrary user-provided functions.
    @property
    def has_reusable_values(self) -> Any:
        return recursive_property(self, "has_reusable_values", True)

    # Whether this strategy is suitable for holding onto in a cache.
    @property
    def is_cacheable(self) -> Any:
        return recursive_property(self, "is_cacheable", True)

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return True

    def calc_is_empty(self, recur: RecurT) -> bool:
        # Note: It is correct and significant that the default return value
        # from calc_is_empty is False despite the default value for is_empty
        # being true. The reason for this is that strategies should be treated
        # as empty absent evidence to the contrary, but most basic strategies
        # are trivially non-empty and it would be annoying to have to override
        # this method to show that.
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
        if getattr(sys, "ps1", None) is None:  # pragma: no branch
            # The other branch *is* covered in cover/test_examples.py; but as that
            # uses `pexpect` for an interactive session `coverage` doesn't see it.
            warnings.warn(
                "The `.example()` method is good for exploring strategies, but should "
                "only be used interactively.  We recommend using `@given` for tests - "
                "it performs better, saves and replays failures to avoid flakiness, "
                "and reports minimal examples. (strategy: %r)" % (self,),
                NonInteractiveExampleWarning,
                stacklevel=2,
            )

        context = _current_build_context.value
        if context is not None:
            if context.data is not None and context.data.depth > 0:
                raise HypothesisException(
                    "Using example() inside a strategy definition is a bad "
                    "idea. Instead consider using hypothesis.strategies.builds() "
                    "or @hypothesis.strategies.composite to define your strategy."
                    " See https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#hypothesis.strategies.builds or "
                    "https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#composite-strategies for more details."
                )
            else:
                raise HypothesisException(
                    "Using example() inside a test function is a bad "
                    "idea. Instead consider using hypothesis.strategies.data() "
                    "to draw more examples during testing. See "
                    "https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#drawing-interactively-in-tests for more details."
                )

        try:
            return self.__examples.pop()
        except (AttributeError, IndexError):
            self.__examples: List[Ex] = []

        from hypothesis.core import given

        # Note: this function has a weird name because it might appear in
        # tracebacks, and we want users to know that they can ignore it.
        @given(self)
        @settings(
            database=None,
            # generate only a few examples at a time to avoid slow interactivity
            # for large strategies. The overhead of @given is very small relative
            # to generation, so a small batch size is fine.
            max_examples=10,
            deadline=None,
            verbosity=Verbosity.quiet,
            phases=(Phase.generate,),
            suppress_health_check=list(HealthCheck),
        )
        def example_generating_inner_function(
            ex: Ex,  # type: ignore # mypy is overzealous in preventing covariant params
        ) -> None:
            self.__examples.append(ex)

        example_generating_inner_function()
        shuffle(self.__examples)
        return self.__examples.pop()

    def map(self, pack: Callable[[Ex], T]) -> "SearchStrategy[T]":
        """Returns a new strategy that generates values by generating a value
        from this strategy and then calling pack() on the result, giving that.

        This method is part of the public API.
        """
        if is_identity_function(pack):
            return self  # type: ignore  # Mypy has no way to know that `Ex == T`
        return MappedStrategy(self, pack=pack)

    def flatmap(
        self, expand: Callable[[Ex], "SearchStrategy[T]"]
    ) -> "SearchStrategy[T]":
        """Returns a new strategy that generates values by generating a value
        from this strategy, say x, then generating a value from
        strategy(expand(x))

        This method is part of the public API.
        """
        from hypothesis.strategies._internal.flatmapped import FlatMapStrategy

        return FlatMapStrategy(expand=expand, strategy=self)

    def filter(self, condition: PredicateT) -> "SearchStrategy[Ex]":
        """Returns a new strategy that generates values from this strategy
        which satisfy the provided condition. Note that if the condition is too
        hard to satisfy this might result in your tests failing with
        Unsatisfiable.

        This method is part of the public API.
        """
        return FilteredStrategy(conditions=(condition,), strategy=self)

    def _filter_for_filtered_draw(self, condition: PredicateT) -> "SearchStrategy[Ex]":
        # Hook for parent strategies that want to perform fallible filtering
        # on one of their internal strategies (e.g. UniqueListStrategy).
        # The returned object must have a `.do_filtered_draw(data)` method
        # that behaves like `do_draw`, but returns the sentinel object
        # `filter_not_satisfied` if the condition could not be satisfied.

        # This is separate from the main `filter` method so that strategies
        # can override `filter` without having to also guarantee a
        # `do_filtered_draw` method.
        return FilteredStrategy(conditions=(condition,), strategy=self)

    @property
    def branches(self) -> Sequence["SearchStrategy[Ex]"]:
        return [self]

    def __or__(self, other: "SearchStrategy[T]") -> "SearchStrategy[Union[Ex, T]]":
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
            self.is_empty
            self.has_reusable_values
        except Exception:
            self.validate_called = False
            raise

    LABELS: ClassVar[