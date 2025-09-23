import math
from collections.abc import Generator
from random import Random
from typing import TYPE_CHECKING, AbstractSet, Dict, Final, List, Optional, Set, Tuple, Union, cast
import attr
from hypothesis.errors import FlakyReplay, FlakyStrategyDefinition, HypothesisException, StopTest
from hypothesis.internal import floats as flt
from hypothesis.internal.conjecture.choice import BooleanKWargs, BytesKWargs, ChoiceKwargsT, ChoiceT, ChoiceTypeT, FloatKWargs, IntegerKWargs, StringKWargs, choice_from_index
from hypothesis.internal.conjecture.data import ConjectureData, DataObserver, Status
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.floats import count_between_floats, float_to_int, int_to_float, sign_aware_lte
if TYPE_CHECKING:
    from typing import TypeAlias
    from hypothesis.vendor.pretty import RepresentationPrinter
ChildrenCacheValueT = Tuple[Generator[ChoiceT, None, None], List[ChoiceT], Set[ChoiceT]]

class PreviouslyUnseenBehaviour(HypothesisException):
    pass
_FLAKY_STRAT_MSG = 'Inconsistent data generation! Data generation behaved differently between different runs. Is your data generation depending on external state?'
EMPTY: Final[frozenset[int]] = frozenset()

@attr.s(slots=True)
class Killed:
    """Represents a transition to part of the tree which has been marked as
    "killed", meaning we want to treat it as not worth exploring, so it will
    be treated as if it were completely explored for the purposes of
    exhaustion."""
    next_node: 'TreeNode' = attr.ib()

    def _repr_pretty_(self, p: 'RepresentationPrinter', cycle: bool) -> None:
        assert cycle is False
        p.text('Killed')

def _node_pretty(choice_type: ChoiceTypeT, value: ChoiceT, kwargs: ChoiceKwargsT, *, forced: bool) -> str:
    forced_marker = ' [forced]' if forced else ''
    return f'{choice_type} {value!r}{forced_marker} {kwargs}'

@attr.s(slots=True)
class Branch:
    """Represents a transition where multiple choices can be made as to what
    to drawn."""
    kwargs: ChoiceKwargsT = attr.ib()
    choice_type: ChoiceTypeT = attr.ib()
    children: Dict[ChoiceT, 'TreeNode'] = attr.ib(repr=False)

    @property
    def max_children(self) -> int:
        max_children = compute_max_children(self.choice_type, self.kwargs)
        assert max_children > 0
        return max_children

    def _repr_pretty_(self, p: 'RepresentationPrinter', cycle: bool) -> None:
        assert cycle is False
        for (i, (value, child)) in enumerate(self.children.items()):
            if i > 0:
                p.break_()
            p.text(_node_pretty(self.choice_type, value, self.kwargs, forced=False))
            with p.indent(2):
                p.break_()
                p.pretty(child)

@attr.s(slots=True, frozen=True)
class Conclusion:
    """Represents a transition to a finished state."""
    status: Status = attr.ib()
    interesting_origin: Optional[InterestingOrigin] = attr.ib()

    def _repr_pretty_(self, p: 'RepresentationPrinter', cycle: bool) -> None:
        assert cycle is False
        o = self.interesting_origin
        origin = '' if o is None else f', {o.exc_type.__name__} at {o.filename}:{o.lineno}'
        p.text(f'Conclusion ({self.status!r}{origin})')
MAX_CHILDREN_EFFECTIVELY_INFINITE: Final[int] = 100000

def _count_distinct_strings(*, alphabet_size: int, min_size: int, max_size: int) -> int:
    definitely_too_large = max_size * math.log(alphabet_size) > math.log(MAX_CHILDREN_EFFECTIVELY_INFINITE)
    if definitely_too_large:
        return MAX_CHILDREN_EFFECTIVELY_INFINITE
    return sum((alphabet_size ** k for k in range(min_size, max_size + 1)))

def compute_max_children(choice_type: ChoiceTypeT, kwargs: ChoiceKwargsT) -> int:
    if choice_type == 'integer':
        kwargs = cast(IntegerKWargs, kwargs)
        min_value = kwargs['min_value']
        max_value = kwargs['max_value']
        if min_value is None and max_value is None:
            return 2 ** 128 - 1
        if min_value is not None and max_value is not None:
            return max_value - min_value + 1
        assert (min_value is None) ^ (max_value is None)
        return 2 ** 127
    elif choice_type == 'boolean':
        kwargs = cast(BooleanKWargs, kwargs)
        p = kwargs['p']
        if p <= 2 ** (-64) or p >= 1 - 2 ** (-64):
            return 1
        return 2
    elif choice_type == 'bytes':
        kwargs = cast(BytesKWargs, kwargs)
        return _count_distinct_strings(alphabet_size=2 ** 8, min_size=kwargs['min_size'], max_size=kwargs['max_size'])
    elif choice_type == 'string':
        kwargs = cast(StringKWargs, kwargs)
        min_size = kwargs['min_size']
        max_size = kwargs['max_size']
        intervals = kwargs['intervals']
        if len(intervals) == 0:
            return 1
        if len(intervals) == 1 and max_size > MAX_CHILDREN_EFFECTIVELY_INFINITE:
            return MAX_CHILDREN_EFFECTIVELY_INFINITE
        return _count_distinct_strings(alphabet_size=len(intervals), min_size=min_size, max_size=max_size)
    elif choice_type == 'float':
        kwargs = cast(FloatKWargs, kwargs)
        min_value_f = kwargs['min_value']
        max_value_f = kwargs['max_value']
        smallest_nonzero_magnitude = kwargs['smallest_nonzero_magnitude']
        count = count_between_floats(min_value_f, max_value_f)
        min_point = max(min_value_f, -flt.next_down(smallest_nonzero_magnitude))
        max_point = min(max_value_f, flt.next_down(smallest_nonzero_magnitude))
        if min_point > max_point:
            return count
        count -= count_between_floats(min_point, max_point)
        if sign_aware_lte(min_value_f, -0.0) and sign_aware_lte(-0.0, max_value_f):
            count += 1
        if sign_aware_lte(min_value_f, 0.0) and sign_aware_lte(0.0, max_value_f):
            count += 1
        return count
    raise NotImplementedError(f'unhandled choice_type {choice_type}')

def _floats_between(a: float, b: float) -> Generator[float, None, None]:
    for n in range(float_to_int(a), float_to_int(b) + 1):
        yield int_to_float(n)

def all_children(choice_type: ChoiceTypeT, kwargs: ChoiceKwargsT) -> Generator[ChoiceT, None, None]:
    if choice_type != 'float':
        for index in range(compute_max_children(choice_type, kwargs)):
            yield choice_from_index(index, choice_type, kwargs)
    else:
        kwargs = cast(FloatKWargs, kwargs)
        min_value = kwargs['min_value']
        max_value = kwargs['max_value']
        smallest_nonzero_magnitude = kwargs['smallest_nonzero_magnitude']
        if sign_aware_lte(min_value, -0.0) and sign_aware_lte(-0.0, max_value):
            yield (-0.0)
        if sign_aware_lte(min_value, 0.0) and sign_aware_lte(0.0, max_value):
            yield 0.0
        if flt.is_negative(min_value):
            if flt.is_negative(max_value):
                max_point = min(max_value, -smallest_nonzero_magnitude)
                yield from _floats_between(max_point, min_value)
            else:
                yield from _floats_between(-smallest_nonzero_magnitude, min_value)
                yield from _floats_between(smallest_nonzero_magnitude, max_value)
        else:
            min_point = max(min_value, smallest_nonzero_magnitude)
            yield from _floats_between(min_point, max_value)

@attr.s(slots=True)
class TreeNode:
    """
    A node, or collection of directly descended nodes, in a DataTree.

    We store the DataTree as a radix tree (https://en.wikipedia.org/wiki/Radix_tree),
    which means that nodes that are the only child of their parent are collapsed
    into their parent to save space.

    Conceptually, you can unfold a single TreeNode storing n values in its lists
    into a sequence of n nodes, each a child of the last. In other words,
    (kwargs[i], values[i], choice_types[i]) corresponds to the single node at index
    i.

    Note that if a TreeNode represents a choice (i.e. the nodes cannot be compacted
    via the radix tree definition), then its lists will be empty and it will
    store a `Branch` representing that choce in its `transition`.

    Examples
    --------

    Consider sequentially drawing a boolean, then an integer.

            data.draw_boolean()
            data.draw_integer(1, 3)

    If we draw True and then 2, the tree may conceptually look like this.

                      ┌──────┐
                      │ root │
                      └──┬───┘
                      ┌──┴───┐
                      │ True │
                      └──┬───┘
                      ┌──┴───┐
                      │  2   │
                      └──────┘

    But since 2 is the only child of True, we will compact these nodes and store
    them as a single TreeNode.

                      ┌──────┐
                      │ root │
                      └──┬───┘
                    ┌────┴──────┐
                    │ [True, 2] │
                    └───────────┘

    If we then draw True and then 3, True will have multiple children and we
    can no longer store this compacted representation. We would call split_at(0)
    on the [True, 2] node to indicate that we need to add a choice at 0-index
    node (True).

                      ┌──────┐
                      │ root │
                      └──┬───┘
                      ┌──┴───┐
                    ┌─┤ True ├─┐
                    │ └──────┘ │
                  ┌─┴─┐      ┌─┴─┐
                  │ 2 │      │ 3 │
                  └───┘      └───┘
    """
    kwargs: List[ChoiceKwargsT] = attr.ib(factory=list)
    values: List[ChoiceT] = attr.ib(factory=list)
    choice_types: List[ChoiceTypeT] = attr.ib(factory=list)
    __forced: Optional[Set[int]] = attr.ib(default=None, init=False)
    transition: Union[None, Branch, Conclusion, Killed] = attr.ib(default=None)
    is_exhausted: bool = attr.ib(default=False, init=False)

    @property
    def forced(self) -> AbstractSet[int]:
        if not self.__forced:
            return EMPTY
        return self.__forced

    def mark_forced(self, i: int) -> None:
        """
        Note that the draw at node i was forced.
        """
        assert 0 <= i < len(self.values)
        if self.__forced is None:
            self.__forced = set()
        self.__forced.add(i)

    def split_at(self, i: int) -> None:
        """
        Splits the tree so that it can incorporate a decision at the draw call
        corresponding to the node at position i.

        Raises FlakyStrategyDefinition if node i was forced.
        """
        if i in self.forced:
            raise FlakyStrategyDefinition(_FLAKY_STRAT_MSG)
        assert not self.is_exhausted
        key = self.values[i]
        child = TreeNode(choice_types=self.choice_types[i + 1:], kwargs=self.kwargs[i + 1:], values=self.values[i + 1:], transition=self.transition)
        self.transition = Branch(kwargs=self.kwargs[i], choice_type=self.choice_types[i], children={key: child})
        if self.__forced is not None:
            child.__forced = {j - i - 1 for j in self.__forced if j > i}
            self.__forced = {j for j in self.__forced if j < i}
        child.check_exhausted()
        del self.choice_types[i:]
        del self.values[i:]
        del self.kwargs[i:]
        assert len(self.values) == len(self.kwargs) == len(self.choice_types) == i

    def check_exhausted(self) -> bool:
        """
        Recalculates is_exhausted if necessary, and then returns it.

        A node is exhausted if:
        - Its transition is Conclusion or Killed
        - It has the maximum number of children (i.e. we have found all of its
          possible children), and all its children are exhausted

        Therefore, we only need to compute this for a node when:
        - We first create it in split_at
        - We set its transition to either Conclusion or Killed
          (TreeRecordingObserver.conclude_test or TreeRecordingObserver.kill_branch)
        - We exhaust any of its children
        """
        if not self.is_exhausted and self.transition is not None and (len(self.forced) == len(self.values)):
            if isinstance(self.transition, (Conclusion, Killed)):
                self.is_exhausted = True
            elif len(self.transition.children) == self.transition.max_children:
                self.is_exhausted = all((v.is_exhausted for v in self.transition.children.values()))
        return self.is_exhausted

    def _repr_pretty_(self, p: 'RepresentationPrinter', cycle: bool) -> None:
        assert cycle is False
        indent = 0
        for (i, (choice_type, kwargs, value)) in enumerate(zip(self.choice_types, self.kwargs, self.values)):
            with p.indent(indent):
                if i > 0:
                    p.break_()
                p.text(_node_pretty(choice_type, value, kwargs, forced=i in self.forced))
            indent += 2
        with p.indent(indent):
            if len(self.values) > 0:
                p.break_()
            if self.transition is not None:
                p.pretty(self.transition)
            else:
                p.text('unknown')

class DataTree:
    """
    A DataTree tracks the structured history of draws in some test function,
    across multiple ConjectureData objects.

    This information is used by ConjectureRunner to generate novel prefixes of
    this tree (see generate_novel_prefix). A novel prefix is a sequence of draws
    which the tree has not seen before, and therefore the ConjectureRunner has
    not generated as an input to the test function before.

    DataTree tracks the following:

    - Drawn choices in the typed choice sequence
      - ConjectureData.draw_integer()
      - ConjectureData.draw_float()
      - ConjectureData.draw_string()
      - ConjectureData.draw_boolean()
      - ConjectureData.draw_bytes()
    - Test conclusions (with some Status, e.g. Status.VALID)
      - ConjectureData.conclude_test()

    A DataTree is — surprise — a *tree*. A node in this tree is either a choice draw
    with some value, a test conclusion with some Status, or a special `Killed` value,
    which denotes that further draws may exist beyond this node but should not be
    considered worth exploring when generating novel prefixes. A node is a leaf
    iff it is a conclusion or Killed.

    A branch from node A to node B indicates that we have previously seen some
    sequence (a, b) of draws, where a and b are the values in nodes A and B.
    Similar intuition holds for conclusion and Killed nodes.

    Examples
    --------

    To see how a DataTree gets built through successive sets of draws, consider
    the following code that calls through to some ConjecutreData object `data`.
    The first call can be either True or False, and the second call can be any
    integer in the range [1, 3].

        data.draw_boolean()
        data.draw_integer(1, 3)

    To start, the corresponding DataTree object is completely empty.

                      ┌──────┐
                      │ root │
                      └──────┘

    We happen to draw True and then 2 in the above code. The tree tracks this.
    (2 also connects to a child Conclusion node with Status.VALID since it's the
    final draw in the code. I'll omit Conclusion nodes in diagrams for brevity.)

                      ┌──────┐
                      │ root │
                      └──┬───┘
                      ┌──┴───┐
                      │ True │
                      └──┬───┘
                      ┌──┴───┐
                      │  2   │
                      └──────┘

    This is a very boring tree so far! But now we happen to draw False and
    then 1. This causes a split in the tree. Remember, DataTree tracks history
    over all invocations of a function, not just one. The end goal is to know
    what invocations haven't been tried yet, after all.

                      ┌──────┐
                  ┌───┤ root ├───┐
                  │   └──────┘   │
               ┌──┴───┐        ┌─┴─────┐
               │ True │        │ False │
               └──┬───┘